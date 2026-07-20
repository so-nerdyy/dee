// dee/engine.cpp
#include "dee/engine.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#ifdef DEE_CUDA
#include <cuda_runtime.h>
#include "dee/cuda_check.h"
#include "dee/swiglu_cuda.h"
#endif

namespace dee {

// ---------------------------------------------------------------------------
// static SwiGLU kernel (raw C++; CUDA/ggml slots in later)
// blob layout: [gate: INTER*HIDDEN][up: INTER*HIDDEN][down: HIDDEN*INTER]
//   gate/up are [INTER, HIDDEN] (row-major); down is [HIDDEN, INTER].
// x is the input hidden (length HIDDEN). acc is accumulated (length HIDDEN).
// ---------------------------------------------------------------------------
void Engine::swiglu(const float* blob, const float* x,
                    int inter, int hidden, float* acc) {
    const float* Wg = blob;
    const float* Wu = blob + (size_t)inter * hidden;
    const float* Wd = blob + 2 * (size_t)inter * hidden;
    for (int i = 0; i < inter; ++i) {
        float g = 0.0f, f = 0.0f;
        const float* rg = Wg + (size_t)i * hidden;
        const float* ru = Wu + (size_t)i * hidden;
        for (int j = 0; j < hidden; ++j) { g += rg[j] * x[j]; f += ru[j] * x[j]; }
        float s = g / (1.0f + std::exp(-g));   // SiLU
        float h = s * f;
        for (int o = 0; o < hidden; ++o) acc[o] += Wd[(size_t)o * inter + i] * h;
    }
}

int Engine::avail_layer(int layer) const {
    // synthetic single-layer shard exposes only layer 0; map everything to it.
    // (Real multi-layer shards return `layer` directly.)
    TensorView requested = resolver_.resolve_expert(layer, 0, TensorResolver::GATE_PROJ);
    return requested.ok() ? layer : 0;
}

const float* Engine::get_staging(int expert) {
    auto it = staging_.find(expert);
    if (it != staging_.end()) return it->second.data();

    int sl = avail_layer(0);
    TensorView gv = resolver_.resolve_expert(sl, expert, TensorResolver::GATE_PROJ);
    TensorView uv = resolver_.resolve_expert(sl, expert, TensorResolver::UP_PROJ);
    TensorView dv = resolver_.resolve_expert(sl, expert, TensorResolver::DOWN_PROJ);
    const size_t gh = (size_t)inter_ * hidden_;
    const size_t dh = (size_t)hidden_ * inter_;
    if (!gv.ok() || !uv.ok() || !dv.ok() || gv.dtype != DType::BF16 || uv.dtype != DType::BF16 || dv.dtype != DType::BF16 ||
        gv.nbytes != gh * sizeof(uint16_t) || uv.nbytes != gh * sizeof(uint16_t) || dv.nbytes != dh * sizeof(uint16_t)) {
        std::fprintf(stderr, "[engine] expert %d has unsupported or inconsistent BF16 tensor layout\n", expert);
        return nullptr;
    }

    std::vector<float> blob(blob_elems_);
    for (size_t i = 0; i < gh; ++i) blob[i] = bf16_to_f32(*(const uint16_t*)(gv.data + i * 2));
    for (size_t i = 0; i < gh; ++i) blob[gh + i] = bf16_to_f32(*(const uint16_t*)(uv.data + i * 2));
    for (size_t i = 0; i < dh; ++i) blob[2 * gh + i] = bf16_to_f32(*(const uint16_t*)(dv.data + i * 2));

    auto res = staging_.emplace(expert, std::move(blob));
    return res.first->second.data();
}

bool Engine::stage_expert(int layer, int expert, int priority) {
    const float* blob = get_staging(expert);
    if (!blob) {
        std::fprintf(stderr, "[engine] missing source weights for expert (%d,%d)\n", layer, expert);
        return false;
    }
    return prefetcher_.prefetch(layer, expert, blob, blob_bytes_, priority) >= 0;
}

void Engine::forward_layer(int layer, const float* h_in, float* h_out) {
    std::vector<int> experts;
    oracle_.predict(layer, h_in, cfg_.topk, experts);

    // CPU/mock mode processes one expert at a time. This preserves the fixed
    // budget contract even when top-K is larger than the cache capacity.
    std::vector<float> acc(hidden_, 0.0f);
    for (size_t k = 0; k < experts.size(); ++k) {
        int e = experts[k];
        if (!stage_expert(layer, e, static_cast<int>(cfg_.topk - k)) || !prefetcher_.wait(layer, e)) {
            stats_.fallbacks++;
            continue;
        }
        const void* p = cache_.data(layer, e);
        if (!p) {  // should not happen (wait guarantees resident)
            stats_.fallbacks++;
            continue;
        }
        swiglu((const float*)p, h_in, inter_, hidden_, acc.data());
        cache_.touch(layer, e);
    }
    // 3) combine (mean over top-K) then stabilize (frozen layer-norm stand-in).
    //    The mock has no residual/LN, so without this the recurrent loop
    //    diverges to Inf over 40 layers. RMS-normalizing to 1.0 per layer keeps
    //    the signal bounded and finite while preserving the DEE pipeline logic.
    float inv = (experts.empty() ? 1.0f : 1.0f / (float)experts.size());
    double ss = 0.0;
    for (int i = 0; i < hidden_; ++i) { float v = acc[i] * inv; h_out[i] = v; ss += (double)v * v; }
    double rms = std::sqrt(ss / hidden_);
    if (rms > 1e-8) {
        float s = 1.0f / (float)rms;   // rescale RMS -> 1.0
        for (int i = 0; i < hidden_; ++i) h_out[i] *= s;
    }
}

bool Engine::init(const EngineConfig& cfg) {
    cfg_ = cfg;
    if (cfg.num_tokens <= 0 || cfg.topk <= 0 || cfg.num_layers <= 0 || cfg.hidden <= 0) {
        std::fprintf(stderr, "[engine] tokens, topk, layers, and hidden must be positive\n");
        return false;
    }
    hidden_ = cfg.hidden;
    // Expert dims are taken from the SHARD (so a mock inter=64 and the real
    // inter=256 are both handled). The Oracle's own MLP width (H=256) is a
    // separate, fixed quantity passed to oracle.load().
    inter_  = 256;            // provisional; overwritten below from the shard
    blob_elems_ = 3ULL * (size_t)inter_ * hidden_;
    blob_bytes_ = blob_elems_ * sizeof(float);

    if (!std::filesystem::is_regular_file(cfg.shard_path)) {
        std::fprintf(stderr, "[engine] shard does not exist or is not a file: %s\n", cfg.shard_path.c_str());
        return false;
    }
    if (!mmap_.open(cfg.shard_path)) {
        fprintf(stderr, "[engine] cannot open shard %s\n", cfg.shard_path.c_str());
        return false;
    }
    resolver_.register_shard(&mmap_);

    // verify expert dims against the shard; derive inter_/hidden_ from it
    TensorView gv = resolver_.resolve_expert(avail_layer(0), 0, TensorResolver::GATE_PROJ);
    if (!gv.ok()) {
        fprintf(stderr, "[engine] cannot resolve expert 0 gate_proj in shard\n");
        return false;
    }
    if (gv.shape.size() != 2) {
        fprintf(stderr, "[engine] gate_proj shape rank != 2\n");
        return false;
    }
    inter_  = (int)gv.shape[0];
    hidden_ = (int)gv.shape[1];
    blob_elems_ = 3ULL * (size_t)inter_ * hidden_;
    blob_bytes_ = blob_elems_ * sizeof(float);
    if (hidden_ != cfg.hidden) {
        fprintf(stderr, "[engine] shard hidden %d != configured %d\n", hidden_, cfg.hidden);
        return false;
    }

    std::ifstream oracle_file(cfg.oracle_path, std::ios::binary);
    if (!oracle_file) {
        std::fprintf(stderr, "[engine] oracle does not exist or is unreadable: %s\n", cfg.oracle_path.c_str());
        return false;
    }
    char oracle_prefix[64]{};
    oracle_file.read(oracle_prefix, sizeof(oracle_prefix) - 1);
    if (std::string(oracle_prefix).find("version https://git-lfs.github.com/spec/v1") != std::string::npos) {
        std::fprintf(stderr, "[engine] oracle is an unresolved Git LFS pointer: %s\n"
                             "[engine] run: git lfs pull\n", cfg.oracle_path.c_str());
        return false;
    }
    if (!oracle_.load(cfg.oracle_path, hidden_, 256, 256)) {
        fprintf(stderr, "[engine] oracle load failed: %s\n", oracle_.error().c_str());
        return false;
    }
    int nl = std::min(cfg.num_layers, oracle_.num_layers());
    cfg_.num_layers = nl;

    // VRAM budget. default: 4 experts. The arena backend is cudaMalloc when the
    // CUDA path is active, else a malloc'd host arena (mock backend).
    size_t budget = cfg.budget_bytes ? cfg.budget_bytes : (4 * blob_bytes_);
    cfg_.budget_bytes = budget;
    Arena::Backend be;
    if (cfg.use_cuda) {
#ifdef DEE_CUDA
        be.kind = "cuda";
        be.alloc = [](size_t n) -> void* { void* p = nullptr; return DEE_CUDA_CHECK_NAMED(cudaMalloc(&p, n), "cudaMalloc(expert cache)") ? p : nullptr; };
        be.free  = [](void* p) { if (p) DEE_CUDA_CHECK_NAMED(cudaFree(p), "cudaFree(expert cache)"); };
#else
        fprintf(stderr, "[engine] --cuda requested but this build has DEE_CUDA=OFF\n");
        return false;
#endif
    } else {
        be.kind = "host";
        be.alloc = [](size_t n) -> void* { return std::malloc(n); };
        be.free  = [](void* p) { std::free(p); };
    }
    if (!cache_.init(budget, be)) {
        fprintf(stderr, "[engine] cache init failed (budget %zu bytes)\n", budget);
        return false;
    }

#ifdef DEE_CUDA
    if (cfg.use_cuda) {
        int device = -1;
        if (!DEE_CUDA_CHECK_NAMED(cudaGetDevice(&device), "cudaGetDevice")) return false;
        cudaDeviceProp prop{};
        if (!DEE_CUDA_CHECK_NAMED(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) return false;
        if (!DEE_CUDA_CHECK_NAMED(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking),
                                  "cudaStreamCreateWithFlags(compute)")) return false;
        auto dev_alloc = [](size_t n) -> float* { float* p = nullptr; return DEE_CUDA_CHECK_NAMED(cudaMalloc(reinterpret_cast<void**>(&p), n), "cudaMalloc(engine work buffer)") ? p : nullptr; };
        d_h_in_  = dev_alloc((size_t)hidden_ * sizeof(float));
        d_h_out_ = dev_alloc((size_t)hidden_ * sizeof(float));
        d_hbuf_  = dev_alloc((size_t)inter_  * sizeof(float));
        d_ybuf_  = dev_alloc((size_t)cfg.topk * hidden_ * sizeof(float));
        if (!d_h_in_ || !d_h_out_ || !d_hbuf_ || !d_ybuf_) {
            fprintf(stderr, "[engine] device work-buffer allocation failed\n");
            return false;
        }
        size_t freeB = 0, totalB = 0;
        if (DEE_CUDA_CHECK_NAMED(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo")) {
            cuda_total_ = totalB; cuda_free_ = freeB;
            stats_.cuda_total = totalB; stats_.cuda_free = freeB;
        }
        int runtime_version = 0;
        if (DEE_CUDA_CHECK_NAMED(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion")) {
            stats_.cuda_runtime_version = runtime_version;
        }
        stats_.cuda_device_name = prop.name;
        stats_.cuda_compute_major = prop.major;
        stats_.cuda_compute_minor = prop.minor;
    }
#endif

    if (!prefetcher_.init(cfg.use_cuda)) {
        fprintf(stderr, "[engine] prefetcher init failed\n");
        return false;
    }

    hidden_buf_[0].assign(hidden_, 0.0f);
    hidden_buf_[1].assign(hidden_, 0.0f);
    return true;
}

bool Engine::generate() {
    auto t0 = std::chrono::steady_clock::now();

    // deterministic "prompt" embedding for token 0, layer 0 input
    std::vector<float>& h0 = hidden_buf_[0];
    for (int i = 0; i < hidden_; ++i) h0[i] = std::sin(0.01f * i) * 0.5f;

    int cur = 0, nxt = 1;
    for (int t = 0; t < cfg_.num_tokens; ++t) {
        float* h_in  = hidden_buf_[cur].data();
        float* h_out = hidden_buf_[nxt].data();
        for (int L = 0; L < cfg_.num_layers; ++L) {
#ifdef DEE_CUDA
            if (cfg_.use_cuda) {
                if (!forward_layer_cuda(L, h_in, h_out)) return false;
            }
            else              forward_layer(L, h_in, h_out);
#else
            forward_layer(L, h_in, h_out);
#endif
        }
        // autoregressive (recurrent mock): next token input = this token output
        std::swap(cur, nxt);
        // track peak VRAM (mock arena high-water / CUDA arena used)
        size_t used = cache_.used_bytes();
        if (used > stats_.peak_vram) stats_.peak_vram = used;
    }

    prefetcher_.reset();   // bound inflight_/event churn between tokens

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    // gather stats
    const VramCacheManager::Stats& cs = cache_.stats();
    const AsyncPrefetcher::Stats& ps = prefetcher_.stats();
    stats_.tokens = cfg_.num_tokens;
    stats_.elapsed_sec = sec;
    stats_.tok_per_sec = sec > 0 ? cfg_.num_tokens / sec : 0.0;
    stats_.cache_hits  = cs.hits;
    stats_.cache_loads = cs.loads;
    stats_.evictions   = cs.evictions;
    stats_.fallbacks   = cs.fallbacks;
    stats_.prefetch_issued = ps.issued;
    stats_.prefetch_fallbacks = ps.fallbacks;

    // validate output hidden is finite
    const float* out = hidden_buf_[cur].data();
    stats_.hidden_finite = true;
    for (int i = 0; i < hidden_; ++i) {
        if (!std::isfinite(out[i])) { stats_.hidden_finite = false; break; }
    }
    return true;
}

#ifdef DEE_CUDA
void Engine::cuda_cleanup() {
    if (d_h_in_)  { DEE_CUDA_CHECK_NAMED(cudaFree(d_h_in_), "cudaFree(d_h_in)");  d_h_in_  = nullptr; }
    if (d_h_out_) { DEE_CUDA_CHECK_NAMED(cudaFree(d_h_out_), "cudaFree(d_h_out)"); d_h_out_ = nullptr; }
    if (d_hbuf_)  { DEE_CUDA_CHECK_NAMED(cudaFree(d_hbuf_), "cudaFree(d_hbuf)");  d_hbuf_  = nullptr; }
    if (d_ybuf_)  { DEE_CUDA_CHECK_NAMED(cudaFree(d_ybuf_), "cudaFree(d_ybuf)");  d_ybuf_  = nullptr; }
    if (compute_stream_) { DEE_CUDA_CHECK_NAMED(cudaStreamDestroy(compute_stream_), "cudaStreamDestroy(compute)"); compute_stream_ = nullptr; }
}

// GPU forward: Oracle predicts experts -> H2D weight copies (secondary stream)
// -> SwiGLU kernels on the compute stream -> mean combine -> D2H. The host
// gates each kernel launch on prefetcher.wait() (cudaEventSynchronize of that
// expert's copy), so the compute stream only blocks when a weight isn't ready.
bool Engine::forward_layer_cuda(int layer, const float* h_in, float* h_out) {
    std::vector<int> experts;
    oracle_.predict(layer, h_in, cfg_.topk, experts);
    int K = (int)experts.size();
    if (K == 0) { for (int i = 0; i < hidden_; ++i) h_out[i] = 0.0f; return true; }

    // 1) issue all expert weight H2D copies (async, secondary stream)
    if (!DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(d_h_in_, h_in, (size_t)hidden_ * sizeof(float),
                                              cudaMemcpyHostToDevice, compute_stream_),
                              "cudaMemcpyAsync(hidden host to device)")) return false;

    const int batch_size = std::max(1, static_cast<int>(cache_.budget_bytes() / blob_bytes_));
    for (int first = 0; first < K; first += batch_size) {
        const int last = std::min(K, first + batch_size);
        for (int k = first; k < last; ++k) {
            if (!stage_expert(layer, experts[k], cfg_.topk - k)) return false;
        }
        for (int k = first; k < last; ++k) {
            const int e = experts[k];
            if (!prefetcher_.wait(layer, e) || !cache_.pin(layer, e)) return false;
            const float* d_blob = static_cast<const float*>(cache_.data(layer, e));
            if (!d_blob || !swiglu_expert_cuda(d_blob, d_h_in_, d_hbuf_,
                                                d_ybuf_ + (size_t)k * hidden_, inter_, hidden_, compute_stream_)) {
                cache_.unpin(layer, e);
                return false;
            }
        }
        // Keep cache blocks pinned until the compute stream has consumed them.
        if (!DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_), "cudaStreamSynchronize(expert batch)")) return false;
        for (int k = first; k < last; ++k) cache_.unpin(layer, experts[k]);
    }
    if (!combine_cuda(d_ybuf_, d_h_out_, K, hidden_, compute_stream_) ||
        !DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(h_out, d_h_out_, (size_t)hidden_ * sizeof(float),
                                              cudaMemcpyDeviceToHost, compute_stream_),
                              "cudaMemcpyAsync(hidden device to host)") ||
        !DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_), "cudaStreamSynchronize(layer output)")) return false;

    // 3) stabilize (frozen layer-norm stand-in) — same as CPU path
    double ss = 0.0;
    for (int i = 0; i < hidden_; ++i) ss += (double)h_out[i] * h_out[i];
    double rms = std::sqrt(ss / hidden_);
    if (rms > 1e-8) {
        float s = 1.0f / (float)rms;
        for (int i = 0; i < hidden_; ++i) h_out[i] *= s;
    }
    return true;
}

Engine::~Engine() {
    if (cfg_.use_cuda) cuda_cleanup();
}
#else
Engine::~Engine() = default;
#endif

} // namespace dee
