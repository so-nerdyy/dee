// dee/engine.cpp
#include "dee/engine.h"

#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_set>

#ifdef DEE_CUDA
#include <cuda_runtime.h>
#include "dee/cuda_check.h"
#include "dee/swiglu_cuda.h"
#endif

namespace dee {
namespace {

uint64_t staging_key(int layer, int expert) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
           static_cast<uint32_t>(expert);
}

}  // namespace

const char* benchmark_scenario_name(BenchmarkScenario scenario) {
    switch (scenario) {
        case BenchmarkScenario::EndToEnd: return "end-to-end";
        case BenchmarkScenario::FullResident: return "full-resident";
        case BenchmarkScenario::ResidentBypass: return "resident-bypass";
        case BenchmarkScenario::TransferOnly: return "transfer-only";
        case BenchmarkScenario::ComputeOnly: return "compute-only";
        case BenchmarkScenario::OracleOnly: return "oracle-only";
        case BenchmarkScenario::CacheMetadataOnly: return "cache-metadata-only";
    }
    return "unknown";
}

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

const float* Engine::get_staging(int source_layer, int expert) {
    const auto profile_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    const uint64_t key = staging_key(source_layer, expert);
    auto it = staging_.find(key);
    if (it != staging_.end()) {
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
        return it->second.data();
    }

    TensorView gv = resolver_.resolve_expert(source_layer, expert, TensorResolver::GATE_PROJ);
    TensorView uv = resolver_.resolve_expert(source_layer, expert, TensorResolver::UP_PROJ);
    TensorView dv = resolver_.resolve_expert(source_layer, expert, TensorResolver::DOWN_PROJ);
    const size_t gh = (size_t)inter_ * hidden_;
    const size_t dh = (size_t)hidden_ * inter_;
    if (!gv.ok() || !uv.ok() || !dv.ok() || gv.dtype != DType::BF16 || uv.dtype != DType::BF16 || dv.dtype != DType::BF16 ||
        gv.nbytes != gh * sizeof(uint16_t) || uv.nbytes != gh * sizeof(uint16_t) || dv.nbytes != dh * sizeof(uint16_t)) {
        std::fprintf(stderr, "[engine] expert %d has unsupported or inconsistent BF16 tensor layout\n", expert);
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
        return nullptr;
    }

    std::vector<float> blob(blob_elems_);
    for (size_t i = 0; i < gh; ++i) blob[i] = bf16_to_f32(*(const uint16_t*)(gv.data + i * 2));
    for (size_t i = 0; i < gh; ++i) blob[gh + i] = bf16_to_f32(*(const uint16_t*)(uv.data + i * 2));
    for (size_t i = 0; i < dh; ++i) blob[2 * gh + i] = bf16_to_f32(*(const uint16_t*)(dv.data + i * 2));

    auto res = staging_.emplace(key, std::move(blob));
    if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
    return res.first->second.data();
}

const uint16_t* Engine::get_staging_bf16(int source_layer, int expert) {
    const auto profile_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    const uint64_t key = staging_key(source_layer, expert);
    auto pinned = pinned_staging_bf16_.find(key);
    if (pinned != pinned_staging_bf16_.end()) {
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
        return static_cast<const uint16_t*>(pinned->second);
    }
    auto it = staging_bf16_.find(key);
    if (it != staging_bf16_.end()) {
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
        return it->second.data();
    }

    TensorView gv = resolver_.resolve_expert(source_layer, expert, TensorResolver::GATE_PROJ);
    TensorView uv = resolver_.resolve_expert(source_layer, expert, TensorResolver::UP_PROJ);
    TensorView dv = resolver_.resolve_expert(source_layer, expert, TensorResolver::DOWN_PROJ);
    const size_t projection = static_cast<size_t>(inter_) * hidden_;
    const size_t projection_bytes = projection * sizeof(uint16_t);
    if (!gv.ok() || !uv.ok() || !dv.ok() || gv.dtype != DType::BF16 ||
        uv.dtype != DType::BF16 || dv.dtype != DType::BF16 ||
        gv.nbytes != projection_bytes || uv.nbytes != projection_bytes ||
        dv.nbytes != projection_bytes) {
        std::fprintf(stderr, "[engine] expert %d has unsupported or inconsistent BF16 tensor layout\n",
                     expert);
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
        return nullptr;
    }

#ifdef DEE_CUDA
    const size_t source_bytes = blob_elems_ * sizeof(uint16_t);
    if (cfg_.use_cuda && pinned_staging_bytes_ + source_bytes <= kPinnedStagingLimit) {
        void* allocation = nullptr;
        if (DEE_CUDA_CHECK_NAMED(cudaHostAlloc(&allocation, source_bytes, cudaHostAllocDefault),
                                 "cudaHostAlloc(persistent BF16 expert source)")) {
            // Tensor lookup/allocation and host copying are distinct profile categories.
            if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
            const auto copy_begin = profiler_.enabled()
                ? StageProfiler::now() : StageProfiler::TimePoint{};
            auto* destination = static_cast<uint16_t*>(allocation);
            std::memcpy(destination, gv.data, projection_bytes);
            std::memcpy(destination + projection, uv.data, projection_bytes);
            std::memcpy(destination + 2 * projection, dv.data, projection_bytes);
            if (profiler_.enabled()) {
                profiler_.add_cpu(CpuStage::MmapToPinned, copy_begin);
                profiler_.note_mmap_copy(source_bytes);
            }
            pinned_staging_bf16_.emplace(key, allocation);
            pinned_staging_bytes_ += source_bytes;
            return destination;
        }
    }
#endif

    std::vector<uint16_t> blob(blob_elems_);
    std::memcpy(blob.data(), gv.data, projection_bytes);
    std::memcpy(blob.data() + projection, uv.data, projection_bytes);
    std::memcpy(blob.data() + 2 * projection, dv.data, projection_bytes);
    auto result = staging_bf16_.emplace(key, std::move(blob));
    if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
    return result.first->second.data();
}

bool Engine::stage_expert(int logical_layer, int source_layer, int expert, int priority) {
    if (cfg_.use_cuda) {
        const uint16_t* blob = get_staging_bf16(source_layer, expert);
        if (!blob) {
            std::fprintf(stderr, "[engine] missing BF16 source weights for expert (%d,%d)\n",
                         source_layer, expert);
            return false;
        }
        const bool source_pinned = pinned_staging_bf16_.find(
            staging_key(source_layer, expert)) != pinned_staging_bf16_.end();
        return prefetcher_.prefetch_bf16_to_f32(
                   source_layer, expert, blob, blob_elems_, priority,
                   current_token_, logical_layer, source_pinned) >= 0;
    }
    const float* blob = get_staging(source_layer, expert);
    if (!blob) {
        std::fprintf(stderr, "[engine] missing source weights for expert (%d,%d)\n", source_layer, expert);
        return false;
    }
    return prefetcher_.prefetch(source_layer, expert, blob, blob_bytes_, priority,
                                current_token_, logical_layer) >= 0;
}

bool Engine::preload_all_experts() {
    std::unordered_set<int> source_layers;
    for (int layer = 0; layer < cfg_.num_layers; ++layer) source_layers.insert(avail_layer(layer));

    size_t loaded = 0;
    for (int source_layer : source_layers) {
        for (int expert = 0; expert < oracle_.num_experts(); ++expert) {
            prefetcher_.begin_batch();
            if (!stage_expert(source_layer, source_layer, expert, oracle_.num_experts() - expert) ||
                !prefetcher_.wait(source_layer, expert)) {
                std::fprintf(stderr, "[engine] failed to preload expert (%d,%d) for scenario %s\n",
                             source_layer, expert, benchmark_scenario_name(cfg_.scenario));
                return false;
            }
            if (++loaded % 64 == 0) prefetcher_.reset();
        }
    }
    prefetcher_.reset();
    return true;
}

bool Engine::prepare_profile_scenario() {
    const bool preload = cfg_.scenario == BenchmarkScenario::FullResident ||
                         cfg_.scenario == BenchmarkScenario::ResidentBypass ||
                         cfg_.scenario == BenchmarkScenario::ComputeOnly;
    if (preload) {
        // Preload is scenario setup, not measured work. Detach the profiler so
        // its bounded CUDA event pool cannot be consumed by setup transfers.
        cache_.set_profiler(nullptr);
        prefetcher_.set_profiler(nullptr);
        if (!preload_all_experts()) return false;
    }

    cache_.reset_stats();
    prefetcher_.reset_stats();
    scenario_requests_ = 0;
    scenario_resident_hits_ = 0;
    scenario_cold_loads_ = 0;
    profiler_.configure(cfg_.profile_stages, cfg_.trace_requests,
                        blob_bytes_, oracle_.num_experts());
    cache_.set_profiler(cfg_.profile_stages ? &profiler_ : nullptr);
    prefetcher_.set_profiler(cfg_.profile_stages ? &profiler_ : nullptr);
    oracle_.set_profiler(cfg_.profile_stages ? &profiler_ : nullptr);
    return true;
}

void Engine::forward_layer(int layer, const float* h_in, float* h_out) {
    std::vector<int> experts;
    const auto oracle_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    oracle_.predict(layer, h_in, cfg_.topk, experts);
    if (profiler_.enabled()) profiler_.add_cpu(CpuStage::Oracle, oracle_begin);

    // CPU/mock mode processes one expert at a time. This preserves the fixed
    // budget contract even when top-K is larger than the cache capacity.
    std::vector<float> acc(hidden_, 0.0f);
    const int source_layer = avail_layer(layer);
    profiler_.note_prediction(current_token_, layer, source_layer, experts);
    for (size_t k = 0; k < experts.size(); ++k) {
        int e = experts[k];
        prefetcher_.begin_batch();
        if (!stage_expert(layer, source_layer, e, static_cast<int>(cfg_.topk - k)) || !prefetcher_.wait(source_layer, e)) {
            stats_.fallbacks++;
            continue;
        }
        const void* p = cache_.data(source_layer, e);
        if (!p) {  // should not happen (wait guarantees resident)
            stats_.fallbacks++;
            continue;
        }
        swiglu((const float*)p, h_in, inter_, hidden_, acc.data());
        cache_.touch(source_layer, e);
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
    profiler_.configure(cfg.profile_stages, cfg.trace_requests, blob_bytes_, oracle_.num_experts());
    cache_.set_profiler(cfg.profile_stages ? &profiler_ : nullptr);
    prefetcher_.set_profiler(cfg.profile_stages ? &profiler_ : nullptr);
    oracle_.set_profiler(cfg.profile_stages ? &profiler_ : nullptr);

    // VRAM budget. default: 4 experts. The arena backend is cudaMalloc when the
    // CUDA path is active, else a malloc'd host arena (mock backend).
    size_t budget = cfg.budget_bytes ? cfg.budget_bytes : (4 * blob_bytes_);
    const bool needs_full_residency = cfg_.scenario == BenchmarkScenario::FullResident ||
                                      cfg_.scenario == BenchmarkScenario::ResidentBypass ||
                                      cfg_.scenario == BenchmarkScenario::ComputeOnly;
    if (needs_full_residency) {
        std::unordered_set<int> source_layers;
        for (int layer = 0; layer < cfg_.num_layers; ++layer) source_layers.insert(avail_layer(layer));
        const size_t physical_experts = source_layers.size() * static_cast<size_t>(oracle_.num_experts());
        if (physical_experts > std::numeric_limits<size_t>::max() / blob_bytes_) {
            std::fprintf(stderr, "[engine] full-resident scenario cache size overflow\n");
            return false;
        }
        budget = physical_experts * blob_bytes_;
    }
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
        if (!DEE_CUBLAS_CHECK_NAMED(cublasCreate(&cublas_handle_), "cublasCreate") ||
            !DEE_CUBLAS_CHECK_NAMED(cublasSetStream(cublas_handle_, compute_stream_), "cublasSetStream(compute)")) return false;
        auto dev_alloc = [](size_t n) -> float* { float* p = nullptr; return DEE_CUDA_CHECK_NAMED(cudaMalloc(reinterpret_cast<void**>(&p), n), "cudaMalloc(engine work buffer)") ? p : nullptr; };
        d_h_in_  = dev_alloc((size_t)hidden_ * sizeof(float));
        d_h_out_ = dev_alloc((size_t)hidden_ * sizeof(float));
        d_hbuf_  = dev_alloc((size_t)inter_  * sizeof(float));
        d_ubuf_  = dev_alloc((size_t)inter_  * sizeof(float));
        d_ybuf_  = dev_alloc((size_t)cfg.topk * hidden_ * sizeof(float));
        if (!d_h_in_ || !d_h_out_ || !d_hbuf_ || !d_ubuf_ || !d_ybuf_) {
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
    if (!prepare_profile_scenario()) return false;
    auto t0 = std::chrono::steady_clock::now();

    // deterministic "prompt" embedding for token 0, layer 0 input
    std::vector<float>& h0 = hidden_buf_[0];
    for (int i = 0; i < hidden_; ++i) h0[i] = std::sin(0.01f * i) * 0.5f;

    int cur = 0, nxt = 1;
    for (int t = 0; t < cfg_.num_tokens; ++t) {
        current_token_ = t;
        const auto token_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
        float* h_in  = hidden_buf_[cur].data();
        float* h_out = hidden_buf_[nxt].data();
        for (int L = 0; L < cfg_.num_layers; ++L) {
            const auto layer_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
#ifdef DEE_CUDA
            if (cfg_.use_cuda) {
                if (!forward_layer_cuda(L, h_in, h_out)) return false;
            }
            else              forward_layer(L, h_in, h_out);
#else
            forward_layer(L, h_in, h_out);
#endif
            if (profiler_.enabled()) profiler_.add_layer_latency(layer_begin);
        }
        // autoregressive (recurrent mock): next token input = this token output
        std::swap(cur, nxt);
        // track peak VRAM (mock arena high-water / CUDA arena used)
        size_t used = cache_.used_bytes();
        if (used > stats_.peak_vram) stats_.peak_vram = used;
        if (profiler_.enabled()) profiler_.add_token_latency(token_begin);
    }

    prefetcher_.reset();   // bound inflight_/event churn between tokens

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    // gather stats
    const VramCacheManager::Stats& cs = cache_.stats();
    const AsyncPrefetcher::Stats& ps = prefetcher_.stats();
    const bool metadata_only = cfg_.scenario == BenchmarkScenario::CacheMetadataOnly;
    const uint64_t requests = metadata_only ? scenario_requests_ : ps.requests;
    const uint64_t resident_hits = metadata_only ? scenario_resident_hits_ : ps.resident_hits;
    const uint64_t inflight_hits = metadata_only ? 0 : ps.inflight_hits;
    const uint64_t cold_loads = metadata_only ? scenario_cold_loads_ : ps.cold_loads;
    if (requests != resident_hits + inflight_hits + cold_loads) {
        std::fprintf(stderr,
                     "[engine] request accounting invariant failed: requests=%llu resident=%llu inflight=%llu cold=%llu\n",
                     static_cast<unsigned long long>(requests),
                     static_cast<unsigned long long>(resident_hits),
                     static_cast<unsigned long long>(inflight_hits),
                     static_cast<unsigned long long>(cold_loads));
#ifdef DEE_CUDA_VALIDATE
        assert(requests == resident_hits + inflight_hits + cold_loads);
#endif
        return false;
    }
    if (cs.hits != resident_hits || cs.loads != cold_loads) {
        std::fprintf(stderr,
                     "[engine] cache/request accounting mismatch: cache_hits=%llu resident=%llu cache_loads=%llu cold=%llu\n",
                     static_cast<unsigned long long>(cs.hits),
                     static_cast<unsigned long long>(resident_hits),
                     static_cast<unsigned long long>(cs.loads),
                     static_cast<unsigned long long>(cold_loads));
#ifdef DEE_CUDA_VALIDATE
        assert(cs.hits == resident_hits && cs.loads == cold_loads);
#endif
        return false;
    }
    stats_.tokens = cfg_.num_tokens;
    stats_.elapsed_sec = sec;
    stats_.tok_per_sec = sec > 0 ? cfg_.num_tokens / sec : 0.0;
    stats_.cache_hits  = cs.hits;
    stats_.cache_loads = cs.loads;
    stats_.evictions   = cs.evictions;
    stats_.fallbacks   = cs.fallbacks;
    stats_.prefetch_issued = requests;
    stats_.prefetch_fallbacks = ps.fallbacks;
    stats_.resident_hits = resident_hits;
    stats_.inflight_hits = inflight_hits;
    stats_.cold_loads = cold_loads;
    stats_.duplicate_requests = ps.duplicate_requests;
    stats_.profile = profiler_.finish(sec * 1000.0, resident_hits, inflight_hits,
                                      cold_loads, ps.duplicate_requests, cs.evictions,
                                      cs.pinned_blocks_skipped);

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
    if (d_ubuf_)  { DEE_CUDA_CHECK_NAMED(cudaFree(d_ubuf_), "cudaFree(d_ubuf)");  d_ubuf_  = nullptr; }
    if (d_ybuf_)  { DEE_CUDA_CHECK_NAMED(cudaFree(d_ybuf_), "cudaFree(d_ybuf)");  d_ybuf_  = nullptr; }
    if (cublas_handle_) { DEE_CUBLAS_CHECK_NAMED(cublasDestroy(cublas_handle_), "cublasDestroy"); cublas_handle_ = nullptr; }
    if (compute_stream_) { DEE_CUDA_CHECK_NAMED(cudaStreamDestroy(compute_stream_), "cudaStreamDestroy(compute)"); compute_stream_ = nullptr; }
}

// GPU forward: Oracle predicts experts -> H2D weight copies (secondary stream)
// -> SwiGLU kernels on the compute stream -> mean combine -> D2H. The host
// gates each kernel launch on prefetcher.wait() (cudaEventSynchronize of that
// expert's copy), so the compute stream only blocks when a weight isn't ready.
bool Engine::forward_layer_cuda(int layer, const float* h_in, float* h_out) {
    std::vector<int> experts;
    if (cfg_.scenario == BenchmarkScenario::ComputeOnly) {
        experts.reserve(static_cast<size_t>(cfg_.topk));
        for (int k = 0; k < cfg_.topk; ++k) {
            experts.push_back((layer * cfg_.topk + k) % oracle_.num_experts());
        }
    } else {
        const auto oracle_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
        oracle_.predict(layer, h_in, cfg_.topk, experts);
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::Oracle, oracle_begin);
    }
    int K = (int)experts.size();
    if (K == 0) { for (int i = 0; i < hidden_; ++i) h_out[i] = 0.0f; return true; }

    const int source_layer = avail_layer(layer);
    profiler_.note_prediction(current_token_, layer, source_layer, experts);

    if (cfg_.scenario == BenchmarkScenario::OracleOnly) {
        std::memcpy(h_out, h_in, static_cast<size_t>(hidden_) * sizeof(float));
        return true;
    }

    if (cfg_.scenario == BenchmarkScenario::CacheMetadataOnly) {
        const int batch_size = std::max(1, static_cast<int>(cache_.budget_bytes() / blob_bytes_));
        for (int first = 0; first < K; first += batch_size) {
            const int last = std::min(K, first + batch_size);
            std::vector<int> pinned;
            pinned.reserve(static_cast<size_t>(last - first));
            for (int k = first; k < last; ++k) {
                const int expert = experts[k];
                const bool resident = cache_.is_resident(source_layer, expert);
                if (!cache_.ensure(source_layer, expert, blob_bytes_, cfg_.topk - k) ||
                    !cache_.pin(source_layer, expert)) return false;
                pinned.push_back(expert);
                ++scenario_requests_;
                if (resident) ++scenario_resident_hits_;
                else ++scenario_cold_loads_;
                const VramCacheManager::EnsureInfo info = cache_.last_ensure_info();
                profiler_.note_request(current_token_, layer, source_layer, expert,
                                       resident ? RequestKind::ResidentHit : RequestKind::ColdLoad,
                                       cache_.used_bytes(),
                                       info.evicted ? info.evicted_key.layer : -1,
                                       info.evicted ? info.evicted_key.expert : -1,
                                       cfg_.topk - k);
            }
            for (int expert : pinned) cache_.unpin(source_layer, expert);
        }
        std::memcpy(h_out, h_in, static_cast<size_t>(hidden_) * sizeof(float));
        return true;
    }

    // 1) issue all expert weight H2D copies (async, secondary stream)
    if (cfg_.scenario != BenchmarkScenario::TransferOnly) {
        const size_t input_h2d_ticket = profiler_.enabled()
            ? profiler_.cuda_begin(GpuStage::H2D, static_cast<void*>(compute_stream_)) : static_cast<size_t>(-1);
        if (!DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(d_h_in_, h_in, (size_t)hidden_ * sizeof(float),
                                                  cudaMemcpyHostToDevice, compute_stream_),
                                  "cudaMemcpyAsync(hidden host to device)")) return false;
        if (profiler_.enabled()) {
            if (!profiler_.cuda_end(input_h2d_ticket, static_cast<void*>(compute_stream_))) return false;
            profiler_.note_h2d_copy(static_cast<size_t>(hidden_) * sizeof(float));
        }
    }

    const auto batch_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    const int batch_size = std::max(1, static_cast<int>(cache_.budget_bytes() / blob_bytes_));
    if (profiler_.enabled()) profiler_.add_cpu(CpuStage::BatchConstruction, batch_begin);
    const bool bypass_cache = cfg_.scenario == BenchmarkScenario::ResidentBypass ||
                              cfg_.scenario == BenchmarkScenario::ComputeOnly;
    for (int first = 0; first < K; first += batch_size) {
        const int last = std::min(K, first + batch_size);
        if (!bypass_cache) {
            prefetcher_.begin_batch();
            for (int k = first; k < last; ++k) {
                if (!stage_expert(layer, source_layer, experts[k], cfg_.topk - k)) return false;
            }
        }
        for (int k = first; k < last; ++k) {
            const int e = experts[k];
            if (!bypass_cache && !prefetcher_.wait(source_layer, e)) return false;
            if (cfg_.scenario == BenchmarkScenario::TransferOnly) continue;
            if (!bypass_cache && !cache_.pin(source_layer, e)) return false;
            const float* d_blob = static_cast<const float*>(cache_.data(source_layer, e));
            if (!d_blob || !swiglu_expert_cuda(cublas_handle_, d_blob, d_h_in_, d_hbuf_, d_ubuf_,
                                                d_ybuf_ + (size_t)k * hidden_, inter_, hidden_, compute_stream_,
                                                profiler_.enabled() ? &profiler_ : nullptr)) {
                if (!bypass_cache) cache_.unpin(source_layer, e);
                return false;
            }
        }
        if (cfg_.scenario == BenchmarkScenario::TransferOnly) {
            // wait() has already completed every transfer in this batch, so
            // reclaim timing events without introducing another CUDA sync.
            if (profiler_.enabled() && !profiler_.cuda_collect_ready()) return false;
            continue;
        }
        // Keep cache blocks pinned until the compute stream has consumed them.
        const auto sync_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
        if (!DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_), "cudaStreamSynchronize(expert batch)")) return false;
        if (profiler_.enabled()) {
            profiler_.add_cpu(CpuStage::Synchronization, sync_begin);
            profiler_.note_host_synchronization();
            if (!profiler_.cuda_collect_ready()) return false;
        }
        if (!bypass_cache) {
            for (int k = first; k < last; ++k) cache_.unpin(source_layer, experts[k]);
        }
    }
    if (cfg_.scenario == BenchmarkScenario::TransferOnly) {
        std::memcpy(h_out, h_in, static_cast<size_t>(hidden_) * sizeof(float));
        return true;
    }
    if (!combine_cuda(d_ybuf_, d_h_out_, K, hidden_, compute_stream_,
                      profiler_.enabled() ? &profiler_ : nullptr) ||
        !DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(h_out, d_h_out_, (size_t)hidden_ * sizeof(float),
                                              cudaMemcpyDeviceToHost, compute_stream_),
                              "cudaMemcpyAsync(hidden device to host)")) return false;
    const auto output_sync_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    if (!DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_), "cudaStreamSynchronize(layer output)")) return false;
    if (profiler_.enabled()) {
        profiler_.add_cpu(CpuStage::Synchronization, output_sync_begin);
        profiler_.note_host_synchronization();
        if (!profiler_.cuda_collect_ready()) return false;
    }

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
    if (cfg_.use_cuda) {
        prefetcher_.synchronize_all();
        if (compute_stream_) {
            DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_),
                                 "cudaStreamSynchronize(engine teardown)");
        }
        if (profiler_.enabled()) profiler_.cuda_collect_ready();
        for (const auto& entry : pinned_staging_bf16_) {
            DEE_CUDA_CHECK_NAMED(cudaFreeHost(entry.second),
                                 "cudaFreeHost(persistent BF16 expert source)");
        }
        pinned_staging_bf16_.clear();
        pinned_staging_bytes_ = 0;
        cuda_cleanup();
    }
}
#else
Engine::~Engine() = default;
#endif

} // namespace dee
