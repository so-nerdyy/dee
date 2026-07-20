// dee.cpp - Step 8: CLI benchmark driver.
//
// Drives the full autoregressive Generation loop on a synthetic MoE model and
// reports decode throughput + cache/miss telemetry. Runs on the mock backend
// (no CUDA toolkit) or, with -DDEE_CUDA (and a toolkit), on whatever GPU is
// present - the arch list in CMakeLists targets Pascal -> Hopper, so one build
// runs on a 3090 (sm_86), L40S/4090 (sm_89), H100 (sm_90), etc.
//
// The weights / router / sampler here are deterministic stand-ins so the engine
// can be exercised end-to-end without a trained checkpoint. Swapping in the
// real WeightMmap + trained Oracle/Router is a drop-in (same interfaces).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dee/async_prefetcher.h"
#include "dee/cuda_device.h"
#include "dee/generation.h"
#include "dee/oracle.h"
#include "dee/vram_cache_manager.h"
#include "dee/weight_mmap.h"
#include "dee/weight_source.h"

namespace {

// All experts share the lowest eviction priority -> ties broken by LRU, which
// makes the benchmark's cache churn deterministic.
class ZeroPriorityOracle : public dee::Oracle {
public:
    float priority(int, int) const override { return 0.f; }
};

// Synthetic FP32 weight source: deterministic per (layer, expert) so runs are
// reproducible. GateUp at offset 0, Down at offset gate_up_bytes.
class SyntheticWeightSource : public dee::WeightSource {
public:
    SyntheticWeightSource(int nl, int ne, int hidden, int inter)
        : nl_(nl), ne_(ne), H_(hidden), I_(inter) {
        const size_t gu = static_cast<size_t>(2) * I_ * H_;
        const size_t dn = static_cast<size_t>(H_) * I_;
        for (int L = 0; L < nl_; ++L)
            for (int E = 0; E < ne_; ++E) {
                std::vector<float> a(gu), b(dn);
                for (size_t i = 0; i < a.size(); ++i)
                    a[i] = static_cast<float>((L * 131 + E * 17 + i * 3 + 1) & 0xFF);
                for (size_t i = 0; i < b.size(); ++i)
                    b[i] = static_cast<float>((L * 197 + E * 11 + i * 5 + 2) & 0xFF);
                data_[{L, E}] = {std::move(a), std::move(b)};
            }
        gu_bytes_ = gu * sizeof(float);
        dn_bytes_ = dn * sizeof(float);
    }
    int num_layers() const override { return nl_; }
    int experts_per_layer() const override { return ne_; }
    int num_tensors_per_expert() const override { return dee::kNumExpertTensors; }
    void expert_tensors(int layer, int expert_id, dee::ExpertTensorView* views) override {
        auto& d = data_.at(dee::ExpertKey{layer, expert_id});
        views[0] = dee::ExpertTensorView{d.gu.data(), gu_bytes_, 0, dee::ExpertTensor::GateUp};
        views[1] = dee::ExpertTensorView{d.dn.data(), dn_bytes_, gu_bytes_, dee::ExpertTensor::Down};
    }
    size_t expert_total_bytes(int, int) const override { return gu_bytes_ + dn_bytes_; }
    std::string tensor_name(int layer, int e, dee::ExpertTensor t) const override {
        return "synth." + std::to_string(layer) + "." + std::to_string(e) + "." +
               (t == dee::ExpertTensor::GateUp ? "gu" : "dn");
    }
    void decode_bf16_to_fp32(const void* bf16, float* out, size_t n) const override {
        std::memcpy(out, bf16, n * sizeof(float));
    }

private:
    struct Entry { std::vector<float> gu, dn; };
    struct KeyHash { size_t operator()(const dee::ExpertKey& k) const noexcept { return dee::ExpertKeyHash()(k); } };
    std::unordered_map<dee::ExpertKey, Entry, KeyHash> data_;
    int nl_, ne_, H_, I_;
    size_t gu_bytes_, dn_bytes_;
};

// Oracle: predicts experts derived from the token (varies across the sequence
// so the prefetcher stays busy and we see a realistic mix of hits/misses).
class TokenOracle : public dee::ExpertOracle {
public:
    TokenOracle(int ne, int hd) : ne_(ne), hd_(hd) {}
    void predict(int layer, const float* h, int, int topk, int* out, float* scores = nullptr) const override {
        int base = static_cast<int>(h[0] * 1000.0f) % ne_;
        if (base < 0) base += ne_;
        for (int i = 0; i < topk; ++i) {
            out[i] = (base + i * 7 + layer * 3) % ne_;
            if (scores) scores[i] = 1.0f - static_cast<float>(i) / topk;
        }
    }
    int num_experts() const override { return ne_; }
    int hidden_dim() const override { return hd_; }
    int lookahead() const override { return 1; }

private:
    int ne_, hd_;
};

// Router: selects a different-but-deterministic set than the Oracle, so some
// experts are Oracle-mispredicted (reactive prefetches) while others are hits.
class TokenRouter : public dee::ExpertRouter {
public:
    TokenRouter(int ne, int hd) : ne_(ne), hd_(hd) {}
    void route(int layer, const float* h, int, int topk, int* out, float* scores = nullptr) const override {
        int base = static_cast<int>(h[0] * 1000.0f) % ne_;
        if (base < 0) base += ne_;
        for (int i = 0; i < topk; ++i) {
            out[i] = (base + i * 5 + layer * 11) % ne_;
            if (scores) scores[i] = 1.0f - static_cast<float>(i) / (2 * topk);
        }
    }

private:
    int ne_, hd_;
};

class LinearEmbedder : public dee::TokenEmbedder {
public:
    void embed(int token_id, float* out, int hidden_dim) const override {
        // Add a slowly-varying positional term so successive decode steps get
        // distinct embeddings. Without this (and with no cross-token attention
        // in the synthetic model) the loop collapses to a fixed point.
        const float pos = std::fmod(static_cast<float>(step_++) * 0.137f, 1.0f);
        for (int i = 0; i < hidden_dim; ++i)
            out[i] = std::fmod(static_cast<float>(token_id * 31 + i * 7 + 1), 10.0f) - 5.0f
                     + pos + std::fmod(static_cast<float>(i) * 0.01f, 1.0f);
    }

private:
    mutable size_t step_ = 0;
};

class ArgmaxSampler : public dee::TokenSampler {
public:
    explicit ArgmaxSampler(int vocab) : vocab_(vocab) {}
    int sample(const float* h, int hidden_dim) const override {
        // Hash the whole hidden vector so generated tokens vary across the
        // sequence (a single fixed element would collapse to a constant token).
        long acc = 0;
        for (int i = 0; i < hidden_dim; ++i) acc = (acc * 31 + static_cast<long>(h[i] * 1000.0f)) % vocab_;
        if (acc < 0) acc += vocab_;
        return static_cast<int>(acc % vocab_);
    }

private:
    int vocab_;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s [--layers N] [--experts N] [--hidden N] [--inter N]\n"
        "            [--topk N] [--slots N] [--prompt TOK] [--max-new N] [--eos N] [--vocab N]\n"
        "            [--weights-dir DIR]\n"
        "  Runs the autoregressive Generation loop and reports decode throughput +\n"
        "  cache/miss telemetry. With --weights-dir DIR the real BF16 MoE shards in\n"
        "  DIR are used (decoded to FP32 on prefetch); otherwise a synthetic model\n"
        "  is used. Dims are taken from the weight files when --weights-dir is set.\n", prog);
}

}  // namespace

int main(int argc, char** argv) {
    int layers = 4, experts = 16, hidden = 64, inter = 128, topk = 2;
    int slots = 32, prompt_token = 1, max_new = 64, eos = -1, vocab = 1000;
    std::string weights_dir;  // empty => synthetic model

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](int def) { return (i + 1 < argc) ? std::atoi(argv[++i]) : def; };
        if (a == "--layers") layers = next(layers);
        else if (a == "--experts") experts = next(experts);
        else if (a == "--hidden") hidden = next(hidden);
        else if (a == "--inter") inter = next(inter);
        else if (a == "--topk") topk = next(topk);
        else if (a == "--slots") slots = next(slots);
        else if (a == "--prompt") prompt_token = next(prompt_token);
        else if (a == "--max-new") max_new = next(max_new);
        else if (a == "--eos") eos = next(eos);
        else if (a == "--vocab") vocab = next(vocab);
        else if (a == "--weights-dir") weights_dir = (i + 1 < argc) ? argv[++i] : "";
        else { usage(argv[0]); return 2; }
    }
    if (max_new <= 0) { usage(argv[0]); return 2; }

    auto dev = dee::make_cuda_device();

    // Weight source: real BF16 shards (--weights-dir) or synthetic FP32. Real
    // shards carry actual layer numbers (e.g. model.layers.10); the Generator's
    // layer_index() mapping addresses them from 0-based positions automatically.
    std::unique_ptr<dee::WeightSource> wsrc_owner;
    dee::WeightSource* wsrc = nullptr;
    int H = hidden, I = inter, E = experts, NL = layers;
    size_t slot_bytes = 0;
    std::string mode;
    if (!weights_dir.empty()) {
        try {
            wsrc_owner.reset(new dee::WeightMmap(weights_dir));
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "failed to load weights from %s: %s\n", weights_dir.c_str(), ex.what());
            return 2;
        }
        wsrc = wsrc_owner.get();
        auto* wm = static_cast<dee::WeightMmap*>(wsrc);
        H = wm->hidden_dim();
        I = wm->inter_dim();
        E = wm->experts_per_layer();
        NL = wm->num_layers();
        slot_bytes = wm->expert_total_bytes(wm->layer_index(0), 0) + 64;
        mode = "real";
    } else {
        wsrc_owner.reset(new SyntheticWeightSource(layers, experts, hidden, inter));
        wsrc = wsrc_owner.get();
        H = hidden; I = inter; E = experts; NL = layers;
        slot_bytes = (static_cast<size_t>(2) * inter * hidden +
                      static_cast<size_t>(hidden) * inter) * sizeof(float) + 64;
        mode = "synth";
    }

    ZeroPriorityOracle coracle;  // zero priority => LRU eviction
    dee::VramCacheManager cache(*dev, slot_bytes, slots, coracle);
    dee::AsyncPrefetcher pf(*dev, cache, *wsrc);
    TokenOracle oracle(E, H);
    TokenRouter router(E, H);
    LinearEmbedder embed;
    ArgmaxSampler sampler(vocab);
    dee::GeneratorConfig cfg{NL, topk, H, I, eos};
    dee::Generator gen(*dev, cache, pf, *wsrc, oracle, router, embed, sampler, cfg);

    std::vector<int> prompt = {prompt_token};
    std::printf("dee_bench: backend=%s mode=%s layers=%d experts=%d hidden=%d inter=%d topk=%d "
                "slots=%d prompt=%d max_new=%d\n",
                dev->backend_name(), mode.c_str(), NL, E, H, I, topk, slots, prompt_token, max_new);

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int> out = gen.generate(prompt, max_new);
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double sec = ms / 1000.0;
    std::printf("tokens generated : %d\n", static_cast<int>(out.size()));
    std::printf("decode time      : %.2f ms (%.1f tok/s)\n", ms, sec > 0 ? out.size() / sec : 0.0);
    std::printf("cache hits       : %ld\n", gen.cache_hits());
    std::printf("cache misses     : %ld\n", gen.cache_misses());
    std::printf("reactive prefetch: %ld\n", gen.reactive_prefetches());
    std::printf("oracle calls     : %ld\n", gen.prediction_calls());
    std::printf("first tokens     :");
    for (size_t i = 0; i < out.size() && i < 12; ++i) std::printf(" %d", out[i]);
    std::printf("\n");
    return 0;
}
