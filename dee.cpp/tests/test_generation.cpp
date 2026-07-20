// dee.cpp - Step 7 test: autoregressive generation loop + cache-miss-gated sync.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "dee/async_prefetcher.h"
#include "dee/cuda_device.h"
#include "dee/generation.h"
#include "dee/moe_forward.h"
#include "dee/oracle.h"
#include "dee/test_util.h"
#include "dee/vram_cache_manager.h"
#include "dee/weight_source.h"

namespace {
using dee::ExpertKey;
using dee::ExpertKeyHash;

// --- mocks ---------------------------------------------------------------

class ConstPriorityOracle : public dee::Oracle {
public:
    float priority(int, int) const override { return 0.f; }
};

// Oracle that always predicts the same fixed experts (the "good" DEE case).
class ConstPredictOracle : public dee::ExpertOracle {
public:
    ConstPredictOracle(int ne, int hd, std::vector<int> preds)
        : ne_(ne), hd_(hd), preds_(std::move(preds)) {}
    void predict(int, const float*, int, int topk, int* out, float* scores = nullptr) const override {
        for (int i = 0; i < topk; ++i) {
            out[i] = preds_[i % static_cast<int>(preds_.size())];
            if (scores) scores[i] = 1.0f - static_cast<float>(i) / topk;
        }
    }
    int num_experts() const override { return ne_; }
    int hidden_dim() const override { return hd_; }
    int lookahead() const override { return 1; }

private:
    int ne_, hd_;
    std::vector<int> preds_;
};

// Router that always selects the same fixed experts.
class ConstRouter : public dee::ExpertRouter {
public:
    ConstRouter(std::vector<int> ids, std::vector<float> scores)
        : ids_(std::move(ids)), scores_(std::move(scores)) {}
    void route(int, const float*, int, int topk, int* out, float* out_scores) const override {
        for (int i = 0; i < topk; ++i) {
            out[i] = ids_[i % static_cast<int>(ids_.size())];
            if (out_scores) out_scores[i] = scores_[i % static_cast<int>(scores_.size())];
        }
    }

private:
    std::vector<int> ids_;
    std::vector<float> scores_;
};

// Projection-aware FP32 mock weight source (mirrors test_step6).
class MockWeightSource : public dee::WeightSource {
public:
    MockWeightSource(int nl, int ne, int hidden, int inter)
        : nl_(nl), ne_(ne), H_(hidden), I_(inter) {
        const size_t gu = static_cast<size_t>(2) * I_ * H_;
        const size_t dn = static_cast<size_t>(H_) * I_;
        for (int L = 0; L < nl_; ++L)
            for (int E = 0; E < ne_; ++E) {
                std::vector<float> a(gu), b(dn);
                std::fill(a.begin(), a.end(), static_cast<float>((L * 7 + E * 13 + 1) & 0xFF));
                std::fill(b.begin(), b.end(), static_cast<float>((L * 11 + E * 17 + 2) & 0xFF));
                data_[{L, E}] = {std::move(a), std::move(b)};
            }
        gu_bytes_ = gu * sizeof(float);
        dn_bytes_ = dn * sizeof(float);
    }
    int num_layers() const override { return nl_; }
    int experts_per_layer() const override { return ne_; }
    int num_tensors_per_expert() const override { return dee::kNumExpertTensors; }
    void expert_tensors(int layer, int expert_id, dee::ExpertTensorView* views) override {
        auto& d = data_.at(ExpertKey{layer, expert_id});
        views[0] = dee::ExpertTensorView{d.gu.data(), gu_bytes_, 0, dee::ExpertTensor::GateUp};
        views[1] = dee::ExpertTensorView{d.dn.data(), dn_bytes_, gu_bytes_, dee::ExpertTensor::Down};
    }
    size_t expert_total_bytes(int, int) const override { return gu_bytes_ + dn_bytes_; }
    std::string tensor_name(int layer, int e, dee::ExpertTensor t) const override {
        return "mock." + std::to_string(layer) + "." + std::to_string(e) + "." +
               (t == dee::ExpertTensor::GateUp ? "gu" : "dn");
    }
    void decode_bf16_to_fp32(const void* bf16, float* out, size_t n) const override {
        std::memcpy(out, bf16, n * sizeof(float));
    }
    const std::vector<float>& gate_up(int L, int e) const { return data_.at(ExpertKey{L, e}).gu; }
    const std::vector<float>& down(int L, int e) const { return data_.at(ExpertKey{L, e}).dn; }

private:
    struct Entry { std::vector<float> gu, dn; };
    struct KeyHash { size_t operator()(const ExpertKey& k) const noexcept { return ExpertKeyHash()(k); } };
    std::unordered_map<ExpertKey, Entry, KeyHash> data_;
    int nl_, ne_, H_, I_;
    size_t gu_bytes_, dn_bytes_;
};

// Deterministic embedder / sampler so generation is reproducible.
class LinearEmbedder : public dee::TokenEmbedder {
public:
    void embed(int token_id, float* out, int hidden_dim) const override {
        for (int i = 0; i < hidden_dim; ++i)
            out[i] = std::fmod(static_cast<float>(token_id * 31 + i * 7 + 1), 10.0f) - 5.0f;
    }
};

class LinearSampler : public dee::TokenSampler {
public:
    explicit LinearSampler(int vocab) : V_(vocab) {}
    int sample(const float* h, int) const override {
        int v = static_cast<int>(h[0] * 1000.0f);
        v %= V_;
        if (v < 0) v += V_;
        return v;
    }

private:
    int V_;
};

// Sampler that always emits a fixed token (used to test EOS early-stop).
class ConstSampler : public dee::TokenSampler {
public:
    explicit ConstSampler(int tok) : tok_(tok) {}
    int sample(const float*, int) const override { return tok_; }

private:
    int tok_;
};

// Independent reference of run_layers (reads weights straight from the mock,
// never touches VRAM) so we can cross-check the Generator's wiring.
std::vector<float> reference_run_layers(const MockWeightSource& ws, const ConstRouter& r,
                                        const std::vector<float>& xin, int nl, int topk, int H,
                                        int I) {
    std::vector<float> x = xin;
    std::vector<float> moe(H, 0.f), eout(H), g(I), u(I), a(I);
    std::vector<int> ids(topk);
    std::vector<float> scores(topk);
    for (int L = 0; L < nl; ++L) {
        r.route(L, x.data(), H, topk, ids.data(), scores.data());
        std::fill(moe.begin(), moe.end(), 0.f);
        for (int k = 0; k < topk; ++k) {
            const float* gu = ws.gate_up(L, ids[k]).data();
            const float* dn = ws.down(L, ids[k]).data();
            dee::expert_mlp(x.data(), H, I, gu, dn, eout.data(), g.data(), u.data(), a.data());
            for (int i = 0; i < H; ++i) moe[i] += scores[k] * eout[i];
        }
        for (int i = 0; i < H; ++i) x[i] += moe[i];
    }
    return x;
}

}  // namespace

int main() {
    const int nl = 2, topk = 2, H = 4, I = 4, ne = 4;
    const size_t slot_bytes = (static_cast<size_t>(2) * I * H + static_cast<size_t>(H) * I) * sizeof(float) + 64;

    // --- Test 1: oracle == router, enough slots -> misses only on first sight ---
    {
        auto dev = dee::make_cuda_device();
        ConstPriorityOracle coracle;
        MockWeightSource ws(nl, ne, H, I);
        dee::VramCacheManager cache(*dev, slot_bytes, /*num_slots=*/8, coracle);
        dee::AsyncPrefetcher pf(*dev, cache, ws);
        ConstPredictOracle oracle(ne, H, {0, 1});
        ConstRouter router({0, 1}, {1.0f, 0.5f});
        LinearEmbedder embed;
        LinearSampler sampler(64);
        dee::GeneratorConfig cfg{nl, topk, H, I, /*eos=*/-1};
        dee::Generator gen(*dev, cache, pf, ws, oracle, router, embed, sampler, cfg);

        auto out = gen.generate({5, 9}, /*max_new=*/3);

        DEE_CHECK_EQ(out.size(), 3u);
        DEE_CHECK_EQ(gen.prediction_calls(), 10L);  // 2 layers * 5 decoded tokens
        DEE_CHECK_EQ(gen.cache_misses(), 4L);        // first token's 4 experts, then all hits
        DEE_CHECK_EQ(gen.cache_hits(), 16L);
        DEE_CHECK_EQ(gen.reactive_prefetches(), 0L);  // oracle predicted what the router used
        DEE_CHECK_EQ(gen.cache_hits() + gen.cache_misses(), 20L);  // 2 layers * 5 tokens * topk
    }

    // --- Test 2: oracle mispredicts -> reactive prefetches on the used experts ---
    {
        auto dev = dee::make_cuda_device();
        ConstPriorityOracle coracle;
        MockWeightSource ws(nl, ne, H, I);
        dee::VramCacheManager cache(*dev, slot_bytes, 8, coracle);
        dee::AsyncPrefetcher pf(*dev, cache, ws);
        // Oracle predicts [0,1]; router actually uses [2,3].
        ConstPredictOracle oracle(ne, H, {0, 1});
        ConstRouter router({2, 3}, {1.0f, 0.5f});
        LinearEmbedder embed;
        LinearSampler sampler(64);
        dee::GeneratorConfig cfg{nl, topk, H, I, -1};
        dee::Generator gen(*dev, cache, pf, ws, oracle, router, embed, sampler, cfg);

        auto out = gen.generate({5, 9}, 3);

        DEE_CHECK_EQ(out.size(), 3u);
        DEE_CHECK_EQ(gen.cache_misses(), 4L);       // first token's 4 used experts
        DEE_CHECK_EQ(gen.cache_hits(), 16L);
        DEE_CHECK_EQ(gen.reactive_prefetches(), 4L);  // oracle never predicted [2,3]
        // We only synchronize experts we actually use: the predicted-but-unused
        // experts linger in-flight (prefetched) but are never synced.
        DEE_CHECK(cache.contains(ExpertKey{0, 0}));
        DEE_CHECK(!pf.is_ready(0, 0));   // predicted, never used -> never synced
        DEE_CHECK(pf.is_ready(0, 2));    // used -> synced
    }

    // --- Test 3: forward math matches a reference (prefetch -> slot -> expert_mlp) ---
    {
        auto dev = dee::make_cuda_device();
        ConstPriorityOracle coracle;
        MockWeightSource ws(nl, ne, H, I);
        dee::VramCacheManager cache(*dev, slot_bytes, 8, coracle);
        dee::AsyncPrefetcher pf(*dev, cache, ws);
        ConstPredictOracle oracle(ne, H, {0, 1});
        ConstRouter router({0, 1}, {1.0f, 0.5f});
        LinearEmbedder embed;
        LinearSampler sampler(64);
        dee::GeneratorConfig cfg{nl, topk, H, I, -1};
        dee::Generator gen(*dev, cache, pf, ws, oracle, router, embed, sampler, cfg);

        auto out = gen.generate({5}, /*max_new=*/1);
        DEE_CHECK_EQ(out.size(), 1u);

        // Reference: run the prompt through the same layers, read straight from
        // the mock weights, then sample. Must equal the Generator's first token.
        std::vector<float> x(H);
        embed.embed(5, x.data(), H);
        std::vector<float> ref = reference_run_layers(ws, router, x, nl, topk, H, I);
        int expected = sampler.sample(ref.data(), H);
        DEE_CHECK_EQ(out[0], expected);
    }

    // --- Test 4: tiny cache (forced eviction) still produces correct output ---
    {
        auto dev = dee::make_cuda_device();
        ConstPriorityOracle coracle;
        MockWeightSource ws(nl, ne, H, I);
        // Only 2 slots -> experts are constantly evicted, but each is read while
        // resident, so the math must match the no-eviction reference.
        dee::VramCacheManager cache(*dev, slot_bytes, /*num_slots=*/2, coracle);
        dee::AsyncPrefetcher pf(*dev, cache, ws);
        ConstPredictOracle oracle(ne, H, {0, 1});
        ConstRouter router({0, 1}, {1.0f, 0.5f});
        LinearEmbedder embed;
        LinearSampler sampler(64);
        dee::GeneratorConfig cfg{nl, topk, H, I, -1};
        dee::Generator gen(*dev, cache, pf, ws, oracle, router, embed, sampler, cfg);

        auto out = gen.generate({5}, 1);
        DEE_CHECK_EQ(out.size(), 1u);
        std::vector<float> x(H);
        embed.embed(5, x.data(), H);
        std::vector<float> ref = reference_run_layers(ws, router, x, nl, topk, H, I);
        DEE_CHECK_EQ(out[0], sampler.sample(ref.data(), H));
    }

    // --- Test 5: EOS stops generation early ---
    {
        auto dev = dee::make_cuda_device();
        ConstPriorityOracle coracle;
        MockWeightSource ws(nl, ne, H, I);
        dee::VramCacheManager cache(*dev, slot_bytes, 8, coracle);
        dee::AsyncPrefetcher pf(*dev, cache, ws);
        ConstPredictOracle oracle(ne, H, {0, 1});
        ConstRouter router({0, 1}, {1.0f, 0.5f});
        LinearEmbedder embed;
        const int eos = 7;
        ConstSampler sampler(eos);
        dee::GeneratorConfig cfg{nl, topk, H, I, eos};
        dee::Generator gen(*dev, cache, pf, ws, oracle, router, embed, sampler, cfg);

        auto out = gen.generate({5}, /*max_new=*/5);
        DEE_CHECK_EQ(out.size(), 1u);
        DEE_CHECK_EQ(out[0], eos);
        // Only the prefill ran layers; the EOS break skips the post-sample step.
        DEE_CHECK_EQ(gen.prediction_calls(), 2L);
    }

    return dee::test::report("dee.cpp Step 7: autoregressive generation loop");
}
