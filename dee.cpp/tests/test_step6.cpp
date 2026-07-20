// dee.cpp - Step 6 test: OracleScheduler wiring + MoE expert MLP forward.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "dee/async_prefetcher.h"
#include "dee/cuda_device.h"
#include "dee/moe_forward.h"
#include "dee/oracle.h"
#include "dee/oracle_scheduler.h"
#include "dee/test_util.h"
#include "dee/vram_cache_manager.h"
#include "dee/weight_source.h"

namespace {

using dee::ExpertKey;
using dee::ExpertKeyHash;

class ConstOracle : public dee::Oracle {
public:
    float priority(int, int) const override { return 0.f; }
};

// Projection-aware mock weight source (same shape as the Step 5 test).
class MockWeightSource : public dee::WeightSource {
public:
    MockWeightSource(int nl, int ne, size_t gu, size_t dn) : nl_(nl), ne_(ne), gu_(gu), dn_(dn) {
        for (int L = 0; L < nl_; ++L)
            for (int E = 0; E < ne_; ++E) {
                std::vector<float> a(gu_), b(dn_);
                std::fill(a.begin(), a.end(), static_cast<float>((L * 7 + E * 13 + 1) & 0xFF));
                std::fill(b.begin(), b.end(), static_cast<float>((L * 11 + E * 17 + 2) & 0xFF));
                data_[{L, E}] = {std::move(a), std::move(b)};
            }
    }
    int num_layers() const override { return nl_; }
    int experts_per_layer() const override { return ne_; }
    int num_tensors_per_expert() const override { return dee::kNumExpertTensors; }
    void expert_tensors(int layer, int expert_id, dee::ExpertTensorView* views) override {
        auto& d = data_.at(ExpertKey{layer, expert_id});
        size_t gb = d.gu.size() * sizeof(float);
        size_t db = d.dn.size() * sizeof(float);
        views[0] = dee::ExpertTensorView{d.gu.data(), gb, 0, dee::ExpertTensor::GateUp};
        views[1] = dee::ExpertTensorView{d.dn.data(), db, gb, dee::ExpertTensor::Down};
    }
    size_t expert_total_bytes(int, int) const override { return (gu_ + dn_) * sizeof(float); }
    std::string tensor_name(int layer, int expert_id, dee::ExpertTensor t) const override {
        return "mock." + std::to_string(layer) + "." + std::to_string(expert_id) + "." +
               (t == dee::ExpertTensor::GateUp ? "gu" : "dn");
    }
    void decode_bf16_to_fp32(const void* bf16, float* out, size_t n) const override {
        std::memcpy(out, bf16, n * sizeof(float));
    }

private:
    struct Entry { std::vector<float> gu, dn; };
    struct KeyHash { size_t operator()(const ExpertKey& k) const noexcept { return ExpertKeyHash()(k); } };
    std::unordered_map<ExpertKey, Entry, KeyHash> data_;
    int nl_, ne_;
    size_t gu_, dn_;
};

// Deterministic mock oracle: base expert derived from hidden[0].
class MockExpertOracle : public dee::ExpertOracle {
public:
    MockExpertOracle(int ne, int hd, int lk = 10) : ne_(ne), hd_(hd), lk_(lk) {}
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
    int lookahead() const override { return lk_; }

private:
    int ne_, hd_, lk_;
};

}  // namespace

int main() {
    using namespace dee;

    // --- Test 1: oracle determinism ---
    {
        MockExpertOracle oracle(16, 8);
        float h[8]{};
        h[0] = 0.123f;  // base = (int)123 % 16 = 11
        int pred[4];
        oracle.predict(0, h, 8, 4, pred);
        int expected[4] = {11, 2, 9, 0};
        for (int i = 0; i < 4; ++i) DEE_CHECK_EQ(pred[i], expected[i]);
    }

    // --- Test 2: scheduler prefetches predicted experts ---
    {
        auto dev = make_cuda_device();
        ConstOracle coracle;
        VramCacheManager cache(*dev, /*slot_bytes=*/384, /*num_slots=*/8, coracle);
        MockWeightSource ws(2, 16, 64, 32);
        AsyncPrefetcher pf(*dev, cache, ws);
        MockExpertOracle oracle(16, 8);
        OracleScheduler sched(oracle, pf, cache, /*topk=*/4);

        float h[8]{};
        h[0] = 0.123f;  // predicts {11,2,9,0}
        sched.on_token_decoded(0, h, 8);
        DEE_CHECK_EQ(pf.in_flight_count(), 4);  // 4 new prefetches enqueued

        DEE_CHECK(sched.ensure_ready(0, 11));  // sync one specific expert
        DEE_CHECK(pf.is_ready(0, 11));
        DEE_CHECK(!pf.is_ready(0, 2));  // others still in flight

        sched.synchronize_all();
        for (int e : {11, 2, 9, 0}) DEE_CHECK(pf.is_ready(0, e));
        DEE_CHECK_EQ(cache.num_cached(), 4);
    }

    // --- Test 3: scheduler does not re-prefetch already-resident experts ---
    {
        auto dev = make_cuda_device();
        ConstOracle coracle;
        VramCacheManager cache(*dev, 384, 8, coracle);
        MockWeightSource ws(2, 16, 64, 32);
        AsyncPrefetcher pf(*dev, cache, ws);
        MockExpertOracle oracle(16, 8);
        OracleScheduler sched(oracle, pf, cache, 4);

        float h[8]{};
        h[0] = 0.123f;
        sched.on_token_decoded(0, h, 8);
        sched.synchronize_all();
        DEE_CHECK_EQ(cache.num_cached(), 4);

        // Same prediction again -> nothing new should be prefetched.
        sched.on_token_decoded(0, h, 8);
        DEE_CHECK_EQ(pf.in_flight_count(), 0);
        DEE_CHECK_EQ(cache.num_cached(), 4);
    }

    // --- Test 4: MoE expert MLP forward (hand-checked reference) ---
    {
        const int hidden = 4, inter = 3;
        std::vector<float> gate_up(2 * inter * hidden);
        std::vector<float> down(hidden * inter);
        for (size_t i = 0; i < gate_up.size(); ++i)
            gate_up[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
        for (size_t i = 0; i < down.size(); ++i)
            down[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
        float x[4] = {1.f, 2.f, -1.f, 0.5f};

        std::vector<float> gate(inter), up(inter), act(inter), out(hidden);
        expert_mlp(x, hidden, inter, gate_up.data(), down.data(), out.data(), gate.data(),
                   up.data(), act.data());

        // Independent reference computation.
        auto matvec = [](const std::vector<float>& W, const float* xv, std::vector<float>& y, int rows,
                        int cols) {
            for (int r = 0; r < rows; ++r) {
                float s = 0.f;
                for (int c = 0; c < cols; ++c) s += W[r * cols + c] * xv[c];
                y[r] = s;
            }
        };
        auto silu = [](float z) { return z / (1.0f + std::exp(-z)); };
        std::vector<float> rgate(inter), rup(inter), ract(inter), rout(hidden);
        matvec(gate_up, x, rgate, inter, hidden);
        std::vector<float> w_up(gate_up.begin() + inter * hidden, gate_up.end());
        matvec(w_up, x, rup, inter, hidden);
        for (int i = 0; i < inter; ++i) ract[i] = silu(rgate[i]) * rup[i];
        matvec(down, act.data(), rout, hidden, inter);

        for (int i = 0; i < hidden; ++i)
            DEE_CHECK(std::fabs(out[i] - rout[i]) < 1e-5f);
        for (int i = 0; i < inter; ++i)
            DEE_CHECK(std::fabs(gate[i] - rgate[i]) < 1e-5f);

        // Independent reference (naive nested loops, not using matvec) to guard
        // against a matvec/algorithm bug: out should match to ~1e-5.
        float eout[4] = {0};
        for (int h = 0; h < hidden; ++h) {
            for (int k = 0; k < inter; ++k) {
                float g = 0.f, u = 0.f;
                for (int j = 0; j < hidden; ++j) {
                    g += gate_up[k * hidden + j] * x[j];
                    u += gate_up[(inter + k) * hidden + j] * x[j];
                }
                float a = silu(g) * u;
                eout[h] += down[h * inter + k] * a;
            }
        }
        for (int i = 0; i < hidden; ++i)
            DEE_CHECK(std::fabs(out[i] - eout[i]) < 1e-2f);
    }

    return dee::test::report("dee.cpp Step 6: Oracle scheduler + MoE forward");
}
