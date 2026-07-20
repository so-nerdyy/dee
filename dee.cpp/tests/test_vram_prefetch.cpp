// dee.cpp - Step 5: smoke test (CPU / mock CUDA).
//
// Exercises the VRAM cache + async prefetcher end-to-end without a GPU:
//   - arena capacity and free-list accounting
//   - prefetch -> copy (all projections) -> verify bytes landed in the right slot
//   - Oracle-priority eviction and LRU tie-break
//   - dedicated prefetch stream + per-expert event isolation
//   - cache-hit semantics (no eviction / re-copy on re-request)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "dee/async_prefetcher.h"
#include "dee/cuda_device.h"
#include "dee/test_util.h"
#include "dee/vram_cache_manager.h"
#include "dee/weight_source.h"

namespace {

using dee::ExpertKey;
using dee::ExpertKeyHash;

// Oracle whose priority is an arbitrary function of (layer, expert).
class LambdaOracle : public dee::Oracle {
public:
    explicit LambdaOracle(std::function<float(int, int)> fn) : fn_(std::move(fn)) {}
    float priority(int layer, int expert_id) const override {
        return fn_ ? fn_(layer, expert_id) : 0.f;
    }

private:
    std::function<float(int, int)> fn_;
};

// Mock weight source: each expert has two FP32 projection buffers (gate_up,
// down) filled with deterministic patterns so prefetch copies can be verified.
class MockWeightSource : public dee::WeightSource {
public:
    MockWeightSource(int num_layers, int experts_per_layer, size_t gate_up_elems, size_t down_elems)
        : nl_(num_layers), ne_(experts_per_layer), gu_(gate_up_elems), dn_(down_elems) {
        for (int L = 0; L < nl_; ++L) {
            for (int E = 0; E < ne_; ++E) {
                std::vector<float> gu(gu_), dn(dn_);
                std::fill(gu.begin(), gu.end(), static_cast<float>((L * 7 + E * 13 + 1) & 0xFF));
                std::fill(dn.begin(), dn.end(), static_cast<float>((L * 11 + E * 17 + 2) & 0xFF));
                data_[{L, E}] = {std::move(gu), std::move(dn)};
            }
        }
    }

    int num_layers() const override { return nl_; }
    int experts_per_layer() const override { return ne_; }
    int num_tensors_per_expert() const override { return dee::kNumExpertTensors; }

    void expert_tensors(int layer, int expert_id, dee::ExpertTensorView* views) override {
        auto& d = data_.at(ExpertKey{layer, expert_id});
        size_t gu_bytes = d.gu.size() * sizeof(float);
        size_t dn_bytes = d.dn.size() * sizeof(float);
        views[0] = dee::ExpertTensorView{d.gu.data(), gu_bytes, 0, dee::ExpertTensor::GateUp};
        views[1] = dee::ExpertTensorView{d.dn.data(), dn_bytes, gu_bytes, dee::ExpertTensor::Down};
    }
    size_t expert_total_bytes(int, int) const override { return (gu_ + dn_) * sizeof(float); }
    std::string tensor_name(int layer, int expert_id, dee::ExpertTensor t) const override {
        return "mock.layers." + std::to_string(layer) + ".expert." + std::to_string(expert_id) + "." +
               (t == dee::ExpertTensor::GateUp ? "gate_up" : "down");
    }
    void decode_bf16_to_fp32(const void* bf16, float* out, size_t n) const override {
        std::memcpy(out, bf16, n * sizeof(float));  // mock already stores FP32
    }

private:
    struct Entry {
        std::vector<float> gu, dn;
    };
    struct KeyHash {
        size_t operator()(const ExpertKey& k) const noexcept { return ExpertKeyHash()(k); }
    };
    std::unordered_map<ExpertKey, Entry, KeyHash> data_;
    int nl_, ne_;
    size_t gu_, dn_;
};

float vgu_val(int L, int E) { return static_cast<float>((L * 7 + E * 13 + 1) & 0xFF); }
float vdn_val(int L, int E) { return static_cast<float>((L * 11 + E * 17 + 2) & 0xFF); }
bool floats_match(const std::byte* p, size_t n, float v) {
    const float* f = reinterpret_cast<const float*>(p);
    for (size_t i = 0; i < n; ++i)
        if (f[i] != v) return false;
    return true;
}

}  // namespace

int main() {
    using namespace dee;

    // --- Test 1: arena capacity + free-list accounting -------------------
    {
        auto dev = make_cuda_device();
        LambdaOracle oracle([](int, int) { return 0.f; });
        VramCacheManager cache(*dev, /*slot_bytes=*/1024, /*num_slots=*/4, oracle);
        DEE_CHECK_EQ(cache.num_slots(), 4);
        DEE_CHECK_EQ(cache.free_slots(), 4);
        DEE_CHECK_EQ(cache.num_cached(), 0);
    }

    // --- Test 2: prefetch fills cache and copies all projections ----------
    {
        auto dev = make_cuda_device();
        LambdaOracle oracle([](int, int) { return 1.f; });
        VramCacheManager cache(*dev, 1024, 4, oracle);
        MockWeightSource ws(2, 8, /*gate_up_elems=*/64, /*down_elems=*/32);
        AsyncPrefetcher pf(*dev, cache, ws);

        for (int E = 0; E < 4; ++E) pf.prefetch_expert(0, E);
        pf.synchronize_all();

        for (int E = 0; E < 4; ++E) {
            DEE_CHECK(pf.is_ready(0, E));
            DEE_CHECK(cache.contains(ExpertKey{0, E}));
        }
        DEE_CHECK_EQ(cache.num_cached(), 4);
        DEE_CHECK_EQ(cache.free_slots(), 0);

        // gate_up is 64 floats (256 B) then down is 32 floats (128 B).
        std::vector<std::byte> buf(384);
        dev->memcpy_d2h_sync(buf.data(), cache.ptr_of(ExpertKey{0, 0}), 384);
        DEE_CHECK(floats_match(buf.data(), 64, vgu_val(0, 0)));
        DEE_CHECK(floats_match(buf.data() + 256, 32, vdn_val(0, 0)));
        dev->memcpy_d2h_sync(buf.data(), cache.ptr_of(ExpertKey{0, 3}), 384);
        DEE_CHECK(floats_match(buf.data(), 64, vgu_val(0, 3)));
        DEE_CHECK(floats_match(buf.data() + 256, 32, vdn_val(0, 3)));
    }

    // --- Test 3: Oracle priority drives eviction -------------------------
    {
        auto dev = make_cuda_device();
        LambdaOracle oracle([](int, int E) { return static_cast<float>(E); });
        VramCacheManager cache(*dev, 1024, 4, oracle);
        MockWeightSource ws(2, 16, 64, 32);
        AsyncPrefetcher pf(*dev, cache, ws);

        for (int E = 0; E < 4; ++E) pf.prefetch_expert(0, E);
        pf.synchronize_all();

        // Insert a new expert (priority 4); among cached {0..3} the lowest
        // priority is (0,0) -> it is evicted.
        pf.prefetch_expert(0, 4);
        pf.synchronize_expert(0, 4);

        DEE_CHECK(cache.contains(ExpertKey{0, 4}));
        DEE_CHECK(!cache.contains(ExpertKey{0, 0}));
        DEE_CHECK_EQ(cache.last_evicted().layer, 0);
        DEE_CHECK_EQ(cache.last_evicted().expert_id, 0);
        DEE_CHECK_EQ(cache.num_cached(), 4);

        std::vector<std::byte> buf(384);
        dev->memcpy_d2h_sync(buf.data(), cache.ptr_of(ExpertKey{0, 4}), 384);
        DEE_CHECK(floats_match(buf.data(), 64, vgu_val(0, 4)));
    }

    // --- Test 4: LRU tie-break when priorities are equal -----------------
    {
        auto dev = make_cuda_device();
        LambdaOracle constant([](int, int) { return 0.f; });
        VramCacheManager cache(*dev, 1024, 4, constant);
        MockWeightSource ws(2, 8, 64, 32);
        AsyncPrefetcher pf(*dev, cache, ws);

        for (int E = 0; E < 4; ++E) pf.prefetch_expert(0, E);
        pf.synchronize_all();

        cache.touch(ExpertKey{0, 0});
        cache.touch(ExpertKey{0, 1});

        pf.prefetch_expert(0, 4);
        pf.synchronize_expert(0, 4);

        // With equal priority, the LRU (lowest last_used) victim is (0,2).
        DEE_CHECK(!cache.contains(ExpertKey{0, 2}));
        DEE_CHECK(cache.contains(ExpertKey{0, 0}));
        DEE_CHECK(cache.contains(ExpertKey{0, 1}));
        DEE_CHECK(cache.contains(ExpertKey{0, 3}));
        DEE_CHECK_EQ(cache.last_evicted().expert_id, 2);
    }

    // --- Test 5: dedicated stream + per-expert event isolation -----------
    {
        auto dev = make_cuda_device();
        LambdaOracle constant([](int, int) { return 0.f; });
        VramCacheManager cache(*dev, 1024, 8, constant);
        MockWeightSource ws(2, 8, 64, 32);
        AsyncPrefetcher pf(*dev, cache, ws);

        pf.prefetch_expert(0, 0);
        pf.prefetch_expert(0, 1);
        pf.prefetch_expert(0, 2);
        DEE_CHECK_EQ(pf.in_flight_count(), 3);
        DEE_CHECK(!pf.is_ready(0, 0));
        DEE_CHECK(!pf.is_ready(0, 1));
        DEE_CHECK(!pf.is_ready(0, 2));

        DEE_CHECK(pf.synchronize_expert(0, 1));
        DEE_CHECK(pf.is_ready(0, 1));
        DEE_CHECK(!pf.is_ready(0, 0));
        DEE_CHECK(!pf.is_ready(0, 2));
        DEE_CHECK_EQ(pf.in_flight_count(), 2);

        DEE_CHECK(!pf.synchronize_expert(0, 1));  // already synced -> no-op

        pf.synchronize_all();
        DEE_CHECK(pf.is_ready(0, 0));
        DEE_CHECK(pf.is_ready(0, 2));
        DEE_CHECK_EQ(pf.in_flight_count(), 0);
    }

    // --- Test 6: cache-hit semantics (no eviction on re-request) ---------
    {
        auto dev = make_cuda_device();
        LambdaOracle constant([](int, int) { return 0.f; });
        VramCacheManager cache(*dev, 1024, 4, constant);

        auto a1 = cache.acquire(ExpertKey{0, 0}, 384);
        DEE_CHECK(!a1.hit);
        auto a2 = cache.acquire(ExpertKey{0, 0}, 384);
        DEE_CHECK(a2.hit);
        DEE_CHECK_EQ(a1.slot, a2.slot);
        DEE_CHECK_EQ(cache.num_cached(), 1);

        bool threw = false;
        try {
            cache.acquire(ExpertKey{0, 1}, 2048);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        DEE_CHECK(threw);
    }

    // --- Test 7: lower-priority cached expert evicted for new low expert -
    {
        auto dev = make_cuda_device();
        LambdaOracle oracle([](int, int E) { return static_cast<float>(E); });
        VramCacheManager cache(*dev, 1024, 4, oracle);
        MockWeightSource ws(2, 16, 64, 32);
        AsyncPrefetcher pf(*dev, cache, ws);

        for (int E = 5; E <= 8; ++E) pf.prefetch_expert(0, E);
        pf.synchronize_all();

        pf.prefetch_expert(0, 0);
        pf.synchronize_expert(0, 0);

        DEE_CHECK(cache.contains(ExpertKey{0, 0}));
        DEE_CHECK(!cache.contains(ExpertKey{0, 5}));
        DEE_CHECK_EQ(cache.last_evicted().expert_id, 5);
        DEE_CHECK(cache.contains(ExpertKey{0, 6}));
        DEE_CHECK(cache.contains(ExpertKey{0, 7}));
        DEE_CHECK(cache.contains(ExpertKey{0, 8}));
    }

    return dee::test::report("dee.cpp Step 5: GPU memory & async prefetcher");
}
