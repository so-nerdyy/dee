// tests/test_host_pack_cache.cpp
//
// Unit tests for the Stage-1 HostPackCache LRU (packed expert byte residency).
// Validates: basic get/fill, hit/miss accounting, LRU eviction order under a
// byte budget, budget clamping (entry larger than budget never caches), and
// clear() resetting everything. Runs on any host (no CUDA dependency).

#include "dee/host_pack_cache.h"

#include <cstdint>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// Fill a buffer with a deterministic pattern derived from (key, offset).
void pattern_fill(uint8_t* dst, size_t n, uint64_t key) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<uint8_t>((key * 31 + i * 7) & 0xFF);
    }
}

bool matches_pattern(const uint8_t* data, size_t n, uint64_t key) {
    for (size_t i = 0; i < n; ++i) {
        if (data[i] != static_cast<uint8_t>((key * 31 + i * 7) & 0xFF)) {
            return false;
        }
    }
    return true;
}

struct BatchContext {
    uint64_t key = 0;
    std::atomic<int>* active = nullptr;
    std::atomic<int>* max_active = nullptr;
    bool fail = false;
    std::vector<uint64_t>* invocation_order = nullptr;
};

bool batch_fill(void* raw, uint8_t* dst, size_t n) {
    auto* context = static_cast<BatchContext*>(raw);
    if (!context || !dst) return false;
    const int active = context->active->fetch_add(1) + 1;
    int observed = context->max_active->load();
    while (observed < active &&
           !context->max_active->compare_exchange_weak(observed, active)) {}
    if (context->invocation_order) {
        context->invocation_order->push_back(context->key);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    pattern_fill(dst, n, context->key);
    context->active->fetch_sub(1);
    return !context->fail;
}

// Trivially-succeeding batch fill for capacity/rejection tests.
bool ok_fill(void* /*context*/, uint8_t* dst, size_t n) {
    if (!dst) return false;
    std::memset(dst, 0xC3, n);
    return true;
}

void test_basic_hit_miss() {
    dee::HostPackCache cache;
    cache.set_budget(1024);
    int fills = 0;
    const uint8_t* a = cache.get(
        1, 256, [&](uint8_t* dst, size_t n) {
            ++fills;
            pattern_fill(dst, n, 1);
        });
    check(a != nullptr, "first get allocates");
    check(fills == 1, "miss fills once");
    const uint8_t* a2 = cache.get(
        1, 256, [&](uint8_t* dst, size_t n) {
            ++fills;
            pattern_fill(dst, n, 1);
        });
    check(a2 == a, "hit reuses the same buffer");
    check(fills == 1, "hit does not refill");
    check(matches_pattern(a2, 256, 1), "cached bytes intact");
    check(cache.stats().hits == 1 && cache.stats().misses == 1,
          "hit/miss counters");
    check(cache.stats().entries == 1 && cache.stats().bytes == 256,
          "entry accounting");
}

void test_lru_eviction() {
    dee::HostPackCache cache;
    cache.set_budget(600);  // fits 2 x 256 + 1 x 64, not a third 256
    std::vector<int> fill_count(4, 0);
    auto getter = [&](int key) {
        return cache.get(
            static_cast<uint64_t>(key), 256,
            [&](uint8_t* dst, size_t n) {
                ++fill_count[key];
                pattern_fill(dst, n, static_cast<uint64_t>(key));
            });
    };
    check(getter(1) != nullptr, "insert 1");
    check(getter(2) != nullptr, "insert 2");
    check(fill_count[1] == 1 && fill_count[2] == 1, "first fills");
    // Touch 1 (most recent), then insert 3 -> evicts 2 (LRU).
    check(getter(1) != nullptr, "touch 1");
    check(getter(3) != nullptr, "insert 3 evicts 2");
    check(cache.stats().evictions == 1, "one eviction");
    check(fill_count[2] == 1, "2 not refilled (evicted once)");
    check(!cache.contains(2), "2 evicted");
    check(cache.contains(1) && cache.contains(3), "1 and 3 resident");
    // Re-get 2 -> refill (evicts LRU = 1).
    check(getter(2) != nullptr, "re-insert 2");
    check(fill_count[2] == 2, "2 refilled after eviction");
    check(cache.stats().misses >= 4, "miss accounting");
    const uint8_t* b = cache.get(2, 256, [&](uint8_t* dst, size_t n) {
        ++fill_count[2];
        pattern_fill(dst, n, 2);
    });
    check(matches_pattern(b, 256, 2), "refilled content correct");
}

void test_oversize_and_clear() {
    dee::HostPackCache cache;
    cache.set_budget(128);
    const uint8_t* big = cache.get(
        99, 256, [](uint8_t*, size_t) {});
    check(big == nullptr, "entry larger than budget never caches");
    check(cache.stats().misses == 1, "oversize counted as miss");
    check(cache.stats().entries == 0, "oversize not stored");
    const uint8_t* small = cache.get(
        5, 16, [](uint8_t* dst, size_t n) { std::memset(dst, 0xAB, n); });
    check(small != nullptr, "small entry caches");
    cache.clear();
    check(cache.stats().entries == 0 && cache.stats().bytes == 0,
          "clear resets accounting");
    check(!cache.contains(5), "clear drops entries");
}

void test_same_buffer_stays_valid_until_evicted() {
    dee::HostPackCache cache;
    // Budget fits 7(128) + 8(256) + 9(256) but not 10(256) more.
    cache.set_budget(128 + 256 + 256 + 128);
    const uint8_t* a = cache.get(
        7, 128, [](uint8_t* dst, size_t n) { std::memset(dst, 0x11, n); });
    const uint8_t* b = cache.get(
        8, 256, [](uint8_t* dst, size_t n) { std::memset(dst, 0x22, n); });
    check(a != nullptr && b != nullptr, "both fit");
    // Touch 8 (MRU), insert 9 -> fits. Insert 10 -> evicts 7 (LRU).
    cache.get(8, 256, [](uint8_t*, size_t) {});
    cache.get(9, 256, [](uint8_t* dst, size_t n) { std::memset(dst, 0x33, n); });
    const uint8_t* ten = cache.get(
        10, 256, [](uint8_t* dst, size_t n) { std::memset(dst, 0x44, n); });
    check(ten != nullptr, "insert 10 fits after evicting 7");
    check(cache.contains(8) && cache.contains(9) && cache.contains(10),
          "8/9/10 retained after evict 7");
    check(!cache.contains(7), "7 gone (LRU)");
    const uint8_t* nine_again = cache.get(
        9, 256, [](uint8_t* dst, size_t n) { std::memset(dst, 0x55, n); });
    check(nine_again != nullptr && nine_again[0] == 0x33,
          "re-touch returns prior payload");
}

void test_bounded_batch_fill_and_rollback() {
    dee::HostPackCache cache;
    cache.set_budget(4096);
    check(cache.set_fill_lanes(3), "three bounded fill lanes initialize");
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
    BatchContext contexts[4];
    dee::HostPackCache::BatchRequest requests[4];
    dee::HostPackCache::BatchResult results[4];
    for (size_t index = 0; index < 4; ++index) {
        contexts[index] = {
            static_cast<uint64_t>(20 + index), &active, &max_active, false};
        requests[index] = {
            contexts[index].key, 256, &batch_fill, &contexts[index],
            static_cast<uint64_t>(100 - index)};
    }
    check(cache.get_batch(requests, 4, results),
          "bounded concurrent batch succeeds");
    check(max_active.load() >= 2 && max_active.load() <= 3,
          "fill concurrency obeys configured lane bound");
    for (size_t index = 0; index < 4; ++index) {
        check(results[index].success && results[index].fill_executed,
              "each cold batch request executes exactly one fill");
        check(matches_pattern(results[index].data, 256, contexts[index].key),
              "concurrent fill preserves deterministic bytes");
    }
    check(cache.stats().fill_batches == 1 &&
          cache.stats().concurrent_fill_batches == 1 &&
          cache.stats().fill_requests == 4 &&
          cache.stats().max_fill_queue_depth == 4 &&
          cache.stats().max_fill_lanes == 3,
          "bounded fill telemetry is exact");
    check(cache.stats().fill_worker_ms >= cache.stats().fill_batch_wall_ms &&
          cache.stats().fill_overlap_ms > 0.0,
          "concurrent worker overlap is measured");

    // A short/error fill is fail-closed and does not leave a readable cache
    // entry behind. Existing successful entries remain intact.
    BatchContext bad{99, &active, &max_active, true};
    dee::HostPackCache::BatchRequest bad_request{
        99, 128, &batch_fill, &bad};
    dee::HostPackCache::BatchResult bad_result;
    check(!cache.get_batch(&bad_request, 1, &bad_result),
          "failed fill rejects the complete batch");
    check(!cache.contains(99) && cache.get_if_present(99) == nullptr,
          "failed reservation is rolled back");
    check(cache.contains(20) && cache.contains(21),
          "failed fill does not corrupt prior cache entries");

    // A failure in a multi-record batch rolls back every reservation created
    // by that batch, not merely the record whose callback returned false.
    BatchContext mixed_contexts[2] = {
        {100, &active, &max_active, false},
        {101, &active, &max_active, true},
    };
    dee::HostPackCache::BatchRequest mixed_requests[2] = {
        {100, 128, &batch_fill, &mixed_contexts[0]},
        {101, 128, &batch_fill, &mixed_contexts[1]},
    };
    dee::HostPackCache::BatchResult mixed_results[2];
    check(!cache.get_batch(mixed_requests, 2, mixed_results),
          "a mixed-success batch fails closed");
    check(!cache.contains(100) && !cache.contains(101) &&
          mixed_results[0].data == nullptr && mixed_results[1].data == nullptr,
          "mixed batch leaves no partial reservation readable");
}

void test_batch_duplicate_is_single_materialization() {
    dee::HostPackCache cache;
    cache.set_budget(1024);
    check(cache.set_fill_lanes(2), "two bounded fill lanes initialize");
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
    BatchContext context{7, &active, &max_active, false};
    dee::HostPackCache::BatchRequest requests[2] = {
        {7, 128, &batch_fill, &context},
        {7, 128, &batch_fill, &context},
    };
    dee::HostPackCache::BatchResult results[2];
    check(cache.get_batch(requests, 2, results),
          "duplicate batch keys resolve");
    check(results[0].fill_executed && !results[1].fill_executed &&
          results[0].data == results[1].data,
          "duplicate key materializes once and shares exact bytes");
    check(cache.stats().misses == 1 && cache.stats().hits == 1,
          "duplicate batch preserves sequential hit/miss semantics");
}

void test_source_order_preserves_request_identity() {
    dee::HostPackCache cache;
    cache.set_budget(2048);
    check(cache.set_fill_lanes(1), "single deterministic fill lane initializes");
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
    std::vector<uint64_t> invocation_order;
    const uint64_t keys[4] = {40, 41, 42, 43};
    const uint64_t source_order[4] = {30, 10, 20, 20};
    BatchContext contexts[4];
    dee::HostPackCache::BatchRequest requests[4];
    dee::HostPackCache::BatchResult results[4];
    for (size_t index = 0; index < 4; ++index) {
        contexts[index] = {
            keys[index], &active, &max_active, false, &invocation_order};
        requests[index] = {
            keys[index], 128, &batch_fill, &contexts[index],
            source_order[index]};
    }

    check(cache.get_batch(requests, 4, results),
          "source-ordered batch succeeds");
    const std::vector<uint64_t> expected_order{41, 42, 43, 40};
    check(invocation_order == expected_order,
          "source order is stable and deterministic for equal hints");
    for (size_t index = 0; index < 4; ++index) {
        check(results[index].success &&
              matches_pattern(results[index].data, 128, keys[index]),
              "source scheduling preserves original request/result identity");
    }

    invocation_order.clear();
    dee::HostPackCache::BatchResult hit_results[4];
    check(cache.get_batch(requests, 4, hit_results),
          "source-ordered batch hits remain valid");
    check(invocation_order.empty(),
          "cache hits do not re-run source callbacks");
    for (size_t index = 0; index < 4; ++index) {
        check(hit_results[index].cache_hit && hit_results[index].success &&
              matches_pattern(hit_results[index].data, 128, keys[index]),
              "hit result order and expert identity remain unchanged");
    }
}

void test_scalar_fill_false_is_not_published() {
    dee::HostPackCache cache;
    cache.set_budget(1024);
    bool succeed = false;
    int fill_calls = 0;
    const std::function<bool(uint8_t*, size_t)> fill =
        [&](uint8_t* dst, size_t n) -> bool {
            ++fill_calls;
            if (!succeed) return false;
            pattern_fill(dst, n, 42);
            return true;
        };
    check(cache.get(42, 256, fill) == nullptr,
          "fill returning false fails the get");
    check(!cache.contains(42) && cache.get_if_present(42) == nullptr,
          "failed fill publishes no readable entry");
    check(cache.stats().misses == 1 && cache.stats().hits == 0 &&
          cache.stats().fill_failures == 1 && cache.stats().entries == 0,
          "failed fill counts one miss + one fill failure, no entry");
    succeed = true;
    const uint8_t* data = cache.get(42, 256, fill);
    check(data != nullptr && matches_pattern(data, 256, 42),
          "a later successful fill caches real bytes");
    check(fill_calls == 2 && cache.stats().misses == 2 &&
          cache.stats().entries == 1,
          "recovery re-runs the fill instead of serving a poisoned hit");
    check(cache.get(42, 256, fill) == data && fill_calls == 2,
          "post-recovery hit serves the recovered bytes");
}

void test_scalar_fill_throw_is_not_published() {
    dee::HostPackCache cache;
    cache.set_budget(1024);
    bool should_throw = true;
    const std::function<bool(uint8_t*, size_t)> fill =
        [&](uint8_t* dst, size_t n) -> bool {
            if (should_throw) throw std::runtime_error("fill boom");
            pattern_fill(dst, n, 7);
            return true;
        };
    check(cache.get(7, 128, fill) == nullptr,
          "throwing fill fails the get");
    check(!cache.contains(7), "throwing fill publishes nothing");
    check(cache.stats().fill_failures == 1 && cache.stats().misses == 1,
          "throw counted as fill failure + miss");
    should_throw = false;
    check(cache.get(7, 128, fill) != nullptr && cache.contains(7),
          "post-throw fill caches normally");
    // The legacy void-fill delegate must contain throws the same way.
    check(cache.get(8, 128, [](uint8_t*, size_t) {
              throw std::runtime_error("void fill boom");
          }) == nullptr,
          "void-fill delegate converts a throw into fail-closed nullptr");
    check(!cache.contains(8) && cache.stats().fill_failures == 2,
          "void-fill throw publishes nothing and counts a failure");
}

void test_same_key_size_mismatch_fails_closed() {
    dee::HostPackCache cache;
    cache.set_budget(1024);
    const std::function<bool(uint8_t*, size_t)> fill =
        [](uint8_t* dst, size_t n) -> bool {
            pattern_fill(dst, n, 5);
            return true;
        };
    const uint8_t* data = cache.get(5, 128, fill);
    check(data != nullptr, "resident insert at 128 bytes");
    check(cache.get(5, 256, fill) == nullptr,
          "same-key different-size bool fill returns nullptr");
    check(cache.get(5, 512, [](uint8_t*, size_t) {}) == nullptr,
          "same-key different-size void fill returns nullptr");
    check(cache.stats().size_mismatches == 2 &&
          cache.stats().misses == 3 && cache.stats().hits == 0,
          "mismatches counted as misses plus size_mismatches");
    const uint8_t* again = cache.get(5, 128, fill);
    check(again == data && matches_pattern(again, 128, 5),
          "original entry still served intact at its own size");
    check(cache.stats().hits == 1, "matching-size hit still counted");
    // get_batch fails the same request closed as well.
    dee::HostPackCache::BatchRequest request{5, 64, &ok_fill, nullptr};
    dee::HostPackCache::BatchResult result;
    check(!cache.get_batch(&request, 1, &result),
          "batch same-key different-size rejected");
    check(cache.stats().size_mismatches == 3, "batch mismatch counted");
    check(cache.get_if_present(5) == data,
          "mismatched batch leaves resident entry intact");
}

void test_budget_rejection_counters() {
    dee::HostPackCache cache;
    cache.set_budget(128);
    check(cache.get(1, 256, [](uint8_t*, size_t) {}) == nullptr,
          "oversize get rejected");
    check(cache.stats().budget_rejections == 1 &&
          cache.stats().misses == 1 && cache.stats().entries == 0,
          "oversize get counted once as miss + budget rejection");
    // Batch-level oversize reject.
    dee::HostPackCache::BatchRequest big{2, 256, &ok_fill, nullptr};
    dee::HostPackCache::BatchResult big_result;
    check(!cache.get_batch(&big, 1, &big_result),
          "oversize batch request rejected");
    check(cache.stats().budget_rejections == 2,
          "batch oversize request counted as budget rejection");
    // Batch capacity reject: individually-legal misses that together exceed
    // the budget.
    dee::HostPackCache::BatchRequest pair[2] = {
        {3, 96, &ok_fill, nullptr},
        {4, 96, &ok_fill, nullptr},
    };
    dee::HostPackCache::BatchResult pair_results[2];
    check(!cache.get_batch(pair, 2, pair_results),
          "batch whose unique misses exceed budget is rejected");
    check(cache.stats().budget_rejections == 3 &&
          cache.stats().entries == 0,
          "batch capacity reject counted, nothing published");
}

void test_evict_observer_reports_victims() {
    dee::HostPackCache cache;
    cache.set_budget(512);  // two 256-byte entries
    std::vector<std::pair<uint64_t, size_t>> evicted;
    cache.set_evict_observer(
        [&](uint64_t key, size_t nbytes) {
            evicted.emplace_back(key, nbytes);
        });
    auto put = [&](uint64_t key) {
        return cache.get(key, 256, [key](uint8_t* dst, size_t n) {
            pattern_fill(dst, n, key);
        });
    };
    check(put(1) != nullptr && put(2) != nullptr, "two residents fit");
    check(evicted.empty(), "no eviction before overflow");
    check(put(3) != nullptr, "overflow insert succeeds");
    check(evicted.size() == 1 && evicted[0].first == 1 &&
          evicted[0].second == 256,
          "observer reports the exact LRU victim key and size");
    check(put(4) != nullptr, "second overflow insert succeeds");
    check(evicted.size() == 2 && evicted[1].first == 2 &&
          evicted[1].second == 256,
          "observer reports the second victim");
    // The batch victim loop observes evictions identically.
    dee::HostPackCache::BatchRequest request{9, 256, &ok_fill, nullptr};
    dee::HostPackCache::BatchResult result;
    check(cache.get_batch(&request, 1, &result),
          "batch insert over a full cache succeeds");
    check(evicted.size() == 3 && evicted[2].first == 3 &&
          evicted[2].second == 256,
          "batch-path victim observed with key and size");
    check(cache.stats().evictions == 3,
          "observer calls match the eviction counter exactly");
    const size_t observed_before_clear = evicted.size();
    cache.clear();
    check(evicted.size() == observed_before_clear,
          "clear() is not reported as eviction");
}

void test_scalar_fill_instrumentation() {
    dee::HostPackCache cache;
    cache.set_budget(1024);
    check(cache.stats().scalar_fills == 0 &&
          cache.stats().scalar_fill_ms == 0.0,
          "scalar fill counters start at zero");
    cache.get(1, 128, [](uint8_t* dst, size_t n) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        std::memset(dst, 0x5A, n);
    });
    check(cache.stats().scalar_fills == 1 &&
          cache.stats().scalar_fill_ms > 0.0,
          "single-lane miss counts the fill and its wall time");
    cache.get(1, 128, [](uint8_t*, size_t) {});
    check(cache.stats().scalar_fills == 1,
          "hits do not count as scalar fills");
    const std::function<bool(uint8_t*, size_t)> bad =
        [](uint8_t*, size_t) -> bool { return false; };
    cache.get(2, 128, bad);
    check(cache.stats().scalar_fills == 2 &&
          cache.stats().fill_failures == 1,
          "failed scalar fills are still instrumented");
    // Batch fills keep their own counters; they must not alias scalar ones.
    dee::HostPackCache::BatchRequest request{3, 128, &ok_fill, nullptr};
    dee::HostPackCache::BatchResult result;
    check(cache.get_batch(&request, 1, &result),
          "batch fill succeeds alongside scalar stats");
    check(cache.stats().scalar_fills == 2 &&
          cache.stats().fill_requests == 1,
          "batch fills stay out of the scalar counters");
}

void test_batch_allprotected_stats_coherence() {
    dee::HostPackCache cache;
    cache.set_budget(768);  // three 256-byte entries
    auto put = [&](uint64_t key) {
        return cache.get(key, 256, [key](uint8_t* dst, size_t n) {
            pattern_fill(dst, n, key);
        });
    };
    check(put(1) != nullptr && put(2) != nullptr && put(3) != nullptr,
          "cache filled to budget");
    check(cache.stats().entries == 3 && cache.stats().bytes == 768,
          "pre-batch accounting exact");
    // Keys 1 and 2 hit (and become protected batch keys); keys 4 and 5 are
    // misses needing 512 bytes.  Evicting 3 frees only 256, then every
    // remaining resident is protected -> the batch must fail closed while
    // stats still reflect the one real eviction.
    dee::HostPackCache::BatchRequest requests[4] = {
        {1, 256, &ok_fill, nullptr},
        {2, 256, &ok_fill, nullptr},
        {4, 256, &ok_fill, nullptr},
        {5, 256, &ok_fill, nullptr},
    };
    dee::HostPackCache::BatchResult results[4];
    check(!cache.get_batch(requests, 4, results),
          "all-protected victim set fails the batch closed");
    check(cache.stats().evictions == 1 &&
          cache.stats().bytes == 512 && cache.stats().entries == 2,
          "stats reflect the partial eviction on early return");
    check(cache.contains(1) && cache.contains(2) && !cache.contains(3) &&
          !cache.contains(4) && !cache.contains(5),
          "map contents match the reported stats exactly");
    check(cache.stats().budget_rejections == 1,
          "all-protected reject counted as budget rejection");
    // The cache keeps serving coherently afterwards.
    check(put(6) != nullptr && cache.stats().entries == 3 &&
          cache.stats().bytes == 768,
          "post-reject inserts behave normally");
}

void test_peek_bytes_is_non_perturbing() {
    dee::HostPackCache cache;
    cache.set_budget(600);  // fits 2 x 256 + 1 x 64
    const uint8_t* a = cache.get(1, 256, [&](uint8_t* dst, size_t n) {
        pattern_fill(dst, n, 1); });
    const uint8_t* b = cache.get(2, 256, [&](uint8_t* dst, size_t n) {
        pattern_fill(dst, n, 2); });
    check(a && b, "seed two entries");
    const dee::HostPackCache::Stats before = cache.stats();
    size_t nbytes = 0;
    const uint8_t* peeked = cache.peek_bytes(1, &nbytes);
    check(peeked == a && nbytes == 256, "peek returns live bytes + size");
    check(cache.peek_bytes(99, &nbytes) == nullptr,
          "peek of absent key returns nullptr");
    check(cache.stats().hits == before.hits &&
          cache.stats().misses == before.misses &&
          cache.stats().entries == before.entries,
          "peek does not touch hit/miss/entry counters");
    // LRU order must be untouched: key 1 stays the oldest, so inserting
    // key 3 must still evict 1 — if peek had refreshed it, 2 would go.
    cache.peek_bytes(1, nullptr);
    cache.peek_bytes(1, nullptr);
    check(cache.get(3, 256, [&](uint8_t* dst, size_t n) {
            pattern_fill(dst, n, 3); }) != nullptr,
          "third insert fits after evicting one");
    check(!cache.contains(1) && cache.contains(2) && cache.contains(3),
          "peek left LRU order untouched (1 still evicted first)");
}

// ---------------------------------------------------------------------------
// Next-use eviction and prefetch hints.
// ---------------------------------------------------------------------------

uint64_t lkey(int layer, int expert) {
    return dee::NextUseEstimator::make_key(layer, expert);
}

void test_next_use_estimator_model() {
    dee::NextUseEstimator est;
    check(!est.enabled() && est.expected_next_use(lkey(1, 1)) == 0.0,
          "unconfigured estimator is inert");
    est.configure(4, 0.5, 0.25);
    est.begin_layer(0);
    est.on_use(lkey(0, 7));
    const double once = est.probability(lkey(0, 7));
    est.on_use(lkey(0, 7));
    check(est.probability(lkey(0, 7)) == once,
          "on_use is idempotent within one layer visit");
    check(once == 0.25 * 0.5 + 0.5, "first use = p0 decayed once plus alpha");
    check(est.distance(lkey(0, 7)) == 0 && est.distance(lkey(1, 0)) == 1 &&
          est.distance(lkey(3, 0)) == 3, "distance follows the layer cycle");
    est.begin_layer(1);
    check(est.distance(lkey(0, 7)) == 3, "passed layer is a full cycle away");
    check(est.probability(lkey(0, 7)) == once,
          "no decay before the layer is visited again");
    // Never-seen expert of the NEXT layer vs a used one of a far layer.
    check(est.expected_next_use(lkey(2, 9)) >
              est.expected_next_use(lkey(0, 7)),
          "an unseen expert ranks behind a recently used one");
    // Skip key (0,7) for two visits of layer 0: p decays by (1-alpha)^2.
    for (int visit = 0; visit < 2; ++visit) {
        est.begin_layer(0);
        est.begin_layer(1);
    }
    check(std::abs(est.probability(lkey(0, 7)) - once * 0.25) < 1e-12,
          "each missed visit decays the reuse estimate");
    est.clear();
    check(est.tracked() == 0 && est.probability(lkey(0, 7)) == 0.25,
          "clear forgets history");
}

// The Phase 6 r5 failure in miniature: a cyclic sweep that needs more
// records than fit.  LRU evicts every record right before its next use.
uint64_t run_cycle(dee::HostEvictionPolicy policy, uint64_t* lookups) {
    const size_t kBytes = 64;
    dee::HostPackCache cache;
    cache.set_budget(kBytes * 4);            // 4 of the 6 records fit
    cache.set_eviction_policy(policy);
    cache.set_next_use_cycle(3);
    for (int sweep = 0; sweep < 12; ++sweep) {
        for (int layer = 0; layer < 3; ++layer) {
            cache.begin_layer(layer);
            for (int expert = 0; expert < 2; ++expert) {
                const uint64_t key = lkey(layer, expert);
                const uint8_t* data = cache.get(
                    key, kBytes, [&](uint8_t* dst, size_t n) {
                        pattern_fill(dst, n, key);
                    });
                check(data && matches_pattern(data, kBytes, key),
                      "cyclic sweep returns the right bytes");
            }
        }
    }
    *lookups = cache.stats().hits + cache.stats().misses;
    return cache.stats().hits;
}

void test_next_use_fixes_cyclic_thrash() {
    uint64_t lru_lookups = 0, next_lookups = 0;
    const uint64_t lru_hits = run_cycle(dee::HostEvictionPolicy::Lru, &lru_lookups);
    const uint64_t next_hits =
        run_cycle(dee::HostEvictionPolicy::NextUse, &next_lookups);
    check(lru_lookups == 72 && next_lookups == 72, "both policies see 72 lookups");
    check(lru_hits == 0, "LRU never hits on an over-budget cyclic sweep");
    check(next_hits * 4 >= next_lookups,
          "NextUse keeps at least a quarter of the sweep resident");
    std::printf("cyclic sweep: lru %llu/72 hits, next_use %llu/72 hits\n",
                static_cast<unsigned long long>(lru_hits),
                static_cast<unsigned long long>(next_hits));
}

void test_next_use_without_cycle_is_lru() {
    dee::HostPackCache cache;
    cache.set_budget(256 * 2);
    cache.set_eviction_policy(dee::HostEvictionPolicy::NextUse);  // no cycle
    for (uint64_t key = 1; key <= 3; ++key) {
        cache.get(key, 256, [&](uint8_t* dst, size_t n) {
            pattern_fill(dst, n, key);
        });
    }
    check(!cache.contains(1) && cache.contains(2) && cache.contains(3),
          "NextUse with no layer cycle evicts exactly like LRU");
}

// A gate a hint fill can block on, so tests control what is queued/running.
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool open = false;
    std::atomic<int> entered{0};
    void wait() {
        entered.fetch_add(1);
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return open; });
    }
    void release() {
        { std::lock_guard<std::mutex> lock(mutex); open = true; }
        cv.notify_all();
    }
    void wait_entered(int count) {
        while (entered.load() < count) std::this_thread::yield();
    }
};

// Let every accepted hint finish and be published.  (drain_hints() is the
// other thing: it CANCELS whatever has not started.)
void settle_hints(dee::HostPackCache& cache) {
    while (cache.pending_hints() != 0) {
        cache.absorb_hints();
        std::this_thread::yield();
    }
}

dee::HostPackCache::HintRequest make_hint(
        uint64_t key, size_t nbytes, uint64_t deadline,
        std::atomic<int>* fills = nullptr, Gate* gate = nullptr,
        bool fail = false, std::vector<uint64_t>* order = nullptr,
        std::mutex* order_mutex = nullptr) {
    dee::HostPackCache::HintRequest request;
    request.key = key;
    request.nbytes = nbytes;
    request.deadline = deadline;
    request.fill = [=](uint8_t* dst, size_t n) {
        if (order) {
            std::lock_guard<std::mutex> lock(*order_mutex);
            order->push_back(key);
        }
        if (gate) gate->wait();
        if (fills) fills->fetch_add(1);
        if (fail) return false;
        pattern_fill(dst, n, key);
        return true;
    };
    return request;
}

void test_hint_brings_record_in_before_demand() {
    dee::HostPackCache cache;
    cache.set_budget(256 * 8);
    std::vector<uint64_t> observed;
    cache.set_hint_read_observer(
        [&](uint64_t key, std::size_t, double, bool ok) {
            if (ok) observed.push_back(key);
        });
    auto refused = make_hint(5, 256, 1);
    check(cache.submit_hints(&refused, 1) == 0 &&
          cache.stats().hints_rejected == 1,
          "hints are refused while there are no hint lanes");
    check(cache.set_hint_lanes(2) && cache.hint_lanes() == 2, "hint lanes start");
    std::atomic<int> hint_fills{0};
    dee::HostPackCache::HintRequest hints[2] = {
        make_hint(5, 256, 1, &hint_fills), make_hint(6, 256, 1, &hint_fills)};
    check(cache.submit_hints(hints, 2) == 2, "two hints accepted");
    check(cache.stats().hint_reserved_bytes == 512 || cache.pending_hints() <= 2,
          "accepted hints reserve their bytes");
    settle_hints(cache);
    check(hint_fills.load() == 2 && cache.contains(5) && cache.contains(6),
          "finished hints are published");
    check(cache.stats().hint_reads == 2 && cache.stats().hint_reserved_bytes == 0 &&
          observed.size() == 2, "hint reads are counted and observed");
    check(cache.stats().hits == 0 && cache.stats().misses == 0,
          "a hint is not a lookup");
    int demand_fills = 0;
    const uint8_t* data = cache.get(5, 256, [&](uint8_t* dst, size_t n) {
        ++demand_fills;
        pattern_fill(dst, n, 5);
    });
    check(data && matches_pattern(data, 256, 5) && demand_fills == 0,
          "demand is served the hinted bytes without a second read");
    check(cache.stats().hits == 1 && cache.stats().hint_hits == 1,
          "the hit is attributed to the hint");
    cache.get(5, 256, [&](uint8_t* dst, size_t n) { pattern_fill(dst, n, 5); });
    check(cache.stats().hits == 2 && cache.stats().hint_hits == 1,
          "only the first use of a hinted record counts as a hint hit");
    auto duplicate = make_hint(6, 256, 1);
    check(cache.submit_hints(&duplicate, 1) == 0,
          "a hint for a resident record is refused");
    for (uint64_t key = 20; key < 28; ++key) {
        cache.get(key, 256, [&](uint8_t* dst, size_t n) {
            pattern_fill(dst, n, key); });
    }
    check(!cache.contains(6) && cache.stats().hints_wasted == 1,
          "a hinted record evicted unused is counted as wasted");
    cache.clear();
    check(cache.stats().hints_wasted == 0 && cache.stats().hint_reads == 0,
          "clear resets the hint counters with the rest");
}

void test_failed_hint_publishes_nothing() {
    dee::HostPackCache cache;
    cache.set_budget(256 * 4);
    cache.set_hint_lanes(1);
    auto hint = make_hint(9, 256, 1, nullptr, nullptr, /*fail=*/true);
    check(cache.submit_hints(&hint, 1) == 1, "failing hint accepted");
    settle_hints(cache);
    check(!cache.contains(9) && cache.stats().hint_failures == 1 &&
          cache.stats().hint_reads == 0 && cache.stats().hint_reserved_bytes == 0,
          "a failed hint publishes nothing and releases its reservation");
    int demand_fills = 0;
    const uint8_t* data = cache.get(9, 256, [&](uint8_t* dst, size_t n) {
        ++demand_fills;
        pattern_fill(dst, n, 9);
    });
    check(data && demand_fills == 1 && matches_pattern(data, 256, 9) &&
          cache.stats().misses == 1,
          "demand reads the record itself after a failed hint");
}

void test_demand_never_waits_behind_a_queued_hint() {
    dee::HostPackCache cache;
    cache.set_budget(256 * 8);
    cache.set_hint_reserve_limit(256 * 8);
    cache.set_hint_lanes(1);
    Gate gate;
    std::atomic<int> hint_fills{0};
    dee::HostPackCache::HintRequest hints[2] = {
        make_hint(1, 256, 5, &hint_fills, &gate),
        make_hint(2, 256, 6, &hint_fills)};
    check(cache.submit_hints(hints, 2) == 2, "blocked + queued hint accepted");
    gate.wait_entered(1);                       // hint 1 is being read
    // Key 2 is queued behind it: demand takes the record over at once.
    int demand_fills = 0;
    const uint8_t* two = cache.get(2, 256, [&](uint8_t* dst, size_t n) {
        ++demand_fills;
        pattern_fill(dst, n, 2);
    });
    check(two && demand_fills == 1 && cache.stats().hints_cancelled == 1 &&
          cache.stats().misses == 1,
          "a queued hint is cancelled and demand reads the record itself");
    // Key 1 is mid-read: demand waits for that read instead of duplicating it.
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        gate.release();
    });
    const uint8_t* one = cache.get(1, 256, [&](uint8_t* dst, size_t n) {
        ++demand_fills;
        pattern_fill(dst, n, 1);
    });
    releaser.join();
    check(one && matches_pattern(one, 256, 1) && demand_fills == 1 &&
          cache.stats().hint_waits == 1 && cache.stats().hint_hits == 1,
          "demand waits for a hint read in flight and reuses its bytes");
    cache.drain_hints();
    check(hint_fills.load() == 1, "the cancelled hint never ran");
}

std::vector<uint64_t> run_hint_order(dee::HintQueueOrder order_mode) {
    dee::HostPackCache cache;
    cache.set_budget(256 * 16);
    cache.set_hint_reserve_limit(256 * 16);
    cache.set_hint_queue_order(order_mode);
    cache.set_hint_lanes(1);
    Gate gate;
    std::vector<uint64_t> order;
    std::mutex order_mutex;
    auto first = make_hint(100, 256, 1, nullptr, &gate, false, &order, &order_mutex);
    cache.submit_hints(&first, 1);
    gate.wait_entered(1);                      // the lane is busy with 100
    dee::HostPackCache::HintRequest rest[3] = {
        make_hint(11, 256, 9, nullptr, nullptr, false, &order, &order_mutex),
        make_hint(12, 256, 3, nullptr, nullptr, false, &order, &order_mutex),
        make_hint(13, 256, 5, nullptr, nullptr, false, &order, &order_mutex)};
    cache.submit_hints(rest, 3);
    gate.release();
    settle_hints(cache);
    return order;
}

void test_hint_queue_order() {
    const std::vector<uint64_t> deadline =
        run_hint_order(dee::HintQueueOrder::Deadline);
    const std::vector<uint64_t> fifo = run_hint_order(dee::HintQueueOrder::Fifo);
    check(deadline == std::vector<uint64_t>({100, 12, 13, 11}),
          "deadline order serves the nearest layer first");
    check(fifo == std::vector<uint64_t>({100, 11, 12, 13}),
          "fifo order serves hints in arrival order");
}

void test_hint_expiry_and_budget() {
    dee::HostPackCache cache;
    cache.set_budget(256 * 4);                 // default hint cap = 1 record
    cache.set_hint_lanes(1);
    Gate gate;
    auto blocked = make_hint(1, 256, 2, nullptr, &gate);
    check(cache.submit_hints(&blocked, 1) == 1, "first hint fits the hint cap");
    gate.wait_entered(1);
    auto over = make_hint(2, 256, 3);
    check(cache.submit_hints(&over, 1) == 0 && cache.stats().hints_rejected == 1,
          "hints beyond the reserve cap are refused");
    cache.set_hint_reserve_limit(256 * 3);
    dee::HostPackCache::HintRequest queued[2] = {
        make_hint(2, 256, 3), make_hint(3, 256, 7)};
    check(cache.submit_hints(queued, 2) == 2, "raised cap admits two more");
    check(cache.stats().hint_reserved_bytes == 256 * 3,
          "reserved bytes cover running and queued hints");
    // Demand for other records must leave the reservation intact: only one
    // more record fits beside three reserved ones.
    for (uint64_t key = 50; key < 53; ++key) {
        check(cache.get(key, 256, [&](uint8_t* dst, size_t n) {
                  pattern_fill(dst, n, key); }) != nullptr,
              "demand still fits beside the hint reservation");
        check(cache.stats().bytes + cache.stats().hint_reserved_bytes <= 256 * 4,
              "resident plus reserved bytes never exceed the budget");
    }
    check(cache.expire_hints(3) == 1 && cache.stats().hints_cancelled == 1 &&
          cache.stats().hint_reserved_bytes == 256 * 2,
          "expire drops queued hints whose layer has started");
    gate.release();
    cache.drain_hints();
    check(cache.stats().hint_reserved_bytes == 0 && cache.pending_hints() == 0,
          "drain leaves no reservation behind");
    check(cache.stats().bytes <= 256 * 4, "budget holds after publication");
    check(cache.set_hint_lanes(0) && cache.hint_lanes() == 0, "hint lanes stop");
}

void test_running_layer_is_the_last_victim() {
    dee::HostPackCache cache;
    cache.set_budget(256 * 3);
    cache.set_eviction_policy(dee::HostEvictionPolicy::NextUse);
    cache.set_next_use_cycle(4);
    const auto demand = [&](uint64_t key) {
        return cache.get(key, 256, [&](uint8_t* dst, size_t n) {
            pattern_fill(dst, n, key); });
    };
    // (2,0) becomes a well-established record: used on five visits.
    for (int visit = 0; visit < 5; ++visit) {
        cache.begin_layer(2);
        demand(lkey(2, 0));
    }
    cache.begin_layer(1);
    demand(lkey(1, 0));                         // first-ever use: low estimate
    demand(lkey(1, 1));
    const double running = cache.next_use().expected_next_use(lkey(1, 0));
    const double other = cache.next_use().expected_next_use(lkey(2, 0));
    check(running > other,
          "setup: the running layer's record has the larger expected next use");
    demand(lkey(1, 2));                         // needs one eviction
    check(cache.contains(lkey(1, 0)) && cache.contains(lkey(1, 1)) &&
          !cache.contains(lkey(2, 0)),
          "records of the running layer are evicted last");
    // LRU + hints: a guess may not evict the running layer either.
    dee::HostPackCache lru;
    lru.set_budget(256 * 2);
    lru.set_hint_reserve_limit(256 * 2);
    lru.set_next_use_cycle(4);                  // layer clock on, policy LRU
    lru.set_hint_lanes(1);
    lru.begin_layer(1);
    for (int expert = 0; expert < 2; ++expert) {
        const uint64_t key = lkey(1, expert);
        lru.get(key, 256, [&](uint8_t* dst, size_t n) { pattern_fill(dst, n, key); });
    }
    auto guess = make_hint(lkey(2, 0), 256, 2);
    check(lru.submit_hints(&guess, 1) == 0 && lru.contains(lkey(1, 0)),
          "under LRU a hint cannot evict the running layer");
    lru.begin_layer(2);
    auto later = make_hint(lkey(3, 0), 256, 3);
    check(lru.submit_hints(&later, 1) == 1 && !lru.contains(lkey(1, 0)),
          "under LRU a hint evicts the oldest record of a finished layer");
    lru.drain_hints();
}

void test_hint_under_next_use_only_displaces_later_records() {
    dee::HostPackCache cache;
    cache.set_budget(256 * 2);
    cache.set_hint_reserve_limit(256 * 2);
    cache.set_eviction_policy(dee::HostEvictionPolicy::NextUse);
    cache.set_next_use_cycle(4);
    cache.set_hint_lanes(1);
    const auto demand = [&](uint64_t key) {
        return cache.get(key, 256, [&](uint8_t* dst, size_t n) {
            pattern_fill(dst, n, key); });
    };
    cache.begin_layer(3);
    demand(lkey(3, 0));                         // gives (3,0) a use history
    cache.begin_layer(1);
    demand(lkey(1, 0));
    demand(lkey(1, 1));                         // evicts (3,0): furthest away
    check(!cache.contains(lkey(3, 0)) && cache.stats().evictions == 1,
          "setup: the far record left, the running layer's stayed");
    // Residents belong to the layer running now (needed soonest): a guess
    // may not push them out.
    auto guess = make_hint(lkey(2, 0), 256, 2);
    check(cache.submit_hints(&guess, 1) == 0 &&
          cache.contains(lkey(1, 0)) && cache.contains(lkey(1, 1)),
          "a hint cannot displace a record needed sooner");
    // One layer on, layer 1 is three calls away and (3,0) only one.
    cache.begin_layer(2);
    auto later = make_hint(lkey(3, 0), 256, 3);
    check(cache.submit_hints(&later, 1) == 1,
          "a hint may displace a record needed later");
    settle_hints(cache);
    check(cache.contains(lkey(3, 0)) && cache.stats().evictions == 2,
          "exactly one later-needed record made room for the hint");
}

}  // namespace

int main() {
    test_basic_hit_miss();
    test_lru_eviction();
    test_peek_bytes_is_non_perturbing();
    test_oversize_and_clear();
    test_same_buffer_stays_valid_until_evicted();
    test_bounded_batch_fill_and_rollback();
    test_batch_duplicate_is_single_materialization();
    test_source_order_preserves_request_identity();
    test_scalar_fill_false_is_not_published();
    test_scalar_fill_throw_is_not_published();
    test_same_key_size_mismatch_fails_closed();
    test_budget_rejection_counters();
    test_evict_observer_reports_victims();
    test_scalar_fill_instrumentation();
    test_batch_allprotected_stats_coherence();
    test_next_use_estimator_model();
    test_next_use_fixes_cyclic_thrash();
    test_next_use_without_cycle_is_lru();
    test_hint_brings_record_in_before_demand();
    test_failed_hint_publishes_nothing();
    test_demand_never_waits_behind_a_queued_hint();
    test_hint_queue_order();
    test_hint_expiry_and_budget();
    test_running_layer_is_the_last_victim();
    test_hint_under_next_use_only_displaces_later_records();
    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}
