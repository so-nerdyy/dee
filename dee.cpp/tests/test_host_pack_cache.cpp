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
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
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

}  // namespace

int main() {
    test_basic_hit_miss();
    test_lru_eviction();
    test_oversize_and_clear();
    test_same_buffer_stays_valid_until_evicted();
    test_bounded_batch_fill_and_rollback();
    test_batch_duplicate_is_single_materialization();
    test_source_order_preserves_request_identity();
    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}
