// tests/test_host_pack_cache.cpp
//
// Unit tests for the Stage-1 HostPackCache LRU (packed expert byte residency).
// Validates: basic get/fill, hit/miss accounting, LRU eviction order under a
// byte budget, budget clamping (entry larger than budget never caches), and
// clear() resetting everything. Runs on any host (no CUDA dependency).

#include "dee/host_pack_cache.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
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

}  // namespace

int main() {
    test_basic_hit_miss();
    test_lru_eviction();
    test_oversize_and_clear();
    test_same_buffer_stays_valid_until_evicted();
    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}