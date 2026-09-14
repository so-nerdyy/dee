#include "dee/engine.h"
#include "dee/expert_tiers.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace {
using namespace dee;
int failures = 0;
void check(bool condition, const char* text) {
    std::cout << (condition ? "PASS " : "FAIL ") << text << '\n';
    if (!condition) ++failures;
}
// Test fixture only: explicit first-idle policy, not a proposed cache policy.
struct TestPolicy : HostPlacementPolicy {
    HostResidency residency(const TierExpertKey& key) const override {
        return key.expert == 99 ? HostResidency::PolicyResident : HostResidency::Dynamic;
    }
    std::optional<size_t> victim(const TierExpertKey&, const std::vector<HostVictim>& candidates) const override {
        return candidates.empty() ? std::nullopt : std::optional<size_t>(candidates.front().slot);
    }
};
struct MemoryCounts { size_t allocs = 0, frees = 0, pins = 0, unpins = 0; };
HostMemoryBackend counted_backend(MemoryCounts& count, bool pin_success = true, size_t fail_alloc = 0) {
    auto native = host_memory_backend(false);
    return {
        [&, native, fail_alloc](size_t n, size_t a) {
            ++count.allocs;
            return count.allocs == fail_alloc ? nullptr : native.allocate(n, a);
        },
        [&, native](void* p) { ++count.frees; native.free(p); },
        [&, pin_success](void*, size_t) { ++count.pins; return pin_success; },
        [&](void*) { ++count.unpins; }
    };
}
HostTierConfig config(size_t dynamic_slots = 1, size_t policy_slots = 0) {
    return {64, 4096, policy_slots, dynamic_slots, 4096 * (dynamic_slots + policy_slots), true};
}
StorageRecord record(int expert, int layer = 0) {
    return {{"model-sha256:A", layer, expert, "packed-test-v1"}, 64, 64, "identity-v1"};
}
struct SyntheticStore : ColdExpertStore {
    std::atomic<size_t> reads{0};
    int failure = 0;
    uint8_t* last_destination = nullptr;
    static uint8_t byte(const TierExpertKey& key, size_t i) {
        return static_cast<uint8_t>(key.expert * 17 + key.layer * 31 + i);
    }
    ColdReadResult read(const TierExpertKey& key, uint8_t* dst, size_t n) override {
        ++reads; last_destination = dst;
        for (size_t i = 0; i < n; ++i) dst[i] = byte(key, i);
        if (failure == 1) return {false, n / 2};
        if (failure == 2) throw std::runtime_error("injected storage failure");
        if (failure == 3) return {true, n - 1};
        return {true, n};
    }
};
bool exact(const void* data, const TierExpertKey& key, size_t bytes = 64) {
    if (!data) return false;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) if (p[i] != SyntheticStore::byte(key, i)) return false;
    return true;
}
Arena::Backend device_mock() {
    return {[](size_t bytes) { return std::malloc(bytes); }, [](void* p) { std::free(p); }, "host"};
}

void host_lifecycle() {
    IdentityCodec codec; SyntheticStore store; MemoryCounts count;
    HostExpertLease survivor;
    {
        HostExpertTier host(config(), counted_backend(count), std::make_shared<TestPolicy>());
        auto a = host.acquire(record(1), store, codec);
        const auto ptr = a.lease.data(); const auto generation = a.lease.generation();
        check(a.status == HostAcquireStatus::Ready && a.lease.pinned(), "host miss yields registered lease");
        check(reinterpret_cast<uintptr_t>(ptr) % 4096 == 0, "slot aligned for storage fill/registration");
        check(ptr == store.last_destination && exact(ptr, record(1).key), "fill lands directly in final slot, exact packed bytes");
        auto again = host.acquire(record(1), store, codec);
        check(again.lease.data() == ptr && store.reads == 1, "host hit does not reread or copy record");
        auto malformed = record(1); malformed.stored_bytes = 63;
        check(!host.acquire(malformed, store, codec).lease, "invalid codec descriptor rejected even on a host hit");
        check(host.acquire(record(2), store, codec).status == HostAcquireStatus::Capacity, "leased slot cannot be evicted");
        check(!host.evict(record(1).key), "explicit eviction cannot revoke lease");
        a.lease.reset(); again.lease.reset();
        auto b = host.acquire(record(2), store, codec);
        check(b.lease.data() == ptr && b.lease.generation() > generation, "pinned slot reused with fresh generation");
        check(exact(b.lease.data(), record(2).key), "reused slot never returns stale expert bytes");
        check(count.allocs == 1 && count.pins == 1 && count.frees == 0, "no payload alloc/free/register per request");
        check(host.stats().allocated_bytes == 4096 && host.stats().resident_bytes == 64 &&
              host.stats().peak_resident_bytes == 64 && host.stats().evictions == 1,
              "bounded allocation, resident peak, and eviction accounting");
        survivor = b.lease;
    }
    check(count.frees == 0 && exact(survivor.data(), record(2).key), "lease outlives tier without use-after-free");
    survivor.reset();
    check(count.frees == 1 && count.unpins == 1, "last lease unregisters and frees exactly once");
}

void host_residency_identity_failures() {
    IdentityCodec codec; SyntheticStore store; MemoryCounts count;
    HostExpertTier host(config(1, 1), counted_backend(count, false), std::make_shared<TestPolicy>());
    auto resident = host.acquire(record(99), store, codec); resident.lease.reset();
    for (int expert : {1, 2, 3}) {
        auto item = host.acquire(record(expert), store, codec);
        check(item.lease && !item.lease.pinned(), "registration failure retains bounded pageable slot");
    }
    const auto reads = store.reads.load();
    auto static_hit = host.acquire(record(99), store, codec);
    check(static_hit.lease && store.reads == reads, "policy-resident slot survives dynamic eviction");
    check(host.stats().pin_failures == 2 && host.stats().pinned_bytes == 0, "pin failure accounting");
    static_hit.lease.reset(); check(host.evict(record(99).key), "explicit static withdrawal supported");
    for (int change = 0; change < 3; ++change) {
        auto item = record(3);
        if (change == 0) item.key.model = "model-sha256:B";
        if (change == 1) item.key.representation = "packed-test-v2";
        if (change == 2) item.key.layer = 7;
        const auto before = store.reads.load();
        auto result = host.acquire(item, store, codec);
        check(result.lease && store.reads == before + 1 && result.lease.key() == item.key,
              "model/layer/representation are part of cache identity");
    }
    for (int failure : {1, 2, 3}) {
        store.failure = failure;
        auto result = host.acquire(record(10 + failure), store, codec);
        check(result.status == HostAcquireStatus::FillFailed && !result.lease,
              "partial/throw/short-success fill fails closed");
        store.failure = 0;
        auto retry = host.acquire(record(10 + failure), store, codec);
        check(retry.lease && exact(retry.lease.data(), retry.lease.key()), "failed reservation reusable only after exact refill");
    }
    auto oversize = record(50); oversize.exact_bytes = 65;
    check(host.acquire(oversize, store, codec).status == HostAcquireStatus::Invalid, "oversize record rejected");
    auto unknown = record(51); unknown.codec = "unselected-codec";
    check(!host.acquire(unknown, store, codec).lease, "unselected codec rejected");
    auto invalid = record(52); invalid.key.expert = -1;
    check(!host.acquire(invalid, store, codec).lease, "invalid expert identity rejected");
    check(count.allocs == 2, "failure/retry/eviction keeps original payload allocations");
}

void plain_lru_host_policy() {
    IdentityCodec codec; SyntheticStore store; MemoryCounts count;
    HostExpertTier host(config(2), counted_backend(count),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    auto first = host.acquire(record(1), store, codec); first.lease.reset();
    auto second = host.acquire(record(2), store, codec); second.lease.reset();
    auto refreshed = host.acquire(record(1), store, codec); refreshed.lease.reset();
    auto third = host.acquire(record(3), store, codec); third.lease.reset();
    check(host.stats().evictions == 1, "plain host LRU performs one replacement");
    const auto before = store.reads.load();
    auto recent = host.acquire(record(1), store, codec); recent.lease.reset();
    check(store.reads == before, "plain host LRU retains the most recent record");
    const auto before_old = store.reads.load();
    auto old = host.acquire(record(2), store, codec); old.lease.reset();
    check(store.reads == before_old + 1, "plain host LRU evicts the older record");
}

struct GatedStore : SyntheticStore {
    std::mutex mutex; std::condition_variable cv;
    bool entered = false, proceed = false;
    ColdReadResult read(const TierExpertKey& key, uint8_t* dst, size_t n) override {
        { std::unique_lock<std::mutex> lock(mutex); entered = true; cv.notify_all();
          cv.wait(lock, [&] { return proceed; }); }
        return SyntheticStore::read(key, dst, n);
    }
};
void concurrent_duplicate(bool fail) {
    IdentityCodec codec; GatedStore store; MemoryCounts count;
    HostExpertTier host(config(), counted_backend(count), std::make_shared<TestPolicy>());
    store.failure = fail ? 1 : 0;
    auto first = std::async(std::launch::async, [&] { return host.acquire(record(1), store, codec); });
    { std::unique_lock<std::mutex> lock(store.mutex); store.cv.wait(lock, [&] { return store.entered; }); }
    auto second = std::async(std::launch::async, [&] { return host.acquire(record(1), store, codec); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (host.stats().coalesced != 1 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    check(host.stats().coalesced == 1, "concurrent duplicate joins reserved generation");
    { std::lock_guard<std::mutex> lock(store.mutex); store.proceed = true; } store.cv.notify_all();
    auto a = first.get(); auto b = second.get();
    check(store.reads == 1, "concurrent requests perform exactly one cold fill");
    if (fail) {
        check(!a.lease && !b.lease && host.stats().leased_slots == 0, "fill failure wakes all waiters and cleans references");
        store.failure = 0;
        check(bool(host.acquire(record(1), store, codec).lease), "retry after concurrent failure succeeds");
    } else {
        check(a.lease.data() == b.lease.data() && a.lease.generation() == b.lease.generation(), "duplicate leases share exact bytes/generation");
    }
}

void budgets_and_init_cleanup() {
    for (int bad : {0, 1, 2}) {
        MemoryCounts counts;
        auto cfg = config(2);
        if (bad == 0) cfg.budget_bytes -= 1;
        if (bad == 1) cfg.alignment = 4000;
        if (bad == 2) cfg.slot_bytes = std::numeric_limits<size_t>::max();
        bool rejected = false;
        try { HostExpertTier host(cfg, counted_backend(counts), std::make_shared<TestPolicy>()); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected && counts.allocs == 0, "invalid/padded/overflow budget rejected before allocation");
    }
    MemoryCounts counts;
    try { HostExpertTier host(config(2), counted_backend(counts, true, 2), std::make_shared<TestPolicy>()); }
    catch (const std::bad_alloc&) {}
    check(counts.allocs == 2 && counts.frees == 1 && counts.unpins == 1,
          "partial initialization unregisters/frees prior slots");
}

void device_integration() {
    IdentityCodec codec; SyntheticStore store; MemoryCounts counts;
    HostExpertTier host(config(), counted_backend(counts), std::make_shared<TestPolicy>());
    VramCacheManager cache; check(cache.init(128, device_mock()), "bounded device arena initialized");
    AsyncPrefetcher prefetcher(cache); check(prefetcher.init(false), "mock transfer stream initialized");
    DeviceExpertTier device(cache, prefetcher, record(0).key);
    auto stage = [&](const StorageRecord& r) { return device.stage(r, store, host, codec, 0); };
    check(stage(record(1)), "cold expert stages through host/device tiers");
    check(host.stats().leased_slots == 1 && cache.pin_count(0, 1) == 1, "pending DMA owns both source and destination");
    check(stage(record(1)) && store.reads == 1, "duplicate device request coalesces without host request");
    check(stage(record(2)), "host pressure completes one DMA and reuses final slot");
    check(cache.pin_count(0, 1) == 1 && exact(cache.data(0, 1), record(1).key), "source reclaim retains device consumer pin and exact prior bytes");
    check(!stage(record(3)) && cache.used_bytes() <= 128, "device budget forbids eviction of staged consumers");
    check(device.wait(record(1).key) && device.wait(record(2).key), "per-expert waits consume both reservations");
    check(exact(cache.data(0, 1), record(1).key) && exact(cache.data(0, 2), record(2).key), "host eviction cannot alter device bytes");
    check(stage(record(3)) && device.wait(record(3).key), "idle device slot evicts and refills");
    check(exact(cache.data(0, 3), record(3).key), "device refill preserves expert identity");
    auto alien = record(3); alien.key.model = "another-checkpoint";
    check(!stage(alien), "device rejects wrong model even on layer/expert hit");
    alien = record(3); alien.key.representation = "another-layout";
    check(!stage(alien), "device rejects wrong representation even on hit");
    auto huge = record(4); huge.exact_bytes = huge.stored_bytes = 129;
    check(!stage(huge), "oversize device request rejected before fill");
    check(stage(record(3, 1)) && device.wait(record(3, 1).key), "same expert across layers has distinct transfer identity");
    check(exact(cache.data(1, 3), record(3, 1).key), "cross-layer bytes are never aliased");
    const auto metrics = device.metrics(host, 2);
    check(metrics.device_hit == 1 && metrics.H2D_bytes == 256 &&
          metrics.bytes_per_token_valid && metrics.H2D_bytes_per_token == 128,
          "tier hit/miss/traffic/token metrics count successful transfers");
    check(metrics.device_evictions == 2 && metrics.device_bytes <= metrics.device_budget &&
          metrics.device_peak_bytes == 128,
          "device eviction/peak/budget accounting");
    check(!device.metrics(host).bytes_per_token_valid, "unknown token denominator stays explicitly invalid");
    std::string error;
    check(cache.validate_invariants(&error) && prefetcher.validate_invariants(&error), "existing cache and transfer invariants hold");
    check(host.stats().leased_slots == 0, "all consumed DMA host leases released");
}
void disabled_configuration() {
    EngineConfig cfg;
    check(!cfg.phase2.enabled && !cfg.phase2.host_enabled &&
          !cfg.phase2.vram_priority_fix_enabled, "Phase 2 switches default OFF");
    Engine engine;
    check(engine.phase2_metrics().host.allocated_bytes == 0, "disabled engine has no tier allocation");
    cfg.phase2.enabled = true;
    check(!engine.init(cfg), "unsupported experimental runtime fails closed");

    EngineConfig legacy;
    legacy.shard_path = "tests/data/layer0_shard.safetensors";
    legacy.hidden = 16; legacy.inter = 8; legacy.num_layers = 1;
    legacy.num_experts = 3; legacy.topk = 2;
    Engine ordinary, explicitly_disabled;
    const bool ordinary_ready = ordinary.init(legacy);
    legacy.phase2.enabled = false;
    legacy.phase2.host.budget_bytes = 1; // invalid if accidentally activated
    legacy.phase2.host.dynamic_slots = 100;
    const bool disabled_ready = explicitly_disabled.init(legacy);
    check(ordinary_ready && disabled_ready, "disabled integration initializes both synthetic engines without Oracle assets");
    if (ordinary_ready && disabled_ready) {
        float input[16]; for (int i = 0; i < 16; ++i) input[i] = 0.001f * (i + 1);
        float a[32] = {}, b[32] = {};
        for (const auto& routes : {std::vector<int>{0, 1}, std::vector<int>{2, 0}, std::vector<int>{0, 1}}) {
            check(ordinary.moe_forward_experts(0, input, a, routes) &&
                  explicitly_disabled.moe_forward_experts(0, input, b, routes) &&
                  std::memcmp(a, b, sizeof(a)) == 0,
                  "default/explicit-OFF produce bit-identical synthetic expert outputs");
        }
        check(explicitly_disabled.phase2_metrics().host.allocated_bytes == 0,
              "OFF ignores experimental budgets and allocates no host tier");
    }
}
}
int main() {
    host_lifecycle(); host_residency_identity_failures();
    plain_lru_host_policy();
    concurrent_duplicate(false); concurrent_duplicate(true);
    budgets_and_init_cleanup(); device_integration(); disabled_configuration();
    std::cout << "Phase 2 failures: " << failures << '\n';
    return failures ? 1 : 0;
}
