// tests/test_phase2_tier_audit.cpp
//
// Independent adversarial audit tests for the Phase-2 host/device tier seams
// (commit 08d3d51). CPU-only: exercises HostExpertTier, HostExpertLease,
// IdentityCodec, ExpertStoreColdAdapter seams (via a fake ExpertStore), the
// AsyncPrefetcher managed-source path on the mock stream, and the
// VramCacheManager::discard_unpinned guard.
//
// These tests deliberately attack documented invariants: lease refcounting,
// fill-failure publication, residency partitions, eviction-while-leased,
// cross-layer key identity on LLP64, and teardown with in-flight fills.

#include "dee/expert_tiers.h"
#include "dee/expert_store.h"
#include "dee/vram_cache.h"
#include "dee/async_prefetcher.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace dee;

int failures = 0;
void check(bool condition, const char* text) {
    std::cout << (condition ? "PASS " : "FAIL ") << text << '\n';
    if (!condition) ++failures;
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
struct AuditPolicy : HostPlacementPolicy {
    // expert >= 90 => PolicyResident, else Dynamic.
    HostResidency residency(const TierExpertKey& key) const override {
        return key.expert >= 90 ? HostResidency::PolicyResident : HostResidency::Dynamic;
    }
    std::optional<size_t> victim(const TierExpertKey&,
                                 const std::vector<HostVictim>& c) const override {
        return c.empty() ? std::nullopt : std::optional<size_t>(c.front().slot);
    }
};

// Records every victim candidate it is offered (keys copied out of the
// borrowed pointers, which is legal only during the callback).
struct RecordingPolicy : HostPlacementPolicy {
    mutable std::mutex mutex;
    mutable std::vector<std::vector<TierExpertKey>> offered;
    HostResidency residency(const TierExpertKey& key) const override {
        return key.expert >= 90 ? HostResidency::PolicyResident : HostResidency::Dynamic;
    }
    std::optional<size_t> victim(const TierExpertKey&,
                               const std::vector<HostVictim>& c) const override {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<TierExpertKey> keys;
        for (const auto& v : c) keys.push_back(*v.key);
        offered.push_back(std::move(keys));
        return c.empty() ? std::nullopt : std::optional<size_t>(c.front().slot);
    }
};

struct MemoryCounts { size_t allocs = 0, frees = 0, pins = 0, unpins = 0; };
HostMemoryBackend counted_backend(MemoryCounts& c, bool pin_ok = true) {
    auto native = host_memory_backend(false);
    return {
        [&, native](size_t n, size_t a) { ++c.allocs; return native.allocate(n, a); },
        [&, native](void* p) { ++c.frees; native.free(p); },
        [&, pin_ok](void*, size_t) { ++c.pins; return pin_ok; },
        [&](void*) { ++c.unpins; }
    };
}

HostTierConfig cfg(size_t dyn, size_t pol = 0, size_t slot_bytes = 64) {
    return {slot_bytes, 64, pol, dyn, 64 * (dyn + pol), true};
}
StorageRecord rec(int expert, int layer = 0, size_t bytes = 64) {
    return {{"audit-model", layer, expert, "audit-rep-v1"}, bytes, bytes, "identity-v1"};
}

struct ByteStore : ColdExpertStore {
    std::atomic<size_t> reads{0};
    std::atomic<int> mode{0}; // 0 ok, 1 fail, 2 throw, 3 short-success, 4 lie-bytes
    static uint8_t byte_at(const TierExpertKey& k, size_t i) {
        return static_cast<uint8_t>(k.expert * 13 + k.layer * 29 + i);
    }
    ColdReadResult read(const TierExpertKey& key, uint8_t* dst, size_t n) override {
        ++reads;
        if (mode == 2) throw std::runtime_error("injected");
        for (size_t i = 0; i < n; ++i) dst[i] = byte_at(key, i);
        if (mode == 1) return {false, n};
        if (mode == 3) return {true, n - 1};
        if (mode == 4) return {true, n + 1}; // lies about bytes materialized
        return {true, n};
    }
};
bool exact(const uint8_t* p, const TierExpertKey& k, size_t n = 64) {
    if (!p) return false;
    for (size_t i = 0; i < n; ++i) if (p[i] != ByteStore::byte_at(k, i)) return false;
    return true;
}

struct GatedStore : ByteStore {
    std::mutex m; std::condition_variable cv;
    int entered = 0; int release_after = 1; // which entry index opens the gate
    bool open = false;
    ColdReadResult read(const TierExpertKey& k, uint8_t* d, size_t n) override {
        {   std::unique_lock<std::mutex> l(m);
            ++entered;
            cv.notify_all();
            if (entered >= release_after) cv.wait(l, [&] { return open; });
        }
        return ByteStore::read(k, d, n);
    }
    void wait_entered(int n) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return entered >= n; });
    }
    void release() { { std::lock_guard<std::mutex> l(m); open = true; } cv.notify_all(); }
};

Arena::Backend mock_device() {
    return {[](size_t b) { return std::malloc(b); }, [](void* p) { std::free(p); }, "host"};
}

// A minimal in-memory ExpertStore: six region buffers per expert so the
// adapter's non-contiguous gather is exercised; can flip contiguity, layout,
// or integrity identity between calls.
struct FakeStore : ExpertStore {
    static constexpr int kExperts = 8;
    std::vector<uint8_t> region[6]; // w0 w1 w2 s0 s1 s2 shared across experts
    std::string identity = "fake-store-sha256";
    std::atomic<int> gets{0};
    bool flip_identity = false;
    bool misorder = false;      // break contiguous tiling order
    bool shrink_nbytes = false; // break same_layout
    std::atomic<size_t> materialize_calls{0};

    FakeStore(size_t w = 16, size_t s = 4) {
        for (int i = 0; i < 3; ++i) {
            region[i].assign(w, static_cast<uint8_t>(0x10 + i));
            region[3 + i].assign(s, static_cast<uint8_t>(0x60 + i));
        }
    }
    size_t record_bytes() const {
        size_t n = 0; for (auto& r : region) n += r.size(); return n;
    }
    bool get(int layer, int expert, ExpertView* out) override {
        ++gets;
        if (!out || layer < 0 || expert < 0 || expert >= kExperts) {
            record_lookup(false);
            return false;
        }
        ExpertView v;
        if (layer > 4) { record_lookup(false); return false; }
        v.codec = ExpertCodec::DeepSeekFp4E2m1E8m0;
        v.record_index = static_cast<size_t>(expert);
        v.integrity_identity = integrity_identity().c_str();
        // Fabricate a contiguous image the tensors tile exactly.
        contig.assign(record_bytes(), 0);
        size_t off = 0;
        if (!misorder) {
            for (int i = 0; i < 6; ++i) { std::memcpy(contig.data() + off, region[i].data(), region[i].size()); off += region[i].size(); }
        } else {
            // same bytes, wrong tiling: scales before weights
            for (int i = 3; i < 6; ++i) { std::memcpy(contig.data() + off, region[i].data(), region[i].size()); off += region[i].size(); }
            for (int i = 0; i < 3; ++i) { std::memcpy(contig.data() + off, region[i].data(), region[i].size()); off += region[i].size(); }
        }
        v.contiguous_data = contig.data();
        v.contiguous_nbytes = contig.size();
        off = 0;
        for (int i = 0; i < 3; ++i) {
            v.weights[i].data = contig.data() + off;
            v.weights[i].nbytes = region[i].size();
            v.weights[i].dtype = DType::I8;
            v.weights[i].shape = {2, static_cast<int64_t>(region[i].size() / 2)};
            off += region[i].size();
        }
        for (int i = 0; i < 3; ++i) {
            v.scales[i].data = contig.data() + off;
            v.scales[i].nbytes = region[3 + i].size();
            v.scales[i].dtype = DType::F8;
            v.scales[i].shape = {2, static_cast<int64_t>(region[3 + i].size() / 2)};
            off += region[3 + i].size();
        }
        if (misorder) { // point tensors at the (wrong-order) positions they claim
            off = 0;
            for (int i = 0; i < 3; ++i) { v.scales[i].data = contig.data() + off; off += region[3 + i].size(); }
            for (int i = 0; i < 3; ++i) { v.weights[i].data = contig.data() + off; off += region[i].size(); }
        }
        if (shrink_nbytes) v.weights[1].nbytes -= 1;
        record_lookup(true);
        *out = std::move(v);
        return true;
    }
    const char* backend_name() const override { return "fake"; }
    const std::string& integrity_identity() const override {
        return flip_identity ? flipped_ : identity;
    }
    std::string flipped_ = "fake-store-sha256-CHANGED";
    std::vector<uint8_t> contig; // scratch backing for the fabricated view
};

// ===========================================================================
// 1. LLP64 map_key collision: the legacy key map truncates layer bits when
//    sizeof(long)==4. The commit deliberately preserves this for the legacy
//    path (default-OFF equivalence) and uses full keys when the experimental
//    host tier is enabled.
// ===========================================================================
void llp64_layer_collision() {
    std::cout << "-- llp64 layer/expert key collision --\n";
    VramCacheManager cache;
    check(cache.init(256, mock_device()), "collision: cache init");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "collision: mock init");
    const uint8_t src[64] = {7};
    const long a = pf.prefetch(0, 5, src, 64, 0);
    const long b = pf.prefetch(1, 5, src, 64, 0);
    check(a >= 0, "collision: first prefetch issued");
    if (sizeof(long) == 4) {
        // DOCUMENTED PRE-EXISTING DEFECT (LLP64): (1,5) coalesces onto (0,5)'s
        // transfer; the layer-1 expert is never loaded and wait() fails.
        check(b == a, "collision: LLP64 legacy map drops layer bits (pre-existing)");
        check(pf.wait(0, 5), "collision: layer-0 expert completes");
        check(!pf.wait(1, 5), "collision: layer-1 expert was silently coalesced away");
        check(!cache.is_resident(1, 5), "collision: layer-1 block absent");
    } else {
        check(b != a, "collision: LP64 keeps layers distinct");
        check(pf.wait(1, 5), "collision: LP64 layer-1 completes");
    }
}

void experimental_mapkey_distinct() {
    std::cout << "-- experimental map_key keeps full layer bits --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(4), counted_backend(mc), std::make_shared<AuditPolicy>());
    VramCacheManager cache;
    check(cache.init(256, mock_device()), "distinct: cache init");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "distinct: mock init");
    DeviceExpertTier dev(cache, pf, rec(0).key);
    check(dev.stage(rec(5, 0), store, host, codec, 0), "distinct: stage L0E5");
    check(dev.stage(rec(5, 1), store, host, codec, 0), "distinct: stage L1E5 is not coalesced");
    check(dev.wait(rec(5, 0).key) && dev.wait(rec(5, 1).key),
          "distinct: both layers complete and resident");
    check(exact(cache.data(0, 5) ? static_cast<const uint8_t*>(cache.data(0, 5)) : nullptr, rec(5, 0).key) &&
          exact(cache.data(1, 5) ? static_cast<const uint8_t*>(cache.data(1, 5)) : nullptr, rec(5, 1).key),
          "distinct: per-layer bytes intact");
}

// ===========================================================================
// 2. Lease lifecycle: copies/moves/self-assign; evict blocked until ALL refs
//    drop; facade destruction mid-fill is survived by the caller-held state.
// ===========================================================================
void lease_refcount_lifecycle() {
    std::cout << "-- lease refcount lifecycle --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc), std::make_shared<AuditPolicy>());
    auto a = host.acquire(rec(1), store, codec);
    check(a.lease && a.status == HostAcquireStatus::Ready, "lifecycle: fill");
    HostExpertLease copies[3];
    for (auto& c : copies) c = a.lease;             // copy-assign onto empty
    HostExpertLease moved = std::move(copies[0]);   // move ctor
    copies[1] = copies[1];                          // self copy-assign
    moved = moved;                                  // self move-assign (guarded)
    check(host.stats().leased_slots == 1, "lifecycle: copies share one slot ref-wise");
    check(!host.evict(rec(1).key), "lifecycle: evict blocked while copies live");
    a.lease.reset();
    for (int i = 1; i < 3; ++i) copies[i].reset();
    check(!host.evict(rec(1).key), "lifecycle: evict still blocked by moved lease");
    moved.reset();
    check(host.evict(rec(1).key), "lifecycle: evict succeeds after last reset");
    check(host.stats().resident_bytes == 0, "lifecycle: resident bytes drained");
    check(mc.frees == 0, "lifecycle: payload not freed while tier alive");
}

void empty_lease_contract() {
    std::cout << "-- empty lease contract --\n";
    HostExpertLease empty;
    check(!empty && empty.data() == nullptr && empty.size() == 0 && !empty.pinned(),
          "empty lease: null fields");
    bool threw = false;
    try { (void)empty.key(); } catch (const std::logic_error&) { threw = true; }
    check(threw, "empty lease: key() throws logic_error");
    empty.reset(); // must be a no-op
    check(!empty, "empty lease: reset() idempotent");
}

void facade_teardown_mid_fill() {
    std::cout << "-- facade teardown mid-fill --\n";
    IdentityCodec codec; GatedStore store; MemoryCounts mc;
    auto host = std::make_unique<HostExpertTier>(cfg(1), counted_backend(mc),
                                               std::make_shared<AuditPolicy>());
    auto worker = std::async(std::launch::async, [&] {
        return host->acquire(rec(1), store, codec);
    });
    store.wait_entered(1); // fill is inside read(); acquire already copied state_
    host.reset();          // destroy the facade while acquire is in flight
    store.release();
    auto result = worker.get();
    check(result.status == HostAcquireStatus::Ready && result.lease,
          "teardown: in-flight fill completes after facade destruction");
    check(exact(result.lease.data(), result.lease.key()),
          "teardown: lease data intact past facade lifetime");
    result.lease.reset();
    check(mc.frees == 1 && mc.unpins == 1,
          "teardown: pool released exactly once at last lease");
}

// ===========================================================================
// 3. Fill-failure publication: coalesced waiters observe THEIR generation's
//    failure; a concurrent retry filling another slot does not migrate them.
// ===========================================================================
void fill_failure_publication() {
    std::cout << "-- fill failure publication across generations --\n";
    IdentityCodec codec; GatedStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc), std::make_shared<AuditPolicy>());

    store.mode = 1;                 // first fill will fail
    store.release_after = 1;        // first entrant blocks until opened
    auto first = std::async(std::launch::async, [&] {
        return host.acquire(rec(1), store, codec);
    });
    store.wait_entered(1);
    auto waiter = std::async(std::launch::async, [&] {
        return host.acquire(rec(1), store, codec); // coalesced waiter
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (host.stats().coalesced != 1 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    check(host.stats().coalesced == 1, "publication: waiter joined the doomed fill");
    store.release();
    check(first.get().status == HostAcquireStatus::FillFailed,
          "publication: filler reports failure");
    check(waiter.get().status == HostAcquireStatus::FillFailed,
          "publication: waiter reports ITS generation's failure");
    check(host.stats().failures == 1 && host.stats().leased_slots == 0,
          "publication: single failure, all references drained");

    // A retry for the same key must fill fresh — the stale Empty slot is
    // invisible to the hit path. Fresh tier/store for the retry check.
    IdentityCodec codec2; GatedStore store2; MemoryCounts mc2;
    HostExpertTier host2(cfg(2), counted_backend(mc2), std::make_shared<AuditPolicy>());
    store2.mode = 1; store2.open = true; // fail immediately
    auto f1 = host2.acquire(rec(1), store2, codec2);
    check(f1.status == HostAcquireStatus::FillFailed, "publication: failing fill");
    store2.mode = 0;
    auto ok = host2.acquire(rec(1), store2, codec2);
    check(ok.lease && exact(ok.lease.data(), rec(1).key),
          "publication: retry re-fills and serves exact bytes");
    check(host2.stats().fills == 1 && host2.stats().failures == 1,
          "publication: fills/failures accounted separately");
}

// ===========================================================================
// 4. Residency partitions: policy slots are never offered as dynamic victims;
//    dynamic acquires cannot steal empty policy slots; explicit evict() can
//    withdraw either class when idle.
// ===========================================================================
void residency_partitions() {
    std::cout << "-- residency partition isolation --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    auto policy = std::make_shared<RecordingPolicy>();
    HostExpertTier host(cfg(1, 1), counted_backend(mc), policy);

    auto dyn = host.acquire(rec(1), store, codec);   // dynamic slot
    auto pol = host.acquire(rec(99), store, codec);  // policy slot
    check(dyn.lease && pol.lease, "partition: both fills");
    dyn.lease.reset(); pol.lease.reset();

    // Dynamic capacity full: a new dynamic acquire must evict via policy.
    auto victim_dyn = host.acquire(rec(2), store, codec);
    check(victim_dyn.lease && host.stats().evictions == 1, "partition: dynamic eviction");
    check(policy->offered.size() == 1 && policy->offered[0].size() == 1 &&
          policy->offered[0][0].expert == 1,
          "partition: only the dynamic key was offered as victim");

    // Policy partition full of a PolicyResident record: a second policy key
    // must get Capacity (policy slots are never victimized by acquire).
    auto p2 = host.acquire(rec(98), store, codec);
    check(p2.status == HostAcquireStatus::Capacity,
          "partition: second policy key rejected, static slot protected");
    // A dynamic key cannot spill into the empty-looking policy partition even
    // when its own partition is exhausted... first fill the dynamic slot again.
    check(host.evict(rec(99).key), "partition: explicit evict withdraws policy slot");
    auto p3 = host.acquire(rec(98), store, codec);
    check(p3.lease && p3.lease.key().expert == 98, "partition: policy slot refillable by policy key");
}

void residency_flip_stays_in_slot() {
    std::cout << "-- residency reclassification keeps existing slot --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    struct FlipPolicy : HostPlacementPolicy {
        std::atomic<bool>* flip;
        explicit FlipPolicy(std::atomic<bool>* f) : flip(f) {}
        HostResidency residency(const TierExpertKey&) const override {
            return flip->load() ? HostResidency::PolicyResident : HostResidency::Dynamic;
        }
        std::optional<size_t> victim(const TierExpertKey&,
                                     const std::vector<HostVictim>& c) const override {
            return c.empty() ? std::nullopt : std::optional<size_t>(c.front().slot);
        }
    };
    std::atomic<bool> flip{false};
    HostExpertTier host(cfg(1, 1), counted_backend(mc), std::make_shared<FlipPolicy>(&flip));
    auto a = host.acquire(rec(5), store, codec);   // lands in dynamic partition
    check(static_cast<bool>(a.lease), "flip: dynamic fill");
    flip = true;                                    // reclassify ALL keys as policy-resident
    auto b = host.acquire(rec(5), store, codec);    // same key
    check(b.lease && b.lease.data() == a.lease.data() && store.reads == 1,
          "flip: hit path does not relocate a reclassified key to its protected class");
    a.lease.reset(); b.lease.reset();
    auto p = host.acquire(rec(99), store, codec);  // fills the policy slot
    check(static_cast<bool>(p.lease), "flip: policy fill on empty policy slot");
    p.lease.reset();
    auto c = host.acquire(rec(98), store, codec);  // policy-resident, partition full
    check(c.status == HostAcquireStatus::Capacity,
          "flip: reclassified key cannot victimize the policy partition");
    // And the record stranded in the dynamic partition is only reclaimable by
    // a Dynamic-classified acquire or an explicit evict — a subtle
    // policy-contract consequence, verified here so rewrites don't regress it.
    check(host.evict(rec(5).key), "flip: stranded dynamic record withdrawable by evict");
    auto d = host.acquire(rec(98), store, codec);
    check(d.status == HostAcquireStatus::Capacity,
          "flip: an empty dynamic slot cannot host a policy-resident key");
    check(host.evict(rec(99).key) && host.acquire(rec(98), store, codec).lease,
          "flip: explicit policy withdrawal restores the policy partition");
}

// ===========================================================================
// 5. Store-side guards: lying/throwing stores, size mismatch, codec rejection,
//    identity/layout mutation between get() calls.
// ===========================================================================
void store_guard_matrix() {
    std::cout << "-- store guard matrix --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(3), counted_backend(mc), std::make_shared<AuditPolicy>());
    for (int m = 1; m <= 4; ++m) {
        store.mode = m;
        auto r = host.acquire(rec(10 + m), store, codec);
        check(r.status == HostAcquireStatus::FillFailed && !r.lease,
              "guard: fail/throw/short/lied fill all fail closed");
    }
    store.mode = 0;
    // resident-key size mismatch poisons lookup but not the stored slot
    auto good = host.acquire(rec(20), store, codec);
    check(static_cast<bool>(good.lease), "guard: baseline fill");
    auto bad = rec(20); bad.exact_bytes = 32; bad.stored_bytes = 32;
    check(host.acquire(bad, store, codec).status == HostAcquireStatus::Invalid,
          "guard: same-key different-size record rejected");
    auto still = host.acquire(rec(20), store, codec);
    check(still.lease && still.lease.data() == good.lease.data(),
          "guard: mismatched acquire did not poison the resident slot");
    // oversize never reaches the store
    auto big = rec(30); big.exact_bytes = big.stored_bytes = 4096 + 1;
    const auto before = store.reads.load();
    check(host.acquire(big, store, codec).status == HostAcquireStatus::Invalid &&
          store.reads == before, "guard: oversize rejected before touching store");
    // codec mismatch
    auto foreign = rec(31); foreign.codec = "zlib-nope";
    check(host.acquire(foreign, store, codec).status == HostAcquireStatus::Invalid,
          "guard: foreign codec rejected");
}

void adapter_layout_guards() {
    std::cout << "-- cold adapter identity/layout guards --\n";
    FakeStore store;
    ExpertView layout;
    check(store.get(0, 1, &layout) && layout.ok(), "adapter: layout reference resolves");
    ExpertStoreColdAdapter cold(store, "audit-ckpt", "packed-v1", layout);
    const size_t n = cold.record(0, 1).exact_bytes;
    check(n == store.record_bytes(), "adapter: record bytes = sum of six regions");
    std::vector<uint8_t> dst(n, 0);
    check(cold.read(cold.record(3, 1).key, dst.data(), n).success, "adapter: read ok");
    check(std::equal(dst.begin(), dst.end(), store.contig.begin()),
          "adapter: contiguous materialize preserves order");

    // wrong-byte-count read must fail without touching dst
    std::vector<uint8_t> dst2(n + 8, 0xee);
    check(!cold.read(cold.record(3, 1).key, dst2.data(), n + 8).success &&
          dst2[0] == 0xee, "adapter: wrong byte count rejected before write");
    // wrong layer lookup fails (expert out of range)
    check(!cold.read(cold.record(99, 0).key, dst.data(), n).success,
          "adapter: out-of-range expert fails closed");
    // identity flip is detected per read
    store.flip_identity = true;
    check(!cold.read(cold.record(3, 1).key, dst.data(), n).success,
          "adapter: store identity change detected per-read");
    store.flip_identity = false;
    // layout mutation detected
    store.shrink_nbytes = true;
    check(!cold.read(cold.record(3, 1).key, dst.data(), n).success,
          "adapter: layout nbytes mutation detected");
    store.shrink_nbytes = false;
    // mis-ordered contiguous tiling rejected rather than reinterpreted
    store.misorder = true;
    check(!cold.read(cold.record(3, 1).key, dst.data(), n).success,
          "adapter: non-canonical region ordering rejected");
    store.misorder = false;
    check(cold.read(cold.record(3, 1).key, dst.data(), n).success,
          "adapter: canonical store reads again");
}

// ===========================================================================
// 6. Host-tier eviction semantics + accounting.
// ===========================================================================
void evict_semantics() {
    std::cout << "-- evict semantics --\n";
    IdentityCodec codec; GatedStore store; MemoryCounts mc;
    HostExpertTier host(cfg(1), counted_backend(mc), std::make_shared<AuditPolicy>());
    store.release_after = 1;
    auto fill = std::async(std::launch::async, [&] {
        return host.acquire(rec(1), store, codec);
    });
    store.wait_entered(1);
    check(!host.evict(rec(1).key), "evict: Filling slot not evictable");
    check(host.stats().leased_slots == 1, "evict: filling slot counted as leased");
    store.release();
    auto a = fill.get();
    check(static_cast<bool>(a.lease), "evict: fill completed");
    check(!host.evict(rec(1).key), "evict: leased Ready slot not evictable");
    a.lease.reset();
    check(host.evict(rec(1).key), "evict: idle Ready slot evicted");
    check(!host.evict(rec(1).key), "evict: stale key on Empty slot is a no-op");
    check(host.stats().resident_bytes == 0 && host.stats().evictions == 1,
          "evict: accounting consistent");
}

void budget_stride_boundary() {
    std::cout << "-- budget/stride boundary --\n";
    MemoryCounts mc;
    // slot_bytes=100, align=64 -> stride=128. budget 256 -> 2 slots fit exactly.
    HostTierConfig c{100, 64, 0, 2, 256, true};
    { HostExpertTier ok(c, counted_backend(mc), std::make_shared<AuditPolicy>());
      check(mc.allocs == 2, "budget: exact stride-fit accepted"); }
    c.budget_bytes = 255;
    bool threw = false;
    try { HostExpertTier bad(c, counted_backend(mc), std::make_shared<AuditPolicy>()); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "budget: one byte short of stride total rejected");
    // zero dynamic partition: dynamic acquires are always Capacity
    HostTierConfig c2{64, 64, 1, 0, 64, true};
    IdentityCodec codec; ByteStore store;
    HostExpertTier host(c2, counted_backend(mc), std::make_shared<AuditPolicy>());
    check(host.acquire(rec(5), store, codec).status == HostAcquireStatus::Capacity,
          "budget: zero-slot dynamic partition always Capacity");
    check(static_cast<bool>(host.acquire(rec(99), store, codec).lease), "budget: policy partition still usable");
}

void pin_fallback_accounting() {
    std::cout << "-- pin fallback accounting --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostTierConfig c = cfg(2); c.try_pin = false;
    HostExpertTier host(c, counted_backend(mc), std::make_shared<AuditPolicy>());
    check(mc.pins == 0 && host.stats().pin_failures == 0 && host.stats().pinned_bytes == 0,
          "pin: try_pin=false skips registration entirely");
    auto a = host.acquire(rec(1), store, codec);
    check(a.lease && !a.lease.pinned(), "pin: unregistered slot still serves");
    MemoryCounts mc2;
    HostExpertTier host2(cfg(2), counted_backend(mc2, false), std::make_shared<AuditPolicy>());
    check(mc2.pins == 2 && host2.stats().pin_failures == 2 && host2.stats().pinned_bytes == 0,
          "pin: per-slot failure counted once each at init");
}

// ===========================================================================
// 7. discard_unpinned: generation and pin guards on the device side.
// ===========================================================================
void discard_unpinned_guards() {
    std::cout << "-- discard_unpinned guards --\n";
    VramCacheManager cache;
    check(cache.init(256, mock_device()), "discard: init");
    check(cache.ensure(0, 7, 64, 0), "discard: ensure");
    const uint64_t gen = cache.generation_of(0, 7);
    check(gen != 0 && cache.is_resident(0, 7), "discard: resident generation");
    check(cache.pin(0, 7), "discard: pin");
    check(!cache.discard_unpinned(0, 7, gen), "discard: pinned block protected");
    check(cache.unpin(0, 7), "discard: unpin");
    check(!cache.discard_unpinned(0, 7, gen + 1), "discard: wrong generation refused");
    check(cache.is_resident(0, 7), "discard: block survives wrong-generation call");
    check(cache.discard_unpinned(0, 7, gen), "discard: exact generation discards");
    check(!cache.is_resident(0, 7) && cache.used_bytes() == 0,
          "discard: arena space reclaimed");
    check(!cache.discard_unpinned(0, 7, gen), "discard: absent key refused");
    // a replacement block has a NEW generation; old generation cannot erase it
    check(cache.ensure(0, 7, 64, 0), "discard: refill");
    const uint64_t gen2 = cache.generation_of(0, 7);
    check(gen2 != gen && !cache.discard_unpinned(0, 7, gen),
          "discard: stale generation cannot erase replacement block");
}

// ===========================================================================
// 8. Seam guards on the managed transfer path (mock stream).
// ===========================================================================
void prefetch_host_lease_guards() {
    std::cout << "-- prefetch_host_lease scope guards --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc), std::make_shared<AuditPolicy>());
    VramCacheManager cache;
    check(cache.init(256, mock_device()), "guard: cache");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "guard: prefetcher");
    auto a = host.acquire(rec(1), store, codec);
    check(static_cast<bool>(a.lease), "guard: lease");
    check(pf.prefetch_host_lease(a.lease, 0) < 0,
          "guard: host lease rejected before experimental enablement");
    a.lease.reset(); // free the slot for the foreign-lease acquisitions below
    {
        DeviceExpertTier dev(cache, pf, rec(0).key);
        auto alien = rec(2); alien.key.model = "foreign-checkpoint";
        auto alien_lease = host.acquire(alien, store, codec);
        check(alien_lease.lease && pf.prefetch_host_lease(alien_lease.lease, 0) < 0,
              "guard: lease whose model mismatches scope rejected");
        auto alien2 = rec(2); alien2.key.representation = "foreign-layout";
        auto alien2_lease = host.acquire(alien2, store, codec);
        check(alien2_lease.lease && pf.prefetch_host_lease(alien2_lease.lease, 0) < 0,
              "guard: lease whose representation mismatches scope rejected");
        HostExpertLease empty;
        check(pf.prefetch_host_lease(empty, 0) < 0, "guard: empty lease rejected");
    }
}

void device_tier_ctor_guards() {
    std::cout << "-- device tier exclusivity guards --\n";
    VramCacheManager cache;
    check(cache.init(256, mock_device()), "ctor: cache");
    check(cache.ensure(0, 0, 64, 0), "ctor: seed resident block");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "ctor: prefetcher");
    bool threw = false;
    try { DeviceExpertTier dev(cache, pf, rec(0).key); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "ctor: non-empty cache rejected");
    VramCacheManager cache2;
    check(cache2.init(256, mock_device()), "ctor2: cache");
    AsyncPrefetcher pf2(cache2);
    check(pf2.init(false), "ctor2: prefetcher");
    const uint8_t src[64] = {0};
    check(pf2.prefetch(0, 0, src, 64, 0) >= 0, "ctor2: seed in-flight transfer");
    threw = false;
    try { DeviceExpertTier dev(cache2, pf2, rec(0).key); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "ctor: in-flight prefetcher rejected");
}

// ===========================================================================
// 9. experimental flag outlives the tier facade (auditor note: there is no
//    disable API; map_key stays 64-bit and prefetch_host_lease stays armed).
// ===========================================================================
void experimental_flag_persistence() {
    std::cout << "-- experimental flag persistence after tier teardown --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc), std::make_shared<AuditPolicy>());
    VramCacheManager cache;
    check(cache.init(512, mock_device()), "persist: cache");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "persist: prefetcher");
    { DeviceExpertTier dev(cache, pf, rec(0).key); }
    // After the tier died, a lease is still accepted by prefetch_host_lease
    // (armed flag + scope are never cleared — documented quirk, not a crash).
    auto a = host.acquire(rec(1), store, codec);
    check(a.lease && pf.prefetch_host_lease(a.lease, 0) >= 0,
          "persist: lease still accepted after DeviceExpertTier destruction");
    check(pf.wait(0, 1), "persist: transfer completes");
    if (sizeof(long) == 4) {
        // map_key remains 64-bit: legacy calls no longer collide either.
        const uint8_t src[64] = {0};
        const long x = pf.prefetch(0, 6, src, 64, 0);
        const long y = pf.prefetch(1, 6, src, 64, 0);
        check(x != y, "persist: LLP64 legacy collision permanently fixed for this prefetcher");
    }
}

// ===========================================================================
// 10. Concurrency stress: N threads, same key -> exactly one fill; distinct
//     keys over few slots -> no crashes, allocator balanced at teardown.
// ===========================================================================
void concurrent_stress() {
    std::cout << "-- concurrent stress --\n";
    {
        IdentityCodec codec; ByteStore store; MemoryCounts mc;
        HostExpertTier host(cfg(4), counted_backend(mc), std::make_shared<AuditPolicy>());
        constexpr int kThreads = 8;
        std::vector<std::future<HostAcquireResult>> fs;
        for (int i = 0; i < kThreads; ++i)
            fs.push_back(std::async(std::launch::async,
                [&] { return host.acquire(rec(42), store, codec); }));
        size_t ready = 0;
        for (auto& f : fs) ready += f.get().lease ? 1 : 0;
        check(ready == kThreads && store.reads == 1,
              "stress: N-way same-key race performs exactly one fill");
        check(host.stats().fills == 1 &&
              host.stats().host_hit == kThreads - 1,
              "stress: one miss + N-1 coalesced/hit accounted");
    }
    {
        IdentityCodec codec; ByteStore store; MemoryCounts mc;
        HostExpertTier host(cfg(4), counted_backend(mc), std::make_shared<AuditPolicy>());
        constexpr int kThreads = 4, kIters = 64;
        std::atomic<int> errors{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < kThreads; ++t)
            ts.emplace_back([&, t] {
                for (int i = 0; i < kIters; ++i) {
                    const int expert = (t + i) % 8;
                    auto r = host.acquire(rec(expert), store, codec);
                    if (r.lease) {
                        if (!exact(r.lease.data(), r.lease.key())) ++errors;
                        std::this_thread::yield(); // hold the lease briefly
                    } else if (r.status != HostAcquireStatus::Capacity) {
                        ++errors; // Capacity is legal; Invalid/FillFailed are not
                    }
                }
            });
        for (auto& th : ts) th.join();
        check(errors == 0, "stress: no torn leases, stale bytes, or unexpected failures");
        check(host.stats().fills + host.stats().host_hit > 0, "stress: stats alive");
    }
    // allocator balance happens at HostExpertTier destruction
}

// ===========================================================================
// 11. Mock managed-path details: collect() is a no-op without wait_one;
//     wait() performs the copy; reset() releases outstanding leases.
// ===========================================================================
void mock_collect_and_reset() {
    std::cout << "-- mock collect/reset source lifecycle --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(1), counted_backend(mc), std::make_shared<AuditPolicy>());
    VramCacheManager cache;
    check(cache.init(128, mock_device()), "mock: cache");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "mock: prefetcher");
    {
        DeviceExpertTier dev(cache, pf, rec(0).key);
        check(dev.stage(rec(1), store, host, codec, 0), "mock: stage");
        check(host.stats().leased_slots == 1, "mock: pending mock-DMA holds lease");
        check(!pf.collect_host_sources(false),
              "mock: non-blocking collect cannot reclaim an un-drained mock transfer");
        check(host.stats().leased_slots == 1, "mock: lease still held");
        check(pf.collect_host_sources(true),
              "mock: blocking collect performs the copy and frees the slot");
        check(host.stats().leased_slots == 0 &&
              exact(static_cast<const uint8_t*>(cache.data(0, 1)), rec(1).key),
              "mock: collect wrote exact bytes to the arena block");
        check(pf.wait(0, 1), "mock: wait sees the completed transfer");
        check(dev.stage(rec(2), store, host, codec, 0), "mock: slot reused for second expert");
        check(host.stats().leased_slots == 1, "mock: second pending DMA holds lease");
    }
    // DeviceExpertTier gone; outstanding lease must be released by reset().
    pf.reset();
    check(host.stats().leased_slots == 0, "mock: reset() releases outstanding source lease");
    check(cache.is_resident(0, 2), "mock: drained block stays resident after reset");
    check(static_cast<bool>(host.acquire(rec(3), store, codec).lease), "mock: host slot reusable");
}

} // namespace

int main() {
    llp64_layer_collision();
    experimental_mapkey_distinct();
    lease_refcount_lifecycle();
    empty_lease_contract();
    facade_teardown_mid_fill();
    fill_failure_publication();
    residency_partitions();
    residency_flip_stays_in_slot();
    store_guard_matrix();
    adapter_layout_guards();
    evict_semantics();
    budget_stride_boundary();
    pin_fallback_accounting();
    discard_unpinned_guards();
    prefetch_host_lease_guards();
    device_tier_ctor_guards();
    experimental_flag_persistence();
    concurrent_stress();
    mock_collect_and_reset();
    std::cout << "Phase 2 audit failures: " << failures << '\n';
    return failures ? 1 : 0;
}
