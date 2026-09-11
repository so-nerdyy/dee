// tests/test_phase2_tier_audit_v2b.cpp
//
// Second independent adversarial audit suite for the Phase-2 tier seams —
// this run targets Luna's rewrite at 56dad3c1 ("feat(phase2): integrate plain
// LRU tiers and VRAM repair"), which the first audit (08d3d51, worktree v2a)
// predates. All first-audit checks are ported verbatim; new sections cover
// the rewrite deltas:
//
//   * PlainLruHostPlacementPolicy::victim — offered-candidate invariant
//     (only Ready, unleased, Dynamic slots) under the min-last_use pick, and
//     last_use tick semantics under hit / evict / recycle.
//   * The enabled x host_enabled x vram_priority_fix_enabled sub-switch
//     matrix (8 combos) at Engine::init, plus host-arm-without-model_identity
//     fail-closed and the policy_slots>0-with-null-policy rejection.
//   * eviction_score precedence: experimental_plain_lru_ vs an injected
//     experimental_score_ callback (vram_cache.h).
//   * Engine::phase2_metrics VRAM-only arm (engine.h ~208-219): device_bytes/
//     budget reported with no tier objects.
//
// CPU-only (MinGW/GCC, DEE_CUDA=OFF). Engine-level switch checks capture
// stderr to distinguish "rejected by the Phase-2 gate" from "passed the gate
// and failed later at shard loading" — both return false from init().

#include "dee/expert_tiers.h"
#include "dee/expert_store.h"
#include "dee/vram_cache.h"
#include "dee/async_prefetcher.h"
#include "dee/engine.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define AUDIT_DUP    _dup
#define AUDIT_DUP2   _dup2
#define AUDIT_CLOSE  _close
#define AUDIT_FILENO _fileno
#else
#include <unistd.h>
#define AUDIT_DUP    dup
#define AUDIT_DUP2   dup2
#define AUDIT_CLOSE  close
#define AUDIT_FILENO fileno
#endif

namespace {
using namespace dee;

int failures = 0;
void check(bool condition, const char* text) {
    std::cout << (condition ? "PASS " : "FAIL ") << text << '\n';
    if (!condition) ++failures;
}

// ---------------------------------------------------------------------------
// Fixtures (identical to the first audit suite)
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

// Gates only one expert's fill inside read(); all other keys pass through.
// (GatedStore's entered>=release_after gate blocks every later entrant, which
// would deadlock a test that needs other fills to proceed.)
struct SelectiveGatedStore : ByteStore {
    std::mutex m; std::condition_variable cv;
    int gate_expert = -1;
    bool open = false;
    std::atomic<int> entered_gate{0};
    ColdReadResult read(const TierExpertKey& k, uint8_t* d, size_t n) override {
        if (k.expert == gate_expert) {
            std::unique_lock<std::mutex> l(m);
            ++entered_gate;
            cv.notify_all();
            cv.wait(l, [&] { return open; });
        }
        return ByteStore::read(k, d, n);
    }
    void wait_entered() {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return entered_gate.load() >= 1; });
    }
    void release() { { std::lock_guard<std::mutex> l(m); open = true; } cv.notify_all(); }
};

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

// ---------------------------------------------------------------------------
// stderr capture for Engine::init gate diagnostics. Both "rejected by the
// Phase-2 switch gate" and "passed the gate, failed at shard load" return
// false; only the stderr text distinguishes them.
// ---------------------------------------------------------------------------
struct StderrCapture {
    int saved_fd = -1;
    FILE* tmp = nullptr;
    bool ok = false;
    bool begin() {
        std::fflush(stderr);
        saved_fd = AUDIT_DUP(AUDIT_FILENO(stderr));
        tmp = std::tmpfile();
        if (saved_fd < 0 || !tmp) { end(); return false; }
        ok = AUDIT_DUP2(AUDIT_FILENO(tmp), AUDIT_FILENO(stderr)) == 0;
        return ok;
    }
    std::string end() {
        std::fflush(stderr);
        if (saved_fd >= 0) {
            AUDIT_DUP2(saved_fd, AUDIT_FILENO(stderr));
            AUDIT_CLOSE(saved_fd);
            saved_fd = -1;
        }
        std::string out;
        if (tmp) {
            std::rewind(tmp);
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, tmp)) > 0) out.append(buf, n);
            std::fclose(tmp);
            tmp = nullptr;
        }
        return out;
    }
    ~StderrCapture() { end(); }
};

// ===========================================================================
// 1. LLP64 map_key collision: the legacy key map truncates layer bits when
//    sizeof(long)==4. Deliberately preserved for the legacy path (default-OFF
//    equivalence); full keys are used once the experimental tier is enabled.
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
// 9. experimental flag outlives the tier facade (no disable API; map_key
//    stays 64-bit and prefetch_host_lease stays armed).
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

// ===========================================================================
// 12. NEW (v2b): PlainLruHostPlacementPolicy — the rewrite's production host
//     policy. Victim must be the min-last_use offered slot; the offered set
//     is gathered by the tier as {Ready, references==0, Dynamic} only, so the
//     min-pick cannot subvert the invariant (and any_of revalidates anyway).
// ===========================================================================
void plain_lru_victim_unit() {
    std::cout << "-- plain LRU victim: unit semantics --\n";
    PlainLruHostPlacementPolicy lru;
    TierExpertKey k{"m", 0, 0, "r"};
    check(lru.residency(k) == HostResidency::Dynamic, "lru: every key is Dynamic");
    check(!lru.victim(k, {}).has_value(), "lru: empty candidate set -> nullopt");
    TierExpertKey k1{"m", 0, 1, "r"}, k2{"m", 0, 2, "r"}, k3{"m", 0, 3, "r"};
    // Offered out of order; the min-last_use slot must win.
    std::vector<HostVictim> c{{5, &k1, 1, 40, 64}, {2, &k2, 2, 10, 64}, {9, &k3, 3, 30, 64}};
    check(lru.victim(k, c) == std::optional<size_t>(2), "lru: picks lowest last_use");
    // Equal last_use (defensive; unreachable in practice — ticks are unique):
    // tie breaks to the lower slot index, still inside the offered set.
    std::vector<HostVictim> tie{{7, &k1, 1, 5, 64}, {3, &k2, 2, 5, 64}};
    check(lru.victim(k, tie) == std::optional<size_t>(3), "lru: last_use tie breaks to lower slot");
}

void plain_lru_tier_ordering() {
    std::cout << "-- plain LRU tier ordering (hit/fill tick semantics) --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    auto a1 = host.acquire(rec(1), store, codec); a1.lease.reset(); // t1
    auto a2 = host.acquire(rec(2), store, codec); a2.lease.reset(); // t2
    auto h1 = host.acquire(rec(1), store, codec); h1.lease.reset(); // hit -> t3
    auto a3 = host.acquire(rec(3), store, codec); a3.lease.reset(); // evict -> t4
    check(host.stats().evictions == 1, "lru-tier: one eviction");
    // E2 was the stalest (last_use t2 < hit t3 < fill t4): it must be gone.
    const auto r_before = store.reads.load();
    auto r1 = host.acquire(rec(1), store, codec); r1.lease.reset();
    check(store.reads == r_before, "lru-tier: hit-refreshed record retained");
    auto r3 = host.acquire(rec(3), store, codec); r3.lease.reset();
    check(store.reads == r_before, "lru-tier: newest fill retained");
    auto r2 = host.acquire(rec(2), store, codec); r2.lease.reset();
    check(store.reads == r_before + 1, "lru-tier: least-recently-used record was evicted");
    check(host.stats().evictions == 2, "lru-tier: re-acquire evicts oldest again");
}

void plain_lru_recycle_and_evict_ticks() {
    std::cout << "-- plain LRU tick semantics on evict/recycle --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    auto a1 = host.acquire(rec(1), store, codec); a1.lease.reset(); // t1
    auto a2 = host.acquire(rec(2), store, codec); a2.lease.reset(); // t2
    check(host.evict(rec(1).key), "lru-tick: explicit evict of oldest");
    auto a3 = host.acquire(rec(3), store, codec); a3.lease.reset(); // reuses slot0 -> t3
    // Now E2 (t2) is oldest; E3 (t3) newest. A miss must evict E2.
    auto a4 = host.acquire(rec(4), store, codec); a4.lease.reset();
    const auto rb = store.reads.load();
    auto h3 = host.acquire(rec(3), store, codec); h3.lease.reset();
    check(store.reads == rb, "lru-tick: recycled slot carries its new (fresh) stamp");
    auto m2 = host.acquire(rec(2), store, codec); m2.lease.reset();
    check(store.reads == rb + 1, "lru-tick: stalest resident evicted, not the recycled one");
}

void plain_lru_never_offers_filling_or_leased() {
    std::cout << "-- plain LRU offered-set excludes Filling/leased slots --\n";
    IdentityCodec codec; SelectiveGatedStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    store.gate_expert = 1;               // only E1's fill blocks inside read()
    auto filling = std::async(std::launch::async, [&] {
        return host.acquire(rec(1), store, codec);
    });
    store.wait_entered();                // slot0 = Filling(E1)
    auto b = host.acquire(rec(2), store, codec);
    check(b.status == HostAcquireStatus::Ready && b.lease,
          "lru-offer: second acquire uses the Empty slot, never disturbs the fill");
    b.lease.reset();
    auto c = host.acquire(rec(3), store, codec);
    check(c.status == HostAcquireStatus::Ready && c.lease,
          "lru-offer: third acquire evicts the only Ready+unleased+Dynamic slot");
    c.lease.reset();
    store.release();
    auto a = filling.get();
    check(a.status == HostAcquireStatus::Ready && a.lease, "lru-offer: gated fill completes");
    a.lease.reset();
    // The Filling slot was never offered as a victim: E1 survived two evictions'
    // worth of pressure. Re-acquires of E2/E3 prove both were rotated through
    // the single dynamic candidate while E1 was protected by its phase.
    const auto rb = store.reads.load();
    auto h1 = host.acquire(rec(1), store, codec); h1.lease.reset();
    check(store.reads == rb, "lru-offer: Filling-slot record was never victimized");
}

void plain_lru_leased_never_offered() {
    std::cout << "-- plain LRU offered-set excludes leased slots --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    auto a = host.acquire(rec(1), store, codec);   // keep the lease held
    auto b = host.acquire(rec(2), store, codec); b.lease.reset();
    auto c = host.acquire(rec(3), store, codec);   // must evict E2 (E1 is leased)
    check(static_cast<bool>(c.lease), "lru-lease: fill past the leased slot");
    c.lease.reset();
    const auto rb = store.reads.load();
    auto d = host.acquire(rec(4), store, codec);   // E1 still leased -> evicts E3
    check(static_cast<bool>(d.lease), "lru-lease: second fill past the leased slot");
    d.lease.reset();
    auto h1 = host.acquire(rec(1), store, codec);
    check(h1.lease && store.reads == rb + 1,
          "lru-lease: leased record survived both evictions without a re-read");
    check(host.stats().evictions == 2, "lru-lease: exactly two evictions");
    h1.lease.reset(); a.lease.reset();
}

void hostile_victim_out_of_set() {
    std::cout << "-- victim answer validated against the offered set --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    struct HostilePolicy : HostPlacementPolicy {
        size_t answer;
        explicit HostilePolicy(size_t s) : answer(s) {}
        HostResidency residency(const TierExpertKey&) const override {
            return HostResidency::Dynamic;
        }
        std::optional<size_t> victim(const TierExpertKey&,
                                     const std::vector<HostVictim>&) const override {
            return answer; // never an offered slot
        }
    };
    {   // Answer outside every slot index: must degrade to Capacity.
        HostExpertTier host(cfg(1), counted_backend(mc),
                            std::make_shared<HostilePolicy>(999));
        auto a = host.acquire(rec(1), store, codec); a.lease.reset();
        auto b = host.acquire(rec(2), store, codec);
        check(b.status == HostAcquireStatus::Capacity && host.stats().evictions == 0,
              "hostile: out-of-range victim answer fails closed to Capacity");
    }
    {   // Answer naming a REAL but unoffered (leased) slot: also rejected.
        HostExpertTier host(cfg(2), counted_backend(mc),
                            std::make_shared<HostilePolicy>(0));
        auto a = host.acquire(rec(1), store, codec);   // slot0 stays leased
        auto b = host.acquire(rec(2), store, codec); b.lease.reset(); // slot1
        auto c = host.acquire(rec(3), store, codec);   // offered={slot1}; answer=slot0
        check(c.status == HostAcquireStatus::Capacity,
              "hostile: naming a leased (unoffered) slot fails closed");
        a.lease.reset();
    }
    {   // nullopt answer: also Capacity, no eviction.
        struct NullPolicy : HostPlacementPolicy {
            HostResidency residency(const TierExpertKey&) const override {
                return HostResidency::Dynamic;
            }
            std::optional<size_t> victim(const TierExpertKey&,
                                         const std::vector<HostVictim>&) const override {
                return std::nullopt;
            }
        };
        HostExpertTier host(cfg(1), counted_backend(mc), std::make_shared<NullPolicy>());
        auto a = host.acquire(rec(1), store, codec); a.lease.reset();
        check(host.acquire(rec(2), store, codec).status == HostAcquireStatus::Capacity,
              "hostile: nullopt victim answer fails closed");
    }
}

void plain_lru_policy_slots_stranded() {
    std::cout << "-- plain LRU + policy_slots>0 strands the policy partition --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    // The engine rejects policy_slots>0 with a null host_policy because plain
    // LRU classifies everything Dynamic: the policy-resident slots can never
    // be filled, so configured capacity silently shrinks. Verify the stranded
    // behavior the rejection exists to prevent.
    HostExpertTier host(cfg(1, 1), counted_backend(mc),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    for (int e = 1; e <= 4; ++e) {
        auto r = host.acquire(rec(e), store, codec);
        check(static_cast<bool>(r.lease), "strand: dynamic fill");
        r.lease.reset();
    }
    check(store.reads == 4 && host.stats().evictions == 3,
          "strand: all traffic squeezed through the single dynamic slot");
    // With dynamic_slots==0 every acquire is Capacity even though a
    // policy-resident slot sits allocated and empty.
    HostExpertTier host2(cfg(0, 1), counted_backend(mc),
                         std::make_shared<PlainLruHostPlacementPolicy>());
    check(host2.acquire(rec(1), store, codec).status == HostAcquireStatus::Capacity,
          "strand: all-policy-slot tier rejects every dynamic acquire");
}

void plain_lru_concurrent() {
    std::cout << "-- plain LRU under concurrency --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(4), counted_backend(mc),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    constexpr int kThreads = 4, kIters = 64;
    std::atomic<int> errors{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t)
        ts.emplace_back([&, t] {
            for (int i = 0; i < kIters; ++i) {
                auto r = host.acquire(rec((t + i) % 8), store, codec);
                if (r.lease) {
                    if (!exact(r.lease.data(), r.lease.key())) ++errors;
                    std::this_thread::yield();
                } else if (r.status != HostAcquireStatus::Capacity) ++errors;
            }
        });
    for (auto& th : ts) th.join();
    check(errors == 0, "lru-stress: no torn leases or unexpected failures");
    check(host.stats().evictions > 0, "lru-stress: LRU eviction exercised under races");
}

// ===========================================================================
// 13. NEW (v2b): eviction_score precedence — experimental_plain_lru_ must
//     outrank an injected experimental_score_ callback AND the legacy
//     priority term (vram_cache.h eviction_score order).
// ===========================================================================
void vram_plain_lru_precedence() {
    std::cout << "-- VRAM plain-LRU precedence over callback and priority --\n";
    constexpr size_t BLK = 64;
    {   // flag + injected callback: flag wins (callback alone would evict the
        // NEWEST block — it returns -last_used).
        VramCacheManager cache;
        check(cache.init(BLK * 2, mock_device()), "prec: init");
        cache.set_experimental_eviction_score(
            [](const ExpertBlock& b) { return -b.last_used; });
        cache.set_experimental_plain_lru(true);
        check(cache.ensure(0, 0, BLK, 0) && cache.ensure(0, 1, BLK, 0), "prec: two loads");
        check(cache.ensure(0, 2, BLK, 0), "prec: third load forces eviction");
        check(!cache.is_resident(0, 0) && cache.is_resident(0, 1) &&
              cache.is_resident(0, 2),
              "prec: flag shadows callback — oldest evicted, not callback's newest");
    }
    {   // Same injected callback WITHOUT the flag: newest is evicted, proving
        // the callback is live whenever the flag does not shadow it.
        VramCacheManager cache;
        check(cache.init(BLK * 2, mock_device()), "prec2: init");
        cache.set_experimental_eviction_score(
            [](const ExpertBlock& b) { return -b.last_used; });
        check(cache.ensure(0, 0, BLK, 0) && cache.ensure(0, 1, BLK, 0), "prec2: two loads");
        check(cache.ensure(0, 2, BLK, 0), "prec2: third load forces eviction");
        check(cache.is_resident(0, 0) && !cache.is_resident(0, 1),
              "prec2: callback alone applies (newest evicted)");
    }
    {   // Flag on, no callback: staging priority is ignored entirely.
        VramCacheManager cache;
        check(cache.init(BLK * 2, mock_device()), "prec3: init");
        cache.set_experimental_plain_lru(true);
        check(cache.ensure(0, 0, BLK, 100) && cache.ensure(0, 1, BLK, 0),
              "prec3: loads with opposing priorities");
        check(cache.ensure(0, 2, BLK, 0), "prec3: third load forces eviction");
        check(!cache.is_resident(0, 0) && cache.is_resident(0, 1),
              "prec3: high-priority oldest block evicted — priority ignored");
    }
    {   // Toggle: flag off restores the injected callback's authority.
        VramCacheManager cache;
        check(cache.init(BLK * 3, mock_device()), "prec4: init");
        cache.set_experimental_eviction_score(
            [](const ExpertBlock& b) { return -b.last_used; });
        cache.set_experimental_plain_lru(true);
        check(cache.experimental_plain_lru(), "prec4: flag observable on");
        check(cache.ensure(0, 0, BLK, 0) && cache.ensure(0, 1, BLK, 0), "prec4: loads");
        cache.set_experimental_plain_lru(false);
        check(cache.ensure(0, 2, BLK, 0), "prec4: third block fits, no eviction");
        // Fourth load must evict: with the callback live again, the NEWEST
        // resident (E2, score=-last_used) is the victim — opposite of LRU.
        check(cache.ensure(0, 3, BLK, 0), "prec4: fourth load forces eviction");
        check(!cache.is_resident(0, 2) && cache.is_resident(0, 0) &&
              cache.is_resident(0, 3),
              "prec4: flag off restores callback authority (newest evicted)");
    }
}

// ===========================================================================
// 14. NEW (v2b): Engine::init sub-switch matrix. enabled x host_enabled x
//     vram_priority_fix_enabled, plus host-arm identity/platform requirements.
//     init() returns false for both gate rejections and downstream failures;
//     stderr text is captured to prove WHICH check fired.
// ===========================================================================
EngineConfig base_cfg() {
    EngineConfig c;
    // Defaults already satisfy the non-phase2 gates: num_tokens/topk/layers/
    // hidden positive, base_layer/device_id 0, transfer Bf16, cache Fp32,
    // use_cuda false, source lanes 1/6, prefetch_depth 64. No shard_path =>
    // every config that passes the Phase-2 gate dies at the shard check.
    return c;
}

std::string init_stderr(EngineConfig c, bool* init_ok) {
    StderrCapture cap;
    Engine engine;
    check(cap.begin(), "matrix: stderr capture armed");
    const bool ok = engine.init(c);
    std::string out = cap.end();
    if (init_ok) *init_ok = ok;
    return out;
}

void engine_subswitch_matrix() {
    std::cout << "-- engine sub-switch matrix (enabled x host x vram) --\n";
    const std::string kSubNeed = "subfeatures require phase2.enabled";
    const std::string kAnyNeed = "requires host_enabled or vram_priority_fix_enabled";
    const std::string kHostNeed = "host tier requires packed FP4 CUDA and an immutable model identity";
    const std::string kShard = "at least one shard path is required";

    // (enabled, host, vram) = (0,0,0): default OFF, passes the phase2 gate.
    {
        auto out = init_stderr(base_cfg(), nullptr);
        check(out.find("Phase 2") == std::string::npos &&
              out.find(kShard) != std::string::npos,
              "matrix 000: default OFF passes phase2 gate, dies at shard check");
    }
    // (0,0,1), (0,1,0), (0,1,1): subfeatures without the master switch.
    for (int h = 0; h <= 1; ++h)
        for (int v = (h ? 0 : 1); v <= 1; ++v) {
            EngineConfig c = base_cfg();
            c.phase2.host_enabled = h != 0;
            c.phase2.vram_priority_fix_enabled = v != 0;
            auto out = init_stderr(c, nullptr);
            check(out.find(kSubNeed) != std::string::npos,
                  h ? "matrix 011/010: subfeature without master rejected"
                    : "matrix 001: vram flag without master rejected");
        }
    // (1,0,0): master on, no subfeature armed.
    {
        EngineConfig c = base_cfg(); c.phase2.enabled = true;
        auto out = init_stderr(c, nullptr);
        check(out.find(kAnyNeed) != std::string::npos,
              "matrix 100: master on with no armed subfeature rejected");
    }
    // (1,0,1): VRAM-repair-only arm — legal, must pass the phase2 gate.
    {
        EngineConfig c = base_cfg();
        c.phase2.enabled = true; c.phase2.vram_priority_fix_enabled = true;
        bool ok = true;
        auto out = init_stderr(c, &ok);
        check(!ok && out.find("Phase 2") == std::string::npos &&
              out.find(kShard) != std::string::npos,
              "matrix 101: VRAM-only arm passes phase2 gate, dies at shard check");
    }
    // (1,1,0) and (1,1,1): host arm with all platform/identity requirements —
    // legal configs, must pass the phase2 gate even with DEE_CUDA=OFF (the
    // gate predicates on config flags; the CUDA-less failure comes later).
    for (int v = 0; v <= 1; ++v) {
        EngineConfig c = base_cfg();
        c.phase2.enabled = true; c.phase2.host_enabled = true;
        c.phase2.vram_priority_fix_enabled = v != 0;
        c.use_cuda = true;
        c.cache_dtype = DeviceCacheDType::Fp4E2m1;
        c.transfer_dtype = WeightTransferDType::Fp4E2m1;
        c.phase2.model_identity = "audit-checkpoint@deadbeef";
        auto out = init_stderr(c, nullptr);
        check(out.find("Phase 2") == std::string::npos &&
              out.find(kShard) != std::string::npos,
              v ? "matrix 111: combined arm passes phase2 gate, dies at shard check"
                : "matrix 110: host-only arm passes phase2 gate, dies at shard check");
    }
    // Host arm fail-closed checks: each missing requirement hits the same gate.
    {
        EngineConfig c = base_cfg();
        c.phase2.enabled = true; c.phase2.host_enabled = true;
        c.use_cuda = true;
        c.cache_dtype = DeviceCacheDType::Fp4E2m1;
        c.transfer_dtype = WeightTransferDType::Fp4E2m1;
        // model_identity left empty
        auto out = init_stderr(c, nullptr);
        check(out.find(kHostNeed) != std::string::npos,
              "matrix: host arm without model_identity fails closed");
    }
    {
        EngineConfig c = base_cfg();
        c.phase2.enabled = true; c.phase2.host_enabled = true;
        c.phase2.model_identity = "audit-checkpoint@deadbeef";
        c.use_cuda = false; // missing CUDA
        c.cache_dtype = DeviceCacheDType::Fp4E2m1;
        c.transfer_dtype = WeightTransferDType::Fp4E2m1;
        auto out = init_stderr(c, nullptr);
        check(out.find(kHostNeed) != std::string::npos,
              "matrix: host arm without --cuda fails closed");
    }
    {
        EngineConfig c = base_cfg();
        c.phase2.enabled = true; c.phase2.host_enabled = true;
        c.phase2.model_identity = "audit-checkpoint@deadbeef";
        c.use_cuda = true;
        c.cache_dtype = DeviceCacheDType::Fp4E2m1;
        c.transfer_dtype = WeightTransferDType::Bf16; // wrong transfer dtype
        auto out = init_stderr(c, nullptr);
        check(out.find(kHostNeed) != std::string::npos,
              "matrix: host arm with Bf16 transfer fails closed");
    }
}

// ===========================================================================
// 15. NEW (v2b): phase2_metrics VRAM-only arm. cfg_ is stored before
//     validation, so a config that passed the phase2 gate but failed later
//     (no shard) still reports the armed arm: device fields read through to
//     the (here uninitialized) arena safely, tokens/valid flag tracked.
// ===========================================================================
void phase2_metrics_vram_arm() {
    std::cout << "-- phase2_metrics VRAM-only arm --\n";
    {   // Never-initialized engine: all zero, never crashes.
        Engine engine;
        auto m = engine.phase2_metrics(8);
        check(m.device_bytes == 0 && m.device_budget == 0 && m.tokens == 0 &&
              !m.bytes_per_token_valid,
              "metrics: unconfigured engine reports empty TierMetrics");
    }
    {   // Default-OFF config that failed at the shard check: arm bits off,
        // metrics stay empty even with a token count.
        Engine engine;
        { StderrCapture cap; check(cap.begin(), "metrics: capture"); (void)engine.init(base_cfg()); cap.end(); }
        auto m = engine.phase2_metrics(8);
        check(m.tokens == 0 && !m.bytes_per_token_valid && m.device_bytes == 0,
              "metrics: disabled arm stays empty regardless of tokens");
    }
    {   // VRAM-armed config (passed the gate, failed at shard): the arm is
        // observable — tokens propagate, per-token flag honors the
        // denominator, device fields read the uninitialized arena as zero.
        Engine engine;
        EngineConfig c = base_cfg();
        c.phase2.enabled = true; c.phase2.vram_priority_fix_enabled = true;
        { StderrCapture cap; check(cap.begin(), "metrics: capture"); (void)engine.init(c); cap.end(); }
        auto m0 = engine.phase2_metrics(0);
        check(m0.tokens == 0 && !m0.bytes_per_token_valid,
              "metrics: vram arm honors the zero-token invalid marker");
        auto m8 = engine.phase2_metrics(8);
        check(m8.tokens == 8 && m8.bytes_per_token_valid &&
              m8.device_bytes == 0 && m8.device_budget == 0 &&
              m8.device_peak_bytes == 0 && m8.host.allocated_bytes == 0,
              "metrics: vram arm reports tokens + arena readout (zero pre-cache-init)");
    }
    {   // enabled+host rejected config (empty identity): metrics stay empty —
        // the arm that failed validation is not reported as live.
        Engine engine;
        EngineConfig c = base_cfg();
        c.phase2.enabled = true; c.phase2.host_enabled = true;
        { StderrCapture cap; check(cap.begin(), "metrics: capture"); (void)engine.init(c); cap.end(); }
        auto m = engine.phase2_metrics(8);
        check(m.tokens == 0 && !m.bytes_per_token_valid,
              "metrics: rejected host arm does not spoof metrics");
    }
}

// ===========================================================================
// 16. NEW (v2b pass 2): second-live-tier scope replacement. At 56dad3c the
//     ctor guards are the only exclusivity: after prefetcher.reset() +
//     cache.clear() (the Engine::reset_runtime_cache pattern), a second
//     DeviceExpertTier with a DIFFERENT scope constructs while the first is
//     alive; enable_experimental_host_tier then silently REPLACES
//     experimental_scope_ under the live tier, and ~DeviceExpertTier
//     unconditionally clears the eviction-score callback — including the
//     replacement tier's own callback (unique_ptr assign order: new ctor
//     installs, then old dtor clears). Both are demonstrated live here.
//     688bd98's arming token + scope epoch close both holes.
// ===========================================================================
struct EvictNewestPolicy : DevicePlacementPolicy {
    int priority(const TierExpertKey&, int route) const override { return route; }
    int64_t eviction_score(const TierExpertKey&, int64_t last_use,
                           int) const override { return -last_use; }
};

void second_tier_scope_replacement() {
    std::cout << "-- second live DeviceExpertTier replaces the armed scope --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc), std::make_shared<AuditPolicy>());
    VramCacheManager cache;
    check(cache.init(128, mock_device()), "scope-rep: cache (two 64B blocks)");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "scope-rep: prefetcher");
    auto dev_a = std::make_unique<DeviceExpertTier>(cache, pf, rec(0).key,
        std::make_shared<EvictNewestPolicy>());
    // Prove A's eviction-score callback is live: -last_used evicts the NEWEST.
    check(cache.ensure(0, 1, 64, 0) && cache.ensure(0, 2, 64, 0), "scope-rep: seed two");
    check(cache.ensure(0, 3, 64, 0), "scope-rep: third forces eviction");
    check(cache.is_resident(0, 1) && !cache.is_resident(0, 2) && cache.is_resident(0, 3),
          "scope-rep: tier-A callback live (newest block evicted)");
    // The reset_runtime_cache pattern: drain the transfer table and empty the
    // arena while tier A is still alive. Both are public, routine operations.
    pf.reset();
    cache.clear();
    auto scope_b = rec(0).key; scope_b.model = "foreign-model";
    std::unique_ptr<DeviceExpertTier> dev_b;
    bool threw = false;
    try { dev_b = std::make_unique<DeviceExpertTier>(cache, pf, scope_b); }
    catch (const std::invalid_argument&) { threw = true; }
    check(!threw && dev_b,
          "FINDING v2b-1: second live tier admitted; armed scope silently replaced "
          "(ctor-time-only exclusivity, still live at 56dad3c; 688bd98 arming token fixes)");
    // Tier A still accepts its own scope but every stage now dies at the
    // prefetch scope gate — AFTER consuming a host fill it can never use.
    const auto fills_before = host.stats().fills;
    check(!dev_a->stage(rec(7), store, host, codec, 0),
          "scope-rep: tier-A stage fails closed under the foreign armed scope");
    check(host.stats().fills == fills_before + 1,
          "scope-rep: rejected stage consumed a host fill (record left resident-unservable)");
    check(dev_a->metrics(host, 0).device_failures == 1,
          "scope-rep: rejection attributed to device_failures");
    // ~B unconditionally clears the score callback A installed at its ctor.
    dev_b.reset();
    check(cache.ensure(0, 4, 64, 0) && cache.ensure(0, 5, 64, 0), "scope-rep: refill two");
    check(cache.ensure(0, 6, 64, 0), "scope-rep: eviction after ~B");
    check(!cache.is_resident(0, 4) && cache.is_resident(0, 5) && cache.is_resident(0, 6),
          "FINDING v2b-1: ~tier-B stripped tier-A's eviction callback — legacy "
          "scoring resumed under a live configured tier");
    dev_a.reset();
}

void tier_replace_strips_new_callback() {
    std::cout << "-- unique_ptr tier replacement strips the NEW tier's callback --\n";
    VramCacheManager cache;
    check(cache.init(128, mock_device()), "replace: cache");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "replace: prefetcher");
    const auto scope = rec(0).key;
    auto dev = std::make_unique<DeviceExpertTier>(cache, pf, scope,
        std::make_shared<EvictNewestPolicy>());
    pf.reset(); cache.clear(); // both guards pass while the first tier lives
    // Engine::init re-init does exactly this: make_unique constructs the
    // replacement (installing a fresh callback), then the assignment destroys
    // the old tier — whose dtor clears the callback the NEW tier just set.
    dev = std::make_unique<DeviceExpertTier>(cache, pf, scope,
        std::make_shared<EvictNewestPolicy>());
    check(static_cast<bool>(dev), "replace: replacement tier constructed");
    check(cache.ensure(0, 1, 64, 0) && cache.ensure(0, 2, 64, 0), "replace: seed two");
    check(cache.ensure(0, 3, 64, 0), "replace: third forces eviction");
    check(!cache.is_resident(0, 1) && cache.is_resident(0, 2) && cache.is_resident(0, 3),
          "FINDING v2b-2: old dtor ran after new ctor — live tier's configured "
          "policy silently dead (legacy scoring observed)");
}

// ===========================================================================
// 17. NEW (v2b pass 2): arm interaction — the VRAM-repair-only arm never arms
//     the prefetcher (no DeviceExpertTier is created), so on LLP64 the legacy
//     map_key truncation survives and layer>0 prefetches coalesce. The repair
//     arm is independently selectable but inherits the legacy map defect on
//     Windows. (688bd98 makes map_key full-width unconditionally — fixes.)
// ===========================================================================
void vram_arm_inherits_llp64_mapkey() {
    std::cout << "-- VRAM-only arm leaves the LLP64 map_key collision live --\n";
    if (sizeof(long) != 4) {
        std::cout << "PASS vram-mapkey: LP64 build — no truncation to inherit\n";
        return;
    }
    VramCacheManager cache;
    check(cache.init(256, mock_device()), "vram-mapkey: cache");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "vram-mapkey: prefetcher");
    cache.set_experimental_plain_lru(true); // exactly what the vram arm applies
    const uint8_t src[64] = {9};
    const long a = pf.prefetch(0, 5, src, 64, 0);
    const long b = pf.prefetch(1, 5, src, 64, 0);
    check(a >= 0, "vram-mapkey: first prefetch issued");
    check(b == a && !pf.wait(1, 5) && !cache.is_resident(1, 5),
          "FINDING v2b-3: vram-only arm on LLP64 keeps the truncated legacy "
          "map_key — layer>0 silently coalesced away (688bd98 fixes)");
}

// ===========================================================================
// 18. NEW (v2b pass 2): device-side pin protection + metrics semantics.
// ===========================================================================
void managed_pin_blocks_eviction() {
    std::cout << "-- pinned in-flight managed block survives ensure pressure --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc), std::make_shared<AuditPolicy>());
    VramCacheManager cache;
    check(cache.init(64, mock_device()), "pin-evict: single-block cache");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "pin-evict: prefetcher");
    DeviceExpertTier dev(cache, pf, rec(0).key);
    check(dev.stage(rec(1), store, host, codec, 0), "pin-evict: first stage");
    check(cache.pin_count(0, 1) == 1,
          "pin-evict: pending managed transfer holds the consumer pin");
    check(!dev.stage(rec(2), store, host, codec, 0),
          "pin-evict: second stage cannot evict the pinned in-flight block");
    check(dev.metrics(host, 0).device_failures == 1, "pin-evict: rejection counted");
    check(host.stats().fills == 2,
          "pin-evict: rejected device stage still consumed a host fill "
          "(record stays resident for the retry)");
    check(dev.wait(rec(1).key), "pin-evict: first transfer completes");
    const auto reads_before = store.reads.load();
    check(dev.stage(rec(2), store, host, codec, 0) && dev.wait(rec(2).key),
          "pin-evict: retry stages from the retained host record");
    check(store.reads == reads_before,
          "pin-evict: host hit on retry — no SSD re-read after rejection");
    check(cache.is_resident(0, 2) && !cache.is_resident(0, 1),
          "pin-evict: released block was evictable; exact LRU order held");
}

void stage_dedup_metrics() {
    std::cout << "-- duplicate stage is a device hit: counters stay consistent --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(2), counted_backend(mc), std::make_shared<AuditPolicy>());
    VramCacheManager cache;
    check(cache.init(256, mock_device()), "dedup: cache");
    AsyncPrefetcher pf(cache);
    check(pf.init(false), "dedup: prefetcher");
    DeviceExpertTier dev(cache, pf, rec(0).key);
    check(dev.stage(rec(1), store, host, codec, 0), "dedup: first stage");
    check(dev.stage(rec(1), store, host, codec, 0),
          "dedup: second stage coalesces onto the in-flight transfer");
    const auto m = dev.metrics(host, 0);
    check(m.device_miss == 1 && m.device_hit == 1 && m.H2D_bytes == 64 &&
          m.host.fills == 1 && m.host.host_miss == 1 && store.reads == 1,
          "dedup: one miss+enqueue, one resident hit — no double fill/H2D accounting");
    check(dev.wait(rec(1).key) && exact(
              static_cast<const uint8_t*>(cache.data(0, 1)), rec(1).key),
          "dedup: bytes exact after coalesced stage");
}

void metrics_invariants_and_gaps() {
    std::cout << "-- host metrics invariants and documented gaps --\n";
    IdentityCodec codec; ByteStore store; MemoryCounts mc;
    HostExpertTier host(cfg(1), counted_backend(mc), std::make_shared<AuditPolicy>());
    // Invariant: every reservation either fills or fails → miss == fills+failures.
    store.mode = 1;
    check(host.acquire(rec(1), store, codec).status == HostAcquireStatus::FillFailed,
          "metrics: failed fill");
    store.mode = 0;
    auto ok = host.acquire(rec(2), store, codec);
    check(static_cast<bool>(ok.lease), "metrics: successful fill");
    // Keep the lease held: the only dynamic slot is now unevictable, so the
    // next acquire must fail closed with Capacity (not evict under lease).
    check(host.acquire(rec(3), store, codec).status == HostAcquireStatus::Capacity,
          "metrics: capacity rejection while sole slot leased");
    ok.lease.reset();
    const auto s = host.stats();
    check(s.host_miss == s.fills + s.failures && s.host_miss == 2 &&
          s.budget_rejections == 1,
          "metrics: host_miss == fills + failures; capacity counted separately");
    // Invalid acquires reach no counter at all.
    auto bad = rec(9); bad.exact_bytes = 0; bad.stored_bytes = 0;
    check(host.acquire(bad, store, codec).status == HostAcquireStatus::Invalid,
          "metrics: invalid acquire");
    const auto s2 = host.stats();
    check(s2.host_miss == s.host_miss && s2.host_hit == s.host_hit &&
          s2.failures == s.failures && s2.budget_rejections == s.budget_rejections,
          "FINDING v2b-4: Invalid acquires are invisible in HostTierStats "
          "(no invalid_requests counter)");
    // coalesced is counted at JOIN time; the same request counts again in
    // host_hit when its fill completes — a documented dual counting.
    IdentityCodec codec2; GatedStore gate; MemoryCounts mc2;
    HostExpertTier host2(cfg(2), counted_backend(mc2), std::make_shared<AuditPolicy>());
    gate.release_after = 1;
    auto f1 = std::async(std::launch::async, [&] { return host2.acquire(rec(1), gate, codec2); });
    gate.wait_entered(1);
    auto f2 = std::async(std::launch::async, [&] { return host2.acquire(rec(1), gate, codec2); });
    const auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (host2.stats().coalesced != 1 && std::chrono::steady_clock::now() < dl)
        std::this_thread::yield();
    gate.release();
    check(f1.get().lease && f2.get().lease, "metrics: coalesced pair completes");
    check(host2.stats().coalesced == 1 && host2.stats().host_hit == 1 &&
          host2.stats().host_miss == 1,
          "metrics: joiner counted in coalesced AND host_hit (documented dual count)");
}

void adapter_identity_bound() {
    std::cout << "-- adapter identity bound (post-concat) --\n";
    FakeStore store;
    ExpertView layout;
    check(store.get(0, 1, &layout) && layout.ok(), "ident: layout resolves");
    std::string huge(TierExpertKey::max_identity_bytes, 'x'); // exactly at bound
    bool threw = false;
    try { ExpertStoreColdAdapter cold(store, huge, "packed-v1", layout); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "ident: model at bound + appended store identity exceeds "
                 "max_identity_bytes — rejected");
    std::string over(TierExpertKey::max_identity_bytes + 1, 'y');
    threw = false;
    try { ExpertStoreColdAdapter cold(store, over, "packed-v1", layout); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "ident: model over bound rejected pre-concat");
    ExpertStoreColdAdapter ok(store, "audit-ckpt", "packed-v1", layout); // control
    check(ok.record(0, 0).exact_bytes == store.record_bytes(), "ident: control adapter constructs");
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
    // v2b additions
    plain_lru_victim_unit();
    plain_lru_tier_ordering();
    plain_lru_recycle_and_evict_ticks();
    plain_lru_never_offers_filling_or_leased();
    plain_lru_leased_never_offered();
    hostile_victim_out_of_set();
    plain_lru_policy_slots_stranded();
    plain_lru_concurrent();
    vram_plain_lru_precedence();
    engine_subswitch_matrix();
    phase2_metrics_vram_arm();
    // v2b second-pass additions
    second_tier_scope_replacement();
    tier_replace_strips_new_callback();
    vram_arm_inherits_llp64_mapkey();
    managed_pin_blocks_eviction();
    stage_dedup_metrics();
    metrics_invariants_and_gaps();
    adapter_identity_bound();
    std::cout << "Phase 2 audit (v2b) failures: " << failures << '\n';
    return failures ? 1 : 0;
}
