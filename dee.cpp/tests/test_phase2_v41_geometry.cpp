// tests/test_phase2_v41_geometry.cpp
//
// W1-T7 — DeepSeek-V4.1-geometry conformance proof for the Phase-2 tier seams.
//
// The tier code (ColdExpertStore -> HostExpertTier -> DeviceExpertTier ->
// StorageCodec/TierExpertKey/StorageRecord) must be model-geometry-agnostic.
// DeepSeek-V4.1 is the abstraction sanity check:
//     40 MoE layers, 384 routed experts/layer, top-6, hidden 5120,
//     expert intermediate 2304  =>  packed FP4-e2m1 + e8m0 record of
//     18,800,640 B (~17.93 MiB), ~269 GiB routed pool.
//
// This test drives the REAL tier objects (ExpertStoreColdAdapter,
// HostExpertTier, DeviceExpertTier, IdentityCodec) at that geometry over a
// fully synthetic store.  Every V4.1 quantity below is DERIVED from the four
// dims above — the test never names a DeepSeek-V4 (43x256, 13,369,344 B)
// constant, which is the property under test: nothing in the seam should
// notice the difference.
//
// Layer-index convention exercised here mirrors PHASE3_FULL_EXPERT_STORE.md:
// main routed layers are buckets 0..L-1 and draft/mtp modules are buckets
// L..L+M-1 (DSv4: 43+N for mtp.N).  For V4.1 the same convention is
// layers 40,41,42 — the tiers treat them as plain layer indices, which is
// exactly what the convention requires.
//
// Standalone (no CUDA, no cmake — registration lives in CMakeLists which this
// task does not own):
//   g++ -std=c++17 -O2 -Wall -Wextra -I include \
//       tests/test_phase2_v41_geometry.cpp \
//       src/host_expert_tier.cpp src/expert_tiers.cpp src/vram_cache.cpp \
//       src/async_prefetcher.cpp src/expert_store.cpp src/json_min.cpp \
//       src/weight_mmap.cpp src/profiling.cpp \
//       -pthread -o /tmp/test_phase2_v41_geometry && /tmp/test_phase2_v41_geometry
//
// CMake registration snippet (for T1's CMakeLists, DEE_TEST_SOURCES):
//     tests/test_phase2_v41_geometry.cpp          (after test_phase2_host_tier.cpp)
//     set_tests_properties(test_phase2_v41_geometry PROPERTIES TIMEOUT 60)

#include "dee/expert_tiers.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace dee;

int failures = 0;
void check(bool condition, const char* text) {
    std::cout << (condition ? "PASS " : "FAIL ") << text << '\n';
    if (!condition) ++failures;
}

// ---- V4.1 geometry, derived ------------------------------------------------
// packed e2m1 I8 weight [out, in/2]; per-block e8m0 scale [out, in/32].
// Projection order is the record convention: gate(w1), up(w3), down(w2).
constexpr int kV41Layers = 40;
constexpr int kV41Experts = 384;
constexpr int kV41MtpBuckets = 3;          // layer indices 40,41,42 (L+N convention)
constexpr int kV41LayerSpan = kV41Layers + kV41MtpBuckets;
constexpr int64_t kV41Hidden = 5120;
constexpr int64_t kV41Inter = 2304;

constexpr size_t packed_i8_bytes(int64_t out, int64_t in) {
    return static_cast<size_t>(out) * static_cast<size_t>(in) / 2;
}
constexpr size_t e8m0_scale_bytes(int64_t out, int64_t in) {
    return static_cast<size_t>(out) * static_cast<size_t>(in / 32);
}
// gate/up: [inter, hidden]; down: [hidden, inter].  Per-projection byte sizes
// are symmetric here, but they are computed per projection, not assumed equal.
constexpr size_t kV41WeightBytes[3] = {
    packed_i8_bytes(kV41Inter, kV41Hidden),   // gate 2304x5120/2
    packed_i8_bytes(kV41Inter, kV41Hidden),   // up   2304x5120/2
    packed_i8_bytes(kV41Hidden, kV41Inter),   // down 5120x2304/2
};
constexpr size_t kV41ScaleBytes[3] = {
    e8m0_scale_bytes(kV41Inter, kV41Hidden),  // gate 2304x160
    e8m0_scale_bytes(kV41Inter, kV41Hidden),  // up   2304x160
    e8m0_scale_bytes(kV41Hidden, kV41Inter),  // down 5120x72
};
constexpr size_t kV41RecordBytes =
    kV41WeightBytes[0] + kV41WeightBytes[1] + kV41WeightBytes[2] +
    kV41ScaleBytes[0] + kV41ScaleBytes[1] + kV41ScaleBytes[2];
static_assert(kV41RecordBytes == 18800640, "V4.1 record = 17.93 MiB");

// Deterministic full-record content pattern: depends on identity AND offset so
// byte-exact verification catches swapped/gather-misordered regions.
uint8_t v41_byte(const TierExpertKey& k, size_t i) {
    const uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(k.layer)) * 1315423911u ^
                       static_cast<uint64_t>(static_cast<uint32_t>(k.expert)) * 2654435761u ^
                       (i * 0x9E3779B1u) ^ (i >> 7);
    return static_cast<uint8_t>((x >> 13) & 0xFF);
}
bool v41_exact(const uint8_t* data, const TierExpertKey& k, size_t bytes = kV41RecordBytes) {
    if (!data) return false;
    for (size_t i = 0; i < bytes; ++i) {
        if (data[i] != v41_byte(k, i)) return false;
    }
    return true;
}

// Representation string, built exactly the way Engine::init builds it:
// a fixed representation-family tag plus per-projection geometry/offset
// parameters derived from the resolved layout (engine.cpp:3684-3693).
std::string v41_representation() {
    size_t packed_total = 0, scale_total = 0;
    for (int p = 0; p < 3; ++p) packed_total += kV41WeightBytes[p];
    std::string rep = "fp4-e2m1-e8m0-gate-up-down-v1";
    const int64_t outs[3] = {kV41Inter, kV41Inter, kV41Hidden};
    const int64_t ins[3] = {kV41Hidden, kV41Hidden, kV41Inter};
    for (int p = 0; p < 3; ++p) {
        rep += ":" + std::to_string(outs[p]) + "x" + std::to_string(ins[p]) +
               ":" + std::to_string(packed_total + scale_total);
        scale_total += kV41ScaleBytes[p];
    }
    return rep;
}
const std::string kV41Model =
    "deepseek-ai/DeepSeek-V4.1@0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

// ---- Synthetic stores -------------------------------------------------------
// Raw cold store: synthesizes the exact record byte stream for any in-range
// (layer, expert); range checks live HERE (the store), not in the tiers.
struct V41ColdStore final : ColdExpertStore {
    size_t reads = 0;
    ColdReadResult read(const TierExpertKey& key, uint8_t* dst, size_t n) override {
        if (!dst || !key.valid() || n != kV41RecordBytes ||
            key.layer < 0 || key.layer >= kV41LayerSpan ||
            key.expert < 0 || key.expert >= kV41Experts) return {};
        ++reads;
        for (size_t i = 0; i < n; ++i) dst[i] = v41_byte(key, i);
        return {true, n};
    }
};

// ExpertStore view source at V4.1 layout.  `contiguous` selects between the
// two adapter paths: one contiguous record buffer (materialize()) vs. six
// separately-allocated tensor regions (adapter gather into the final slot).
// Backings are refilled per get(); the adapter's read is caller-serialized.
struct V41ExpertStore final : ExpertStore {
    bool contiguous = false;
    std::vector<uint8_t> blob;
    std::array<std::vector<uint8_t>, 6> region;
    std::string identity = "v41-synthetic-store-integrity-0123456789abcdef0123456789abcdef";

    explicit V41ExpertStore(bool contiguous_mode) : contiguous(contiguous_mode) {
        if (contiguous) blob.resize(kV41RecordBytes);
        else for (int r = 0; r < 6; ++r)
            region[r].resize(r < 3 ? kV41WeightBytes[r] : kV41ScaleBytes[r - 3]);
    }
    bool get(int layer, int expert, ExpertView* out) override {
        const bool in_range = out && layer >= 0 && layer < kV41LayerSpan &&
                              expert >= 0 && expert < kV41Experts;
        record_lookup(in_range);
        if (!in_range) return false;
        TierExpertKey key{kV41Model, layer, expert, {}};
        if (contiguous) {
            for (size_t i = 0; i < kV41RecordBytes; ++i) blob[i] = v41_byte(key, i);
        } else {
            size_t offset = 0;
            for (int r = 0; r < 6; ++r) {
                for (size_t i = 0; i < region[r].size(); ++i)
                    region[r][i] = v41_byte(key, offset + i);
                offset += region[r].size();
            }
        }
        ExpertView view;
        view.codec = ExpertCodec::DeepSeekFp4E2m1E8m0; // same FP4 format family
        view.integrity_identity = identity.c_str();
        const int64_t outs[3] = {kV41Inter, kV41Inter, kV41Hidden};
        const int64_t stored_ins[3] = {kV41Hidden / 2, kV41Hidden / 2, kV41Inter / 2};
        const int64_t scale_ins[3] = {kV41Hidden / 32, kV41Hidden / 32, kV41Inter / 32};
        size_t offset = 0;
        for (int p = 0; p < 3; ++p) {
            auto& w = view.weights[p];
            w.data = contiguous ? blob.data() + offset : region[p].data();
            w.nbytes = kV41WeightBytes[p];
            w.dtype = DType::I8;
            w.shape = {outs[p], stored_ins[p]};
            offset += kV41WeightBytes[p];
        }
        for (int p = 0; p < 3; ++p) {
            auto& s = view.scales[p];
            s.data = contiguous ? blob.data() + offset : region[3 + p].data();
            s.nbytes = kV41ScaleBytes[p];
            s.dtype = DType::F8;
            s.shape = {outs[p], scale_ins[p]};
            offset += kV41ScaleBytes[p];
        }
        if (contiguous) {
            view.contiguous_data = blob.data();
            view.contiguous_nbytes = kV41RecordBytes;
        }
        view.record_index =
            static_cast<uint64_t>(static_cast<uint32_t>(layer) * kV41Experts + expert);
        if (!view.ok()) return false;
        *out = std::move(view);
        return true;
    }
    const char* backend_name() const override { return "v41-synthetic"; }
    const std::string& integrity_identity() const override { return identity; }
};

// ---- Shared fixtures ---------------------------------------------------------
struct MemoryCounts { size_t allocs = 0, frees = 0, pins = 0, unpins = 0; };
HostMemoryBackend counted_backend(MemoryCounts& count, bool pin_success = true) {
    auto native = host_memory_backend(false);
    return {
        [&, native](size_t n, size_t a) { ++count.allocs; return native.allocate(n, a); },
        [&, native](void* p) { ++count.frees; native.free(p); },
        [&, pin_success](void*, size_t) { ++count.pins; return pin_success; },
        [&](void*) { ++count.unpins; }
    };
}
HostTierConfig v41_host_config(size_t dynamic_slots) {
    HostTierConfig cfg;
    cfg.slot_bytes = kV41RecordBytes;
    cfg.alignment = 4096;
    cfg.policy_slots = 0;
    cfg.dynamic_slots = dynamic_slots;
    cfg.budget_bytes = dynamic_slots * kV41RecordBytes; // stride == slot_bytes (4K-aligned)
    cfg.try_pin = true;
    return cfg;
}
Arena::Backend device_mock() {
    return {[](size_t bytes) { return std::malloc(bytes); },
            [](void* p) { std::free(p); }, "host"};
}

// ---- Tests -------------------------------------------------------------------

void key_and_scope_conventions() {
    const std::string rep = v41_representation();
    check(TierExpertKey{kV41Model, 0, 0, rep}.valid(), "V4.1 key (layer 0, expert 0) valid");
    check(TierExpertKey{kV41Model, 39, 383, rep}.valid(), "V4.1 key (last main layer, last expert) valid");
    check(TierExpertKey{kV41Model, 40, 5, rep}.valid() &&
          TierExpertKey{kV41Model, 41, 5, rep}.valid() &&
          TierExpertKey{kV41Model, 42, 5, rep}.valid(),
          "draft/mtp bucket layers (40..42 = L+N convention) are ordinary valid keys");
    check(!TierExpertKey{kV41Model, -1, 0, rep}.valid() &&
          !TierExpertKey{kV41Model, 0, -1, rep}.valid(),
          "negative layer/expert rejected by key validity");
    check(!TierExpertKey{"", 0, 0, rep}.valid() &&
          !TierExpertKey{kV41Model, 0, 0, ""}.valid(),
          "empty model/representation rejected");
    check(!TierExpertKey{std::string(TierExpertKey::max_identity_bytes + 1, 'x'),
                         0, 0, rep}.valid(),
          "identity strings bounded at 1024 bytes");
    // Range admission is the store's job, not the key's: out-of-universe keys
    // are still structurally valid, and the cold store fails closed on them.
    V41ColdStore store;
    std::vector<uint8_t> buf(kV41RecordBytes);
    check(!store.read(TierExpertKey{kV41Model, 43, 0, rep}, buf.data(), buf.size()).success &&
          !store.read(TierExpertKey{kV41Model, 0, 384, rep}, buf.data(), buf.size()).success,
          "cold store owns range admission (43rd+1 bucket / 384th+1 expert fail closed)");
}

void adapter_scope_and_gather() {
    V41ExpertStore store(false); // six non-contiguous regions
    ExpertView layout;
    check(store.get(0, 0, &layout), "synthetic V4.1 layout resolves");
    ExpertStoreColdAdapter adapter(store, kV41Model, v41_representation(), layout);

    const auto rec = adapter.record(39, 383);
    check(rec.key.valid() && rec.key.layer == 39 && rec.key.expert == 383,
          "adapter record carries requested (layer, expert)");
    check(rec.key.model == std::string(kV41Model) + "\nstore:" + store.integrity_identity(),
          "scope model binds caller identity to store integrity identity");
    check(rec.key.representation == v41_representation(),
          "scope representation is the parametric layout string");
    check(rec.stored_bytes == kV41RecordBytes && rec.exact_bytes == kV41RecordBytes &&
          rec.codec == "identity-v1",
          "record size flows from ExpertView layout (18,800,640 B), codec identity-v1");

    std::vector<uint8_t> dst(kV41RecordBytes, 0xCC);
    const auto res = adapter.read(rec.key, dst.data(), dst.size());
    check(res.success && res.bytes_read == kV41RecordBytes,
          "adapter read materializes the 17.93 MiB record");
    check(v41_exact(dst.data(), rec.key),
          "six-region gather order is exact: gate|up|down weights then scales");
    const auto stats = store.stats();
    check(stats.source_regions == 6 && stats.contiguous_source_reads == 0,
          "non-contiguous source consumed as six gathered regions");

    TierExpertKey wrong_model = rec.key; wrong_model.model = "other-model";
    TierExpertKey wrong_rep = rec.key; wrong_rep.representation = "other-rep";
    check(!adapter.read(wrong_model, dst.data(), dst.size()).success &&
          !adapter.read(wrong_rep, dst.data(), dst.size()).success &&
          !adapter.read(rec.key, dst.data(), kV41RecordBytes - 1).success,
          "adapter fails closed on foreign model, foreign representation, short buffer");
    check(!adapter.read(adapter.record(0, 384).key, dst.data(), dst.size()).success,
          "adapter propagates store range rejection");
}

void adapter_contiguous_path() {
    V41ExpertStore store(true); // one contiguous record buffer
    ExpertView layout;
    check(store.get(5, 200, &layout), "contiguous layout reference resolves");
    ExpertStoreColdAdapter adapter(store, kV41Model, v41_representation(), layout);
    const auto rec = adapter.record(5, 200);
    std::vector<uint8_t> dst(kV41RecordBytes);
    const auto res = adapter.read(rec.key, dst.data(), dst.size());
    check(res.success && res.bytes_read == kV41RecordBytes,
          "contiguous materialize delivers the full record");
    check(v41_exact(dst.data(), rec.key),
          "contiguous record content matches the canonical byte stream");
    const auto stats = store.stats();
    check(stats.contiguous_source_reads == 1 && stats.source_regions == 1,
          "contiguous source counted as a single region");
}

void host_tier_at_v41() {
    V41ColdStore store;
    IdentityCodec codec;
    MemoryCounts count;
    constexpr size_t kSlots = 4;
    {
        HostExpertTier host(v41_host_config(kSlots), counted_backend(count),
                            std::make_shared<PlainLruHostPlacementPolicy>());
        auto record = [&](int layer, int expert) {
            return StorageRecord{{kV41Model, layer, expert, v41_representation()},
                                 kV41RecordBytes, kV41RecordBytes, "identity-v1"};
        };
        const size_t stride = kV41RecordBytes; // 4K-aligned exactly
        check(host.stats().allocated_bytes == kSlots * stride &&
              host.stats().pinned_bytes == kSlots * stride,
              "four 17.93 MiB slots allocated and pinned once at construction");

        auto a = host.acquire(record(0, 0), store, codec);
        check(a.status == HostAcquireStatus::Ready && a.lease.pinned(),
              "V4.1 record acquires a pinned host lease");
        check(v41_exact(a.lease.data(), a.lease.key()),
              "host slot holds byte-exact 17.93 MiB V4.1 record");
        const auto reads1 = store.reads;
        auto again = host.acquire(record(0, 0), store, codec);
        check(again.lease && store.reads == reads1, "host hit re-reads nothing");
        check(host.stats().SSD_bytes == reads1 * kV41RecordBytes,
              "SSD byte accounting scales with the V4.1 record size");

        // Fill remaining slots with corner/interior identities, then force LRU.
        auto b = host.acquire(record(39, 383), store, codec);
        auto c = host.acquire(record(20, 191), store, codec);
        auto d = host.acquire(record(42, 7), store, codec); // draft-bucket layer
        check(b.lease && c.lease && d.lease, "corner/interior/mtp-bucket records all resident");
        a.lease.reset(); again.lease.reset();
        auto e = host.acquire(record(7, 300), store, codec);
        check(e.lease && host.stats().evictions == 1,
              "fifth record triggers exactly one LRU eviction at 17.93 MiB stride");
        // Free every lease so refills have evictable candidates (a held lease
        // pins its slot; this is the capacity contract, not geometry).
        b.lease.reset(); c.lease.reset(); d.lease.reset(); e.lease.reset();
        const auto before = store.reads;
        auto refetch = host.acquire(record(0, 0), store, codec);
        check(refetch.lease && store.reads == before + 1,
              "evicted record refills byte-exact from cold store");
        check(v41_exact(refetch.lease.data(), refetch.lease.key()),
              "refilled slot content is exact");

        // Fail-closed surface at this geometry.
        auto oversize = record(1, 1); oversize.exact_bytes = kV41RecordBytes + 1;
        check(host.acquire(oversize, store, codec).status == HostAcquireStatus::Invalid,
              "record larger than slot rejected");
        auto mismatched = record(1, 1); mismatched.stored_bytes = kV41RecordBytes - 8;
        check(!host.acquire(mismatched, store, codec).lease,
              "stored!=exact rejected by IdentityCodec");
        auto wrong_codec = record(1, 1); wrong_codec.codec = "other-codec";
        check(!host.acquire(wrong_codec, store, codec).lease,
              "foreign codec descriptor rejected");
        refetch.lease.reset();
        auto foreign = record(2, 2); foreign.key.model = "other-model";
        check(host.acquire(foreign, store, codec).lease &&
              host.acquire(foreign, store, codec).lease,
              "foreign model identity occupies its own slot, never aliases");
    }
    check(count.frees == kSlots && count.unpins == kSlots,
          "tier teardown frees/unpins every slot exactly once");
}

void device_tier_at_v41() {
    V41ExpertStore store(true);
    ExpertView layout;
    check(store.get(0, 0, &layout), "device-tier layout resolves");
    ExpertStoreColdAdapter adapter(store, kV41Model, v41_representation(), layout);
    IdentityCodec codec;
    MemoryCounts count;
    HostExpertTier host(v41_host_config(3), counted_backend(count),
                        std::make_shared<PlainLruHostPlacementPolicy>());

    VramCacheManager cache;
    check(cache.init(2 * kV41RecordBytes, device_mock()),
          "device arena sized in V4.1 records (2 x 17.93 MiB)");
    AsyncPrefetcher prefetcher(cache);
    check(prefetcher.init(false), "mock transfer stream initialized");
    {
        DeviceExpertTier device(cache, prefetcher, adapter.record(0, 0).key);
        auto stage = [&](int layer, int expert, int priority = 0) {
            return device.stage(adapter.record(layer, expert), adapter, host, codec,
                                priority, /*token*/0, /*logical_layer*/layer);
        };
        check(stage(39, 383), "cold V4.1 record stages through both tiers");
        check(device.wait(adapter.record(39, 383).key) &&
              v41_exact(static_cast<const uint8_t*>(cache.data(39, 383)),
                        adapter.record(39, 383).key),
              "wait() lands byte-exact V4.1 record in the device arena");
        // Resident-hit path pins the block until a wait consumes it; stage a
        // hit and consume it so later stages have an evictable victim.
        check(stage(39, 383) && device.wait(adapter.record(39, 383).key),
              "resident hit re-stages without a cold read");

        check(stage(38, 382) && device.wait(adapter.record(38, 382).key),
              "second record fills the two-record arena");
        check(stage(41, 7) && device.wait(adapter.record(41, 7).key),
              "draft-bucket layer (41 = L+N) stages through the same seam");
        check(!cache.is_resident(39, 383) &&
              v41_exact(static_cast<const uint8_t*>(cache.data(41, 7)),
                        adapter.record(41, 7).key),
              "device LRU evicted oldest V4.1 record, newest is exact");

        auto foreign = adapter.record(41, 7);
        foreign.key.model = "other-model";
        check(!device.stage(foreign, adapter, host, codec, 0),
              "device tier rejects a foreign model identity");
        foreign = adapter.record(41, 7);
        foreign.key.representation = "other-representation";
        check(!device.stage(foreign, adapter, host, codec, 0),
              "device tier rejects a foreign representation");
        auto huge = adapter.record(1, 1);
        huge.exact_bytes = huge.stored_bytes = 2 * kV41RecordBytes + 1;
        check(!device.stage(huge, adapter, host, codec, 0),
              "record larger than device budget rejected before any fill");

        const auto metrics = device.metrics(host, 2);
        check(metrics.device_miss == 3 && metrics.device_hit == 1 &&
              metrics.H2D_bytes == 3 * kV41RecordBytes &&
              metrics.bytes_per_token_valid &&
              metrics.H2D_bytes_per_token == 1.5 * kV41RecordBytes,
              "tier metrics scale with V4.1 record bytes");
        check(metrics.device_bytes <= metrics.device_budget &&
              metrics.device_peak_bytes == 2 * kV41RecordBytes,
              "device budget/peak accounting in V4.1 record units");
        check(host.stats().leased_slots == 0,
              "all host DMA leases released after consumption");
        std::string error;
        check(cache.validate_invariants(&error) && prefetcher.validate_invariants(&error),
              "cache and prefetcher invariants hold at V4.1 geometry");
    }
}

void full_universe_sweep() {
    // Every (layer, expert) in the V4.1 routed universe (40x384 main +
    // 3 draft buckets) forms a valid, distinct tier identity — without
    // touching storage for each pair.
    V41ColdStore store;
    IdentityCodec codec;
    MemoryCounts count;
    HostExpertTier host(v41_host_config(2), counted_backend(count),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    size_t formed = 0;
    for (int layer = 0; layer < kV41LayerSpan; ++layer) {
        for (int expert = 0; expert < kV41Experts; ++expert) {
            const StorageRecord rec{
                {kV41Model, layer, expert, v41_representation()},
                kV41RecordBytes, kV41RecordBytes, "identity-v1"};
            if (!rec.key.valid() || !codec.accepts(rec)) { ++failures; return; }
            ++formed;
        }
    }
    check(formed == static_cast<size_t>(kV41LayerSpan) * kV41Experts,
          "all 16,512 V4.1 universe identities are valid tier records");

    // Strided materialization sample: distinct identities, forced evictions,
    // every byte exact.
    const int layers[] = {0, 7, 13, 25, 39, 40, 41, 42};
    const int experts[] = {0, 1, 127, 191, 255, 383};
    size_t materialized = 0;
    for (int layer : layers) {
        for (int expert : experts) {
            const StorageRecord rec{
                {kV41Model, layer, expert, v41_representation()},
                kV41RecordBytes, kV41RecordBytes, "identity-v1"};
            auto got = host.acquire(rec, store, codec);
            if (!(got.lease && v41_exact(got.lease.data(), got.lease.key()))) {
                check(false, "strided universe sample materializes byte-exact");
                return;
            }
            ++materialized;
        }
    }
    check(materialized == 48 && host.stats().evictions == materialized - 2,
          "two-slot host tier serves strided V4.1 samples with exact refills");
    check(host.stats().SSD_bytes == store.reads * kV41RecordBytes &&
          store.reads == materialized,
          "every sample was one exact cold read at V4.1 record size");
}

void geometry_is_not_dsv4() {
    // The audit property itself: nothing in this test's wiring equals the
    // DeepSeek-V4 canonical record or layer/expert counts, and the tiers
    // never noticed.
    static_assert(kV41RecordBytes != 13369344, "record size differs from DSv4");
    static_assert(kV41Layers != 43 && kV41Experts != 256,
                  "layer/expert counts differ from DSv4");
    check(v41_representation().find("2304x5120") != std::string::npos &&
          v41_representation().find("5120x2304") != std::string::npos,
          "representation string encodes V4.1 projection dims, not DSv4's");
}

} // namespace

int main() {
    std::cout << "V4.1 record bytes: " << kV41RecordBytes
              << " (" << (kV41RecordBytes / 1048576.0) << " MiB)\n"
              << "representation: " << v41_representation() << '\n';
    key_and_scope_conventions();
    adapter_scope_and_gather();
    adapter_contiguous_path();
    host_tier_at_v41();
    device_tier_at_v41();
    full_universe_sweep();
    geometry_is_not_dsv4();
    std::cout << "V4.1 geometry failures: " << failures << '\n';
    return failures ? 1 : 0;
}
