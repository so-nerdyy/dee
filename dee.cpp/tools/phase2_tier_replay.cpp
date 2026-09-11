// tools/phase2_tier_replay.cpp — Phase-2 tier replay validation (host-only).
//
// Drives the REAL Phase-2 mechanism classes — HostExpertTier,
// VramCacheManager, AsyncPrefetcher, DeviceExpertTier — with the sealed v50
// route journal's engine-dedup access stream, and dumps realized counters
// for comparison against tools/phase2_ws_policy_sim_v4.py predictions.
//
// The journal record shape is:
//   {"record_index":N, "forward_step":S, "layer":L,
//    "expert_ids_rank_order":[[e..],[e..],..], ...}
// Per (record), the engine-visible stream is the deduplicated expert set of
// all token rows (see sim load_batches: sorted unique ids per
// (forward_step, layer)).
//
// Arms:
//   host     HostExpertTier + PlainLruHostPlacementPolicy over the stream
//            (the sim "lru" host arm; scope may be full/cuda0/cuda1).
//   bare     VramCacheManager::ensure directly (no prefetcher, no pins) —
//            the purest device-policy analog of the sim touch().
//   vram     AsyncPrefetcher::prefetch + wait per batch — the legacy
//            device-staging path the VRAM-only arm uses, including
//            cache pins held across an in-batch staging window.
//   combined DeviceExpertTier::stage + wait per batch over a real
//            HostExpertTier — the host-enabled / combined arms: the host
//            tier then sees only the device-miss substream, and host
//            slots stay leased for the batch staging window.
//   sweep    Runs the validation matrix and emits one JSON document.
//
// Byte sizes are LOGICAL by default (--record-bytes 4096): hit/miss/
// eviction ordering in both tiers is size-independent (uniform records),
// so the counters are identical at any record size; scaled byte figures
// are computed in the report. --record-bytes 13369344 replays with the
// physical DEE4 record size for spot-checks (RAM-bound).
//
// Build (no CUDA needed):
//   cmake -S dee.cpp -B build -G "MinGW Makefiles" -DDEE_CUDA=OFF \
//         -DDEE_BUILD_TIER_REPLAY=ON
//   cmake --build build --target phase2_tier_replay
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "dee/async_prefetcher.h"
#include "dee/expert_tiers.h"
#include "dee/host_expert_tier.h"
#include "dee/json_min.h"
#include "dee/vram_cache.h"

namespace {

using namespace dee;

constexpr size_t kDee4RecordBytes = 13369344;  // physical packed FP4 record
constexpr int    kSplitGpu0 = 22;              // cuda0: layers 0-21

// ------------------------------------------------------------ journal -----

struct Batch {
    int step = 0;
    int layer = 0;
    std::vector<int> sorted;      // ascending-id dedup order (sim model)
    std::vector<int> first_seen;  // row-major first-appearance order
};

bool load_journal(const std::string& path, std::vector<Batch>* batches) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "tier_replay: cannot open journal %s\n", path.c_str());
        return false;
    }
    struct Raw { long long index; Batch batch; };
    std::vector<Raw> raw;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        bool ok = false;
        json::ValuePtr v = json::parse(line, &ok);
        if (!ok || !v || !v->is_object()) {
            std::fprintf(stderr, "tier_replay: malformed journal line\n");
            return false;
        }
        const json::Value* idx = v->find("record_index");
        const json::Value* step = v->find("forward_step");
        const json::Value* layer = v->find("layer");
        const json::Value* rows = v->find("expert_ids_rank_order");
        if (!idx || !step || !layer || !rows || !rows->is_array()) {
            std::fprintf(stderr, "tier_replay: journal record missing fields\n");
            return false;
        }
        Raw r;
        r.index = idx->i;
        r.batch.step = static_cast<int>(step->i);
        r.batch.layer = static_cast<int>(layer->i);
        std::set<int> seen;
        for (const auto& row : rows->arr) {
            if (!row->is_array()) return false;
            for (const auto& e : row->arr) {
                const int expert = static_cast<int>(e->i);
                if (seen.insert(expert).second)
                    r.batch.first_seen.push_back(expert);
            }
        }
        r.batch.sorted.assign(seen.begin(), seen.end());
        raw.push_back(std::move(r));
    }
    std::sort(raw.begin(), raw.end(),
              [](const Raw& a, const Raw& b) { return a.index < b.index; });
    for (auto& r : raw) batches->push_back(std::move(r.batch));
    return !batches->empty();
}

std::vector<Batch> scope_stream(const std::vector<Batch>& batches,
                                const std::string& scope) {
    if (scope == "full") return batches;
    const bool lo = scope == "cuda0";
    std::vector<Batch> out;
    for (const auto& b : batches)
        if ((b.layer < kSplitGpu0) == lo) out.push_back(b);
    return out;
}

// ------------------------------------------------------------ fakes -------

// Deterministic zero-cost cold store: correctly-sized records whose bytes
// carry a trivial identity pattern. Exercises lookup/fill/lease accounting,
// not contents.
struct ZeroStore final : ColdExpertStore {
    size_t record_bytes = 0;
    uint64_t reads = 0;
    ColdReadResult read(const TierExpertKey& key, uint8_t* dst, size_t n) override {
        if (!dst || n != record_bytes || !key.valid()) return {};
        ++reads;
        std::memset(dst, static_cast<int>((key.layer * 31 + key.expert * 17) & 0xff), n);
        return {true, n};
    }
};

StorageRecord make_record(int layer, int expert, size_t bytes) {
    StorageRecord r;
    r.key.model = "phase2-replay/v50-journal";
    r.key.layer = layer;
    r.key.expert = expert;
    r.key.representation = "dee4-packed-fp4-v1";
    r.stored_bytes = bytes;
    r.exact_bytes = bytes;
    r.codec = "identity-v1";
    return r;
}

HostMemoryBackend replay_host_backend() {
    HostMemoryBackend be = host_memory_backend(false);
    return be;
}

Arena::Backend host_arena() {
    return {[](size_t n) { return std::malloc(n); },
            [](void* p) { std::free(p); }, "host"};
}

HostTierConfig host_config(size_t slots, size_t slot_bytes) {
    HostTierConfig cfg;
    cfg.slot_bytes = slot_bytes;
    cfg.alignment = 4096;
    cfg.policy_slots = 0;
    cfg.dynamic_slots = slots;
    const size_t stride = (slot_bytes + cfg.alignment - 1) & ~(cfg.alignment - 1);
    cfg.budget_bytes = slots * stride;
    cfg.try_pin = false;  // no cudaHostRegister locally; pinning is a perf detail
    return cfg;
}

// ------------------------------------------------------------ arms --------

struct ArmResult {
    std::string arm, scope, score, priority_model, lease_mode;
    size_t host_slots = 0, vram_slots = 0, record_bytes = 0;
    uint64_t accesses = 0;
    // host counters
    uint64_t host_hit = 0, host_miss = 0, host_evictions = 0, host_fills = 0;
    uint64_t host_coalesced = 0, host_budget_rejections = 0, host_failures = 0;
    uint64_t host_ssd_bytes = 0, host_peak_resident_bytes = 0;
    // device counters
    uint64_t device_hit = 0, device_miss = 0, device_evictions = 0;
    uint64_t device_failures = 0, device_pinned_skipped = 0;
    uint64_t h2d_bytes = 0;
    size_t device_resident = 0;
    bool ok = true;
    std::string error;
};

int priority_for(size_t j, size_t K, const std::string& model) {
    (void)model;
    return static_cast<int>(K - j);  // sim model: prio = K - j, j = stream idx
}

ArmResult arm_host(const std::vector<Batch>& stream, size_t slots,
                   size_t record_bytes, const std::string& lease_mode,
                   const std::string& scope) {
    ArmResult r;
    r.arm = "host"; r.scope = scope; r.lease_mode = lease_mode;
    r.host_slots = slots; r.record_bytes = record_bytes;
    ZeroStore store; store.record_bytes = record_bytes;
    IdentityCodec codec;
    HostExpertTier host(host_config(slots, record_bytes), replay_host_backend(),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    std::vector<HostExpertLease> held;
    for (const auto& batch : stream) {
        const size_t K = batch.sorted.size();
        for (size_t j = 0; j < K; ++j) {
            const auto rec = make_record(batch.layer, batch.sorted[j], record_bytes);
            HostAcquireResult a = host.acquire(rec, store, codec);
            // Capacity with held leases: model DMA completion by releasing the
            // batch's outstanding leases, then retry (the DeviceExpertTier
            // collect_host_sources(true) equivalent).
            while (a.status == HostAcquireStatus::Capacity && !held.empty()) {
                held.clear();
                a = host.acquire(rec, store, codec);
            }
            if (a.status != HostAcquireStatus::Ready) {
                r.ok = false;
                r.error = "host.acquire failed status=" +
                          std::to_string(static_cast<int>(a.status));
                return r;
            }
            ++r.accesses;
            if (lease_mode == "batch") held.push_back(a.lease);
            // touch mode: lease destructs immediately (pure sim-LRU analog)
        }
        held.clear();
    }
    const HostTierStats s = host.stats();
    r.host_hit = s.host_hit; r.host_miss = s.host_miss;
    r.host_evictions = s.evictions; r.host_fills = s.fills;
    r.host_coalesced = s.coalesced;
    r.host_budget_rejections = s.budget_rejections;
    r.host_failures = s.failures;
    r.host_ssd_bytes = s.SSD_bytes;
    r.host_peak_resident_bytes = s.peak_resident_bytes;
    if (s.leased_slots != 0) { r.ok = false; r.error = "leased slots at end"; }
    return r;
}

ArmResult arm_bare(const std::vector<Batch>& stream, size_t slots,
                   size_t record_bytes, const std::string& score,
                   const std::string& prio_model, const std::string& scope) {
    ArmResult r;
    r.arm = "bare"; r.scope = scope; r.score = score;
    r.priority_model = prio_model;
    r.vram_slots = slots; r.record_bytes = record_bytes;
    VramCacheManager cache;
    if (!cache.init(slots * record_bytes, host_arena())) {
        r.ok = false; r.error = "arena init failed"; return r;
    }
    if (score == "lru") cache.set_experimental_plain_lru(true);
    for (const auto& batch : stream) {
        const auto& order = prio_model == "first_seen" ? batch.first_seen
                                                     : batch.sorted;
        const size_t K = order.size();
        for (size_t j = 0; j < K; ++j) {
            if (!cache.ensure(batch.layer, order[j], record_bytes,
                              priority_for(j, K, prio_model))) {
                r.ok = false; r.error = "ensure failed"; return r;
            }
            ++r.accesses;
        }
    }
    const auto& s = cache.stats();
    r.device_hit = s.hits; r.device_miss = s.loads;
    r.device_evictions = s.evictions;
    r.device_pinned_skipped = s.pinned_blocks_skipped;
    r.device_resident = cache.resident_count();
    return r;
}

ArmResult arm_vram(const std::vector<Batch>& stream, size_t slots,
                   size_t record_bytes, const std::string& score,
                   const std::string& prio_model, const std::string& scope) {
    ArmResult r;
    r.arm = "vram"; r.scope = scope; r.score = score;
    r.priority_model = prio_model;
    r.vram_slots = slots; r.record_bytes = record_bytes;
    VramCacheManager cache;
    if (!cache.init(slots * record_bytes, host_arena())) {
        r.ok = false; r.error = "arena init failed"; return r;
    }
    if (score == "lru") cache.set_experimental_plain_lru(true);
    AsyncPrefetcher prefetcher(cache);
    if (!prefetcher.init(false)) { r.ok = false; r.error = "prefetcher init"; return r; }
    // Windows/LLP64: the legacy in-flight map key truncates (layer,expert) to
    // the low 32 bits (audit-documented, deliberately retained for default-OFF
    // equivalence). The Phase-2 scope flag restores full keys so multi-layer
    // replay exercises the real bookkeeping instead of aliasing layers.
    // On the LP64 T4 target the legacy key is already full-width.
    if (!prefetcher.enable_experimental_host_tier(
            make_record(0, 0, record_bytes).key)) {
        r.ok = false; r.error = "experimental scope enable failed"; return r;
    }
    std::vector<uint8_t> source(record_bytes, 0xab);  // stable pageable source
    for (const auto& batch : stream) {
        const auto& order = prio_model == "first_seen" ? batch.first_seen
                                                     : batch.sorted;
        const size_t K = order.size();
        // Engine chunks staging at cache_batch = budget/record so in-flight
        // (pinned) blocks can never exceed capacity; mirror that here.
        for (size_t first = 0; first < K; first += slots) {
            const size_t last = std::min(K, first + slots);
            prefetcher.begin_batch();
            for (size_t j = first; j < last; ++j) {
                const long id = prefetcher.prefetch(batch.layer, order[j],
                    source.data(), record_bytes, priority_for(j, K, prio_model),
                    batch.step, batch.layer);
                if (id < 0) { r.ok = false; r.error = "prefetch failed"; return r; }
                ++r.accesses;
            }
            // The engine waits on each staged expert before its compute; on
            // the CPU mock this drains the copy and drops the DMA pin/lease.
            for (size_t j = first; j < last; ++j) {
                if (!prefetcher.wait(batch.layer, order[j])) {
                    r.ok = false; r.error = "wait failed"; return r;
                }
            }
        }
    }
    const auto& s = cache.stats();
    r.device_hit = s.hits; r.device_miss = s.loads;
    r.device_evictions = s.evictions;
    r.device_pinned_skipped = s.pinned_blocks_skipped;
    r.device_resident = cache.resident_count();
    r.h2d_bytes = r.device_miss * record_bytes;  // CPU mock copies on drain
    return r;
}

ArmResult arm_combined(const std::vector<Batch>& stream, size_t host_slots,
                       size_t vram_slots, size_t record_bytes,
                       const std::string& score, const std::string& prio_model,
                       const std::string& scope) {
    ArmResult r;
    r.arm = "combined"; r.scope = scope; r.score = score;
    r.priority_model = prio_model;
    r.host_slots = host_slots; r.vram_slots = vram_slots;
    r.record_bytes = record_bytes;
    ZeroStore store; store.record_bytes = record_bytes;
    IdentityCodec codec;
    HostExpertTier host(host_config(host_slots, record_bytes),
                        replay_host_backend(),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    VramCacheManager cache;
    if (!cache.init(vram_slots * record_bytes, host_arena())) {
        r.ok = false; r.error = "arena init failed"; return r;
    }
    if (score == "lru") cache.set_experimental_plain_lru(true);
    AsyncPrefetcher prefetcher(cache);
    if (!prefetcher.init(false)) { r.ok = false; r.error = "prefetcher init"; return r; }
    DeviceExpertTier device(cache, prefetcher, make_record(0, 0, record_bytes).key);
    for (const auto& batch : stream) {
        const auto& order = prio_model == "first_seen" ? batch.first_seen
                                                     : batch.sorted;
        const size_t K = order.size();
        // Same cache_batch chunking as the engine staging loop.
        for (size_t first = 0; first < K; first += vram_slots) {
            const size_t last = std::min(K, first + vram_slots);
            prefetcher.begin_batch();
            for (size_t j = first; j < last; ++j) {
                const auto rec = make_record(batch.layer, order[j], record_bytes);
                if (!device.stage(rec, store, host, codec,
                                  priority_for(j, K, prio_model),
                                  batch.step, batch.layer)) {
                    r.ok = false;
                    r.error = "stage failed layer=" + std::to_string(batch.layer) +
                              " expert=" + std::to_string(order[j]);
                    return r;
                }
                ++r.accesses;
            }
            for (size_t j = first; j < last; ++j) {
                if (!device.wait(make_record(batch.layer, order[j], record_bytes).key)) {
                    r.ok = false; r.error = "device wait failed"; return r;
                }
            }
        }
    }
    const TierMetrics m = device.metrics(host);
    const HostTierStats hs = m.host;
    r.host_hit = hs.host_hit; r.host_miss = hs.host_miss;
    r.host_evictions = hs.evictions; r.host_fills = hs.fills;
    r.host_coalesced = hs.coalesced;
    r.host_budget_rejections = hs.budget_rejections;
    r.host_failures = hs.failures;
    r.host_ssd_bytes = hs.SSD_bytes;
    r.host_peak_resident_bytes = hs.peak_resident_bytes;
    r.device_hit = m.device_hit; r.device_miss = m.device_miss;
    r.device_evictions = m.device_evictions;
    r.device_failures = m.device_failures;
    r.device_pinned_skipped = cache.stats().pinned_blocks_skipped;
    r.h2d_bytes = m.H2D_bytes;
    r.device_resident = cache.resident_count();
    if (hs.leased_slots != 0) { r.ok = false; r.error = "leased slots at end"; }
    return r;
}

// ------------------------------------------------------------ output ------

void emit(std::ostream& os, const ArmResult& r, bool comma) {
    os << "  {\"arm\":\"" << r.arm << "\",\"scope\":\"" << r.scope
       << "\",\"score\":\"" << r.score << "\",\"priority_model\":\""
       << r.priority_model << "\",\"lease_mode\":\"" << r.lease_mode
       << "\",\"host_slots\":" << r.host_slots
       << ",\"vram_slots\":" << r.vram_slots
       << ",\"record_bytes\":" << r.record_bytes
       << ",\"accesses\":" << r.accesses
       << ",\"ok\":" << (r.ok ? "true" : "false");
    if (!r.ok) os << ",\"error\":\"" << r.error << "\"";
    os << ",\"host\":{\"hits\":" << r.host_hit << ",\"misses\":" << r.host_miss
       << ",\"evictions\":" << r.host_evictions << ",\"fills\":" << r.host_fills
       << ",\"coalesced\":" << r.host_coalesced
       << ",\"budget_rejections\":" << r.host_budget_rejections
       << ",\"failures\":" << r.host_failures
       << ",\"ssd_bytes\":" << r.host_ssd_bytes
       << ",\"peak_resident_bytes\":" << r.host_peak_resident_bytes << "}"
       << ",\"device\":{\"hits\":" << r.device_hit
       << ",\"misses\":" << r.device_miss
       << ",\"evictions\":" << r.device_evictions
       << ",\"failures\":" << r.device_failures
       << ",\"pinned_skipped\":" << r.device_pinned_skipped
       << ",\"resident\":" << r.device_resident
       << ",\"h2d_bytes\":" << r.h2d_bytes << "}}"
       << (comma ? ",\n" : "\n");
}

struct Args {
    std::string journal;
    std::string mode = "sweep";
    std::string scope = "full";
    std::string score = "priority";
    std::string prio_model = "sorted";
    std::string lease_mode = "touch";
    size_t host_slots = 682;
    size_t vram_slots = 281;
    size_t record_bytes = 4096;  // logical default; 13369344 for physical
    std::string out;
};

bool parse_args(int argc, char** argv, Args* a) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name, std::string* dst) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "tier_replay: %s needs a value\n", name);
                return false;
            }
            *dst = argv[++i];
            return true;
        };
        auto next_size = [&](const char* name, size_t* dst) {
            std::string t; if (!next(name, &t)) return false;
            *dst = static_cast<size_t>(std::stoull(t)); return true;
        };
        if (arg == "--journal") { if (!next("--journal", &a->journal)) return false; }
        else if (arg == "--mode") { if (!next("--mode", &a->mode)) return false; }
        else if (arg == "--scope") { if (!next("--scope", &a->scope)) return false; }
        else if (arg == "--score") { if (!next("--score", &a->score)) return false; }
        else if (arg == "--priority-model") { if (!next("--priority-model", &a->prio_model)) return false; }
        else if (arg == "--lease-mode") { if (!next("--lease-mode", &a->lease_mode)) return false; }
        else if (arg == "--host-slots") { if (!next_size("--host-slots", &a->host_slots)) return false; }
        else if (arg == "--vram-slots") { if (!next_size("--vram-slots", &a->vram_slots)) return false; }
        else if (arg == "--record-bytes") { if (!next_size("--record-bytes", &a->record_bytes)) return false; }
        else if (arg == "--out") { if (!next("--out", &a->out)) return false; }
        else { std::fprintf(stderr, "tier_replay: unknown arg %s\n", arg.c_str()); return false; }
    }
    if (a->journal.empty()) {
        std::fprintf(stderr, "tier_replay: --journal is required\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) return 2;
    std::vector<Batch> journal;
    if (!load_journal(args.journal, &journal)) return 2;

    std::vector<ArmResult> results;
    const std::vector<size_t> host_budget_slots =
        {642, 963, 1285, 1606, 1927, 2570, 10280};  // 8/12/16/20/24/32/128 GiB
    const std::vector<size_t> vram_budget_slots =
        {80, 160, 240, 281, 321, 481, 642};        // 1/2/3/3.5/4/6/8 GiB
    const std::vector<std::string> gpu_scopes = {"cuda0", "cuda1"};

    if (args.mode == "sweep") {
        // Host arm: plain LRU, full + per-scope streams, lease touch + batch.
        for (size_t slots : host_budget_slots) {
            results.push_back(arm_host(scope_stream(journal, "full"), slots,
                                       args.record_bytes, "touch", "full"));
        }
        for (const auto& scope : gpu_scopes) {
            for (size_t slots : {682, 1285}) {
                results.push_back(arm_host(scope_stream(journal, scope), slots,
                                           args.record_bytes, "touch", scope));
            }
            results.push_back(arm_host(scope_stream(journal, scope), 682,
                                       args.record_bytes, "batch", scope));
        }
        // Device arm: bare ensure + prefetch path, both scores, both
        // priority models, all sim VRAM budgets.
        for (const auto& scope : gpu_scopes) {
            const auto sub = scope_stream(journal, scope);
            for (size_t slots : vram_budget_slots) {
                for (const auto& score : {"priority", "lru"}) {
                    for (const auto& pm : {"sorted", "first_seen"}) {
                        results.push_back(arm_bare(sub, slots,
                            args.record_bytes, score, pm, scope));
                        results.push_back(arm_vram(sub, slots,
                            args.record_bytes, score, pm, scope));
                    }
                }
            }
        }
        // Combined arm: device 281 + host 682 and device 281 + host 1285.
        for (const auto& scope : gpu_scopes) {
            const auto sub = scope_stream(journal, scope);
            for (size_t hslots : {682, 1285}) {
                for (const auto& score : {"priority", "lru"}) {
                    for (const auto& pm : {"sorted", "first_seen"}) {
                        results.push_back(arm_combined(sub, hslots, 281,
                            args.record_bytes, score, pm, scope));
                    }
                }
            }
        }
    } else if (args.mode == "host") {
        results.push_back(arm_host(scope_stream(journal, args.scope),
            args.host_slots, args.record_bytes, args.lease_mode, args.scope));
    } else if (args.mode == "bare") {
        results.push_back(arm_bare(scope_stream(journal, args.scope),
            args.vram_slots, args.record_bytes, args.score, args.prio_model,
            args.scope));
    } else if (args.mode == "vram") {
        results.push_back(arm_vram(scope_stream(journal, args.scope),
            args.vram_slots, args.record_bytes, args.score, args.prio_model,
            args.scope));
    } else if (args.mode == "combined") {
        results.push_back(arm_combined(scope_stream(journal, args.scope),
            args.host_slots, args.vram_slots, args.record_bytes, args.score,
            args.prio_model, args.scope));
    } else {
        std::fprintf(stderr, "tier_replay: unknown mode %s\n", args.mode.c_str());
        return 2;
    }

    std::ostringstream buf;
    buf << "{\"journal\":\"" << args.journal
        << "\",\"dee4_record_bytes\":" << kDee4RecordBytes
        << ",\"runs\":[\n";
    for (size_t i = 0; i < results.size(); ++i) emit(buf, results[i], i + 1 < results.size());
    buf << "]}\n";
    if (args.out.empty()) {
        std::cout << buf.str();
    } else {
        std::ofstream out(args.out, std::ios::binary | std::ios::trunc);
        if (!out) { std::fprintf(stderr, "tier_replay: cannot write %s\n", args.out.c_str()); return 2; }
        out << buf.str();
        std::fprintf(stderr, "tier_replay: wrote %zu runs to %s\n",
                     results.size(), args.out.c_str());
    }
    for (const auto& r : results) {
        if (!r.ok) {
            std::fprintf(stderr, "tier_replay: FAILED run arm=%s scope=%s "
                         "score=%s slots=%zu/%zu: %s\n", r.arm.c_str(),
                         r.scope.c_str(), r.score.c_str(), r.host_slots,
                         r.vram_slots, r.error.c_str());
        }
    }
    return 0;
}
