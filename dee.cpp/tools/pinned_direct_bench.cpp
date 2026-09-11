// tools/pinned_direct_bench.cpp — GPU Batch #1 segment P3 microbench.
//
// Measures the enqueue-path decomposition the fixed-slot staging prototype
// (research/phase2-fixedslot-proto @ fbc8b205, FIXED_SLOT_STAGING_HOLD) could
// not: the real-CUDA submission cost of the pinned-direct host-lease path
// (AsyncPrefetcher::prefetch_host_lease — one cudaMemcpyAsync straight out of
// the host tier's registered slot plus one event record) versus the legacy
// gather-copy path (source -> pinned-staging memcpy -> cudaMemcpyAsync), at
// the production 12.75 MiB DEE4 record size.
//
// Feeds the HOLD gate only (informational; never gates the arms):
//   PROMOTE iff modeled enqueue drop >= 6 s over the sealed 16-token decode's
//   1,391 + 1,092 host requests AND pool registration is not a regression;
//   REJECT iff drop < 3 s or any byte/lifetime defect; 3-6 s stays HOLD.
//
// Compiles clean under DEE_CUDA=OFF: every CUDA call sits behind #ifdef
// DEE_CUDA and the host-only decomposition still runs on the mock stream +
// pageable backend (submit timings labeled "measured": false). Exit code is
// 0 on a completed measurement, 77 (ctest SKIP convention) when --require-cuda
// is passed and no usable CUDA device exists.
//
// Usage:
//   pinned_direct_bench [--counts 64,256,512,1024] [--slots N]
//                       [--vram-bytes B] [--record-bytes B]
//                       [--require-cuda] [--host-only]
//                       [--out pinned_direct_bench.json]

#include "dee/async_prefetcher.h"
#include "dee/expert_tiers.h"
#include "dee/host_expert_tier.h"
#include "dee/vram_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef DEE_CUDA
#include <cuda_runtime.h>
#endif

namespace {
using namespace dee;

// Production DEE4 packed record: 12.75 MiB (AGENTS.md canonical geometry).
constexpr size_t kProductionRecordBytes = 13369344;
// v60 sealed host-miss counts per engine (campaign doc section 3.3): the
// modeled decode enqueue total multiplies per-record costs by these counts.
constexpr uint64_t kSealedHostRequests[2] = {1391, 1092};
// Sealed VRAM residency budget: 281 slots x 12.75 MiB = 3.5 GiB (v60).
constexpr size_t kSealedVramBudget = 3758096384ull;
// Matched host envelope: ~8.5 GiB per engine (v60 runtime cap, campaign 2.2).
constexpr size_t kSealedHostBudget = 9126805504ull;
// The HOLD gate's predeclared decision bounds (proto section 6 / campaign 3.3).
constexpr double kPromoteDropS = 6.0;
constexpr double kRejectDropS = 3.0;
// The proto's unverifiable enqueue claim this bench replaces with a measurement.
constexpr double kProtoEnqueueClaimS = 9.4;
// The tier key scope every bench record shares (one model, one layout).
const char* kBenchModel = "pinned-direct-bench-v1";
const char* kBenchRepr = "dee4-packed-fp4-v1";

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}
double ns_to_us(uint64_t ns) { return static_cast<double>(ns) / 1000.0; }

struct Args {
    std::vector<size_t> counts{64, 256, 512, 1024};
    size_t slots = 0;  // 0 => floor(kSealedHostBudget / record stride)
    size_t vram_bytes = kSealedVramBudget;
    size_t record_bytes = kProductionRecordBytes;
    bool require_cuda = false;  // SKIP(77) when no usable GPU is present
    bool host_only = false;     // force the mock/no-CUDA decomposition
    std::string out = "pinned_direct_bench.json";
};

bool parse_args(int argc, char** argv, Args* out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name, std::string* dst) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "pinned_direct_bench: %s needs a value\n", name);
                return false;
            }
            *dst = argv[++i];
            return true;
        };
        if (arg == "--counts") {
            std::string text;
            if (!need("--counts", &text)) return false;
            out->counts.clear();
            size_t pos = 0;
            while (pos <= text.size()) {
                const size_t comma = text.find(',', pos);
                const std::string tok =
                    text.substr(pos, comma == std::string::npos
                                       ? std::string::npos : comma - pos);
                if (!tok.empty())
                    out->counts.push_back(
                        static_cast<size_t>(std::stoull(tok)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            if (out->counts.empty()) return false;
        } else if (arg == "--slots") {
            std::string t; if (!need("--slots", &t)) return false;
            out->slots = static_cast<size_t>(std::stoull(t));
        } else if (arg == "--vram-bytes") {
            std::string t; if (!need("--vram-bytes", &t)) return false;
            out->vram_bytes = static_cast<size_t>(std::stoull(t));
        } else if (arg == "--record-bytes") {
            std::string t; if (!need("--record-bytes", &t)) return false;
            out->record_bytes = static_cast<size_t>(std::stoull(t));
        } else if (arg == "--require-cuda") {
            out->require_cuda = true;
        } else if (arg == "--host-only") {
            out->host_only = true;
        } else if (arg == "--out") {
            if (!need("--out", &out->out)) return false;
        } else {
            std::fprintf(stderr, "pinned_direct_bench: unknown arg %s\n", arg.c_str());
            return false;
        }
    }
    for (size_t count : out->counts) {
        if (count == 0 || count > 65536) {
            std::fprintf(stderr, "pinned_direct_bench: --counts out of range\n");
            return false;
        }
    }
    if (out->record_bytes == 0 || out->record_bytes > (size_t(1) << 30)) {
        std::fprintf(stderr, "pinned_direct_bench: --record-bytes out of range\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

// Deterministic key-derived record bytes; also the expected-image check.
uint8_t pattern_at(const TierExpertKey& key, size_t i) {
    return static_cast<uint8_t>((key.expert * 37 + key.layer * 101 + i) & 0xff);
}

// Cold store that materializes deterministic bytes and times its own copy so
// the fill (the gather-equivalent component inside a miss acquire) separates
// from tier bookkeeping.
struct TimedStore : ColdExpertStore {
    size_t record_bytes = 0;
    uint64_t fill_memcpy_ns = 0;
    uint64_t reads = 0;
    ColdReadResult read(const TierExpertKey& key, uint8_t* dst, size_t n) override {
        ++reads;
        const uint64_t begin = now_ns();
        for (size_t i = 0; i < n; ++i) dst[i] = pattern_at(key, i);
        fill_memcpy_ns += now_ns() - begin;
        return {true, n};
    }
};

Arena::Backend host_arena() {
    return {[](size_t b) { return std::malloc(b); },
            [](void* p) { std::free(p); }, "host"};
}

#ifdef DEE_CUDA
Arena::Backend cuda_arena() {
    return {[](size_t b) -> void* {
                void* ptr = nullptr;
                return cudaMalloc(&ptr, b) == cudaSuccess ? ptr : nullptr;
            },
            [](void* p) { cudaFree(p); }, "cuda"};
}
#endif

struct Timing {
    std::vector<double> us;
    void add_ns(uint64_t ns) { us.push_back(ns_to_us(ns)); }
};

struct Summary {
    size_t n = 0;
    double p50 = 0, p95 = 0, max = 0, mean = 0;
};

Summary summarize(const Timing& t) {
    Summary s;
    s.n = t.us.size();
    if (t.us.empty()) return s;
    std::vector<double> sorted = t.us;
    std::sort(sorted.begin(), sorted.end());
    s.p50 = sorted[sorted.size() / 2];
    s.p95 = sorted[std::min(sorted.size() - 1, (sorted.size() * 95) / 100)];
    s.max = sorted.back();
    double total = 0;
    for (double v : sorted) total += v;
    s.mean = total / sorted.size();
    return s;
}

std::string summary_json(const Timing& t, bool measured) {
    const Summary s = summarize(t);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"n\":%zu,\"p50_us\":%.2f,\"p95_us\":%.2f,\"max_us\":%.2f,"
        "\"mean_us\":%.2f,\"measured\":%s}",
        s.n, s.p50, s.p95, s.max, s.mean, measured ? "true" : "false");
    return buf;
}

// ---------------------------------------------------------------------------
// Host pool construction cost: HostExpertTier's ctor allocates every slot then
// calls backend.pin (cudaHostRegister under CUDA) per slot when try_pin. The
// pinned-vs-pageable ctor delta isolates registration from raw allocation.
// ---------------------------------------------------------------------------
struct PoolMeasure {
    double alloc_only_ms = 0;    // try_pin=false ctor: aligned allocs only
    double alloc_and_pin_ms = 0; // try_pin=true ctor: allocs + cudaHostRegister
    double register_per_slot_ms = 0;
    size_t slots = 0;
    size_t budget_bytes = 0;
    size_t pinned_bytes = 0;
    uint64_t pin_failures = 0;
};

HostTierConfig pool_config(size_t slots, size_t record_bytes, bool try_pin) {
    HostTierConfig cfg;
    cfg.slot_bytes = record_bytes;
    cfg.alignment = 4096;
    cfg.policy_slots = 0;
    cfg.dynamic_slots = slots;
    cfg.budget_bytes = slots * record_bytes;
    cfg.try_pin = try_pin;
    return cfg;
}

// ---------------------------------------------------------------------------
// Tier (pinned-direct) sweep: `count` distinct records through
// acquire -> prefetch_host_lease -> wait/release. Pass 2 reruns the same keys
// after a device-cache clear: host-resident source + device-miss = the
// "hit-path submit latency with a resident source" the doc asks for.
// ---------------------------------------------------------------------------
struct TierSweep {
    Timing acquire_miss;   // miss acquire (contains the store fill copy)
    Timing acquire_hit;    // hit acquire (lease return only)
    Timing submit_miss;    // prefetch_host_lease enqueue, miss path
    Timing submit_hit;     // prefetch_host_lease enqueue, resident source
    Timing ops;            // wait + collect + lease release bookkeeping
    uint64_t fill_memcpy_ns = 0;
    uint64_t h2d_bytes = 0;
    uint64_t hits = 0, misses = 0, pin_failures = 0;
    bool submit_measured = false;
    bool exact = true;
};

TierSweep run_tier_sweep(size_t count, bool use_cuda, bool try_pin,
                         size_t vram_bytes, size_t record_bytes,
                         size_t slot_count, uint64_t* io_seq) {
    TierSweep r;
    TimedStore store;
    store.record_bytes = record_bytes;
    IdentityCodec codec;
    HostExpertTier host(pool_config(slot_count, record_bytes, try_pin),
                        host_memory_backend(use_cuda, 0),
                        std::make_shared<PlainLruHostPlacementPolicy>());
    VramCacheManager cache;
    Arena::Backend arena = host_arena();
#ifdef DEE_CUDA
    if (use_cuda) arena = cuda_arena();
#endif
    if (!cache.init(vram_bytes, arena))
        throw std::runtime_error("tier sweep: cache arena init failed");
    AsyncPrefetcher prefetcher(cache);
    if (!prefetcher.init(use_cuda))
        throw std::runtime_error("tier sweep: prefetcher init failed");
    const TierExpertKey scope{kBenchModel, 0, 0, kBenchRepr};
    DeviceExpertTier device(cache, prefetcher, scope);

    const uint64_t fill_before = store.fill_memcpy_ns;
    std::vector<uint8_t> check(record_bytes);

    // Pass 1 reruns the SAME keys as pass 0 after a device-cache clear: the
    // host tier still holds them (capacity permitting), so every stage is a
    // host-hit + device-miss — the "hit-path submit latency with a resident
    // source" the campaign doc asks for. `io_seq` therefore advances only
    // once, after both passes.
    const uint64_t pass_base = *io_seq;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) cache.clear();  // device miss + host hit
        for (size_t i = 0; i < count; ++i) {
            const int layer = static_cast<int>((pass_base + i) / 256) % 43;
            const int expert = static_cast<int>((pass_base + i) % 256);
            const StorageRecord rec{
                {kBenchModel, layer, expert, kBenchRepr},
                record_bytes, record_bytes, "identity-v1"};

            const uint64_t t0 = now_ns();
            HostAcquireResult acquired = host.acquire(rec, store, codec);
            const uint64_t t1 = now_ns();
            if (!acquired.lease) {
                if (acquired.status == HostAcquireStatus::Capacity) continue;
                throw std::runtime_error("tier sweep: acquire failed");
            }
            if (pass == 0) r.acquire_miss.add_ns(t1 - t0);
            else r.acquire_hit.add_ns(t1 - t0);

            const uint64_t t2 = now_ns();
            const long tid = prefetcher.prefetch_host_lease(
                acquired.lease, 0, /*token*/ 0, layer);
            const uint64_t t3 = now_ns();
            if (tid < 0) throw std::runtime_error("tier sweep: submit failed");
            if (pass == 0) r.submit_miss.add_ns(t3 - t2);
            else r.submit_hit.add_ns(t3 - t2);

            const uint64_t t4 = now_ns();
            if (!prefetcher.wait(layer, expert))
                throw std::runtime_error("tier sweep: wait failed");
            acquired.lease.reset();
            prefetcher.collect_host_sources(false);
            const uint64_t t5 = now_ns();
            r.ops.add_ns(t5 - t4);

            // Byte-exactness spot check on the first and last record.
            if (i == 0 || i + 1 == count) {
                const void* data = cache.data(layer, expert);
                if (!data) throw std::runtime_error("tier sweep: block absent");
#ifdef DEE_CUDA
                if (use_cuda) {
                    if (cudaMemcpy(check.data(), data, record_bytes,
                                   cudaMemcpyDeviceToHost) != cudaSuccess)
                        throw std::runtime_error("tier sweep: D2H copyback");
                } else
#endif
                {
                    std::memcpy(check.data(), data, record_bytes);
                }
                for (size_t b = 0; b < record_bytes; ++b) {
                    if (check[b] != pattern_at(rec.key, b)) {
                        r.exact = false;
                        break;
                    }
                }
            }
        }
    }
    *io_seq = pass_base + count;
    prefetcher.synchronize_all();
    r.fill_memcpy_ns = store.fill_memcpy_ns - fill_before;
    const HostTierStats hs = host.stats();
    r.hits = hs.host_hit;
    r.misses = hs.host_miss;
    r.pin_failures = hs.pin_failures;
    r.h2d_bytes = prefetcher.stats().h2d_bytes;
    r.submit_measured = use_cuda;
    return r;
}

// ---------------------------------------------------------------------------
// Gather-copy (legacy) sweep: source -> pinned-staging memcpy (the component
// cuda_submit performs on the calling thread, accounted as mmap_to_pinned
// bytes), then the H2D enqueue + event record. Under CUDA both are real;
// without it the submit is mock and marked unmeasured.
// ---------------------------------------------------------------------------
struct GatherSweep {
    Timing gather;   // source -> pinned staging memcpy (calling thread)
    Timing submit;   // ensure + cudaMemcpyAsync + cudaEventRecord (+destroy)
    Timing ops;      // completion wait + unpin bookkeeping
    uint64_t gathered_bytes = 0;
    bool submit_measured = false;
    bool exact = true;
};

GatherSweep run_gather_sweep(size_t count, bool use_cuda, size_t vram_bytes,
                             size_t record_bytes, uint64_t io_base) {
    GatherSweep r;
    VramCacheManager cache;
    Arena::Backend arena = host_arena();
#ifdef DEE_CUDA
    if (use_cuda) arena = cuda_arena();
#endif
    if (!cache.init(vram_bytes, arena))
        throw std::runtime_error("gather sweep: cache arena init failed");

    // One reusable staging slot, matching production's bounded pinned ring
    // (serial submits synchronize before slot reuse here, the conservative end).
    uint8_t* staging = nullptr;
#ifdef DEE_CUDA
    if (use_cuda) {
        if (cudaHostAlloc(reinterpret_cast<void**>(&staging), record_bytes,
                          cudaHostAllocDefault) != cudaSuccess)
            throw std::runtime_error("gather sweep: cudaHostAlloc failed");
    } else
#endif
    {
        staging = static_cast<uint8_t*>(std::malloc(record_bytes));
    }
    if (!staging) throw std::runtime_error("gather sweep: staging alloc failed");

    std::vector<uint8_t> source(record_bytes);
    std::vector<uint8_t> check(record_bytes);
#ifdef DEE_CUDA
    cudaStream_t stream = nullptr;
    if (use_cuda &&
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("gather sweep: stream init failed");
#endif

    for (size_t i = 0; i < count; ++i) {
        const int layer = static_cast<int>((io_base + i) / 256) % 43;
        const int expert = static_cast<int>((io_base + i) % 256);
        const TierExpertKey key{kBenchModel, layer, expert, kBenchRepr};
        for (size_t b = 0; b < record_bytes; ++b) source[b] = pattern_at(key, b);

        const uint64_t t0 = now_ns();
        std::memcpy(staging, source.data(), record_bytes);
        const uint64_t t1 = now_ns();
        r.gather.add_ns(t1 - t0);
        r.gathered_bytes += record_bytes;

        const uint64_t t2 = now_ns();
        if (!cache.ensure(layer, expert, record_bytes, 0))
            throw std::runtime_error("gather sweep: ensure failed");
        // Production pins the destination block before submitting the copy
        // (prefetch_impl pins inside the submit window) and releases it on
        // completion. Mirror that so the release below is a real op and the
        // cache underflow warning cannot fire.
        const bool held = cache.pin(layer, expert);
        void* dst = cache.data(layer, expert);
#ifdef DEE_CUDA
        cudaEvent_t event = nullptr;
        if (use_cuda) {
            if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming)
                    != cudaSuccess)
                throw std::runtime_error("gather sweep: event create");
            if (cudaMemcpyAsync(dst, staging, record_bytes,
                                cudaMemcpyHostToDevice, stream) != cudaSuccess)
                throw std::runtime_error("gather sweep: memcpyAsync");
            if (cudaEventRecord(event, stream) != cudaSuccess)
                throw std::runtime_error("gather sweep: event record");
        }
#else
        (void)dst;
#endif
        const uint64_t t3 = now_ns();
        r.submit.add_ns(t3 - t2);

        const uint64_t t4 = now_ns();
#ifdef DEE_CUDA
        if (use_cuda) {
            if (cudaEventSynchronize(event) != cudaSuccess)
                throw std::runtime_error("gather sweep: event sync");
            if (cudaEventDestroy(event) != cudaSuccess)
                throw std::runtime_error("gather sweep: event destroy");
        } else
#endif
        {
            std::memcpy(dst, staging, record_bytes);  // mock device write
        }
        if (held) cache.unpin(layer, expert);  // wait() releases the pin
        const uint64_t t5 = now_ns();
        r.ops.add_ns(t5 - t4);

        if (i == 0 || i + 1 == count) {
#ifdef DEE_CUDA
            if (use_cuda) {
                if (cudaMemcpy(check.data(), cache.data(layer, expert),
                               record_bytes, cudaMemcpyDeviceToHost)
                        != cudaSuccess)
                    throw std::runtime_error("gather sweep: D2H copyback");
            } else
#endif
            {
                std::memcpy(check.data(), cache.data(layer, expert), record_bytes);
            }
            for (size_t b = 0; b < record_bytes; ++b) {
                if (check[b] != pattern_at(key, b)) { r.exact = false; break; }
            }
        }
    }
#ifdef DEE_CUDA
    if (stream) cudaStreamDestroy(stream);
    if (use_cuda && staging) cudaFreeHost(staging);
    else
#endif
    std::free(staging);
    r.submit_measured = use_cuda;
    return r;
}

// ---------------------------------------------------------------------------
// JSON emission (hand-rolled; tool avoids a json dependency for portability)
// ---------------------------------------------------------------------------
std::string esc(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        std::fprintf(stderr,
            "usage: pinned_direct_bench [--counts 64,256,512,1024] "
            "[--slots N] [--vram-bytes B] [--record-bytes B] "
            "[--require-cuda] [--host-only] [--out FILE]\n");
        return 2;
    }

    bool cuda_available = false;
    std::string device_name;
#ifdef DEE_CUDA
    if (!args.host_only) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0) {
            cuda_available = true;
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
                device_name = prop.name;
        }
    }
#endif
    const bool use_cuda = cuda_available && !args.host_only;

    const size_t slot_count = args.slots
        ? args.slots
        : args.record_bytes <= kSealedHostBudget
            ? kSealedHostBudget / args.record_bytes
            : 1;

    if (args.require_cuda && !use_cuda) {
        std::ostringstream stub;
        stub << "{\"tool\":\"pinned_direct_bench\",\"status\":\"SKIP_NO_GPU\","
             << "\"dee_cuda_build\":"
#ifdef DEE_CUDA
             << "true"
#else
             << "false"
#endif
             << ",\"cuda_available\":false,\"record_bytes\":"
             << args.record_bytes << "}\n";
        if (!args.out.empty()) {
            std::ofstream f(args.out);
            f << stub.str();
        }
        std::cout << "SKIP: no usable CUDA device; stub written to "
                  << args.out << "\n";
        return 77;
    }

    try {
        // ---- Pool cost: alloc-only ctor vs alloc+register ctor ------------
        PoolMeasure pool;
        pool.slots = slot_count;
        pool.budget_bytes = slot_count * args.record_bytes;
        {
            const uint64_t t0 = now_ns();
            HostExpertTier pageable(
                pool_config(slot_count, args.record_bytes, false),
                host_memory_backend(use_cuda, 0),
                std::make_shared<PlainLruHostPlacementPolicy>());
            pool.alloc_only_ms = ns_to_us(now_ns() - t0) / 1000.0;
        }
        std::unique_ptr<HostExpertTier> pinned_pool;
        const uint64_t t1 = now_ns();
        pinned_pool.reset(new HostExpertTier(
            pool_config(slot_count, args.record_bytes, use_cuda /*try_pin*/),
            host_memory_backend(use_cuda, 0),
            std::make_shared<PlainLruHostPlacementPolicy>()));
        pool.alloc_and_pin_ms = ns_to_us(now_ns() - t1) / 1000.0;
        const HostTierStats ps = pinned_pool->stats();
        pool.pinned_bytes = ps.pinned_bytes;
        pool.pin_failures = ps.pin_failures;
        pool.register_per_slot_ms =
            (pool.alloc_and_pin_ms - pool.alloc_only_ms) / slot_count;
        // The pinned pool above is dropped before the sweeps so total host
        // memory stays near one envelope; sweeps rebuild their own tier.
        pinned_pool.reset();

        // ---- Sweeps ------------------------------------------------------
        std::ostringstream direct_json, gather_json;
        uint64_t io_seq = 0;
        bool all_exact = true;
        bool any_exact_checked = false;
        TierSweep last_tier;      // kept for the enqueue model below
        GatherSweep last_gather;  // (largest configured count)
        direct_json << "{";
        gather_json << "{";
        for (size_t ci = 0; ci < args.counts.size(); ++ci) {
            const size_t count = args.counts[ci];
            const TierSweep tier = run_tier_sweep(
                count, use_cuda, /*try_pin*/ use_cuda, args.vram_bytes,
                args.record_bytes, slot_count, &io_seq);
            const GatherSweep gather = run_gather_sweep(
                count, use_cuda, args.vram_bytes, args.record_bytes,
                /*io_base*/ io_seq - count);
            last_tier = tier;
            last_gather = gather;
            all_exact = all_exact && tier.exact && gather.exact;
            any_exact_checked = true;
            direct_json << (ci ? ",\"" : "\"") << count << "\":{"
                << "\"acquire_miss_us\":" << summary_json(tier.acquire_miss, true)
                << ",\"acquire_hit_us\":" << summary_json(tier.acquire_hit, true)
                << ",\"submit_miss_us\":" << summary_json(tier.submit_miss, tier.submit_measured)
                << ",\"submit_hit_us\":" << summary_json(tier.submit_hit, tier.submit_measured)
                << ",\"ops_us\":" << summary_json(tier.ops, true)
                << ",\"fill_memcpy_total_ms\":" << (tier.fill_memcpy_ns / 1e6)
                << ",\"host_hits\":" << tier.hits
                << ",\"host_misses\":" << tier.misses
                << ",\"pin_failures\":" << tier.pin_failures
                << ",\"h2d_bytes\":" << tier.h2d_bytes
                << ",\"exact\":" << (tier.exact ? "true" : "false") << "}";
            gather_json << (ci ? ",\"" : "\"") << count << "\":{"
                << "\"gather_us\":" << summary_json(gather.gather, true)
                << ",\"submit_us\":" << summary_json(gather.submit, gather.submit_measured)
                << ",\"ops_us\":" << summary_json(gather.ops, true)
                << ",\"gathered_bytes\":" << gather.gathered_bytes
                << ",\"exact\":" << (gather.exact ? "true" : "false") << "}";
            std::fprintf(stderr,
                "[bench] count=%zu direct submit p50=%.1fus gather p50=%.1fus "
                "gsubmit p50=%.1fus\n",
                count, summarize(tier.submit_miss).p50,
                summarize(gather.gather).p50, summarize(gather.submit).p50);
        }
        direct_json << "}";
        gather_json << "}";

        // ---- Pageable-source direct sweep (forced try_pin=false) ----------
        const TierSweep pageable = run_tier_sweep(
            args.counts.back(), use_cuda, /*try_pin*/ false, args.vram_bytes,
            args.record_bytes, slot_count, &io_seq);
        all_exact = all_exact && pageable.exact;

        // ---- Modeled enqueue split vs the HOLD gate -----------------------
        // Uses the largest pinned sweep's measured means (already run above).
        // Per-record enqueue: gather path = gather + submit (both on the
        // calling thread in production); direct path = prefetch_host_lease.
        const uint64_t requests = kSealedHostRequests[0] + kSealedHostRequests[1];
        const double direct_us = summarize(last_tier.submit_miss).mean;
        const double gather_us = summarize(last_gather.gather).mean
                               + summarize(last_gather.submit).mean;
        const double direct_total_s = direct_us * requests / 1e6;
        const double gather_total_s = gather_us * requests / 1e6;
        const double drop_s = gather_total_s - direct_total_s;
        // "No pinned-alloc regression": one-time slot-pool registration must
        // stay a small fraction of the modeled per-decode prize (<=25%).
        const double pin_overhead_s = std::max(0.0,
            (pool.alloc_and_pin_ms - pool.alloc_only_ms) / 1000.0);
        const bool pin_overhead_ok =
            pin_overhead_s <= 0.25 * drop_s;
        // Under the mock path there is no real submission cost to model —
        // label the verdict instead of emitting a misleading gate decision.
        const char* verdict = !all_exact ? "REJECT_CORRECTNESS"
            : !use_cuda ? "MOCK_NO_CUDA"
            : (drop_s >= kPromoteDropS && pin_overhead_ok) ? "PROMOTE"
            : drop_s < kRejectDropS ? "REJECT" : "HOLD";

        std::ostringstream out;
        out << "{\n"
            << "\"tool\":\"pinned_direct_bench\",\n"
            << "\"status\":\"MEASURED\",\n"
            << "\"purpose\":\"Phase-2D enqueue split: pinned-direct host-lease "
               "vs gather-copy submission at production record size\",\n"
            << "\"dee_cuda_build\":" <<
#ifdef DEE_CUDA
            "true"
#else
            "false"
#endif
            << ",\n"
            << "\"cuda_available\":" << (cuda_available ? "true" : "false") << ",\n"
            << "\"cuda_used\":" << (use_cuda ? "true" : "false") << ",\n"
            << "\"device\":\"" << esc(device_name) << "\",\n"
            << "\"record_bytes\":" << args.record_bytes << ",\n"
            << "\"vram_budget_bytes\":" << args.vram_bytes << ",\n"
            << "\"pool\":{\"slots\":" << pool.slots
            << ",\"budget_bytes\":" << pool.budget_bytes
            << ",\"alloc_only_ms\":" << pool.alloc_only_ms
            << ",\"alloc_and_pin_ms\":" << pool.alloc_and_pin_ms
            << ",\"register_per_slot_ms\":" << pool.register_per_slot_ms
            << ",\"pinned_bytes\":" << pool.pinned_bytes
            << ",\"pin_failures\":" << pool.pin_failures << "},\n"
            << "\"sweeps\":{\"pinned_direct\":" << direct_json.str()
            << ",\"gather_copy\":" << gather_json.str() << "},\n"
            << "\"pageable_direct_sweep\":{\"submit_miss_us\":"
            << summary_json(pageable.submit_miss, pageable.submit_measured)
            << ",\"submit_hit_us\":"
            << summary_json(pageable.submit_hit, pageable.submit_measured)
            << ",\"pin_failures\":" << pageable.pin_failures
            << ",\"exact\":" << (pageable.exact ? "true" : "false") << "},\n"
            << "\"model\":{\n"
            << "  \"sealed_host_requests\":[" << kSealedHostRequests[0]
            << "," << kSealedHostRequests[1] << "],\n"
            << "  \"per_record_us\":{\"gather_copy\":" << gather_us
            << ",\"pinned_direct\":" << direct_us << "},\n"
            << "  \"enqueue_total_s\":{\"gather_copy\":" << gather_total_s
            << ",\"pinned_direct\":" << direct_total_s
            << ",\"drop_s\":" << drop_s << "},\n"
            << "  \"proto_claim_enqueue_s\":" << kProtoEnqueueClaimS << ",\n"
            << "  \"pin_registration_overhead_s\":" << pin_overhead_s << ",\n"
            << "  \"gate\":{\"promote_s\":" << kPromoteDropS
            << ",\"reject_s\":" << kRejectDropS
            << ",\"pin_overhead_fraction_of_prize\":0.25"
            << ",\"pin_overhead_ok\":" << (pin_overhead_ok ? "true" : "false")
            << ",\"verdict\":\"" << verdict << "\"}\n"
            << "},\n"
            << "\"exact_bytes\":" << (all_exact ? "true" : "false") << ",\n"
            << "\"exact_checked\":" << (any_exact_checked ? "true" : "false")
            << "\n}\n";
        if (!args.out.empty()) {
            std::ofstream f(args.out);
            f << out.str();
        }
        std::cout << out.str();
        std::fprintf(stderr, "[bench] modeled enqueue drop %.3fs -> %s\n",
                     drop_s, verdict);
        return all_exact ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pinned_direct_bench FAIL: %s\n", e.what());
        return 1;
    }
}
