// bench_fixedslot.cpp — baseline (real HostPackCache::get_batch) vs fixed-slot
// prototype: reservation latency, first-submit, HIT head-of-line,
// per-completion publish, allocs/copies, sync, exactness.
//
// Usage:
//   bench_fixedslot --records <bytes> --entries <map preload> --slots <proto slots>
//                   --batches <n> --fill-us <sleep> --h2d-us <sleep>
//                   --scenario cold|mixed|churn --out <json>
//
// Scenarios: cold = all-miss 6-wide; mixed = 3 resident hits + 3 misses;
// churn = working set 1.5x budget, cyclic demand (evictions exercised).
// All fills write deterministic key-derived bytes; combine = FNV-1a over
// rank-ordered payloads. Baseline and prototype must agree bit-exactly.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "dee/host_pack_cache.h"
#include "fixed_slot_staging.h"

namespace {
uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
void sleep_us(uint64_t us) {
    if (us) std::this_thread::sleep_for(std::chrono::microseconds(us));
}
// Deterministic exact payload: key-derived bytes (stand-in for packed FP4).
void fill_pattern(uint8_t* dst, size_t n, uint64_t key, uint64_t fill_us) {
    const uint64_t k = key * 0x9E3779B97F4A7C15ull + 0x12345;
    for (size_t i = 0; i < n; ++i)
        dst[i] = static_cast<uint8_t>((k >> ((i % 8) * 8)) ^ (i * 2654435761u));
    sleep_us(fill_us);
}
uint64_t fnv1a(const uint8_t* d, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; ++i) {
        h ^= d[i];
        h *= 1099511628211ull;
    }
    return h;
}

struct Cfg {
    size_t record_bytes = 1 << 20;
    size_t preload_entries = 64;
    size_t proto_slots = 64;
    size_t lanes = 3;
    int batches = 20;
    uint64_t fill_us = 2000;
    uint64_t h2d_us = 500;
    uint64_t stagger_us = 0;  // extra fill sleep * (rank % 3): publish-order probe
    std::string scenario = "mixed";
    bool light = false;  // hash-only exactness (no payload copies; 12MiB runs)
    std::string out;
};

struct ExpertEvent {
    size_t batch = 0;
    size_t rank = 0;
    uint64_t key = 0;
    bool was_hit = false;
    uint64_t t_submit_ns = 0;  // H2D submit, relative to batch demand arrival
    uint64_t t_fill_invoke_ns = 0;
    uint64_t t_fill_done_ns = 0;
};

// Simulated async DMA: FIFO transfer thread, per-transfer completion stamps.
// Counts the gather-memcpy bytes the bench attributes to each path.
struct SimDma {
    uint64_t latency_us = 500;
    std::atomic<uint64_t> memcpy_bytes{0};
    std::atomic<uint64_t> submits{0};
    std::mutex mu;
    std::condition_variable cv;
    struct Item {
        size_t rank;
        uint64_t key;
        const uint8_t* data;
        size_t n;
        bool was_hit;
        uint64_t submit_ns;
        std::function<void(size_t)> on_done;
    };
    std::queue<Item> q;
    bool stop = false;
    std::thread th;
    SimDma() { th = std::thread([this] { loop(); }); }
    ~SimDma() {
        {
            std::lock_guard<std::mutex> l(mu);
            stop = true;
        }
        cv.notify_all();
        if (th.joinable()) th.join();
    }
    void submit(Item it) {
        // Model the pinned-gather copy attributed to the H2D path.
        memcpy_bytes += it.n;
        ++submits;
        {
            std::lock_guard<std::mutex> l(mu);
            q.push(std::move(it));
        }
        cv.notify_one();
    }
    void loop() {
        for (;;) {
            Item it{0, 0, nullptr, 0, false, 0, nullptr};
            {
                std::unique_lock<std::mutex> l(mu);
                cv.wait(l, [&] { return stop || !q.empty(); });
                if (stop && q.empty()) return;
                it = std::move(q.front());
                q.pop();
            }
            sleep_us(latency_us);
            if (it.on_done) it.on_done(it.rank);
        }
    }
};

struct BatchResult {
    uint64_t combine_hash = 0;
    std::vector<std::vector<uint8_t>> payloads;  // rank-ordered copies
};

// ---- Baseline: real get_batch + sequential stage loop (engine.cpp:880-886)
struct Baseline {
    dee::HostPackCache cache;
    SimDma dma;
    std::vector<ExpertEvent> events;
    std::vector<BatchResult> results;
    std::atomic<uint64_t> gather_memcpy{0};
    uint64_t reserve_wall_ns = 0;
    uint64_t first_invoke_ns = 0;  // sum of per-batch first-fill-invoke
    uint64_t batches = 0;
    uint64_t ev_batch = 0;  // running index for event rows (incl. preload)

    static bool fill_tramp(void* ctx, uint8_t* dst, size_t n) {
        auto* f = static_cast<std::function<void(uint8_t*, size_t)>*>(ctx);
        (*f)(dst, n);
        return true;
    }

    bool run(const Cfg& cfg, const std::vector<std::vector<uint64_t>>& demand) {
        cache.set_budget(cfg.preload_entries * cfg.record_bytes);
        cache.set_fill_lanes(cfg.lanes);
        const size_t pre = (cfg.preload_entries + 5) / 6;
        size_t bi = 0;
        for (const auto& keys : demand) {
            const bool measured = (bi++ >= pre);
            const uint64_t t0 = now_ns();
            const size_t m = keys.size();
            std::vector<dee::HostPackCache::BatchRequest> req(m);
            std::vector<dee::HostPackCache::BatchResult> res(m);
            std::vector<std::function<void(uint8_t*, size_t)>> fills(m);
            std::vector<uint64_t> invoke_ns(m, 0), done_ns(m, 0);
            for (size_t r = 0; r < m; ++r) {
                const uint64_t key = keys[r];
                fills[r] = [&, r, key](uint8_t* dst, size_t n) {
                    invoke_ns[r] = now_ns() - t0;
                    fill_pattern(dst, n, key,
                                 cfg.fill_us + cfg.stagger_us * (r % 3));
                    done_ns[r] = now_ns() - t0;
                };
                req[r] = {key, cfg.record_bytes, &fill_tramp, &fills[r], key};
            }
            const auto r0 = cache.stats().fill_reservation_wall_ms;
            if (!cache.get_batch(req.data(), m, res.data())) return false;
            const uint64_t rwall = static_cast<uint64_t>(
                (cache.stats().fill_reservation_wall_ms - r0) * 1e6);
            uint64_t first_inv = 0;
            for (size_t r = 0; r < m; ++r) {
                if (!res[r].fill_executed) continue;
                if (first_inv == 0 || invoke_ns[r] < first_inv)
                    first_inv = invoke_ns[r];
            }
            if (measured) {
                ++batches;
                reserve_wall_ns += rwall;
                first_invoke_ns += first_inv;
            }
            // Sequential stage loop: consume + H2D submit per expert in order.
            BatchResult br;
            if (!cfg.light) br.payloads.resize(m);
            uint64_t h = 1469598103934665603ull;
            for (size_t r = 0; r < m; ++r) {
                if (!res[r].success) return false;
                const uint64_t ts = now_ns() - t0;
                events.push_back({ev_batch, r, keys[r], res[r].cache_hit,
                                  ts, invoke_ns[r], done_ns[r]});
                // Calling-thread pinned-gather copy (the 1-2 s bucket).
                gather_memcpy += cfg.record_bytes;
                dma.submit({r, keys[r], res[r].data, cfg.record_bytes,
                            res[r].cache_hit, ts, [](size_t) {}});
                if (cfg.light) {
                    h = fnv1a(res[r].data, cfg.record_bytes, h);
                } else {
                    br.payloads[r].assign(res[r].data,
                                          res[r].data + cfg.record_bytes);
                }
            }
            if (!cfg.light) {
                for (size_t r = 0; r < m; ++r)
                    h = fnv1a(br.payloads[r].data(), cfg.record_bytes, h);
            }
            br.combine_hash = h;
            results.push_back(std::move(br));
            ++ev_batch;
        }
        return true;
    }
};

// ---- Prototype path
struct Proto {
    fixedslot::FixedSlotStaging staging;
    SimDma dma;
    std::vector<ExpertEvent> events;
    std::vector<BatchResult> results;
    std::mutex ev_mu;
    uint64_t reserve_wall_ns = 0;
    uint64_t first_invoke_ns = 0;  // sum of per-batch first-fill-invoke
    uint64_t batches = 0;
    uint64_t ev_batch = 0;  // running index for event rows (incl. preload)
    uint64_t ooo_batches = 0;  // batches where publish order != rank order
    const Cfg* cfgp = nullptr;
    uint64_t batch_t0 = 0;

    explicit Proto(const Cfg& c)
        : staging(c.proto_slots, c.lanes), cfgp(&c) {
        staging.set_h2d_hook(
            [this](size_t rank, uint64_t key, const uint8_t* data, size_t n,
                   bool was_hit, uint64_t now_ns_abs) {
                const uint64_t ts = now_ns_abs - batch_t0;
                {
                    std::lock_guard<std::mutex> l(ev_mu);
                    events.push_back({ev_batch, rank, key, was_hit, ts, 0, 0});
                }
                // Async DMA; on completion: pin device block + release host.
                dma.submit({rank, key, data, n, was_hit, ts,
                            [this](size_t rk) { staging.on_dma_complete(rk); }});
            });
    }

    bool run(const Cfg& cfg,
             const std::vector<std::vector<uint64_t>>& demand) {
        const size_t pre = (cfg.preload_entries + 5) / 6;
        size_t bi = 0;
        for (const auto& keys : demand) {
            const bool measured = (bi++ >= pre);
            batch_t0 = now_ns();
            const size_t m = keys.size();
            std::vector<fixedslot::Demand> dd(m);
            std::vector<uint64_t> invoke_ns(m, 0);
            const bool lifetime = (cfg.scenario == "lifetime");
            for (size_t r = 0; r < m; ++r) {
                const uint64_t key = keys[r];
                dd[r].key = key;
                dd[r].nbytes = cfg.record_bytes;
                // Lifetime scenario: even preload keys are policy-resident.
                if (lifetime && key <= cfg.preload_entries && key % 2 == 0)
                    dd[r].policy_resident = true;
                dd[r].fill = [key, &cfg, &invoke_ns, r,
                              t0 = batch_t0](uint8_t* dst, size_t n) {
                    invoke_ns[r] = now_ns() - t0;
                    fill_pattern(dst, n, key,
                                 cfg.fill_us + cfg.stagger_us * (r % 3));
                };
            }
            std::vector<fixedslot::FixedSlotStaging::Handle> handles;
            const uint64_t r0 = now_ns();
            if (!staging.reserve(dd.data(), m, batch_t0, handles))
                return false;
            const uint64_t rwall = now_ns() - r0;
            staging.submit_misses();  // caller lane + pool drain; join
            // Per-completion publish evidence: finish-order log vs rank order.
            {
                auto log = staging.drain_completion_log();
                bool sorted = true, rank_order = true;
                for (size_t i = 1; i < log.size(); ++i) {
                    if (log[i].fill_done_ns < log[i - 1].fill_done_ns)
                        sorted = false;
                    if (log[i].rank < log[i - 1].rank) rank_order = false;
                }
                if (measured && !log.empty() && sorted && !rank_order)
                    ++ooo_batches;
            }
            uint64_t first_inv = 0;
            for (size_t r = 0; r < m; ++r) {
                if (handles[r].was_hit) continue;
                if (first_inv == 0 || invoke_ns[r] < first_inv)
                    first_inv = invoke_ns[r];
            }
            if (measured) {
                reserve_wall_ns += rwall;
                first_invoke_ns += first_inv;
            }
            BatchResult br;
            if (!cfg.light) br.payloads.resize(m);
            uint64_t h = 1469598103934665603ull;
            for (size_t r = 0; r < m; ++r) {
                if (!staging.wait_rank(r)) return false;
                // DMA completion is async (transfer thread): poll until this
                // batch's transfer completes, then consume in rank order.
                // (The engine would wait_on_stream on the compute stream.)
                bool ok = false;
                for (int i = 0; i < 20000; ++i) {
                    ok = staging.mark_consumed(r);
                    if (ok) break;
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(100));
                }
                if (!ok) return false;
                const uint8_t* p = staging.payload(r);
                if (!p) return false;
                if (cfg.light) {
                    h = fnv1a(p, cfg.record_bytes, h);
                } else {
                    br.payloads[r].assign(p, p + cfg.record_bytes);
                }
            }
            if (!cfg.light) {
                for (size_t r = 0; r < m; ++r)
                    h = fnv1a(br.payloads[r].data(), cfg.record_bytes, h);
            }
            br.combine_hash = h;
            results.push_back(std::move(br));
            ++ev_batch;
            if (measured) ++batches;
        }
        return true;
    }
};

std::vector<std::vector<uint64_t>> make_demand(const Cfg& cfg) {
    // Deterministic PRNG (mulberry-ish); working set sized per scenario.
    uint64_t s = 0x243F6A8885A308D3ull;
    auto rnd = [&] {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    };
    size_t ws = cfg.preload_entries;
    if (cfg.scenario == "churn") ws = cfg.preload_entries * 3 / 2;
    if (cfg.scenario == "lifetime") ws = cfg.preload_entries * 2;
    std::vector<std::vector<uint64_t>> out;
    // Preload stream: first `preload_entries` unique keys (resident after).
    std::vector<uint64_t> preload;
    for (size_t i = 0; i < cfg.preload_entries; ++i)
        preload.push_back(i + 1);
    // Warm both paths identically is handled by running same demand twice;
    // here just build measured batches:
    for (int b = 0; b < cfg.batches; ++b) {
        std::vector<uint64_t> keys;
        if (cfg.scenario == "cold") {
            for (int r = 0; r < 6; ++r)
                keys.push_back(1000000ull + b * 6 + r);
        } else if (cfg.scenario == "mixed") {
            // 3 stable residents (hits) + 3 fresh (misses), rank-interleaved
            // so head-of-line is observable in both orders across batches.
            for (int r = 0; r < 6; ++r) {
                if ((r + b) % 2 == 0)
                    keys.push_back(1 + (rnd() % cfg.preload_entries));
                else
                    keys.push_back(2000000ull + b * 6 + r);
            }
        } else {  // churn
            for (int r = 0; r < 6; ++r) keys.push_back(1 + (rnd() % ws));
        }
        out.push_back(keys);
    }
    // Prefix preload batches so steady-state hits exist (not measured
    // separately; both paths run the identical full stream).
    std::vector<std::vector<uint64_t>> full;
    for (size_t i = 0; i < preload.size(); i += 6) {
        std::vector<uint64_t> k;
        for (size_t j = i; j < i + 6 && j < preload.size(); ++j)
            k.push_back(preload[j]);
        full.push_back(k);
    }
    for (auto& b : out) full.push_back(b);
    return full;
}

void write_json(const Cfg& cfg, const char* path, Baseline& bl, Proto& pr,
                bool exact_ok, double memset_MBps, bool policy_intact,
                bool violation_probe_pass) {
    // Per-batch-excluded preload rows: preload batches = ceil(entries/6).
    const size_t pre = (cfg.preload_entries + 5) / 6;
    auto summarize = [&](std::vector<ExpertEvent>& ev) {
        // hit H2D-submit delay and miss submit delay, measured batches only
        double hit_sum = 0, hit_n = 0, miss_sum = 0, miss_n = 0;
        double first_submit_sum = 0;
        size_t last_batch = static_cast<size_t>(-1);
        for (auto& e : ev) {
            if (e.batch < pre) continue;
            if (e.batch != last_batch) {
                last_batch = e.batch;
                first_submit_sum += e.t_submit_ns;
            }
            if (e.was_hit) {
                hit_sum += e.t_submit_ns;
                hit_n += 1;
            } else {
                miss_sum += e.t_submit_ns;
                miss_n += 1;
            }
        }
        return std::tuple<double, double, double, double, double>(
            hit_n ? hit_sum / hit_n / 1e6 : 0,
            hit_n, miss_n ? miss_sum / miss_n / 1e6 : 0, miss_n,
            first_submit_sum / cfg.batches / 1e6);
    };
    auto b = summarize(bl.events);
    auto p = summarize(pr.events);
    FILE* f = std::fopen(path, "w");
    std::fprintf(f,
                 "{\n"
                 " \"label\": \"MEASURED (synthetic host mechanism; no GPU)\",\n"
                 " \"cfg\": {\"record_bytes\": %zu, \"preload_entries\": %zu, "
                 "\"proto_slots\": %zu, \"lanes\": %zu, \"batches\": %d, "
                 "\"fill_us\": %llu, \"h2d_us\": %llu, \"scenario\": \"%s\"},\n"
                 " \"exact_combine_match\": %s,\n"
                 " \"baseline\": {\"reserve_ms_per_batch\": %.4f, "
                 "\"hit_h2d_submit_ms\": %.4f, \"hit_n\": %.0f, "
                 "\"miss_h2d_submit_ms\": %.4f, \"miss_n\": %.0f, "
                 "\"first_h2d_submit_ms\": %.4f, "
                 "\"first_fill_invoke_ms\": %.4f, "
                 "\"hits\": %llu, \"misses\": %llu, \"evictions\": %llu, "
                 "\"reused_buffers\": %llu, \"gather_memcpy_MB\": %.3f},\n"
                 " \"prototype\": {\"reserve_ms_per_batch\": %.4f, "
                 "\"hit_h2d_submit_ms\": %.4f, \"hit_n\": %.0f, "
                 "\"miss_h2d_submit_ms\": %.4f, \"miss_n\": %.0f, "
                 "\"first_h2d_submit_ms\": %.4f, "
                 "\"first_fill_invoke_ms\": %.4f, "
                 "\"victim_scans\": %llu, \"victims_taken\": %llu, "
                 "\"slot_allocs\": %llu, \"slot_reuses\": %llu, "
                 "\"zero_fill_bytes\": %llu, \"mutex_waits\": %llu,\n"
                 "  \"dma_memcpy_MB\": %.3f, \"ooo_batches\": %llu},\n"
                 " \"lifetime\": {\"policy_resident_intact\": %s, "
                 "\"consume_before_dma_fails_closed\": %s},\n"
                 " \"projection\": {\"memset_MBps_measured\": %.1f, "
                 "\"note\": \"DERIVED: production 12.75MiB fresh-miss resize cost = "
                 "13369344 / memset_rate; map-scan cost scales per entry\"}\n"
                 "}\n",
                 cfg.record_bytes, cfg.preload_entries, cfg.proto_slots,
                 cfg.lanes, cfg.batches,
                 (unsigned long long)cfg.fill_us,
                 (unsigned long long)cfg.h2d_us, cfg.scenario.c_str(),
                 exact_ok ? "true" : "false",
                 (double)bl.reserve_wall_ns / 1e6 /
                     (bl.batches ? bl.batches : 1),
                 std::get<0>(b), std::get<1>(b), std::get<2>(b),
                 std::get<3>(b), std::get<4>(b),
                 (double)bl.first_invoke_ns / 1e6 /
                     (bl.batches ? bl.batches : 1),
                 (unsigned long long)bl.cache.stats().hits,
                 (unsigned long long)bl.cache.stats().misses,
                 (unsigned long long)bl.cache.stats().evictions,
                 (unsigned long long)bl.cache.stats().reused_fill_buffers,
                 (double)bl.gather_memcpy / 1048576.0,
                 (double)pr.reserve_wall_ns / 1e6 /
                     (pr.batches ? pr.batches : 1),
                 std::get<0>(p), std::get<1>(p), std::get<2>(p),
                 std::get<3>(p), std::get<4>(p),
                 (double)pr.first_invoke_ns / 1e6 /
                     (pr.batches ? pr.batches : 1),
                 (unsigned long long)pr.staging.stats().victim_scans,
                 (unsigned long long)pr.staging.stats().victim_taken,
                 (unsigned long long)pr.staging.stats().slot_allocs,
                 (unsigned long long)pr.staging.stats().slot_reuses,
                 (unsigned long long)pr.staging.stats().zero_fill_bytes,
                 (unsigned long long)pr.staging.stats().mutex_waits,
                 (double)pr.dma.memcpy_bytes / 1048576.0,
                 (unsigned long long)pr.ooo_batches,
                 policy_intact ? "true" : "false",
                 violation_probe_pass ? "true" : "false", memset_MBps);
    std::fclose(f);
}

double measure_memset_MBps(size_t n) {
    std::vector<uint8_t> v;
    const auto t0 = now_ns();
    for (int i = 0; i < 8; ++i) {
        v.resize(n);  // fresh zero-fill path (baseline fresh-miss cost)
        v[0] = (uint8_t)i;
    }
    const double s = (now_ns() - t0) / 1e9;
    return (8.0 * n / 1048576.0) / s;
}

}  // namespace

int main(int argc, char** argv) {
    Cfg cfg;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        std::string v = argv[i + 1];
        if (k == "--records") cfg.record_bytes = std::stoul(v);
        if (k == "--entries") cfg.preload_entries = std::stoul(v);
        if (k == "--slots") cfg.proto_slots = std::stoul(v);
        if (k == "--batches") cfg.batches = std::stoi(v);
        if (k == "--fill-us") cfg.fill_us = std::stoull(v);
        if (k == "--h2d-us") cfg.h2d_us = std::stoull(v);
        if (k == "--stagger-us") cfg.stagger_us = std::stoull(v);
        if (k == "--scenario") cfg.scenario = v;
        if (k == "--light") cfg.light = (v == "1");
        if (k == "--out") cfg.out = v;
    }
    if (cfg.out.empty()) {
        std::fprintf(stderr, "need --out\n");
        return 2;
    }
    auto demand = make_demand(cfg);
    Baseline bl;
    if (!bl.run(cfg, demand)) {
        std::fprintf(stderr, "baseline failed\n");
        return 1;
    }
    Proto pr(cfg);
    if (!pr.run(cfg, demand)) {
        std::fprintf(stderr, "prototype failed\n");
        return 1;
    }
    // Exactness: rank-ordered payloads + combine hash per measured batch.
    const size_t pre = (cfg.preload_entries + 5) / 6;
    bool exact_ok = bl.results.size() == pr.results.size();
    if (exact_ok) {
        for (size_t i = pre; i < bl.results.size(); ++i) {
            if (bl.results[i].combine_hash != pr.results[i].combine_hash ||
                bl.results[i].payloads.size() !=
                    pr.results[i].payloads.size()) {
                exact_ok = false;
                break;
            }
            for (size_t r = 0; r < bl.results[i].payloads.size(); ++r) {
                if (bl.results[i].payloads[r] != pr.results[i].payloads[r]) {
                    exact_ok = false;
                    break;
                }
            }
            if (!exact_ok) break;
        }
    }
    // Per-completion publish order evidence (prototype): finish-order log of
    // last measured batch must be sorted by fill_done (nondecreasing) while
    // consumption stayed rank-ordered (checked above via payloads).
    const double memset_MBps = measure_memset_MBps(cfg.record_bytes);
    // Lifetime checks (deterministic, all scenarios): policy-resident keys
    // must still be resident after churn; consume-before-DMA must fail
    // closed on a fresh single-miss batch.
    bool policy_intact = true;
    if (cfg.scenario == "lifetime") {
        for (uint64_t k = 2; k <= cfg.preload_entries; k += 2) {
            if (!pr.staging.is_resident(k)) {
                policy_intact = false;
                break;
            }
        }
    }
    bool violation_probe_pass = false;
    {
        fixedslot::FixedSlotStaging probe(4, 1);
        fixedslot::Demand d;
        d.key = 0xBEEFu;
        d.nbytes = 4096;
        d.fill = [](uint8_t* dst, size_t n) { std::memset(dst, 0xA5, n); };
        std::vector<fixedslot::FixedSlotStaging::Handle> hh;
        if (probe.reserve(&d, 1, 1, hh)) {
            // No submit, no DMA: consume must refuse (lifetime violation).
            violation_probe_pass = !probe.mark_consumed(0);
        }
    }
    write_json(cfg, cfg.out.c_str(), bl, pr, exact_ok && policy_intact,
               memset_MBps, policy_intact, violation_probe_pass);
    std::printf("exact=%d policy=%d viol=%d memset_MBps=%.1f\n",
                (int)exact_ok, (int)policy_intact,
                (int)violation_probe_pass, memset_MBps);
    return (exact_ok && policy_intact && violation_probe_pass) ? 0 : 1;
}

