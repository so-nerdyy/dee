// dee/host_pack_cache.h
//
// Bounded LRU cache of packed expert weight bytes (DeepSeek-V4-Flash-0731 FP4
// e2m1 + per-block e8m0 scales, ~12.6 MB per entry).
//
// Why this exists (measured): the engine stages the six packed mmap regions of
// each expert verbatim and the prefetcher gathers them into its pinned slot on
// EVERY transfer.  With a 152.8 GiB checkpoint and only ~30 GiB of page cache,
// the OS LRU evicts an expert's pages between tokens, so each gather re-faults
// cold mmap pages against storage (~4.5 s/expert measured -> ~366 s/token
// decode).  This cache keeps a private RAM copy of each visited pack so
// repeated gathers hit host DRAM.  The unique working set of a 16-token
// response is ~2,365 pairs × 12.6 MB ≈ 30 GiB, so a multi-GiB budget matches
// the box's RAM: after the first touch, the response becomes RAM-resident.
//
// Thread-safety: the engine host path is single-threaded per engine; the cache
// is only touched from get_staging_fp4 / stage paths, all on the same thread.
// The returned buffer pointer stays valid until the next call that evicts this
// key (LRU never evicts the just-inserted/just-touched entry).

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dee {

class HostPackCache {
public:
    static constexpr size_t kMaxFillLanes = 8;
    static constexpr size_t kMaxBatchRequests = 256;

    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        size_t bytes = 0;
        size_t entries = 0;
        uint64_t fill_batches = 0;
        uint64_t concurrent_fill_batches = 0;
        uint64_t fill_requests = 0;
        size_t max_fill_queue_depth = 0;
        size_t max_fill_lanes = 0;
        double fill_batch_wall_ms = 0.0;
        double fill_worker_ms = 0.0;
        double fill_overlap_ms = 0.0;
        // Phase-4 fail-closed + single-lane instrumentation parity.  All are
        // additive; every field above keeps its established semantics.
        // fill_failures: single-lane get() fills that returned false or
        //   threw (nothing is published, so no later lookup can be served a
        //   poisoned zeroed entry).  Also bumped if a reservation insert
        //   itself fails.  Batch fill failures roll back the whole batch and
        //   are not counted here.
        // alloc_failures: get() reservation allocation failures
        //   (entry.bytes.resize throwing, e.g. bad_alloc).  Kept separate
        //   from fill_failures so capacity OOM is distinguishable from a
        //   store-side read failure.
        // size_mismatches: same-key requests refused because the resident
        //   entry's payload size differs (get hit path, and the get_batch
        //   request scan).  Each is also counted as a miss on the get() path.
        // budget_rejections: requests refused because they cannot be admitted
        //   under the byte budget: oversize get(), oversize or capacity-
        //   exceeding get_batch requests, and a get_batch whose victim scan
        //   finds every resident protected by the batch itself.
        // scalar_fills / scalar_fill_ms: executed single-lane get() fills
        //   (attempted count + wall milliseconds, success or failure).
        uint64_t fill_failures = 0;
        uint64_t alloc_failures = 0;
        uint64_t size_mismatches = 0;
        uint64_t budget_rejections = 0;
        uint64_t scalar_fills = 0;
        double scalar_fill_ms = 0.0;
    };

    using BatchFill = bool (*)(void* context, uint8_t* dst, size_t nbytes);
    struct BatchRequest {
        uint64_t key = 0;
        size_t nbytes = 0;
        BatchFill fill = nullptr;
        void* context = nullptr;
        // Stable source-local ordering hint for cold fills.  The cache still
        // returns results in caller order; this only orders the bounded worker
        // queue so positional reads do not inherit rank-order seek churn.
        uint64_t source_order = 0;
    };
    struct BatchResult {
        const uint8_t* data = nullptr;
        bool cache_hit = false;
        bool fill_executed = false;
        bool success = false;
        double fill_milliseconds = 0.0;
        // Worker-observed start offset from the batch submit instant, for
        // occupancy timelines (0 when unmeasured/single-lane inline).
        double fill_start_offset_ms = 0.0;
    };

    HostPackCache() = default;
    ~HostPackCache();
    HostPackCache(const HostPackCache&) = delete;
    HostPackCache& operator=(const HostPackCache&) = delete;

    void set_budget(size_t bytes) { budget_bytes_ = bytes; }
    size_t budget() const { return budget_bytes_; }
    // Configure a persistent bounded worker pool. One lane executes entirely
    // on the caller and is the exact legacy default. Must be called before a
    // fill batch is active.
    bool set_fill_lanes(size_t lanes);
    size_t fill_lanes() const { return fill_lanes_; }

    // Return a pointer to `nbytes` of cached bytes for `key`, running
    // `fill(dst, nbytes)` on a miss (fill must copy exactly nbytes).  Returns
    // nullptr when the entry cannot fit the budget or allocation fails.
    //
    // Fail-closed contract: a fill that returns false or throws publishes
    // NOTHING — no reservation is left behind, the miss is counted exactly
    // once, and fill_failures is incremented — so a later lookup can never be
    // served a poisoned zeroed entry (the pre-repair defect this overload
    // exists to fix).  A resident entry that is not ready, or whose payload
    // size differs from `nbytes`, is likewise refused: nullptr, counted as a
    // miss, plus size_mismatches when the size differs.  Allocation failure
    // during reservation returns nullptr and counts alloc_failures.
    const uint8_t* get(
        uint64_t key, size_t nbytes,
        const std::function<bool(uint8_t* dst, size_t n)>& fill);

    // Legacy void-fill form kept so existing callers compile unchanged: a
    // thin delegate that treats fill completion as success (a throw still
    // fails closed via the bool overload).  New callers that can report
    // failure should prefer the bool-fill overload so a failed
    // materialization cannot publish a readable zeroed entry.
    const uint8_t* get(
        uint64_t key, size_t nbytes,
        const std::function<void(uint8_t* dst, size_t n)>& fill) {
        if (!fill) return nullptr;
        const std::function<bool(uint8_t* dst, size_t n)> wrapper =
            [&fill](uint8_t* dst, size_t n) -> bool {
                fill(dst, n);
                return true;
            };
        return get(key, nbytes, wrapper);
    }

    // Atomically reserve all unique misses, then materialize them through the
    // persistent bounded lane pool. Cache/LRU mutation stays on the caller;
    // workers only write disjoint reserved payloads. Requests must fit the
    // fixed bound and all miss callbacks must succeed, otherwise failed
    // reservations are removed and the call fails closed.
    bool get_batch(const BatchRequest* requests, size_t count,
                   BatchResult* results);

    // Return/touch a complete cached entry without invoking a fill. This is
    // used after get_batch so the unchanged staging/H2D path can consume the
    // prepared bytes without double-counting a second logical cache lookup.
    const uint8_t* get_if_present(uint64_t key, bool count_hit = true);

    // Non-perturbing read-only view for forensic fingerprinting: returns the
    // entry's bytes pointer and sets *nbytes, without touching the LRU or
    // stats.  nullptr when the key is absent or its reservation is unready.
    const uint8_t* peek_bytes(uint64_t key, size_t* nbytes) const;

    bool contains(uint64_t key) const {
        return map_.find(key) != map_.end();
    }

    void clear();
    const Stats& stats() const { return stats_; }
    // Optional fill-path profiler (default nullptr = no observation).
    // Profiling-only: never changes fill behavior, ordering, or results.
    void set_fill_profiler(class StageProfiler* profiler) {
        fill_profiler_ = profiler;
    }
    // Optional per-batch attribution context (token/layer/device, default
    // -1 = unattributed). Profiling-only.
    void set_fill_context(int token, int layer, int device) {
        fill_ctx_token_ = token;
        fill_ctx_layer_ = layer;
        fill_ctx_device_ = device;
    }
    // Optional eviction observer (default nullptr = no observation): invoked
    // once per evicted key with (key, payload nbytes) from both the get()
    // LRU-eviction path and the get_batch() victim loop, on the caller thread
    // after the victim has been removed.  Rollback removals of never-ready
    // reservations and clear() are not evictions and are not observed.
    void set_evict_observer(
        std::function<void(uint64_t key, std::size_t nbytes)> cb) {
        evict_observer_ = std::move(cb);
    }

private:
    struct Entry {
        std::vector<uint8_t> bytes;
        size_t nbytes = 0;
        bool ready = true;
    };
    // map: key -> (payload, LRU position). lru_ front = most recently used.
    using LruIter = typename std::list<uint64_t>::iterator;
    std::unordered_map<uint64_t, std::pair<Entry, LruIter>> map_;
    std::list<uint64_t> lru_;
    size_t budget_bytes_ = 8ull << 30;
    size_t used_bytes_ = 0;
    Stats stats_;

    std::vector<std::thread> fill_workers_;
    size_t fill_lanes_ = 1;
    std::mutex fill_mutex_;
    std::condition_variable fill_start_cv_;
    std::condition_variable fill_done_cv_;
    bool stop_fill_workers_ = false;
    uint64_t fill_epoch_ = 0;
    size_t fill_workers_ready_ = 0;
    size_t fill_workers_pending_ = 0;
    const BatchRequest* active_requests_ = nullptr;
    BatchResult* active_results_ = nullptr;
    std::array<size_t, kMaxBatchRequests> active_fill_order_{};
    size_t active_fill_count_ = 0;
    std::atomic<size_t> next_fill_index_{0};
    // Batch submit instant for per-request start offsets (steady clock).
    std::chrono::steady_clock::time_point active_batch_begin_{};
    uint64_t fill_batch_id_ = 0;
    class StageProfiler* fill_profiler_ = nullptr;
    int fill_ctx_token_ = -1;
    int fill_ctx_layer_ = -1;
    int fill_ctx_device_ = -1;
    std::function<void(uint64_t key, std::size_t nbytes)> evict_observer_;

    void stop_fill_workers();
    void fill_worker_loop();
    void run_fill_lane();
    bool is_batch_key(uint64_t key, const BatchRequest* requests,
                      size_t count) const;
};

}  // namespace dee
