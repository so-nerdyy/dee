// fixed_slot_staging.h — exact-safe fixed-slot / per-completion staging PROTOTYPE.
//
// RESEARCH ONLY (research/phase2-fixedslot-proto). Not production code.
// No CUDA, no engine dependency, stdlib + threads only.
//
// Contract (mirrors the required semantics):
//  - Demand IDs are authoritative input (native router IDs in production).
//    This class never predicts, reorders demand, or changes bytes.
//  - reserve(): O(1) slot-indexed reservation per demanded expert. Hits return
//    immediately; misses pop a free double-buffered slot; victims are chosen
//    only among idle dynamic slots (policy-resident slots are never taken).
//    reserve() NEVER waits for fills.
//  - submit(): each miss fill runs on the lane pool immediately after reserve;
//    each completion is published independently with its finish timestamp.
//  - Host slot lifetime: busy from reserve until DMA-completion callback.
//    Device block lifetime: pinned from H2D completion until mark_consumed().
//  - Consumption is rank-ordered (wait_rank), preserving combine order.
//  - H2D submit is a caller-provided hook so the bench can timestamp submits
//    and simulate DMA without a GPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fixedslot {

struct Demand {
    uint64_t key = 0;
    size_t nbytes = 0;
    // Fill must write exactly nbytes of the expert's exact payload.
    std::function<void(uint8_t* dst, size_t n)> fill;
    bool policy_resident = false;  // placement hint only (Luna vocabulary)
};

struct Completion {
    size_t rank = 0;
    uint64_t key = 0;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    bool was_hit = false;
    uint64_t fill_done_ns = 0;  // steady-clock ns since demand arrival
};

struct Stats {
    uint64_t reserves = 0;
    uint64_t reserve_hits = 0;
    uint64_t reserve_misses = 0;
    uint64_t victim_scans = 0;      // idle-dynamic scans (0 when free slot hit)
    uint64_t victim_taken = 0;
    uint64_t slot_allocs = 0;       // backing-store allocations (amortized)
    uint64_t slot_reuses = 0;       // fills into an already-allocated slot
    uint64_t zero_fill_bytes = 0;   // prototype must keep this 0 (reuse only)
    uint64_t fills_submitted = 0;
    uint64_t fills_completed = 0;
    uint64_t mutex_waits = 0;       // completion-queue waits (consumer side)
};

class TransferDma;  // defined by the bench (simulated H2D + device blocks)

class FixedSlotStaging {
public:
    struct Handle {
        size_t rank = 0;
        uint64_t key = 0;
        size_t nbytes = 0;
        size_t slot = 0;
        bool was_hit = false;
        const uint8_t* data = nullptr;  // valid after wait_rank returns
    };

    // slots: number of double-buffered destination slots.
    // lanes: fill workers (caller acts as one extra lane, mirroring
    //        HostPackCache where the caller shares next_fill_index_).
    FixedSlotStaging(size_t slots, size_t lanes);
    ~FixedSlotStaging();

    FixedSlotStaging(const FixedSlotStaging&) = delete;
    FixedSlotStaging& operator=(const FixedSlotStaging&) = delete;

    // Reserve destinations for one layer's demand in rank order. Returns
    // immediately; fills are NOT waited on. t0_ns anchors completion stamps.
    // H2D-submit hook for hits fires synchronously inside reserve() so hits
    // progress without waiting behind misses.
    bool reserve(const Demand* demands, size_t count, uint64_t t0_ns,
                 std::vector<Handle>& out);

    // Submit all miss fills reserved by the last reserve() call.
    void submit_misses();

    // Block until rank's payload is ready (finish-order publish, rank-order
    // consume). Returns false on fill failure (fail-closed: whole batch bad).
    bool wait_rank(size_t rank);

    // DMA-completion boundary (bench calls from its transfer thread): only
    // after this may the host slot be recycled. Models "host source lifetime
    // until DMA completion".
    void on_dma_complete(size_t rank);

    // Final-consumer completion: releases the device block (device lifetime
    // end) and, for misses, returns the host slot to the free stack. Returns
    // false if this batch's DMA is not yet complete (lifetime violation,
    // fail-closed). Hit ranks only validate DMA (their slot stays resident).
    bool mark_consumed(size_t rank);

    // Current payload pointer for a consumed rank (set by wait_rank for
    // misses, at reserve time for hits).
    const uint8_t* payload(size_t rank);

    // Bench introspection: is key currently resident (LRU + index agree)?
    bool is_resident(uint64_t key);

    // Finish-order completion log (publish order evidence).
    std::vector<Completion> drain_completion_log();

    const Stats& stats() const { return stats_; }
    void set_h2d_hook(std::function<void(size_t rank, uint64_t key,
                                         const uint8_t* data, size_t n,
                                         bool was_hit, uint64_t now_ns)> hook) {
        h2d_hook_ = std::move(hook);
    }

private:
    struct Slot {
        std::vector<uint8_t> buf;
        uint64_t key = 0;
        bool resident = false;         // holds a complete payload
        bool policy_resident = false;  // placement: never victimized
        bool busy = false;             // reserved or filling or DMA-pending
        bool ready = false;            // fill complete, consumable (misses)
        bool failed = false;
        uint64_t dma_gen = 0;          // last DMA-completed batch generation
        uint64_t last_use = 0;
        // Position in lru_ when resident-idle; list end = eviction candidate.
        std::list<uint64_t>::iterator lru_pos;
        bool in_lru = false;
    };
    struct FillWork {
        size_t rank = 0;
        size_t slot = 0;
    };
    void worker_loop();
    bool run_one_fill();  // caller lane + workers cooperatively drain
    bool victimize(size_t need_bytes, size_t& out_slot);

    std::vector<Slot> slots_;
    std::vector<size_t> free_stack_;
    std::vector<Handle> batch_;
    std::vector<FillWork> pending_;
    std::vector<std::function<void(uint8_t*, size_t)>> pending_fills_;
    size_t next_work_ = 0;
    std::vector<Completion> completion_log_;
    std::mutex mutex_;
    std::condition_variable ready_cv_;   // fill-ready publish
    std::condition_variable done_cv_;    // batch drain (submit_misses join)
    std::mutex completion_mutex_;
    std::vector<std::thread> workers_;
    std::mutex work_mutex_;
    std::condition_variable work_cv_;
    bool stop_ = false;
    bool batch_failed_ = false;
    size_t batch_pending_ = 0;
    uint64_t t0_ns_ = 0;
    uint64_t clock_ = 0;
    uint64_t dma_clock_ = 0;    // current batch generation for DMA matching
    uint64_t batch_dma_gen_ = 0;
    std::unordered_map<uint64_t, size_t> resident_index_;
    // Resident-idle slots, front = most recently used (mirrors the HostPackCache
    // LRU so victim choice is O(find-first-evictable), not O(min-search)).
    std::list<uint64_t> lru_;
    std::function<void(size_t, uint64_t, const uint8_t*, size_t, bool, uint64_t)> h2d_hook_;
    Stats stats_;
};

}  // namespace fixedslot
