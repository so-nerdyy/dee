// dee/async_prefetcher.h
//
// Step 6 — AsyncPrefetcher (DEE streaming layer).
//
// Streams expert weights from the WeightMmap host pointers into the
// VramCacheManager's VRAM arena on a SECONDARY stream, decoupled from the
// compute stream. Each in-flight transfer is tagged with an event; the compute
// stream only waits on the specific expert it needs *now* (sync_fallback), so
// independent transfers never stall compute.
//
// Backend seam (same idea as VramCacheManager::Arena): a mock stream/event on
// CPU-only boxes (this WSL build), and a real cudaStream_t / cudaMemcpyAsync /
// cudaEvent_t under DEE_CUDA=ON. The control flow is identical across backends,
// so the streaming + synchronization logic is fully unit-tested without a CUDA
// toolkit. The real CUDA calls are guarded by #ifdef DEE_CUDA.

#pragma once

#include "dee/vram_cache.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace dee {

// A single in-flight weight transfer.
struct Transfer {
    ExpertKey key{};
    void*     dst      = nullptr;  // VRAM arena slot (from VramCacheManager)
    const void* src    = nullptr;  // WeightMmap host pointer
    size_t    nbytes   = 0;
    bool      done     = false;    // mock event "signaled"
    bool      abandoned = false;
    void*     event    = nullptr;  // cudaEvent_t* (DEE_CUDA path only)
    long      id       = -1;
    size_t    staging_slot = static_cast<size_t>(-1);
    bool      cache_pin_held = false;
};

// ---------------------------------------------------------------------------
// AsyncPrefetcher: issues copies on a secondary stream + per-transfer events.
// ---------------------------------------------------------------------------
class AsyncPrefetcher {
public:
    // Construct with the cache it fills. `nstream_slots` reserves the mock
    // staging/event pool (mirrors llama.cpp's ring of host buffers + events).
    explicit AsyncPrefetcher(VramCacheManager& cache, size_t ring_size = 64);
    ~AsyncPrefetcher();

    // Initialize the stream/event backend. `cuda` selects the real CUDA path
    // (only available when built with DEE_CUDA=ON); otherwise a mock stream.
    bool init(bool use_cuda = false);

    // Issue an async copy of `nbytes` from `src` (WeightMmap host ptr) into the
    // cache slot for (layer, expert). Reserves the VRAM slot via the cache.
    // Returns a transfer id (>=0) or -1 on failure. `priority` feeds the cache.
    long prefetch(int layer, int expert, const void* src, size_t nbytes,
                  int priority = 0);

    // Ensure (layer, expert) is resident AND its transfer has completed before
    // compute uses it. Blocks only on this one expert (sync fallback). Returns
    // true if ready.
    bool wait(int layer, int expert);

    // Drain all in-flight transfers (e.g. between sequences). Real CUDA path
    // calls cudaStreamSynchronize; mock drains the queue in order.
    void synchronize_all();

    // Drain then drop all in-flight transfers + events. Events are never
    // destroyed while their copy can still be executing.
    void reset();

    size_t in_flight() const { return inflight_.size(); }

    // Stats (mirrors prototype's fallback accounting)
    struct Stats {
        uint64_t issued     = 0;
        uint64_t waited     = 0;  // wait() calls
        uint64_t fallbacks  = 0;  // wait() had to block on an unfinished xfer
    };
    const Stats& stats() const { return stats_; }

    bool using_cuda() const { return use_cuda_; }

private:
    VramCacheManager& cache_;
    bool              use_cuda_ = false;
    size_t            ring_size_ = 64;

    // mock backend state
    std::vector<Transfer>        inflight_;   // ordered submission queue
    std::unordered_map<long, int> key_to_idx_; // ExpertKey -> idx in inflight_

    struct PinnedStagingSlot {
        void* ptr = nullptr;
        size_t bytes = 0;
        bool busy = false;
    };

    // CUDA uses a bounded ring. Source data is copied into a pinned slot on
    // the host, then the pinned slot is copied asynchronously to device. This
    // avoids describing a pageable mmap as an asynchronous NVMe-to-VRAM path.
    void* stream_ = nullptr;  // cudaStream_t
    std::vector<PinnedStagingSlot> staging_slots_;
    size_t next_staging_slot_ = 0;

    long   next_id_ = 0;
    Stats  stats_{};

    long   find_inflight(int layer, int expert) const;
    void   drain_until(int idx);   // mock: run copies up to idx (inclusive)
    bool   cuda_init();            // guarded real init
    bool   cuda_submit(long idx);  // guarded real submit + event record
    bool   cuda_wait(long idx);    // guarded real event sync
    bool   release_transfer(Transfer& transfer);
};

} // namespace dee
