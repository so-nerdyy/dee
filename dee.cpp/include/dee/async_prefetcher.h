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
    const void* src    = nullptr;  // stable pageable host staging pointer
    size_t    nbytes   = 0;        // destination/cache bytes
    size_t    source_nbytes = 0;   // bytes copied through pinned memory/H2D
    bool      expand_bf16 = false;
    bool      cache_fp16 = false;
    bool      source_pinned = false;
    bool      done     = false;    // mock event "signaled"
    bool      abandoned = false;
    void*     event    = nullptr;  // cudaEvent_t* (DEE_CUDA path only)
    long      id       = -1;
    size_t    staging_slot = static_cast<size_t>(-1);
    bool      cache_pin_held = false;
    bool      active_counted = false;
    int       token = -1;
    int       logical_layer = -1;
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
                  int priority = 0, int token = -1, int logical_layer = -1);

    // CUDA streaming specialization: transfer packed BF16, then expand into
    // the FP32 cache block on the prefetch stream before signaling readiness.
    long prefetch_bf16_to_f32(int layer, int expert, const uint16_t* src,
                              size_t elements, int priority = 0,
                              int token = -1, int logical_layer = -1,
                              bool source_pinned = false);

    // Convert packed BF16 source weights into a persistent FP16 cache block.
    long prefetch_bf16_to_f16(int layer, int expert, const uint16_t* src,
                              size_t elements, int priority = 0,
                              int token = -1, int logical_layer = -1,
                              bool source_pinned = false);

    // Delimit one logical expert batch for duplicate-request accounting.
    void begin_batch() { batch_keys_.clear(); }

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
        uint64_t issued     = 0;  // compatibility alias for requests
        uint64_t requests   = 0;
        uint64_t resident_hits = 0;
        uint64_t inflight_hits = 0;
        uint64_t cold_loads = 0;
        uint64_t duplicate_requests = 0;
        uint64_t waited     = 0;  // wait() calls
        uint64_t fallbacks  = 0;  // wait() had to block on an unfinished xfer
        uint64_t mmap_to_pinned_bytes = 0;
        uint64_t h2d_bytes = 0;
        uint64_t h2d_copies = 0;
    };
    const Stats& stats() const { return stats_; }
    bool accounting_valid() const {
        return stats_.requests == stats_.resident_hits + stats_.inflight_hits + stats_.cold_loads;
    }
    void reset_stats() { stats_ = Stats{}; batch_keys_.clear(); }
    void set_profiler(StageProfiler* profiler) { profiler_ = profiler; }
    bool set_ring_size(size_t ring_size) {
        if (!inflight_.empty() || !staging_slots_.empty() || ring_size == 0) return false;
        ring_size_ = ring_size;
        return true;
    }
    size_t ring_size() const { return ring_size_; }

    bool using_cuda() const { return use_cuda_; }
    void* cuda_stream() const { return stream_; }

private:
    VramCacheManager& cache_;
    bool              use_cuda_ = false;
    size_t            ring_size_ = 64;

    // mock backend state
    std::vector<Transfer>        inflight_;   // ordered submission queue
    std::unordered_map<long, int> key_to_idx_; // ExpertKey -> idx in inflight_
    std::vector<long> batch_keys_;

    struct PinnedStagingSlot {
        void* ptr = nullptr;
        size_t bytes = 0;
        void* device_ptr = nullptr;
        size_t device_bytes = 0;
        bool busy = false;
    };

    // CUDA uses a bounded ring. Source data is copied into a pinned slot on
    // the host, then the pinned slot is copied asynchronously to device. This
    // avoids describing a pageable mmap as an asynchronous NVMe-to-VRAM path.
    void* stream_ = nullptr;  // cudaStream_t
    std::vector<PinnedStagingSlot> staging_slots_;
    size_t next_staging_slot_ = 0;
    size_t active_transfers_ = 0;

    long   next_id_ = 0;
    Stats  stats_{};
    StageProfiler* profiler_ = nullptr;

    long   find_inflight(int layer, int expert) const;
    long   prefetch_impl(int layer, int expert, const void* src,
                         size_t source_nbytes, size_t destination_nbytes,
                         bool expand_bf16, bool cache_fp16, bool source_pinned,
                         int priority, int token,
                         int logical_layer);
    void   drain_until(int idx);   // mock: run copies up to idx (inclusive)
    bool   cuda_init();            // guarded real init
    bool   cuda_submit(long idx);  // guarded real submit + event record
    bool   cuda_wait(long idx, HostWaitReason reason);  // guarded real event sync
    void   release_staging(Transfer& transfer);
    bool   release_transfer(Transfer& transfer);
    void   record_request(RequestKind kind, int token, int logical_layer,
                          int resolved_layer, int expert, int priority,
                          int evicted_layer = -1, int evicted_expert = -1);
};

} // namespace dee
