// dee/parallel_copy.h
//
// Multi-lane host memcpy for the pageable -> pinned staging gather.
//
// Why (measured, p6m-l4-r5 b=8 h64): the prefetcher copied every expert
// (~13.4 MB FP4 pack) from the host pack into its pinned slot with ONE
// memcpy on the engine thread: mmap_to_pinned was 262 s CPU of a 640 s row,
// ~5.3 GB/s, below the L4's PCIe rate, so the CPU copy (not H2D) set the
// step time.  Splitting each copy across a few persistent lanes is
// byte-identical (same bytes, same destination offsets) and ~3x faster on
// an 8-vCPU box (scratch bench: 6.7 -> 20.3 GB/s at 4 lanes).
//
// Lanes: DEE_PINNED_COPY_THREADS (1 = old single memcpy; default 4; max 16).
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace dee {

struct CopyPiece {
    void* dst = nullptr;
    const void* src = nullptr;
    size_t n = 0;
};

class ParallelCopier {
public:
    // Copies below this many total bytes run inline on the caller.
    static constexpr size_t kMinParallelBytes = 1u << 20;
    static constexpr size_t kMaxLanes = 16;

    explicit ParallelCopier(size_t lanes);
    ~ParallelCopier();
    ParallelCopier(const ParallelCopier&) = delete;
    ParallelCopier& operator=(const ParallelCopier&) = delete;

    // Copy every piece; returns when all bytes are in place.  Pieces must
    // not overlap.  Not re-entrant (one caller thread at a time).
    void run(const CopyPiece* pieces, size_t count);
    size_t lanes() const { return lanes_; }

    static size_t lanes_from_env();

private:
    void worker(size_t lane);
    void copy_lane(size_t lane);

    size_t lanes_ = 1;
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    uint64_t generation_ = 0;
    size_t remaining_ = 0;
    bool stop_ = false;
    const CopyPiece* pieces_ = nullptr;
    size_t count_ = 0;
    size_t total_ = 0;
};

}  // namespace dee
