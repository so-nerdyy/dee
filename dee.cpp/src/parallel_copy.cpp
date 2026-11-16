#include "dee/parallel_copy.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace dee {

size_t ParallelCopier::lanes_from_env() {
    const char* v = std::getenv("DEE_PINNED_COPY_THREADS");
    if (!v || !*v) return 4;
    const long n = std::strtol(v, nullptr, 10);
    if (n < 1) return 1;
    return std::min<size_t>(static_cast<size_t>(n), kMaxLanes);
}

ParallelCopier::ParallelCopier(size_t lanes)
    : lanes_(std::max<size_t>(1, std::min(lanes, kMaxLanes))) {
    for (size_t lane = 1; lane < lanes_; ++lane) {
        threads_.emplace_back([this, lane] { worker(lane); });
    }
}

ParallelCopier::~ParallelCopier() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    start_cv_.notify_all();
    for (auto& t : threads_) t.join();
}

// Lane `lane` copies global byte range [lane*total/L, (lane+1)*total/L) of
// the concatenated pieces.  Ranges are disjoint and cover every byte once.
void ParallelCopier::copy_lane(size_t lane) {
    const size_t begin = total_ / lanes_ * lane + std::min(lane, total_ % lanes_);
    const size_t end = begin + total_ / lanes_ + (lane < total_ % lanes_ ? 1 : 0);
    size_t base = 0;
    for (size_t i = 0; i < count_ && base < end; ++i) {
        const CopyPiece& p = pieces_[i];
        const size_t lo = std::max(begin, base), hi = std::min(end, base + p.n);
        if (lo < hi) {
            std::memcpy(static_cast<uint8_t*>(p.dst) + (lo - base),
                        static_cast<const uint8_t*>(p.src) + (lo - base), hi - lo);
        }
        base += p.n;
    }
}

void ParallelCopier::worker(size_t lane) {
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mu_);
            start_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
        }
        copy_lane(lane);
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (--remaining_ == 0) done_cv_.notify_one();
        }
    }
}

void ParallelCopier::run(const CopyPiece* pieces, size_t count) {
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) total += pieces[i].n;
    if (lanes_ == 1 || total < kMinParallelBytes) {
        for (size_t i = 0; i < count; ++i) {
            if (pieces[i].n) std::memcpy(pieces[i].dst, pieces[i].src, pieces[i].n);
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        pieces_ = pieces;
        count_ = count;
        total_ = total;
        remaining_ = lanes_ - 1;
        ++generation_;
    }
    start_cv_.notify_all();
    copy_lane(0);
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [&] { return remaining_ == 0; });
}

}  // namespace dee
