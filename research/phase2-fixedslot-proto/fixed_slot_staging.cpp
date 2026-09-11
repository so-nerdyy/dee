// fixed_slot_staging.cpp — prototype implementation (research only).
#include "fixed_slot_staging.h"

#include <chrono>

namespace fixedslot {
namespace {
uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
}  // namespace

FixedSlotStaging::FixedSlotStaging(size_t slots, size_t lanes) {
    slots_.resize(slots ? slots : 1);
    for (size_t i = 0; i < slots_.size(); ++i)
        free_stack_.push_back(slots_.size() - 1 - i);
    const size_t workers = lanes > 1 ? lanes - 1 : 0;
    for (size_t i = 0; i < workers; ++i)
        workers_.emplace_back(&FixedSlotStaging::worker_loop, this);
}

FixedSlotStaging::~FixedSlotStaging() {
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        stop_ = true;
    }
    work_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

bool FixedSlotStaging::reserve(const Demand* demands, size_t count,
                               uint64_t t0_ns, std::vector<Handle>& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    batch_.clear();
    batch_.reserve(count);
    pending_.clear();
    pending_fills_.clear();
    completion_log_.clear();
    batch_failed_ = false;
    batch_pending_ = 0;
    next_work_ = 0;
    t0_ns_ = t0_ns;
    ++clock_;
    ++dma_clock_;
    batch_dma_gen_ = dma_clock_;
    out.clear();
    for (size_t r = 0; r < count; ++r) {
        const Demand& d = demands[r];
        ++stats_.reserves;
        auto hit = resident_index_.find(d.key);
        if (hit != resident_index_.end()) {
            Slot& s = slots_[hit->second];
            s.last_use = clock_;
            if (d.policy_resident) s.policy_resident = true;
            if (s.in_lru) {
                lru_.erase(s.lru_pos);
                s.in_lru = false;
            }
            lru_.push_front(d.key);
            s.lru_pos = lru_.begin();
            s.in_lru = true;
            Handle h{r, d.key, d.nbytes, hit->second, true, s.buf.data()};
            batch_.push_back(h);
            out.push_back(h);
            ++stats_.reserve_hits;
            // Hits progress immediately: H2D submits inside reserve(),
            // never behind miss fills.
            if (h2d_hook_)
                h2d_hook_(r, d.key, s.buf.data(), d.nbytes, true, now_ns());
            continue;
        }
        ++stats_.reserve_misses;
        size_t slot = static_cast<size_t>(-1);
        if (!free_stack_.empty()) {
            slot = free_stack_.back();
            free_stack_.pop_back();
        } else if (!victimize(d.nbytes, slot)) {
            batch_.clear();
            return false;  // fail-closed: no idle dynamic slot available
        }
        Slot& s = slots_[slot];
        if (s.buf.size() < d.nbytes) {
            s.buf.resize(d.nbytes);  // amortized: ~once per slot lifetime
            ++stats_.slot_allocs;
        } else {
            ++stats_.slot_reuses;
        }
        // No memset: the fill contract overwrites the full payload (same rule
        // as HostPackCache reused-victim payloads). Short fills fail closed.
        s.key = d.key;
        s.busy = true;  // held through fill AND DMA (lifetime rule)
        s.ready = false;
        s.failed = false;
        s.resident = false;
        s.policy_resident = d.policy_resident;
        s.last_use = clock_;
        Handle h{r, d.key, d.nbytes, slot, false, nullptr};
        batch_.push_back(h);
        out.push_back(h);
        pending_.push_back(FillWork{r, slot});
        pending_fills_.push_back(d.fill);
        ++batch_pending_;
    }
    return true;
}

bool FixedSlotStaging::victimize(size_t need_bytes, size_t& out_slot) {
    // LRU-tail search: first idle dynamic slot from the back wins (usually
    // the first probe). Policy-resident and busy slots are skipped.
    // Counts probes so the bench can compare against LRU victim scans.
    for (auto it = lru_.end(); it != lru_.begin();) {
        --it;
        ++stats_.victim_scans;
        auto f = resident_index_.find(*it);
        if (f == resident_index_.end()) continue;  // stale; compacted below
        Slot& s = slots_[f->second];
        if (s.busy || s.policy_resident || !s.resident) continue;
        if (s.buf.size() < need_bytes) continue;
        lru_.erase(it);
        s.in_lru = false;
        resident_index_.erase(f);
        ++stats_.victim_taken;
        out_slot = f->second;
        return true;
    }
    return false;
}

bool FixedSlotStaging::run_one_fill() {
    size_t idx = static_cast<size_t>(-1);
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        if (next_work_ >= pending_.size()) return false;
        idx = next_work_++;
    }
    const FillWork w = pending_[idx];
    std::function<void(uint8_t*, size_t)> fill;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fill = pending_fills_[idx];
    }
    uint8_t* dst = nullptr;
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dst = slots_[w.slot].buf.data();
        n = batch_[w.rank].nbytes;
    }
    bool ok = false;
    if (fill) {
        fill(dst, n);  // disjoint payload, no locks held
        ok = true;
        std::lock_guard<std::mutex> stat_lock(work_mutex_);
        ++stats_.fills_submitted;
    }
    const uint64_t done_ns = now_ns();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Slot& s = slots_[w.slot];
        if (!ok) {
            s.failed = true;
            batch_failed_ = true;
        } else {
            s.ready = true;
            ++stats_.fills_completed;
        }
        Completion c{w.rank, s.key, s.buf.data(), n,
                     false, done_ns >= t0_ns_ ? done_ns - t0_ns_ : 0};
        {
            std::lock_guard<std::mutex> clock(completion_mutex_);
            completion_log_.push_back(c);
        }
        if (ok && h2d_hook_)
            h2d_hook_(w.rank, s.key, s.buf.data(), n, false, done_ns);
        if (--batch_pending_ == 0) done_cv_.notify_all();
    }
    ready_cv_.notify_all();
    return true;
}

void FixedSlotStaging::submit_misses() {
    work_cv_.notify_all();
    while (run_one_fill()) {
    }
    // Wait for workers to drain (join semantics for the batch's misses).
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&] { return batch_pending_ == 0; });
}

void FixedSlotStaging::worker_loop() {
    std::unique_lock<std::mutex> lock(work_mutex_);
    while (!stop_) {
        work_cv_.wait(lock, [&] { return stop_ || next_work_ < pending_.size(); });
        if (stop_) return;
        lock.unlock();
        while (run_one_fill()) {
        }
        lock.lock();
    }
}

bool FixedSlotStaging::wait_rank(size_t rank) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (rank < batch_.size() && batch_[rank].was_hit) return true;  // resident
    ++stats_.mutex_waits;
    ready_cv_.wait(lock, [&] {
        return rank < batch_.size() && slots_[batch_[rank].slot].ready;
    });
    const Slot& s = slots_[batch_[rank].slot];
    if (batch_failed_ || s.failed) return false;
    batch_[rank].data = s.buf.data();
    return true;
}

void FixedSlotStaging::on_dma_complete(size_t rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (rank < batch_.size())
        slots_[batch_[rank].slot].dma_gen = batch_dma_gen_;
}

bool FixedSlotStaging::mark_consumed(size_t rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (rank >= batch_.size()) return false;
    Handle& h = batch_[rank];
    Slot& s = slots_[h.slot];
    if (s.dma_gen != batch_dma_gen_) return false;  // DMA not complete
    if (h.was_hit) return true;  // resident already; nothing to recycle
    // Device-lifetime end (bench releases its simulated device block before
    // calling here). The slot stays resident and slot-indexed; it is NOT
    // returned to the free stack (that stack holds never-allocated slots
    // only). Future misses reuse it via victimize(), which drops the old
    // resident_index_ entry atomically with the take.
    s.busy = false;
    s.resident = true;
    resident_index_[s.key] = h.slot;
    h.data = s.buf.data();
    if (!s.in_lru) {
        lru_.push_front(s.key);
        s.lru_pos = lru_.begin();
        s.in_lru = true;
    }
    return true;
}

const uint8_t* FixedSlotStaging::payload(size_t rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (rank >= batch_.size()) return nullptr;
    return batch_[rank].data;
}

bool FixedSlotStaging::is_resident(uint64_t key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto f = resident_index_.find(key);
    if (f == resident_index_.end()) return false;
    return slots_[f->second].resident && !slots_[f->second].busy;
}

std::vector<Completion> FixedSlotStaging::drain_completion_log() {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    return completion_log_;
}

}  // namespace fixedslot
