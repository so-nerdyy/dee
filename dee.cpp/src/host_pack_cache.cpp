#include "dee/host_pack_cache.h"
#include "dee/profiling.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <new>

namespace dee {

HostPackCache::~HostPackCache() {
    stop_hint_workers();
    stop_fill_workers();
}

void HostPackCache::stop_fill_workers() {
    {
        std::lock_guard<std::mutex> lock(fill_mutex_);
        stop_fill_workers_ = true;
        ++fill_epoch_;
    }
    fill_start_cv_.notify_all();
    for (auto& worker : fill_workers_) {
        if (worker.joinable()) worker.join();
    }
    fill_workers_.clear();
    stop_fill_workers_ = false;
    fill_workers_ready_ = 0;
    fill_workers_pending_ = 0;
    active_requests_ = nullptr;
    active_results_ = nullptr;
    active_fill_count_ = 0;
}

bool HostPackCache::set_fill_lanes(size_t lanes) {
    if (lanes == 0 || lanes > kMaxFillLanes) return false;
    if (lanes == fill_lanes_) return true;
    stop_fill_workers();
    fill_lanes_ = lanes;
    try {
        fill_workers_.reserve(lanes > 1 ? lanes - 1 : 0);
        for (size_t lane = 1; lane < lanes; ++lane) {
            fill_workers_.emplace_back(&HostPackCache::fill_worker_loop, this);
        }
        // Do not permit a batch to count a just-created lane until it has
        // captured the current epoch and begun waiting for a later one.
        std::unique_lock<std::mutex> lock(fill_mutex_);
        fill_done_cv_.wait(lock, [&] {
            return fill_workers_ready_ == fill_workers_.size();
        });
    } catch (...) {
        stop_fill_workers();
        fill_lanes_ = 1;
        return false;
    }
    return true;
}

void HostPackCache::run_fill_lane() {
    for (;;) {
        const size_t order_index = next_fill_index_.fetch_add(
            1, std::memory_order_relaxed);
        if (order_index >= active_fill_count_) return;
        const size_t index = active_fill_order_[order_index];
        const BatchRequest& request = active_requests_[index];
        BatchResult& result = active_results_[index];
        const auto begin = std::chrono::steady_clock::now();
        // Profiling-only start offset for occupancy timelines (clock read
        // only; the fill itself is untouched).
        result.fill_start_offset_ms =
            std::chrono::duration<double, std::milli>(
                begin - active_batch_begin_).count();
        bool success = false;
        try {
            success = request.fill && result.data &&
                request.fill(request.context,
                             const_cast<uint8_t*>(result.data),
                             request.nbytes);
        } catch (...) {
            success = false;
        }
        result.fill_milliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        result.success = success;
    }
}

void HostPackCache::fill_worker_loop() {
    std::unique_lock<std::mutex> lock(fill_mutex_);
    // A prior pool teardown advances the epoch to wake its old workers.  A
    // newly-created worker must start from that current value, rather than
    // treating the stale teardown epoch as a real batch and dereferencing an
    // unset active request array.
    uint64_t observed_epoch = fill_epoch_;
    ++fill_workers_ready_;
    fill_done_cv_.notify_all();
    for (;;) {
        fill_start_cv_.wait(lock, [&] {
            return stop_fill_workers_ || fill_epoch_ != observed_epoch;
        });
        if (stop_fill_workers_) return;
        observed_epoch = fill_epoch_;
        lock.unlock();
        run_fill_lane();
        lock.lock();
        if (fill_workers_pending_ > 0 && --fill_workers_pending_ == 0) {
            fill_done_cv_.notify_one();
        }
    }
}

bool HostPackCache::is_batch_key(
        uint64_t key, const BatchRequest* requests, size_t count) const {
    for (size_t index = 0; index < count; ++index) {
        if (requests[index].key == key) return true;
    }
    return false;
}

const uint8_t* HostPackCache::get(
    uint64_t key, size_t nbytes,
    const std::function<bool(uint8_t* dst, size_t n)>& fill) {
    if (nbytes == 0 || !fill) return nullptr;
    settle_hint_for_demand(key);
    next_use_.on_use(key);
    auto found = map_.find(key);
    if (found != map_.end()) {
        Entry& resident = found->second.first;
        if (!resident.ready || resident.nbytes != nbytes) {
            // Fail closed: never serve an unready reservation or a same-key
            // entry whose payload size differs from this request.  Counted
            // as a miss; a payload-size mismatch is additionally attributed.
            ++stats_.misses;
            if (resident.nbytes != nbytes) ++stats_.size_mismatches;
            return nullptr;
        }
        // Refresh LRU position; never evict the entry we are about to return.
        lru_.erase(found->second.second);
        lru_.push_front(key);
        found->second.second = lru_.begin();
        ++stats_.hits;
        note_served(resident);
        return resident.bytes.data();
    }

    ++stats_.misses;
    if (nbytes > budget_bytes_) {
        // Cannot ever fit; do not allocate.
        ++stats_.budget_rejections;
        return nullptr;
    }
    if (!make_room(nbytes, nullptr, 0)) {
        // Only bytes reserved for prefetch hints can stand in the way of a
        // record that fits the budget: publish/cancel them and retry.  A
        // demand request always wins over a guess.
        drain_hints();
        if (!make_room(nbytes, nullptr, 0)) {
            ++stats_.budget_rejections;
            stats_.bytes = used_bytes_;
            stats_.entries = map_.size();
            return nullptr;
        }
    }
    Entry entry;
    try {
        entry.bytes.resize(nbytes);
    } catch (...) {
        // Allocation failure: nothing is published, the miss is already
        // counted.  Refresh derived stats for any victims evicted above.
        ++stats_.alloc_failures;
        stats_.bytes = used_bytes_;
        stats_.entries = map_.size();
        return nullptr;
    }
    entry.nbytes = nbytes;
    const auto fill_begin = std::chrono::steady_clock::now();
    bool fill_ok = false;
    {
        DemandScope demand(*this);
        try {
            fill_ok = fill(entry.bytes.data(), nbytes);
        } catch (...) {
            fill_ok = false;
        }
    }
    ++stats_.scalar_fills;
    stats_.scalar_fill_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - fill_begin).count();
    if (!fill_ok) {
        // Fail closed: a fill reporting false or throwing never publishes a
        // readable entry, so no later lookup can be served poisoned bytes.
        ++stats_.fill_failures;
        stats_.bytes = used_bytes_;
        stats_.entries = map_.size();
        return nullptr;
    }

    lru_.push_front(key);
    auto inserted = map_.emplace(
        key, std::make_pair(std::move(entry), lru_.begin()));
    if (!inserted.second) {
        // Cannot happen for a key already screened above; fail closed and
        // drop the duplicate LRU front rather than corrupting the index.
        lru_.pop_front();
        ++stats_.fill_failures;
        stats_.bytes = used_bytes_;
        stats_.entries = map_.size();
        return nullptr;
    }
    used_bytes_ += nbytes;
    stats_.bytes = used_bytes_;
    stats_.entries = map_.size();
    return inserted.first->second.first.bytes.data();
}

const uint8_t* HostPackCache::get_if_present(uint64_t key, bool count_hit) {
    auto found = map_.find(key);
    if (found == map_.end() || !found->second.first.ready) return nullptr;
    lru_.erase(found->second.second);
    lru_.push_front(key);
    found->second.second = lru_.begin();
    if (count_hit) {
        ++stats_.hits;
        note_served(found->second.first);
    }
    return found->second.first.bytes.data();
}

const uint8_t* HostPackCache::peek_bytes(uint64_t key, size_t* nbytes) const {
    auto found = map_.find(key);
    if (found == map_.end() || !found->second.first.ready) return nullptr;
    if (nbytes) *nbytes = found->second.first.nbytes;
    return found->second.first.bytes.data();
}

bool HostPackCache::get_batch(
        const BatchRequest* requests, size_t count, BatchResult* results) {
    if (!requests || !results || count == 0 ||
        count > kMaxBatchRequests) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) results[index] = {};
    // A demand request never waits behind a hint that has not started, and
    // never reads a record a hint lane is already reading: settle every
    // requested key first so hit/miss below is decided from the map alone.
    for (size_t index = 0; index < count; ++index) {
        settle_hint_for_demand(requests[index].key);
    }
    // Profiling-only phase clock (steady_clock reads only; no behavior change).
    const auto phase_t0 = std::chrono::steady_clock::now();
    const uint64_t evict_before_batch = stats_.evictions;

    size_t additional_bytes = 0;
    size_t unique_misses = 0;
    for (size_t index = 0; index < count; ++index) {
        const BatchRequest& request = requests[index];
        if (request.nbytes == 0) return false;
        if (request.nbytes > budget_bytes_) {
            // Request can never be admitted under the byte budget.
            ++stats_.budget_rejections;
            return false;
        }
        size_t duplicate = index;
        for (size_t prior = 0; prior < index; ++prior) {
            if (requests[prior].key == request.key) {
                duplicate = prior;
                break;
            }
        }
        if (duplicate != index) {
            ++stats_.hits;
            results[index].cache_hit = true;
            continue;
        }
        next_use_.on_use(request.key);
        auto found = map_.find(request.key);
        if (found != map_.end() && found->second.first.ready) {
            if (found->second.first.nbytes != request.nbytes) {
                // Same-key different-size request: fail closed like get().
                ++stats_.size_mismatches;
                return false;
            }
            lru_.erase(found->second.second);
            lru_.push_front(request.key);
            found->second.second = lru_.begin();
            ++stats_.hits;
            note_served(found->second.first);
            results[index].data = found->second.first.bytes.data();
            results[index].cache_hit = true;
            results[index].success = true;
            continue;
        }
        if (!request.fill) return false;
        if (additional_bytes > budget_bytes_ - request.nbytes) {
            // The unique misses of this batch alone exceed the byte budget;
            // the request set can never be admitted.
            ++stats_.budget_rejections;
            return false;
        }
        additional_bytes += request.nbytes;
        ++unique_misses;
        ++stats_.misses;
    }

    // Protect every key in the incoming batch while selecting victims (by
    // the active policy).  This makes all reservations stable until their
    // disjoint fills finish.
    if (!make_room(additional_bytes, requests, count)) {
        // Bytes reserved for prefetch hints may be what blocks a batch that
        // fits the budget: publish/cancel them and retry once.
        drain_hints();
        if (!make_room(additional_bytes, requests, count)) {
            // Every remaining resident is a protected batch key; the request
            // set cannot be admitted under the byte budget.  Refresh derived
            // stats for victims already removed above before bailing.
            ++stats_.budget_rejections;
            stats_.bytes = used_bytes_;
            stats_.entries = map_.size();
            return false;
        }
    }

    try {
        for (size_t index = 0; index < count; ++index) {
            if (results[index].cache_hit) continue;
            bool duplicate = false;
            for (size_t prior = 0; prior < index; ++prior) {
                if (requests[prior].key == requests[index].key) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            Entry entry;
            entry.bytes.resize(requests[index].nbytes);
            entry.nbytes = requests[index].nbytes;
            entry.ready = false;
            lru_.push_front(requests[index].key);
            auto inserted = map_.emplace(
                requests[index].key,
                std::make_pair(std::move(entry), lru_.begin()));
            if (!inserted.second) {
                // The key was already resident (e.g. an unready reservation
                // predating this batch).  Drop the duplicate LRU front and
                // roll back every reservation this call already published,
                // mirroring the bad_alloc handler, before failing closed.
                lru_.pop_front();
                for (size_t r = 0; r < count; ++r) {
                    if (!results[r].fill_executed) continue;
                    auto prior = map_.find(requests[r].key);
                    if (prior == map_.end()) continue;
                    used_bytes_ -= prior->second.first.nbytes;
                    lru_.erase(prior->second.second);
                    map_.erase(prior);
                    results[r] = {};
                }
                stats_.bytes = used_bytes_;
                stats_.entries = map_.size();
                return false;
            }
            used_bytes_ += requests[index].nbytes;
            results[index].data = inserted.first->second.first.bytes.data();
            results[index].fill_executed = true;
        }
    } catch (const std::bad_alloc&) {
        for (size_t index = 0; index < count; ++index) {
            if (!results[index].fill_executed) continue;
            auto found = map_.find(requests[index].key);
            if (found == map_.end()) continue;
            used_bytes_ -= found->second.first.nbytes;
            lru_.erase(found->second.second);
            map_.erase(found);
            results[index] = {};
        }
        stats_.bytes = used_bytes_;
        stats_.entries = map_.size();
        return false;
    }
    const auto phase_reserved = std::chrono::steady_clock::now();

    const auto batch_begin = std::chrono::steady_clock::now();
    active_batch_begin_ = batch_begin;
    active_requests_ = requests;
    active_results_ = results;
    active_fill_count_ = 0;
    for (size_t index = 0; index < count; ++index) {
        if (results[index].fill_executed) {
            active_fill_order_[active_fill_count_++] = index;
        }
    }
    // DEE4 trace records are laid out by record index, while decode requests
    // arrive in authoritative router-rank order.  Stable sorting only the
    // private fill queue makes each worker consume monotonic positional-read
    // work without changing request/result order, cache identity, or later
    // H2D/compute order. Equal/default hints retain the legacy order exactly.
    std::sort(
        active_fill_order_.begin(),
        active_fill_order_.begin() + active_fill_count_,
        [&](size_t lhs, size_t rhs) {
            if (requests[lhs].source_order != requests[rhs].source_order) {
                return requests[lhs].source_order < requests[rhs].source_order;
            }
            return lhs < rhs;
        });
    next_fill_index_.store(0, std::memory_order_relaxed);
    const auto wake_begin = std::chrono::steady_clock::now();
    // Hint lanes take no new work while this batch is reading.
    DemandScope demand(*this);
    if (!fill_workers_.empty() && unique_misses > 1) {
        {
            std::lock_guard<std::mutex> lock(fill_mutex_);
            fill_workers_pending_ = fill_workers_.size();
            ++fill_epoch_;
        }
        fill_start_cv_.notify_all();
        run_fill_lane();
        std::unique_lock<std::mutex> lock(fill_mutex_);
        fill_done_cv_.wait(lock, [&] { return fill_workers_pending_ == 0; });
    } else {
        run_fill_lane();
    }
    const auto wake_end = std::chrono::steady_clock::now();
    const double batch_wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - batch_begin).count();

    bool success = true;
    double worker_ms = 0.0;
    for (size_t index = 0; index < count; ++index) {
        if (!results[index].fill_executed) continue;
        worker_ms += results[index].fill_milliseconds;
        auto found = map_.find(requests[index].key);
        if (!results[index].success || found == map_.end()) {
            success = false;
            if (found != map_.end()) {
                used_bytes_ -= found->second.first.nbytes;
                lru_.erase(found->second.second);
                map_.erase(found);
            }
            results[index].data = nullptr;
            continue;
        }
        found->second.first.ready = true;
    }
    // get_batch is fail-closed as a unit.  A callback failure must not leave
    // the rest of this batch as silently usable partial state: the caller
    // receives failure and every new reservation from this batch is removed.
    // Entries that predated the batch were never marked fill_executed, so they
    // remain intact.
    if (!success) {
        for (size_t index = 0; index < count; ++index) {
            if (!results[index].fill_executed) continue;
            auto found = map_.find(requests[index].key);
            if (found != map_.end()) {
                used_bytes_ -= found->second.first.nbytes;
                lru_.erase(found->second.second);
                map_.erase(found);
            }
            results[index].data = nullptr;
            results[index].success = false;
        }
    }
    // Resolve duplicate result pointers only after their source reservation is
    // known complete (or failed).
    for (size_t index = 0; index < count; ++index) {
        for (size_t prior = 0; prior < index; ++prior) {
            if (requests[prior].key != requests[index].key) continue;
            results[index].data = results[prior].data;
            results[index].success = results[prior].success;
            break;
        }
    }
    active_requests_ = nullptr;
    active_results_ = nullptr;
    active_fill_count_ = 0;

    ++stats_.fill_batches;
    if (unique_misses > 1 && fill_lanes_ > 1) {
        ++stats_.concurrent_fill_batches;
    }
    stats_.fill_requests += unique_misses;
    stats_.max_fill_queue_depth = std::max(
        stats_.max_fill_queue_depth, unique_misses);
    stats_.max_fill_lanes = std::max(
        stats_.max_fill_lanes,
        std::min(fill_lanes_, std::max<size_t>(1, unique_misses)));
    stats_.fill_batch_wall_ms += batch_wall_ms;
    stats_.fill_worker_ms += worker_ms;
    stats_.fill_overlap_ms += std::max(0.0, worker_ms - batch_wall_ms);
    stats_.bytes = used_bytes_;
    stats_.entries = map_.size();
    // Profiling-only batch record (clock reads + small vector; the fill
    // itself is untouched). phases: submit->dedup->reserve->wake->wait.
    if (fill_profiler_ != nullptr) {
        FillBatchRecord record;
        record.batch_id = ++fill_batch_id_;
        record.token = fill_ctx_token_;
        record.layer = fill_ctx_layer_;
        record.device_id = fill_ctx_device_;
        record.misses = unique_misses;
        record.bytes = additional_bytes;
        record.evictions = stats_.evictions - evict_before_batch;
        record.lanes = std::min(fill_lanes_, std::max<size_t>(1, unique_misses));
        record.reserve_ms = std::chrono::duration<double, std::milli>(
            phase_reserved - phase_t0).count();
        record.wake_ms = std::chrono::duration<double, std::milli>(
            wake_end - wake_begin).count();
        record.batch_wall_ms = batch_wall_ms;
        record.worker_sum_ms = worker_ms;
        for (size_t index = 0; index < count; ++index) {
            FillRequestSample sample;
            sample.key = requests[index].key;
            sample.start_offset_ms = results[index].fill_start_offset_ms;
            sample.service_ms = results[index].fill_milliseconds;
            sample.nbytes = requests[index].nbytes;
            sample.cache_hit = results[index].cache_hit;
            sample.success = results[index].success;
            record.requests.push_back(sample);
        }
        fill_profiler_->note_fill_batch(record);
    }
    return success;
}

void HostPackCache::clear() {
    // No hint lane may still be writing when the cache is wiped; whatever
    // they finished is published first and then dropped with everything else.
    drain_hints();
    next_use_.clear();
    map_.clear();
    lru_.clear();
    used_bytes_ = 0;
    stats_.bytes = 0;
    stats_.entries = 0;
    stats_.evictions = 0;
    stats_.hits = 0;
    stats_.misses = 0;
    stats_.fill_batches = 0;
    stats_.concurrent_fill_batches = 0;
    stats_.fill_requests = 0;
    stats_.max_fill_queue_depth = 0;
    stats_.max_fill_lanes = 0;
    stats_.fill_batch_wall_ms = 0.0;
    stats_.fill_worker_ms = 0.0;
    stats_.fill_overlap_ms = 0.0;
    stats_.fill_failures = 0;
    stats_.alloc_failures = 0;
    stats_.size_mismatches = 0;
    stats_.budget_rejections = 0;
    stats_.scalar_fills = 0;
    stats_.scalar_fill_ms = 0.0;
    stats_.hints_submitted = 0;
    stats_.hints_rejected = 0;
    stats_.hints_cancelled = 0;
    stats_.hint_reads = 0;
    stats_.hint_read_ms = 0.0;
    stats_.hint_failures = 0;
    stats_.hint_hits = 0;
    stats_.hint_waits = 0;
    stats_.hints_wasted = 0;
    stats_.hint_reserved_bytes = hint_reserved_bytes_;
}

// ---------------------------------------------------------------------------
// Policy-aware eviction (caller thread only)
// ---------------------------------------------------------------------------
void HostPackCache::note_served(Entry& entry) {
    if (!entry.hinted_unused) return;
    entry.hinted_unused = false;
    ++stats_.hint_hits;
}

void HostPackCache::evict_entry(uint64_t key) {
    auto found = map_.find(key);
    if (found == map_.end()) return;
    const size_t nbytes = found->second.first.nbytes;
    if (found->second.first.hinted_unused) ++stats_.hints_wasted;
    used_bytes_ -= nbytes;
    lru_.erase(found->second.second);
    map_.erase(found);
    ++stats_.evictions;
    if (evict_observer_) evict_observer_(key, nbytes);
}

bool HostPackCache::make_room(
        size_t need, const BatchRequest* protect, size_t protect_count) {
    const auto fits = [&]() {
        return need <= budget_bytes_ &&
               used_bytes_ + hint_reserved_bytes_ <= budget_bytes_ - need;
    };
    if (fits()) return true;
    if (next_use_active()) {
        // Furthest expected next use leaves first.  One ranking pass serves
        // every victim of this call: ranks only move when a record is used
        // or the layer advances, and neither happens while evicting.
        struct Candidate {
            double rank;
            uint64_t key;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(map_.size());
        for (const auto& kv : map_) {
            if (!kv.second.first.ready) continue;
            if (protect && is_batch_key(kv.first, protect, protect_count)) {
                continue;
            }
            candidates.push_back(
                {next_use_.victim_rank(kv.first), kv.first});
        }
        // Equal ranks fall back to key order so the choice never depends on
        // hash-map iteration order.
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& lhs, const Candidate& rhs) {
                      if (lhs.rank != rhs.rank) return lhs.rank > rhs.rank;
                      return lhs.key < rhs.key;
                  });
        for (const Candidate& candidate : candidates) {
            if (fits()) break;
            evict_entry(candidate.key);
        }
        return fits();
    }
    // LRU: walk from the least recently used end, skipping protected keys.
    auto it = lru_.end();
    while (!fits() && it != lru_.begin()) {
        --it;
        const uint64_t key = *it;
        auto found = map_.find(key);
        if (found == map_.end()) {
            it = lru_.erase(it);
            continue;
        }
        if (!found->second.first.ready ||
            (protect && is_batch_key(key, protect, protect_count))) {
            continue;
        }
        it = std::next(it);
        evict_entry(key);
    }
    return fits();
}

// ---------------------------------------------------------------------------
// Prefetch hints
// ---------------------------------------------------------------------------
HostPackCache::DemandScope::DemandScope(HostPackCache& cache) : cache_(cache) {
    std::lock_guard<std::mutex> lock(cache_.hint_mutex_);
    ++cache_.hint_demand_active_;
}

HostPackCache::DemandScope::~DemandScope() {
    {
        std::lock_guard<std::mutex> lock(cache_.hint_mutex_);
        --cache_.hint_demand_active_;
    }
    cache_.hint_work_cv_.notify_all();
}

size_t HostPackCache::hint_reserve_cap() const {
    if (hint_reserve_limit_ == 0) return budget_bytes_ / 4;
    return std::min(hint_reserve_limit_, budget_bytes_);
}

void HostPackCache::set_hint_queue_order(HintQueueOrder order) {
    std::lock_guard<std::mutex> lock(hint_mutex_);
    hint_order_ = order;
}

size_t HostPackCache::pending_hints() const {
    std::lock_guard<std::mutex> lock(hint_mutex_);
    return hint_jobs_.size();
}

void HostPackCache::hint_worker_loop() {
    std::unique_lock<std::mutex> lock(hint_mutex_);
    for (;;) {
        hint_work_cv_.wait(lock, [&] {
            return hint_stop_ ||
                   (!hint_queue_.empty() && hint_demand_active_ == 0);
        });
        if (hint_stop_) return;
        // Nearest deadline first (ties: arrival order); Fifo ignores deadlines.
        size_t best = 0;
        for (size_t index = 1; index < hint_queue_.size(); ++index) {
            const HintJob& candidate = *hint_queue_[index];
            const HintJob& current = *hint_queue_[best];
            const bool sooner =
                hint_order_ == HintQueueOrder::Deadline &&
                candidate.deadline != current.deadline
                    ? candidate.deadline < current.deadline
                    : candidate.seq < current.seq;
            if (sooner) best = index;
        }
        HintJobPtr job = hint_queue_[best];
        hint_queue_[best] = hint_queue_.back();
        hint_queue_.pop_back();
        job->state = HintJob::State::Running;
        lock.unlock();

        bool ok = false;
        const auto begin = std::chrono::steady_clock::now();
        try {
            job->bytes.resize(job->nbytes);
            ok = job->fill && job->fill(job->bytes.data(), job->nbytes);
        } catch (...) {
            ok = false;
        }
        const double milliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();

        lock.lock();
        job->milliseconds = milliseconds;
        job->fill = nullptr;
        if (!ok) std::vector<uint8_t>().swap(job->bytes);
        job->state = ok ? HintJob::State::Done : HintJob::State::Failed;
        hint_finished_.push_back(job);
        hint_done_cv_.notify_all();
    }
}

void HostPackCache::stop_hint_workers() {
    {
        std::lock_guard<std::mutex> lock(hint_mutex_);
        hint_stop_ = true;
    }
    hint_work_cv_.notify_all();
    for (auto& worker : hint_workers_) {
        if (worker.joinable()) worker.join();
    }
    hint_workers_.clear();
    std::lock_guard<std::mutex> lock(hint_mutex_);
    hint_stop_ = false;
    // Whatever was queued or finished but unpublished is dropped: nothing a
    // hint read is ever visible unless absorb_hints() published it.
    stats_.hints_cancelled += hint_jobs_.size();
    hint_jobs_.clear();
    hint_queue_.clear();
    hint_finished_.clear();
    hint_reserved_bytes_ = 0;
    stats_.hint_reserved_bytes = 0;
}

bool HostPackCache::set_hint_lanes(size_t lanes) {
    if (lanes > kMaxFillLanes) return false;
    if (lanes == hint_workers_.size()) return true;
    stop_hint_workers();
    try {
        hint_workers_.reserve(lanes);
        for (size_t lane = 0; lane < lanes; ++lane) {
            hint_workers_.emplace_back(&HostPackCache::hint_worker_loop, this);
        }
    } catch (...) {
        stop_hint_workers();
        return false;
    }
    return true;
}

void HostPackCache::cancel_queued_hint_locked(const HintJobPtr& job) {
    for (size_t index = 0; index < hint_queue_.size(); ++index) {
        if (hint_queue_[index] != job) continue;
        hint_queue_[index] = hint_queue_.back();
        hint_queue_.pop_back();
        break;
    }
    hint_jobs_.erase(job->key);
}

size_t HostPackCache::submit_hints(HintRequest* requests, size_t count) {
    if (!requests || count == 0) return 0;
    absorb_hints();
    size_t accepted = 0;
    for (size_t index = 0; index < count; ++index) {
        HintRequest& request = requests[index];
        bool ok = !hint_workers_.empty() && request.fill &&
                  request.nbytes != 0 && request.nbytes <= budget_bytes_ &&
                  map_.find(request.key) == map_.end() &&
                  request.nbytes <= hint_reserve_cap() &&
                  hint_reserved_bytes_ <= hint_reserve_cap() - request.nbytes;
        if (ok) {
            std::lock_guard<std::mutex> lock(hint_mutex_);
            ok = hint_jobs_.find(request.key) == hint_jobs_.end();
        }
        if (ok && used_bytes_ + hint_reserved_bytes_ >
                      budget_bytes_ - request.nbytes) {
            if (next_use_active()) {
                // A guess may only displace a record that is expected to be
                // needed LATER than the hinted one.
                const double hint_rank =
                    next_use_.expected_next_use(request.key);
                double worst = -1.0;
                uint64_t victim = 0;
                bool have = false;
                for (const auto& kv : map_) {
                    if (!kv.second.first.ready) continue;
                    const double rank = next_use_.victim_rank(kv.first);
                    if (!have || rank > worst ||
                        (rank == worst && kv.first < victim)) {
                        worst = rank;
                        victim = kv.first;
                        have = true;
                    }
                }
                ok = have && worst > hint_rank &&
                     map_.find(victim)->second.first.nbytes >= request.nbytes;
                if (ok) evict_entry(victim);
                ok = ok && used_bytes_ + hint_reserved_bytes_ <=
                               budget_bytes_ - request.nbytes;
            } else {
                // LRU: a guess may not push out a record of the layer that
                // is running now (it may be prepared but not staged yet).
                ok = !(next_use_.enabled() && !lru_.empty() &&
                       next_use_.victim_rank(lru_.back()) < 0.0) &&
                     make_room(request.nbytes, nullptr, 0);
            }
        }
        if (!ok) {
            ++stats_.hints_rejected;
            continue;
        }
        auto job = std::make_shared<HintJob>();
        job->key = request.key;
        job->nbytes = request.nbytes;
        job->fill = std::move(request.fill);
        job->deadline = request.deadline;
        hint_reserved_bytes_ += request.nbytes;
        {
            std::lock_guard<std::mutex> lock(hint_mutex_);
            job->seq = ++hint_seq_;
            hint_jobs_[job->key] = job;
            hint_queue_.push_back(job);
        }
        ++stats_.hints_submitted;
        ++accepted;
    }
    if (accepted != 0) hint_work_cv_.notify_all();
    stats_.bytes = used_bytes_;
    stats_.entries = map_.size();
    stats_.hint_reserved_bytes = hint_reserved_bytes_;
    return accepted;
}

size_t HostPackCache::expire_hints(uint64_t now) {
    if (hint_reserved_bytes_ == 0) return 0;
    size_t cancelled = 0;
    size_t released = 0;
    {
        std::lock_guard<std::mutex> lock(hint_mutex_);
        for (size_t index = 0; index < hint_queue_.size();) {
            const HintJobPtr job = hint_queue_[index];
            if (job->deadline > now) {
                ++index;
                continue;
            }
            hint_queue_[index] = hint_queue_.back();
            hint_queue_.pop_back();
            hint_jobs_.erase(job->key);
            released += job->nbytes;
            ++cancelled;
        }
    }
    hint_reserved_bytes_ -= released;
    stats_.hints_cancelled += cancelled;
    stats_.hint_reserved_bytes = hint_reserved_bytes_;
    return cancelled;
}

size_t HostPackCache::absorb_hints() {
    if (hint_reserved_bytes_ == 0) return 0;
    std::vector<HintJobPtr> finished;
    {
        std::lock_guard<std::mutex> lock(hint_mutex_);
        if (hint_finished_.empty()) return 0;
        finished.swap(hint_finished_);
        for (const HintJobPtr& job : finished) hint_jobs_.erase(job->key);
    }
    size_t published = 0;
    for (const HintJobPtr& job : finished) {
        hint_reserved_bytes_ -= job->nbytes;
        const bool ok = job->state == HintJob::State::Done;
        if (hint_read_observer_) {
            hint_read_observer_(job->key, job->nbytes, job->milliseconds, ok);
        }
        if (!ok) {
            // Fail closed: a failed hint publishes nothing; a later demand
            // request reads the record through the normal path.
            ++stats_.hint_failures;
            continue;
        }
        ++stats_.hint_reads;
        stats_.hint_read_ms += job->milliseconds;
        if (map_.find(job->key) != map_.end() ||
            job->nbytes > budget_bytes_ ||
            used_bytes_ + hint_reserved_bytes_ > budget_bytes_ - job->nbytes) {
            ++stats_.hints_wasted;
            continue;
        }
        Entry entry;
        entry.bytes = std::move(job->bytes);
        entry.nbytes = job->nbytes;
        entry.ready = true;
        entry.hinted_unused = true;
        lru_.push_front(job->key);
        auto inserted = map_.emplace(
            job->key, std::make_pair(std::move(entry), lru_.begin()));
        if (!inserted.second) {
            lru_.pop_front();
            ++stats_.hints_wasted;
            continue;
        }
        used_bytes_ += job->nbytes;
        ++published;
    }
    stats_.bytes = used_bytes_;
    stats_.entries = map_.size();
    stats_.hint_reserved_bytes = hint_reserved_bytes_;
    return published;
}

void HostPackCache::settle_hint_for_demand(uint64_t key) {
    if (hint_reserved_bytes_ == 0) return;   // no hint outstanding at all
    bool cancelled = false;
    size_t released = 0;
    {
        std::unique_lock<std::mutex> lock(hint_mutex_);
        auto found = hint_jobs_.find(key);
        if (found != hint_jobs_.end()) {
            HintJobPtr job = found->second;
            if (job->state == HintJob::State::Queued) {
                // Not started: the demand path reads it now, on its own lanes.
                cancel_queued_hint_locked(job);
                cancelled = true;
                released = job->nbytes;
            } else if (job->state == HintJob::State::Running) {
                // Already being read: wait for that read instead of issuing
                // a second one for the same record.
                ++stats_.hint_waits;
                hint_done_cv_.wait(lock, [&] {
                    return job->state != HintJob::State::Running;
                });
            }
        }
    }
    if (cancelled) {
        hint_reserved_bytes_ -= released;
        ++stats_.hints_cancelled;
        stats_.hint_reserved_bytes = hint_reserved_bytes_;
    }
    absorb_hints();
}

void HostPackCache::drain_hints() {
    if (hint_reserved_bytes_ == 0) return;
    size_t cancelled = 0;
    size_t released = 0;
    {
        std::unique_lock<std::mutex> lock(hint_mutex_);
        for (const HintJobPtr& job : hint_queue_) {
            hint_jobs_.erase(job->key);
            released += job->nbytes;
            ++cancelled;
        }
        hint_queue_.clear();
        hint_done_cv_.wait(lock, [&] {
            for (const auto& kv : hint_jobs_) {
                if (kv.second->state == HintJob::State::Running) return false;
            }
            return true;
        });
    }
    hint_reserved_bytes_ -= released;
    stats_.hints_cancelled += cancelled;
    stats_.hint_reserved_bytes = hint_reserved_bytes_;
    absorb_hints();
}

}  // namespace dee
