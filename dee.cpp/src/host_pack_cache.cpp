#include "dee/host_pack_cache.h"
#include "dee/profiling.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <new>

namespace dee {

HostPackCache::~HostPackCache() { stop_fill_workers(); }

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
    auto found = map_.find(key);
    if (found != map_.end()) {
        const Entry& resident = found->second.first;
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
        return resident.bytes.data();
    }

    ++stats_.misses;
    if (nbytes > budget_bytes_) {
        // Cannot ever fit; do not allocate.
        ++stats_.budget_rejections;
        return nullptr;
    }
    while (used_bytes_ + nbytes > budget_bytes_ && !lru_.empty()) {
        const uint64_t victim_key = lru_.back();
        lru_.pop_back();
        auto victim = map_.find(victim_key);
        if (victim == map_.end()) continue;
        const size_t victim_nbytes = victim->second.first.nbytes;
        used_bytes_ -= victim_nbytes;
        map_.erase(victim);
        ++stats_.evictions;
        if (evict_observer_) evict_observer_(victim_key, victim_nbytes);
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
    try {
        fill_ok = fill(entry.bytes.data(), nbytes);
    } catch (...) {
        fill_ok = false;
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
    if (count_hit) ++stats_.hits;
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

    // Protect every key in the incoming batch while selecting LRU victims.
    // This makes all reservations stable until their disjoint fills finish.
    while (used_bytes_ > budget_bytes_ - additional_bytes) {
        auto victim = lru_.end();
        for (auto it = lru_.end(); it != lru_.begin();) {
            --it;
            if (!is_batch_key(*it, requests, count)) {
                victim = it;
                break;
            }
        }
        if (victim == lru_.end()) {
            // Every resident is a protected batch key; the request set
            // cannot be admitted under the byte budget.  Refresh derived
            // stats for victims already removed above before bailing.
            ++stats_.budget_rejections;
            stats_.bytes = used_bytes_;
            stats_.entries = map_.size();
            return false;
        }
        const uint64_t victim_key = *victim;
        auto found = map_.find(victim_key);
        lru_.erase(victim);
        if (found == map_.end()) continue;
        const size_t victim_nbytes = found->second.first.nbytes;
        used_bytes_ -= victim_nbytes;
        map_.erase(found);
        ++stats_.evictions;
        if (evict_observer_) evict_observer_(victim_key, victim_nbytes);
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
}

}  // namespace dee
