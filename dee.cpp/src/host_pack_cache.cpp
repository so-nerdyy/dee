#include "dee/host_pack_cache.h"

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
    active_request_count_ = 0;
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
        const size_t index = next_fill_index_.fetch_add(
            1, std::memory_order_relaxed);
        if (index >= active_request_count_) return;
        const BatchRequest& request = active_requests_[index];
        BatchResult& result = active_results_[index];
        if (!result.fill_executed) continue;
        const auto begin = std::chrono::steady_clock::now();
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
    const std::function<void(uint8_t* dst, size_t n)>& fill) {
    if (nbytes == 0 || !fill) return nullptr;
    auto found = map_.find(key);
    if (found != map_.end()) {
        // Refresh LRU position; never evict the entry we are about to return.
        lru_.erase(found->second.second);
        lru_.push_front(key);
        found->second.second = lru_.begin();
        ++stats_.hits;
        return found->second.first.bytes.data();
    }

    ++stats_.misses;
    if (nbytes > budget_bytes_) {
        // Cannot ever fit; do not allocate.
        return nullptr;
    }
    while (used_bytes_ + nbytes > budget_bytes_ && !lru_.empty()) {
        const uint64_t victim_key = lru_.back();
        lru_.pop_back();
        auto victim = map_.find(victim_key);
        if (victim == map_.end()) continue;
        used_bytes_ -= victim->second.first.nbytes;
        map_.erase(victim);
        ++stats_.evictions;
    }
    Entry entry;
    entry.bytes.resize(nbytes);
    entry.nbytes = nbytes;
    fill(entry.bytes.data(), nbytes);

    lru_.push_front(key);
    auto inserted = map_.emplace(
        key, std::make_pair(std::move(entry), lru_.begin()));
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

bool HostPackCache::get_batch(
        const BatchRequest* requests, size_t count, BatchResult* results) {
    if (!requests || !results || count == 0 ||
        count > kMaxBatchRequests) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) results[index] = {};

    size_t additional_bytes = 0;
    size_t unique_misses = 0;
    for (size_t index = 0; index < count; ++index) {
        const BatchRequest& request = requests[index];
        if (request.nbytes == 0 || request.nbytes > budget_bytes_) return false;
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
            if (found->second.first.nbytes != request.nbytes) return false;
            lru_.erase(found->second.second);
            lru_.push_front(request.key);
            found->second.second = lru_.begin();
            ++stats_.hits;
            results[index].data = found->second.first.bytes.data();
            results[index].cache_hit = true;
            results[index].success = true;
            continue;
        }
        if (!request.fill ||
            additional_bytes > budget_bytes_ - request.nbytes) {
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
        if (victim == lru_.end()) return false;
        const uint64_t victim_key = *victim;
        auto found = map_.find(victim_key);
        lru_.erase(victim);
        if (found == map_.end()) continue;
        used_bytes_ -= found->second.first.nbytes;
        map_.erase(found);
        ++stats_.evictions;
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
                lru_.pop_front();
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

    const auto batch_begin = std::chrono::steady_clock::now();
    active_requests_ = requests;
    active_results_ = results;
    active_request_count_ = count;
    next_fill_index_.store(0, std::memory_order_relaxed);
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
    active_request_count_ = 0;

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
}

}  // namespace dee
