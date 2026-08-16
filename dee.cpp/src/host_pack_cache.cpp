#include "dee/host_pack_cache.h"

#include <cstring>

namespace dee {

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

void HostPackCache::clear() {
    map_.clear();
    lru_.clear();
    used_bytes_ = 0;
    stats_.bytes = 0;
    stats_.entries = 0;
    stats_.evictions = 0;
    stats_.hits = 0;
    stats_.misses = 0;
}

}  // namespace dee