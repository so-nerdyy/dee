// dee.cpp - Step 5: VRAM cache manager implementation.

#include "dee/vram_cache_manager.h"

#include <algorithm>
#include <stdexcept>

namespace dee {

VramCacheManager::VramCacheManager(CudaDevice& dev, size_t slot_bytes, int num_slots,
                                   Oracle& oracle)
    : dev_(dev), oracle_(oracle), slot_bytes_(slot_bytes), num_slots_(num_slots) {
    if (num_slots <= 0) throw std::invalid_argument("VramCacheManager: num_slots must be > 0");
    if (slot_bytes == 0) throw std::invalid_argument("VramCacheManager: slot_bytes must be > 0");
    arena_ = dev_.device_malloc(slot_bytes_ * static_cast<size_t>(num_slots_));
    if (!arena_) throw std::bad_alloc();
    free_list_.reserve(num_slots_);
    for (int i = num_slots_ - 1; i >= 0; --i) free_list_.push_back(i);
}

VramCacheManager::~VramCacheManager() {
    if (arena_) dev_.device_free(arena_);
}

DevicePtr VramCacheManager::slot_ptr(int slot) const {
    return static_cast<std::byte*>(arena_) + static_cast<size_t>(slot) * slot_bytes_;
}

int VramCacheManager::evict_victim() {
    // Choose the cached expert with the lowest Oracle priority; break ties by
    // least-recently-used (smallest last_used).
    int victim_slot = -1;
    const CachedExpert* victim = nullptr;
    for (const auto& kv : cached_) {
        const CachedExpert& ce = kv.second;
        if (!victim || ce.priority < victim->priority ||
            (ce.priority == victim->priority && ce.last_used < victim->last_used)) {
            victim = &ce;
            victim_slot = ce.slot;
        }
    }
    // cached_ is guaranteed non-empty here (free list was empty).
    ExpertKey vk = victim->key;
    cached_.erase(vk);
    last_evicted_ = vk;
    return victim_slot;
}

VramCacheManager::AcquireResult VramCacheManager::acquire(const ExpertKey& key, size_t expert_bytes) {
    if (expert_bytes > slot_bytes_) {
        throw std::invalid_argument("VramCacheManager: expert_bytes exceeds slot_bytes");
    }

    auto it = cached_.find(key);
    if (it != cached_.end()) {
        it->second.last_used = ++clock_;
        return {slot_ptr(it->second.slot), it->second.slot, /*hit=*/true};
    }

    int slot;
    if (!free_list_.empty()) {
        slot = free_list_.back();
        free_list_.pop_back();
    } else {
        slot = evict_victim();
    }

    CachedExpert ce;
    ce.key = key;
    ce.slot = slot;
    ce.bytes = expert_bytes;
    ce.last_used = ++clock_;
    ce.priority = oracle_.priority(key.layer, key.expert_id);
    cached_[key] = ce;
    return {slot_ptr(slot), slot, /*hit=*/false};
}

bool VramCacheManager::contains(const ExpertKey& key) const {
    return cached_.find(key) != cached_.end();
}

DevicePtr VramCacheManager::ptr_of(const ExpertKey& key) const {
    auto it = cached_.find(key);
    if (it == cached_.end()) return nullptr;
    return slot_ptr(it->second.slot);
}

int VramCacheManager::slot_of(const ExpertKey& key) const {
    auto it = cached_.find(key);
    return it == cached_.end() ? -1 : it->second.slot;
}

void VramCacheManager::touch(const ExpertKey& key) {
    auto it = cached_.find(key);
    if (it != cached_.end()) it->second.last_used = ++clock_;
}

std::vector<ExpertKey> VramCacheManager::cached_keys() const {
    std::vector<ExpertKey> keys;
    keys.reserve(cached_.size());
    for (const auto& kv : cached_) keys.push_back(kv.first);
    return keys;
}

const CachedExpert* VramCacheManager::info(const ExpertKey& key) const {
    auto it = cached_.find(key);
    return it == cached_.end() ? nullptr : &it->second;
}

}  // namespace dee
