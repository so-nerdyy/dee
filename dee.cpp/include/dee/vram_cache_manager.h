#pragma once
// dee.cpp - Step 5: VRAM cache manager.
//
// Owns a single fixed-size device arena allocated once (e.g. 22GB) and serves
// expert slots out of it via a free-list. There is NO per-expert cudaMalloc;
// every expert lives inside the one arena. When the arena is full, the lowest
// priority (Oracle score) expert is evicted; ties are broken by least-recently
// used (LRU).

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "dee/cuda_device.h"

namespace dee {

// Identifies one expert in the model.
struct ExpertKey {
    int layer = -1;
    int expert_id = -1;
    bool operator==(const ExpertKey& o) const { return layer == o.layer && expert_id == o.expert_id; }
};

struct ExpertKeyHash {
    size_t operator()(const ExpertKey& k) const noexcept {
        // layer in the high bits, expert_id in the low bits (expert_id >= 0).
        return (static_cast<size_t>(k.layer) << 32) ^ static_cast<size_t>(k.expert_id + 1);
    }
};

// Higher score => higher priority to KEEP in VRAM (lower eviction priority).
// The production Oracle is the trained 3-layer MLP from Step 2; tests inject a
// mock that returns deterministic scores.
class Oracle {
public:
    virtual ~Oracle() = default;
    virtual float priority(int layer, int expert_id) const = 0;
};

struct CachedExpert {
    ExpertKey key;
    int slot = -1;        // index into the arena
    size_t bytes = 0;     // bytes this expert occupies
    uint64_t last_used = 0;  // LRU timestamp (monotonic clock)
    float priority = 0.f;    // Oracle priority captured at acquire time
};

class VramCacheManager {
public:
    // Allocates an arena of `slot_bytes * num_slots` device bytes and a free-list
    // of `num_slots` slots.
    VramCacheManager(CudaDevice& dev, size_t slot_bytes, int num_slots, Oracle& oracle);
    ~VramCacheManager();

    // Acquire a VRAM slot for the expert. Returns the device pointer + slot.
    // `hit` is true when the expert was already cached (no eviction needed to
    // make it resident; the caller may still need to (re)fill it).
    // If the free list is empty, the lowest-priority (LRU on tie) cached expert
    // is evicted and its slot reused.
    struct AcquireResult {
        DevicePtr ptr = nullptr;
        int slot = -1;
        bool hit = false;
    };
    AcquireResult acquire(const ExpertKey& key, size_t expert_bytes);

    bool contains(const ExpertKey& key) const;
    DevicePtr ptr_of(const ExpertKey& key) const;
    int slot_of(const ExpertKey& key) const;

    // Refresh the LRU timestamp (call when the model actually uses the expert).
    void touch(const ExpertKey& key);

    int num_cached() const { return static_cast<int>(cached_.size()); }
    int num_slots() const { return num_slots_; }
    int free_slots() const { return static_cast<int>(free_list_.size()); }
    size_t slot_bytes() const { return slot_bytes_; }

    std::vector<ExpertKey> cached_keys() const;
    const CachedExpert* info(const ExpertKey& key) const;

    // Most recently evicted expert (layer/expert_id == -1 if none yet). Useful
    // for tests that verify eviction ordering.
    ExpertKey last_evicted() const { return last_evicted_; }

private:
    // Evict the lowest-priority (LRU on tie) cached expert; returns its slot.
    int evict_victim();
    DevicePtr slot_ptr(int slot) const;

    CudaDevice& dev_;
    Oracle& oracle_;
    size_t slot_bytes_;
    int num_slots_;
    DevicePtr arena_ = nullptr;
    std::vector<int> free_list_;  // available slot indices
    std::unordered_map<ExpertKey, CachedExpert, ExpertKeyHash> cached_;
    uint64_t clock_ = 0;
    ExpertKey last_evicted_{-1, -1};
};

}  // namespace dee
