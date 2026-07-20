// dee/vram_cache.cpp
#include "dee/vram_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace dee {

// ---- Arena ----------------------------------------------------------------
Arena::~Arena() {
    if (base_ && be_.free) be_.free(base_);
    base_ = nullptr;
}

bool Arena::init(size_t capacity, Backend be) {
    if (base_) return false;
    be_ = std::move(be);
    cap_ = capacity;
    used_ = 0;
    high_ = 0;
    free_.clear();
    if (!be_.alloc) return false;
    base_ = be_.alloc(capacity);
    return base_ != nullptr;
}

size_t Arena::alloc(size_t bytes) {
    if (bytes == 0) return size_t(-1);
    // 1) first-fit in the free-list (keep sorted by offset for coalescing).
    for (size_t i = 0; i < free_.size(); ++i) {
        if (free_[i].size >= bytes) {
            size_t off = free_[i].off;
            if (free_[i].size == bytes) {
                free_.erase(free_.begin() + i);
            } else {
                free_[i].off  += bytes;
                free_[i].size -= bytes;
            }
            used_ += bytes;
            return off;
        }
    }
    // 2) bump from high-water if it fits.
    if (high_ + bytes <= cap_) {
        size_t off = high_;
        high_ += bytes;
        used_ += bytes;
        return off;
    }
    return size_t(-1); // caller must evict first
}

void Arena::free(size_t offset, size_t bytes) {
    if (offset == size_t(-1) || bytes == 0) return;
    // insert keeping sorted by offset
    Hole h{offset, bytes};
    auto it = std::lower_bound(free_.begin(), free_.end(), h,
                               [](const Hole& a, const Hole& b) { return a.off < b.off; });
    free_.insert(it, h);
    used_ -= bytes;
    coalesce();
}

void Arena::coalesce() {
    if (free_.size() < 2) return;
    std::vector<Hole> merged;
    merged.push_back(free_[0]);
    for (size_t i = 1; i < free_.size(); ++i) {
        Hole& last = merged.back();
        if (last.off + last.size == free_[i].off) {
            last.size += free_[i].size; // adjacent -> merge
        } else {
            merged.push_back(free_[i]);
        }
    }
    free_.swap(merged);
}

// ---- VramCacheManager -----------------------------------------------------
bool VramCacheManager::init(size_t budget_bytes, Arena::Backend be) {
    return arena_.init(budget_bytes, std::move(be));
}

ExpertBlock* VramCacheManager::find_block(int layer, int expert) {
    ExpertKey k{layer, expert};
    auto it = blocks_.find(k);
    return it == blocks_.end() ? nullptr : &it->second;
}
const ExpertBlock* VramCacheManager::find_block(int layer, int expert) const {
    ExpertKey k{layer, expert};
    auto it = blocks_.find(k);
    return it == blocks_.end() ? nullptr : &it->second;
}

void VramCacheManager::evict_until_free(size_t need) {
    while (arena_.free_space() < need && !blocks_.empty()) {
        // find the resident block with the lowest eviction score
        ExpertBlock* victim = nullptr;
        int64_t worst = 0;
        bool first = true;
        for (auto& kv : blocks_) {
            ExpertBlock& b = kv.second;
            if (!b.resident || b.pins != 0) continue;
            int64_t s = eviction_score(b);
            if (first || s < worst) { worst = s; victim = &b; first = false; }
        }
        if (!victim) break; // nothing resident to evict
        arena_.free(victim->offset, victim->size);
        victim->resident = false;
        victim->ptr = nullptr;
        // remove from map so a later ensure re-loads it
        blocks_.erase(ExpertKey{victim->key.layer, victim->key.expert});
        ++stats_.evictions;
    }
}

bool VramCacheManager::ensure(int layer, int expert, size_t nbytes, int priority) {
    ++stats_.ensures;
    ExpertBlock* b = find_block(layer, expert);
    if (b && b->resident) {
        b->last_used = ++tick_;
        b->priority  = priority;
        ++stats_.hits;
        return true;
    }
    // not resident -> evict to make room, then allocate
    evict_until_free(nbytes);
    size_t off = arena_.alloc(nbytes);
    if (off == size_t(-1)) {
        // still no room (nbytes > budget). Allocate at bump even if over budget
        // so the engine doesn't deadlock; caller's budget should prevent this.
        fprintf(stderr, "VramCacheManager: cannot allocate expert (%d,%d) %zuB in budget %zuB; all remaining blocks may be pinned\n",
                layer, expert, nbytes, arena_.capacity());
        return false;
    }
    ExpertBlock nb;
    nb.key       = ExpertKey{layer, expert};
    nb.ptr       = (uint8_t*)arena_.base() + off;
    nb.offset    = off;
    nb.size      = nbytes;
    nb.last_used = ++tick_;
    nb.priority  = priority;
    nb.resident  = true;
    blocks_[nb.key] = nb;
    ++stats_.loads;
    return true;
}

void VramCacheManager::touch(int layer, int expert) {
    ExpertBlock* b = find_block(layer, expert);
    if (b && b->resident) b->last_used = ++tick_;
}

bool VramCacheManager::is_resident(int layer, int expert) const {
    const ExpertBlock* b = find_block(layer, expert);
    return b && b->resident;
}

void* VramCacheManager::data(int layer, int expert) const {
    const ExpertBlock* b = find_block(layer, expert);
    return (b && b->resident) ? b->ptr : nullptr;
}

size_t VramCacheManager::size_of(int layer, int expert) const {
    const ExpertBlock* b = find_block(layer, expert);
    return (b && b->resident) ? b->size : 0;
}

bool VramCacheManager::sync_fallback(int layer, int expert, size_t nbytes, int priority) {
    if (is_resident(layer, expert)) return true;
    ++stats_.fallbacks;
    return ensure(layer, expert, nbytes, priority);
}

void VramCacheManager::clear() {
    for (auto& kv : blocks_) {
        if (kv.second.resident) arena_.free(kv.second.offset, kv.second.size);
    }
    blocks_.clear();
    tick_ = 0;
    // reset stats? keep cumulative; caller decides.
}

size_t VramCacheManager::resident_count() const {
    size_t n = 0;
    for (const auto& kv : blocks_) if (kv.second.resident) ++n;
    return n;
}

size_t VramCacheManager::pinned_count() const {
    size_t n = 0;
    for (const auto& kv : blocks_) if (kv.second.resident && kv.second.pins != 0) ++n;
    return n;
}

bool VramCacheManager::pin(int layer, int expert) {
    ExpertBlock* b = find_block(layer, expert);
    if (!b || !b->resident) return false;
    ++b->pins;
    return true;
}

void VramCacheManager::unpin(int layer, int expert) {
    ExpertBlock* b = find_block(layer, expert);
    if (!b || b->pins == 0) return;
    --b->pins;
}

} // namespace dee
