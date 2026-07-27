// dee/vram_cache.cpp
#include "dee/vram_cache.h"
#include "dee/trace_alloc.h"  // Milestone 3 v5 teardown-forensics sentinel

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <unordered_set>

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
    int iteration = 0;
    int pinned_skipped_total = 0;
    while (arena_.free_space() < need && !blocks_.empty()) {
        ++iteration;
        // find the resident block with the lowest eviction score
        ExpertBlock* victim = nullptr;
        int64_t worst = 0;
        bool first = true;
        int pinned_skipped_this_iter = 0;
        for (auto& kv : blocks_) {
            ExpertBlock& b = kv.second;
            if (!b.resident) continue;
            if (b.pins != 0) {
                ++stats_.pinned_blocks_skipped;
                ++pinned_skipped_this_iter;
                if (profiler_) profiler_->note_pinned_skip();
                continue;
            }
            int64_t s = eviction_score(b);
            if (first || s < worst) { worst = s; victim = &b; first = false; }
        }
        pinned_skipped_total += pinned_skipped_this_iter;
        if (!victim) {
            std::fprintf(stderr,
                "[dee-cache %s:%d] evict_until_free(need=%zuB) ABORT iter=%d: "
                "no evictable victim (pinned_skipped_this_iter=%d "
                "resident_count=%zu arena_free=%zuB arena_used=%zuB)\n",
                __FILE__, __LINE__, need, iteration, pinned_skipped_this_iter,
                resident_count(), arena_.free_space(), arena_.used());
            last_error_message_ =
                "evict_until_free aborted: no evictable victim (all resident "
                "blocks pinned, pins>0 prevents eviction; the most likely "
                "cause is a leaked cache_.pin() from a prior failed cuBLAS "
                "launch or premature re-entry into the same forward path)";
            break; // nothing resident to evict
        }
        std::fprintf(stderr,
            "[dee-cache %s:%d] evict_until_free(need=%zuB) iter=%d EVICT "
            "layer=%d expert=%d size=%zuB score=%lld priority=%d "
            "last_used=%lld arena_free_before=%zuB\n",
            __FILE__, __LINE__, need, iteration,
            victim->key.layer, victim->key.expert, victim->size,
            (long long)worst, victim->priority,
            (long long)victim->last_used, arena_.free_space());
        arena_.free(victim->offset, victim->size);
        last_ensure_info_.evicted = true;
        last_ensure_info_.evicted_key = victim->key;
        last_ensure_info_.evicted_generation = victim->generation;
        if (profiler_) {
            profiler_->note_generation_evicted(
                victim->key.layer, victim->key.expert, victim->generation);
        }
        victim->resident = false;
        victim->ptr = nullptr;
        // remove from map so a later ensure re-loads it
        blocks_.erase(ExpertKey{victim->key.layer, victim->key.expert});
        ++stats_.evictions;
        if (profiler_) profiler_->note_eviction();
    }
}

bool VramCacheManager::ensure(int layer, int expert, size_t nbytes, int priority) {
    ++stats_.ensures;
    last_ensure_info_ = EnsureInfo{};
    last_error_message_.clear();
    last_ensure_info_.cache_bytes_before = arena_.used();
    last_ensure_info_.cache_entries_before = resident_count();
    const auto lookup_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    ExpertBlock* b = find_block(layer, expert);
    if (profiler_ && profiler_->enabled()) {
        const auto lookup_end = StageProfiler::now();
        profiler_->add_cpu_ms(CpuStage::CacheLookup,
            std::chrono::duration<double, std::milli>(lookup_end - lookup_begin).count());
        profiler_->note_cpu_timeline_interval(CpuTimelineKind::CacheLookup,
            lookup_begin, lookup_end, -1, layer, expert);
    }
    if (b && b->resident) {
        last_ensure_info_.resident_hit = true;
        b->last_used = ++tick_;
        b->priority  = priority;
        ++stats_.hits;
        last_ensure_info_.generation = b->generation;
        last_ensure_info_.cache_bytes_after = arena_.used();
        last_ensure_info_.cache_entries_after = resident_count();
        last_ensure_info_.pin_count_after = b->pins;
        return !debug_validation_ || validate_or_record("ensure resident hit");
    }
    // not resident -> evict to make room, then allocate
    const auto eviction_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    evict_until_free(nbytes);
    if (profiler_ && profiler_->enabled()) {
        const auto eviction_end = StageProfiler::now();
        profiler_->add_cpu_ms(CpuStage::EvictionSelection,
            std::chrono::duration<double, std::milli>(eviction_end - eviction_begin).count());
        profiler_->note_cpu_timeline_interval(CpuTimelineKind::EvictionEligibility,
            eviction_begin, eviction_end, -1, layer, expert);
    }
    size_t off = arena_.alloc(nbytes);
    if (off == size_t(-1)) {
        // Detailed forensic dump.  Enumerate every resident block sorted by
        // eviction score so the analyzer can identify *why* eviction could
        // not free enough space -- expected root cause for this M3 failure
        // is "every resident block is pinned (pins > 0)" because a prior
        // cuBLAS or DMA launch leaked a cache_.pin() reference.
        std::vector<std::pair<int64_t, const ExpertBlock*>> ranked;
        ranked.reserve(blocks_.size());
        for (const auto& kv : blocks_) {
            if (!kv.second.resident) continue;
            ranked.emplace_back(eviction_score(kv.second), &kv.second);
        }
        std::sort(ranked.begin(), ranked.end());
        const size_t live_resident = ranked.size();
        const size_t live_pinned  = pinned_count();
        char err_summary[1024];
        std::snprintf(err_summary, sizeof(err_summary),
            "VramCacheManager::ensure(layer=%d expert=%d nbytes=%zu) failed "
            "[requested=%zuB budget=%zuB arena_used=%zuB arena_free=%zuB "
            "live_resident=%zu live_pinned=%zu]",
            layer, expert, nbytes,
            nbytes, arena_.capacity(), arena_.used(), arena_.free_space(),
            live_resident, live_pinned);
        std::fprintf(stderr,
            "[dee-cache %s:%d] %s\n", __FILE__, __LINE__, err_summary);
        if (live_resident == 0) {
            std::fprintf(stderr,
                "[dee-cache %s:%d] block dump: NO resident blocks.  Eviction "
                "still failed -> fragmented free-list smaller than requested "
                "allocation despite empty resident map (free-space=%zuB "
                "requested=%zuB).\n",
                __FILE__, __LINE__, arena_.free_space(), nbytes);
        } else {
            std::fprintf(stderr,
                "[dee-cache %s:%d] block dump (%zu resident entries, sorted by "
                "ascending eviction_score):\n", __FILE__, __LINE__, live_resident);
            for (size_t i = 0; i < ranked.size(); ++i) {
                const ExpertBlock& bb = *ranked[i].second;
                std::fprintf(stderr,
                    "  [%zu] layer=%d expert=%d size=%zuB offset=%zuB "
                    "last_used=%lld priority=%d pins=%u resident=%s "
                    "score=%lld -> %s\n",
                    i, bb.key.layer, bb.key.expert, bb.size, bb.offset,
                    (long long)bb.last_used, bb.priority, bb.pins,
                    bb.resident ? "true" : "false",
                    (long long)ranked[i].first,
                    bb.pins != 0 ? "SKIP (pinned, pins>0 -> ensure fails)"
                                 : "would-evict candidate");
            }
        }
        last_error_message_ = err_summary;
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
    nb.generation = next_generation_++;
    blocks_[nb.key] = nb;
    ++stats_.loads;
    last_ensure_info_.generation = nb.generation;
    last_ensure_info_.cache_bytes_after = arena_.used();
    last_ensure_info_.cache_entries_after = resident_count();
    last_ensure_info_.pin_count_after = nb.pins;
    return !debug_validation_ || validate_or_record("ensure cold load");
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

uint64_t VramCacheManager::generation_of(int layer, int expert) const {
    const ExpertBlock* b = find_block(layer, expert);
    return (b && b->resident) ? b->generation : 0;
}

uint32_t VramCacheManager::pin_count(int layer, int expert) const {
    const ExpertBlock* b = find_block(layer, expert);
    return (b && b->resident) ? b->pins : 0;
}

bool VramCacheManager::sync_fallback(int layer, int expert, size_t nbytes, int priority) {
    if (is_resident(layer, expert)) return true;
    ++stats_.fallbacks;
    return ensure(layer, expert, nbytes, priority);
}

void VramCacheManager::clear() {
    for (auto& kv : blocks_) {
        if (kv.second.resident) {
            if (profiler_) {
                profiler_->note_generation_evicted(
                    kv.second.key.layer, kv.second.key.expert, kv.second.generation);
            }
            arena_.free(kv.second.offset, kv.second.size);
        }
    }
    blocks_.clear();
    tick_ = 0;
    if (debug_validation_) validate_or_record("clear");
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
    if (b->pins == std::numeric_limits<uint32_t>::max()) {
        last_error_message_ = "VramCacheManager::pin counter overflow";
        std::fprintf(stderr, "[dee-cache] %s layer=%d expert=%d\n",
                     last_error_message_.c_str(), layer, expert);
        return false;
    }
    ++b->pins;
    return !debug_validation_ || validate_or_record("pin");
}

bool VramCacheManager::unpin(int layer, int expert) {
    ExpertBlock* b = find_block(layer, expert);
    if (!b || !b->resident || b->pins == 0) {
        std::ostringstream message;
        message << "VramCacheManager::unpin underflow or missing resident block"
                << " layer=" << layer << " expert=" << expert;
        last_error_message_ = message.str();
        std::fprintf(stderr, "[dee-cache] %s\n", last_error_message_.c_str());
        return false;
    }
    --b->pins;
    return !debug_validation_ || validate_or_record("unpin");
}

bool VramCacheManager::validate_invariants(std::string* error) const {
    auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    const auto base = reinterpret_cast<uintptr_t>(arena_.base());
    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(blocks_.size());
    std::unordered_set<uint64_t> generations;
    generations.reserve(blocks_.size());
    size_t used_sum = 0;
    for (const auto& entry : blocks_) {
        const ExpertKey& map_key = entry.first;
        const ExpertBlock& block = entry.second;
        if (!(map_key == block.key)) {
            return fail("cache map key does not match ExpertBlock key");
        }
        if (!block.resident) {
            return fail("non-resident block retained in resident map");
        }
        if (!block.ptr || block.size == 0 || block.generation == 0) {
            return fail("resident block has null pointer, zero size, or zero generation");
        }
        if (block.generation >= next_generation_ ||
            !generations.insert(block.generation).second) {
            return fail("resident block generation is stale, future, or duplicated");
        }
        if (block.offset > arena_.capacity() ||
            block.size > arena_.capacity() - block.offset) {
            return fail("resident block range exceeds arena capacity");
        }
        if (reinterpret_cast<uintptr_t>(block.ptr) != base + block.offset) {
            return fail("resident block pointer does not match arena base plus offset");
        }
        if (used_sum > std::numeric_limits<size_t>::max() - block.size) {
            return fail("resident block size sum overflow");
        }
        used_sum += block.size;
        ranges.emplace_back(block.offset, block.offset + block.size);
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i - 1].second > ranges[i].first) {
            return fail("resident block ranges overlap");
        }
    }
    if (used_sum != arena_.used()) {
        std::ostringstream message;
        message << "resident size sum " << used_sum
                << " does not match arena used bytes " << arena_.used();
        return fail(message.str());
    }
    if (error) error->clear();
    return true;
}

bool VramCacheManager::validate_or_record(const char* context) {
    std::string error;
    if (validate_invariants(&error)) return true;
    last_error_message_ = std::string("cache invariant failure after ") + context + ": " + error;
    std::fprintf(stderr, "[dee-cache] %s\n", last_error_message_.c_str());
    return false;
}

} // namespace dee
