// dee/vram_cache.h
//
// Step 5 — VRAM Cache Manager (DEE data/control layer).
//
// Owns a fixed VRAM budget and decides which experts are resident. Eviction
// policy = LRU with Oracle priority: an Oracle-predicted expert gets a priority
// boost so it is evicted later than a same-age, non-predicted one.
//
// The manager is device-agnostic. It allocates from an Arena whose backend is
// swappable: a host (malloc) backend for CPU/tests now, and a CUDA backend
// (cudaMalloc / cudaFree) wired in by the Step 6 prefetcher. The cache,
// eviction, and free-list logic is identical across backends.

#pragma once

#include "dee/profiling.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace dee {

// (layer, expert) packed into one key for the block map.
struct ExpertKey {
    int layer;
    int expert;
    bool operator==(const ExpertKey& o) const { return layer == o.layer && expert == o.expert; }
};

struct ExpertKeyHash {
    size_t operator()(const ExpertKey& k) const noexcept {
        return ((size_t)(uint32_t)k.layer << 32) ^ (size_t)(uint32_t)k.expert;
    }
};

// Eviction policy selector (Phase-4 repair, A/B-able at runtime):
//   RankPriority — legacy score: last_used + priority*PRIORITY_WEIGHT. The
//                  priority term is request position in the live batched
//                  path (descending expert-ID order), so one rank step is
//                  worth ~2^20 recency ticks (~4000 tokens).
//   Recency      — pure LRU by last_used; the priority term is ignored for
//                  eviction but still stored on the block for telemetry.
enum class EvictionPolicy { RankPriority, Recency };

// One block evicted during a single ensure() eviction pass. EnsureInfo
// carries ALL victims in eviction order (multi-victim evictions used to
// expose only the last one via evicted_key).
struct EvictedVictim {
    int layer;
    int expert;
    std::uint64_t generation;
};

// Victim list that also reads as a bool ("was anything evicted this pass").
// Existing consumers test `info.evicted ? ... : ...` / `if (info.evicted)`;
// keeping contextual bool conversion here lets them compile unchanged while
// new consumers iterate the full victim list.
struct EvictedVictimList : std::vector<EvictedVictim> {
    operator bool() const { return !this->empty(); }
};

// A resident expert block: a slice of the arena + bookkeeping for eviction.
struct ExpertBlock {
    ExpertKey key{};
    void*     ptr       = nullptr;  // device/host pointer (from Arena backend)
    size_t    offset    = 0;        // arena offset (debug / provenance)
    size_t    size      = 0;        // bytes
    int64_t   last_used = 0;        // monotonic tick of last touch
    int       priority  = 0;        // Oracle priority (higher => keep longer)
    bool      resident  = false;
    uint32_t  pins      = 0;        // DMA or compute users that forbid eviction
    uint64_t  generation = 0;       // monotonic allocation generation in this cache
};

// ---------------------------------------------------------------------------
// Arena: a fixed-size region with a first-fit free-list + coalescing.
// Backend allocation is injected so the same logic serves host and GPU.
// ---------------------------------------------------------------------------
class Arena {
public:
    // Backend: alloc/free a region of `bytes`. kind is "host" or "cuda".
    struct Backend {
        std::function<void*(size_t)> alloc;
        std::function<void(void*)>   free;
        std::string                  kind;
    };

    Arena() = default;
    ~Arena();

    // Initialize with `capacity` bytes using `be`. Returns false on failure.
    bool init(size_t capacity, Backend be);

    // Allocate `bytes`; returns offset into the arena, or size_t(-1) if it
    // cannot fit (caller must free space first via the VramCacheManager).
    size_t alloc(size_t bytes);

    // Free a previously allocated offset.
    void free(size_t offset, size_t bytes);

    void*   base()  const { return base_; }
    size_t  capacity() const { return cap_; }
    size_t  used()  const { return used_; }
    size_t  free_space() const { return cap_ - used_; }

private:
    struct Hole { size_t off; size_t size; };
    Backend       be_{};
    void*         base_   = nullptr;
    size_t        cap_    = 0;
    size_t        used_   = 0;
    size_t        high_   = 0;   // bump-pointer high-water mark
    std::vector<Hole> free_;     // kept sorted by offset for coalescing

    void coalesce();
};

// ---------------------------------------------------------------------------
// VramCacheManager: LRU + Oracle-priority expert cache over a fixed budget.
// ---------------------------------------------------------------------------
class VramCacheManager {
public:
    struct Stats {
        uint64_t ensures   = 0;  // ensure() calls
        uint64_t hits      = 0;  // ensure() found resident
        uint64_t loads     = 0;  // ensure() had to load (alloc)
        uint64_t evictions = 0;  // blocks evicted to make room
        uint64_t fallbacks = 0;  // sync_fallback stalls (miss at compute time)
        uint64_t pinned_blocks_skipped = 0;  // eviction candidates rejected due to active pins
                                             // (counted once per distinct block per ensure() call)
        uint64_t budget_rejections = 0;      // ensure() rejected nbytes > budget before evicting
        uint64_t size_mismatches = 0;        // ensure() hit rejected: resident size != nbytes
        uint64_t invalidations = 0;          // evict_key() removals (invalidation, not LRU eviction)
    };

    struct EnsureInfo {
        bool resident_hit = false;
        // ALL victims of this ensure's eviction pass, in eviction order.
        // Cleared at the start of each ensure()/eviction pass. Also usable
        // in bool context ("was anything evicted") for legacy consumers.
        EvictedVictimList evicted;
        ExpertKey evicted_key{-1, -1};      // LAST victim only (backward compat)
        uint64_t evicted_generation = 0;    // LAST victim's generation (backward compat)
        uint64_t generation = 0;
        size_t cache_bytes_before = 0;
        size_t cache_entries_before = 0;
        size_t cache_bytes_after = 0;
        size_t cache_entries_after = 0;
        uint32_t pin_count_after = 0;
    };

    // Create a manager with a `budget_bytes` arena using backend `be`.
    bool init(size_t budget_bytes, Arena::Backend be);

    // Mark an expert resident (allocating arena space, evicting if needed).
    // `priority` is the Oracle priority (>=0). Returns true if resident after call.
    // NOTE: this does NOT copy weights — the caller (or Step 6 prefetcher)
    // fills block.ptr afterwards via data(layer, expert).
    bool ensure(int layer, int expert, size_t nbytes, int priority = 0);

    // Advance the clock and record a use (called by the forward pass per layer).
    void touch(int layer, int expert);

    bool is_resident(int layer, int expert) const;
    void* data(int layer, int expert) const;   // nullptr if not resident
    size_t size_of(int layer, int expert) const;
    uint64_t generation_of(int layer, int expert) const;
    uint32_t pin_count(int layer, int expert) const;

    // Sync fallback: if the expert is NOT resident, block until it is. In Step 5
    // (host) the block is synchronously allocated by ensure(); this only counts
    // a fallback when compute reached an un-resident expert. Returns true if the
    // expert became resident.
    bool sync_fallback(int layer, int expert, size_t nbytes, int priority = 0);

    // Force-evict everything (used between sequences / for tests).
    void clear();

    const Stats& stats() const { return stats_; }
    const EnsureInfo& last_ensure_info() const { return last_ensure_info_; }
    void reset_stats() { stats_ = Stats{}; last_ensure_info_ = EnsureInfo{}; }
    void set_profiler(StageProfiler* profiler) { profiler_ = profiler; }
    void set_debug_validation(bool enabled) { debug_validation_ = enabled; }
    bool debug_validation_enabled() const { return debug_validation_; }
    bool validate_invariants(std::string* error = nullptr) const;
    size_t  used_bytes() const { return arena_.used(); }
    size_t  budget_bytes() const { return arena_.capacity(); }
    size_t  resident_count() const;
    size_t  pinned_count() const;

    // A pinned block cannot be evicted. AsyncPrefetcher pins during DMA and
    // Engine pins while a CUDA compute stream may still dereference the block.
    bool pin(int layer, int expert);
    bool unpin(int layer, int expert);

    // Invalidation (NOT an LRU eviction): remove the block iff it is resident
    // AND pins==0. Frees the arena range and erases the map entry; returns
    // true. If the key is absent or currently pinned, nothing is removed and
    // false is returned (no partial). Counted via Stats::invalidations —
    // stats_.evictions is deliberately untouched. Used by the prefetcher to
    // invalidate a block whose submit/fill failed.
    bool evict_key(int layer, int expert);

    // Eviction-policy flag (Phase-4 A/B). Default is RankPriority for legacy
    // compatibility; the engine config selects Recency explicitly.
    void set_eviction_policy(EvictionPolicy p) { eviction_policy_ = p; }
    EvictionPolicy eviction_policy() const { return eviction_policy_; }

    // Milestone 3 forensic: capture the most recent ensure/evict failure
    // context so the engine can surface it to Python instead of collapsing
    // to a single "cannot allocate" line.  Cleared by ensure() on entry;
    // written on every failure path.
    void set_last_error(const std::string& m) { last_error_message_ = m; }
    void clear_last_error() { last_error_message_.clear(); }
    const std::string& last_error_message() const { return last_error_message_; }

private:
    Arena arena_;
    std::unordered_map<ExpertKey, ExpertBlock, ExpertKeyHash> blocks_;
    int64_t tick_ = 0;
    uint64_t next_generation_ = 1;
    Stats   stats_{};
    EnsureInfo last_ensure_info_{};
    StageProfiler* profiler_ = nullptr;
    std::string last_error_message_;
    bool debug_validation_ = false;
    EvictionPolicy eviction_policy_ = EvictionPolicy::RankPriority;

    ExpertBlock* find_block(int layer, int expert);
    const ExpertBlock* find_block(int layer, int expert) const;

    // Evict lowest-scored blocks until `need` bytes free (score depends on
    // eviction_policy_).
    void evict_until_free(size_t need);
    bool validate_or_record(const char* context);

    // Lower score => evict first. Under RankPriority higher last_used OR
    // higher priority RAISES the score (both keep an expert resident longer).
    // Under Recency the priority term is ignored — strict LRU by last_used.
    static constexpr int64_t PRIORITY_WEIGHT = 1 << 20;
    int64_t eviction_score(const ExpertBlock& b) const {
        if (eviction_policy_ == EvictionPolicy::Recency) return b.last_used;
        return b.last_used + (int64_t)b.priority * PRIORITY_WEIGHT;
    }
};

} // namespace dee
