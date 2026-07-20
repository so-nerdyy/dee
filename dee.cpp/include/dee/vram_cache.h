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

// A resident expert block: a slice of the arena + bookkeeping for eviction.
struct ExpertBlock {
    ExpertKey key{};
    void*     ptr       = nullptr;  // device/host pointer (from Arena backend)
    size_t    offset    = 0;        // arena offset (debug / provenance)
    size_t    size      = 0;        // bytes
    int64_t   last_used = 0;        // monotonic tick of last touch
    int       priority  = 0;        // Oracle priority (higher => keep longer)
    bool      resident  = false;
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

    // Sync fallback: if the expert is NOT resident, block until it is. In Step 5
    // (host) the block is synchronously allocated by ensure(); this only counts
    // a fallback when compute reached an un-resident expert. Returns true if the
    // expert became resident.
    bool sync_fallback(int layer, int expert, size_t nbytes, int priority = 0);

    // Force-evict everything (used between sequences / for tests).
    void clear();

    const Stats& stats() const { return stats_; }
    size_t  used_bytes() const { return arena_.used(); }
    size_t  budget_bytes() const { return arena_.capacity(); }
    size_t  resident_count() const;

private:
    Arena arena_;
    std::unordered_map<ExpertKey, ExpertBlock, ExpertKeyHash> blocks_;
    int64_t tick_ = 0;
    Stats   stats_{};

    ExpertBlock* find_block(int layer, int expert);
    const ExpertBlock* find_block(int layer, int expert) const;

    // Evict lowest (last_used + priority*PRIORITY_WEIGHT) until `need` bytes free.
    void evict_until_free(size_t need);

    // Lower score => evict first. Higher last_used (more recent) OR higher
    // priority (Oracle-predicted) RAISES the score, so both recency and
    // Oracle prediction keep an expert resident longer.
    static constexpr int64_t PRIORITY_WEIGHT = 1 << 20;
    int64_t eviction_score(const ExpertBlock& b) const {
        return b.last_used + (int64_t)b.priority * PRIORITY_WEIGHT;
    }
};

} // namespace dee
