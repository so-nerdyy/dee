// dee/host_pack_cache.h
//
// Bounded LRU cache of packed expert weight bytes (DeepSeek-V4-Flash-0731 FP4
// e2m1 + per-block e8m0 scales, ~12.6 MB per entry).
//
// Why this exists (measured): the engine stages the six packed mmap regions of
// each expert verbatim and the prefetcher gathers them into its pinned slot on
// EVERY transfer.  With a 152.8 GiB checkpoint and only ~30 GiB of page cache,
// the OS LRU evicts an expert's pages between tokens, so each gather re-faults
// cold mmap pages against storage (~4.5 s/expert measured -> ~366 s/token
// decode).  This cache keeps a private RAM copy of each visited pack so
// repeated gathers hit host DRAM.  The unique working set of a 16-token
// response is ~2,365 pairs × 12.6 MB ≈ 30 GiB, so a multi-GiB budget matches
// the box's RAM: after the first touch, the response becomes RAM-resident.
//
// Thread-safety: the engine host path is single-threaded per engine; the cache
// is only touched from get_staging_fp4 / stage paths, all on the same thread.
// The returned buffer pointer stays valid until the next call that evicts this
// key (LRU never evicts the just-inserted/just-touched entry).

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>
#include <vector>

namespace dee {

class HostPackCache {
public:
    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        size_t bytes = 0;
        size_t entries = 0;
    };

    HostPackCache() = default;
    HostPackCache(const HostPackCache&) = delete;
    HostPackCache& operator=(const HostPackCache&) = delete;

    void set_budget(size_t bytes) { budget_bytes_ = bytes; }
    size_t budget() const { return budget_bytes_; }

    // Return a pointer to `nbytes` of cached bytes for `key`, running
    // `fill(dst, nbytes)` on a miss (fill must copy exactly nbytes).  Returns
    // nullptr when the entry cannot fit the budget or allocation fails.
    const uint8_t* get(
        uint64_t key, size_t nbytes,
        const std::function<void(uint8_t* dst, size_t n)>& fill);

    bool contains(uint64_t key) const {
        return map_.find(key) != map_.end();
    }

    void clear();
    const Stats& stats() const { return stats_; }

private:
    struct Entry {
        std::vector<uint8_t> bytes;
        size_t nbytes = 0;
    };
    // map: key -> (payload, LRU position). lru_ front = most recently used.
    using LruIter = typename std::list<uint64_t>::iterator;
    std::unordered_map<uint64_t, std::pair<Entry, LruIter>> map_;
    std::list<uint64_t> lru_;
    size_t budget_bytes_ = 8ull << 30;
    size_t used_bytes_ = 0;
    Stats stats_;
};

}  // namespace dee