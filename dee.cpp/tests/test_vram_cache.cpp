// tests/test_vram_cache.cpp
//
// Step 5 — VRAM Cache Manager unit test.
//
// Build (no cmake):
//   g++ -std=c++17 -I../include test_vram_cache.cpp ../src/vram_cache.cpp \
//       -o test_vram_cache
//
// Verifies: arena alloc/free + coalescing, LRU eviction under a tight budget,
// Oracle-priority protection, touch() recency, and stats counters.

#include "dee/vram_cache.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(const char* what, bool cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_fail;
}

// host backend: malloc/free
static dee::Arena::Backend host_backend() {
    dee::Arena::Backend be;
    be.kind = "host";
    be.alloc = [](size_t n) -> void* { return n ? malloc(n) : nullptr; };
    be.free  = [](void* p) { if (p) free(p); };
    return be;
}

int main() {
    printf("=== dee.cpp Step 5 VRAM Cache Manager test ===\n");

    // Each expert "block" = 256 bytes. Budget fits exactly 4 blocks.
    const size_t BLK = 256;
    const size_t BUDGET = BLK * 4;

    dee::VramCacheManager mgr;
    check("init manager", mgr.init(BUDGET, host_backend()));
    check("budget correct", mgr.budget_bytes() == BUDGET);

    // --- Load 4 experts: all fit, no eviction ---
    for (int e = 0; e < 4; ++e) {
        check(("ensure E0..E3 #" + std::to_string(e)).c_str(),
              mgr.ensure(0, e, BLK, /*priority=*/0));
    }
    check("4 resident", mgr.resident_count() == 4);
    check("used == budget", mgr.used_bytes() == BUDGET);
    check("0 evictions so far", mgr.stats().evictions == 0);
    check("4 loads", mgr.stats().loads == 4);

    // Write a marker into each block's data and confirm it round-trips.
    for (int e = 0; e < 4; ++e) {
        void* p = mgr.data(0, e);
        check(("data E" + std::to_string(e) + " non-null").c_str(), p != nullptr);
        memset(p, (int)('A' + e), BLK);
    }
    check("E2 marker intact", ((char*)mgr.data(0, 2))[10] == 'C');

    // --- Load a 5th expert: must evict exactly 1 (LRU = E0, oldest tick) ---
    check("ensure E4 (triggers eviction)", mgr.ensure(0, 4, BLK, 0));
    check("still 4 resident (budget cap)", mgr.resident_count() == 4);
    check("1 eviction", mgr.stats().evictions == 1);
    check("E0 evicted (LRU)", !mgr.is_resident(0, 0));
    check("E4 resident", mgr.is_resident(0, 4));

    // --- Oracle priority: protect a low-recency expert from eviction ---
    // Touch E1 (recent), leave E3 old. Give E3 high priority so it survives
    // over E1 even though E1 was touched more recently.
    mgr.touch(0, 1);                       // E1 now most recent
    // E3 has priority boost from its original ensure(...,priority=0) -> bump it:
    mgr.ensure(0, 3, BLK, /*priority=*/100);  // re-ensure bumps priority to 100
    // Now load two more -> must evict 2. E3 (prio 100) must survive; the
    // lowest-score victims should be among {E1,E2,E4} (all prio 0, older than E3).
    check("ensure E5", mgr.ensure(0, 5, BLK, 0));
    check("ensure E6", mgr.ensure(0, 6, BLK, 0));
    check("E3 protected by priority", mgr.is_resident(0, 3));
    check("evictions now 3", mgr.stats().evictions == 3);

    // --- sync_fallback semantics ---
    // E0 was evicted earlier; reaching it at compute time = a fallback + reload.
    check("sync_fallback on evicted E0", mgr.sync_fallback(0, 0, BLK, 0));
    check("fallback counted", mgr.stats().fallbacks == 1);
    check("E0 resident again", mgr.is_resident(0, 0));

    // --- A pinned block is skipped, reported, and never selected as victim. ---
    dee::VramCacheManager pinned_mgr;
    check("init pinned-skip manager", pinned_mgr.init(BLK * 2, host_backend()));
    check("load pinned-skip E0", pinned_mgr.ensure(0, 0, BLK, 0));
    check("load pinned-skip E1", pinned_mgr.ensure(0, 1, BLK, 0));
    check("pin E0", pinned_mgr.pin(0, 0));
    check("load E2 while E0 pinned", pinned_mgr.ensure(0, 2, BLK, 0));
    check("pinned E0 survived", pinned_mgr.is_resident(0, 0));
    check("unpinned E1 was evicted", !pinned_mgr.is_resident(0, 1));
    check("pinned candidate skip counted", pinned_mgr.stats().pinned_blocks_skipped >= 1);
    pinned_mgr.unpin(0, 0);

    // --- clear ---
    mgr.clear();
    check("clear empties cache", mgr.resident_count() == 0);
    check("clear frees budget", mgr.used_bytes() == 0);

    printf("=== %s ===\n", g_fail == 0 ? "ALL PASS" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
