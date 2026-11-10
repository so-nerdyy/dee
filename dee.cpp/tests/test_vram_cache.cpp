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
    mgr.set_debug_validation(true);
    check("budget correct", mgr.budget_bytes() == BUDGET);
    check("default eviction policy is RankPriority (legacy)",
          mgr.eviction_policy() == dee::EvictionPolicy::RankPriority);

    // --- Load 4 experts: all fit, no eviction ---
    for (int e = 0; e < 4; ++e) {
        check(("ensure E0..E3 #" + std::to_string(e)).c_str(),
              mgr.ensure(0, e, BLK, /*priority=*/0));
    }
    check("4 resident", mgr.resident_count() == 4);
    check("used == budget", mgr.used_bytes() == BUDGET);
    check("0 evictions so far", mgr.stats().evictions == 0);
    check("4 loads", mgr.stats().loads == 4);
    const uint64_t e0_first_generation = mgr.generation_of(0, 0);
    check("first residency generation is nonzero", e0_first_generation != 0);
    check("initial cache invariants valid", mgr.validate_invariants());

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

    // --- Oracle priority (RankPriority policy): protect a low-recency
    //     expert from eviction ---
    // Touch E1 (recent), leave E3 old. Give E3 high priority so it survives
    // over E1 even though E1 was touched more recently. (The default policy
    // is RankPriority, so this is the legacy-behavior variant.)
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
    check("reloaded E0 receives a newer generation",
          mgr.generation_of(0, 0) > e0_first_generation);

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
    check("balanced unpin succeeds", pinned_mgr.unpin(0, 0));
    check("unpin underflow is reported", !pinned_mgr.unpin(0, 0));
    check("unpin underflow leaves diagnostic", !pinned_mgr.last_error_message().empty());

    // A large allocation may require more than one victim. Generation remains
    // monotonic and the arena/range accounting must stay valid.
    dee::VramCacheManager multi_mgr;
    check("init multi-victim manager", multi_mgr.init(BLK * 2, host_backend()));
    multi_mgr.set_debug_validation(true);
    check("multi load E0", multi_mgr.ensure(0, 0, BLK / 2));
    check("multi load E1", multi_mgr.ensure(0, 1, BLK / 2));
    check("multi load E2", multi_mgr.ensure(0, 2, BLK / 2));
    check("multi load E3", multi_mgr.ensure(0, 3, BLK / 2));
    // Make the tail-adjacent blocks the first three LRU victims so their holes
    // coalesce into one range large enough for the replacement.
    multi_mgr.touch(0, 2);
    multi_mgr.touch(0, 1);
    multi_mgr.touch(0, 0);
    check("large load evicts multiple blocks", multi_mgr.ensure(0, 4, BLK + BLK / 2));
    check("multi-victim eviction count", multi_mgr.stats().evictions == 3);
    check("multi-victim invariants valid", multi_mgr.validate_invariants());
    // EnsureInfo must expose ALL victims in eviction order: recency after the
    // touches is E3 < E2 < E1 < E0, so victims are E3, E2, E1 in that order.
    {
        const dee::VramCacheManager::EnsureInfo& info = multi_mgr.last_ensure_info();
        check("evicted victim list has all 3 victims", info.evicted.size() == 3);
        check("victim[0] is E3 (coldest)",
              info.evicted.size() > 0 && info.evicted[0].layer == 0 && info.evicted[0].expert == 3);
        check("victim[1] is E2",
              info.evicted.size() > 1 && info.evicted[1].layer == 0 && info.evicted[1].expert == 2);
        check("victim[2] is E1 (last)",
              info.evicted.size() > 2 && info.evicted[2].layer == 0 && info.evicted[2].expert == 1);
        check("victim generations are nonzero",
              info.evicted.size() == 3 && info.evicted[0].generation != 0 &&
              info.evicted[1].generation != 0 && info.evicted[2].generation != 0);
        // Back-compat scalar fields keep the LAST victim.
        check("evicted_key is last victim E1",
              info.evicted_key.layer == 0 && info.evicted_key.expert == 1);
        check("evicted_generation is last victim's",
              info.evicted.size() == 3 &&
              info.evicted_generation == info.evicted[2].generation);
        check("evicted list is truthy when non-empty", info.evicted ? true : false);
    }
    // A subsequent no-eviction ensure clears the victim list.
    check("multi re-ensure E4 hits", multi_mgr.ensure(0, 4, BLK + BLK / 2));
    check("victim list cleared on hit", multi_mgr.last_ensure_info().evicted.empty());
    check("evicted flag false on hit", multi_mgr.last_ensure_info().evicted ? false : true);

    // --- Oversized ensure: early-reject BEFORE evicting; residents survive ---
    dee::VramCacheManager over_mgr;
    check("init oversize manager", over_mgr.init(BUDGET, host_backend()));
    over_mgr.set_debug_validation(true);
    check("over load E0", over_mgr.ensure(0, 0, BLK, 0));
    check("over load E1", over_mgr.ensure(0, 1, BLK, 0));
    memset(over_mgr.data(0, 0), 'x', BLK);
    memset(over_mgr.data(0, 1), 'y', BLK);
    check("oversize ensure rejected", !over_mgr.ensure(0, 2, BUDGET + 1, 0));
    check("budget_rejections counted", over_mgr.stats().budget_rejections == 1);
    check("oversize caused no evictions", over_mgr.stats().evictions == 0);
    check("oversize left error message", !over_mgr.last_error_message().empty());
    check("E0 survived oversize reject", over_mgr.is_resident(0, 0));
    check("E1 survived oversize reject", over_mgr.is_resident(0, 1));
    check("resident bytes intact", over_mgr.used_bytes() == BLK * 2);
    check("E0 data intact", ((char*)over_mgr.data(0, 0))[10] == 'x');
    check("E1 data intact", ((char*)over_mgr.data(0, 1))[10] == 'y');
    check("no victim list on reject", over_mgr.last_ensure_info().evicted.empty());
    check("invariants still valid", over_mgr.validate_invariants());
    // Cache still fully usable afterwards.
    check("post-reject ensure works", over_mgr.ensure(0, 2, BLK, 0));
    check("E2 resident after reject", over_mgr.is_resident(0, 2));

    // --- Same-key different-size hit: fail closed ---
    dee::VramCacheManager size_mgr;
    check("init size-mismatch manager", size_mgr.init(BUDGET, host_backend()));
    check("size load E0", size_mgr.ensure(0, 0, BLK, 0));
    check("same-key larger size rejected", !size_mgr.ensure(0, 0, BLK * 2, 0));
    check("size_mismatches counted", size_mgr.stats().size_mismatches == 1);
    check("mismatch left error message", !size_mgr.last_error_message().empty());
    check("mismatch is not a hit", size_mgr.stats().hits == 0);
    check("E0 still resident after mismatch", size_mgr.is_resident(0, 0));
    check("E0 size unchanged", size_mgr.size_of(0, 0) == BLK);
    check("same-key smaller size rejected", !size_mgr.ensure(0, 0, BLK / 2, 0));
    check("size_mismatches counted again", size_mgr.stats().size_mismatches == 2);
    check("same-key same-size still hits", size_mgr.ensure(0, 0, BLK, 0));
    check("hit counted after correct size", size_mgr.stats().hits == 1);
    // Oversize on a RESIDENT key is also a size mismatch (hit path first).
    check("resident-key oversize is size mismatch",
          !size_mgr.ensure(0, 0, BUDGET + 1, 0));
    check("size_mismatches counted third", size_mgr.stats().size_mismatches == 3);
    check("budget_rejections untouched by hit path",
          size_mgr.stats().budget_rejections == 0);

    // --- Eviction policy A/B: same script, different victim ---
    // Script: E0..E3 loaded (ticks 1..4), E3 re-ensured with priority=100
    // (tick 5), then E0,E1,E2 touched (ticks 6,7,8). Recency order is then
    // E3(oldest, but prio 100) < E0 < E1 < E2(newest).
    {
        dee::VramCacheManager rec_mgr;   // Recency: priority must NOT protect
        check("init recency manager", rec_mgr.init(BUDGET, host_backend()));
        rec_mgr.set_eviction_policy(dee::EvictionPolicy::Recency);
        check("policy set to Recency",
              rec_mgr.eviction_policy() == dee::EvictionPolicy::Recency);
        for (int e = 0; e < 4; ++e)
            check(("rec load E" + std::to_string(e)).c_str(),
                  rec_mgr.ensure(0, e, BLK, 0));
        check("rec E3 high priority", rec_mgr.ensure(0, 3, BLK, 100));
        rec_mgr.touch(0, 0);
        rec_mgr.touch(0, 1);
        rec_mgr.touch(0, 2);
        check("rec ensure E4 evicts", rec_mgr.ensure(0, 4, BLK, 0));
        check("Recency evicts E3 despite priority 100", !rec_mgr.is_resident(0, 3));
        check("Recency keeps E0", rec_mgr.is_resident(0, 0));
        check("Recency keeps E1", rec_mgr.is_resident(0, 1));
        check("Recency keeps E2", rec_mgr.is_resident(0, 2));
        check("Recency victim list is E3",
              rec_mgr.last_ensure_info().evicted.size() == 1 &&
              rec_mgr.last_ensure_info().evicted[0].expert == 3);
        // Strict LRU order: next ensure evicts E0 (now oldest).
        check("rec ensure E5 evicts", rec_mgr.ensure(0, 5, BLK, 0));
        check("Recency evicts E0 next (strict LRU)", !rec_mgr.is_resident(0, 0));

        dee::VramCacheManager rank_mgr;  // RankPriority: priority DOES protect
        check("init rank manager", rank_mgr.init(BUDGET, host_backend()));
        check("rank default policy is RankPriority",
              rank_mgr.eviction_policy() == dee::EvictionPolicy::RankPriority);
        for (int e = 0; e < 4; ++e)
            check(("rank load E" + std::to_string(e)).c_str(),
                  rank_mgr.ensure(0, e, BLK, 0));
        check("rank E3 high priority", rank_mgr.ensure(0, 3, BLK, 100));
        rank_mgr.touch(0, 0);
        rank_mgr.touch(0, 1);
        rank_mgr.touch(0, 2);
        check("rank ensure E4 evicts", rank_mgr.ensure(0, 4, BLK, 0));
        check("RankPriority protects E3 (prio 100)", rank_mgr.is_resident(0, 3));
        check("RankPriority evicts E0 (lowest score)", !rank_mgr.is_resident(0, 0));
        check("RankPriority victim list is E0",
              rank_mgr.last_ensure_info().evicted.size() == 1 &&
              rank_mgr.last_ensure_info().evicted[0].expert == 0);
    }

    // --- evict_key: invalidation, not LRU eviction ---
    dee::VramCacheManager inv_mgr;
    check("init evict_key manager", inv_mgr.init(BUDGET, host_backend()));
    inv_mgr.set_debug_validation(true);
    check("inv load E0", inv_mgr.ensure(0, 0, BLK, 0));
    check("inv load E1", inv_mgr.ensure(0, 1, BLK, 0));
    check("evict_key removes resident", inv_mgr.evict_key(0, 0));
    check("E0 gone after evict_key", !inv_mgr.is_resident(0, 0));
    check("invalidations counted", inv_mgr.stats().invalidations == 1);
    check("evictions NOT counted", inv_mgr.stats().evictions == 0);
    check("arena space reclaimed", inv_mgr.used_bytes() == BLK);
    check("evict_key absent key refused", !inv_mgr.evict_key(0, 0));
    check("absent evict_key counts nothing", inv_mgr.stats().invalidations == 1);
    check("pin E1", inv_mgr.pin(0, 1));
    check("evict_key pinned refused", !inv_mgr.evict_key(0, 1));
    check("pinned E1 still resident", inv_mgr.is_resident(0, 1));
    check("pinned refusal counts nothing", inv_mgr.stats().invalidations == 1);
    check("pinned refusal leaves bytes", inv_mgr.used_bytes() == BLK);
    check("unpin E1", inv_mgr.unpin(0, 1));
    check("evict_key works after unpin", inv_mgr.evict_key(0, 1));
    check("invalidations counted twice", inv_mgr.stats().invalidations == 2);
    check("invariants valid after evict_key", inv_mgr.validate_invariants());
    check("reclaimed arena reusable", inv_mgr.ensure(0, 2, BLK * 2, 0));
    check("E2 resident", inv_mgr.is_resident(0, 2));

    // --- pinned_blocks_skipped: once per distinct block per ensure() ---
    // Two pinned + two unpinned blocks; an ensure needing TWO evictions scans
    // the pinned pair twice but must count each only once.
    dee::VramCacheManager skip_mgr;
    check("init skip-count manager", skip_mgr.init(BUDGET, host_backend()));
    check("skip load E0", skip_mgr.ensure(0, 0, BLK, 0));
    check("skip load E1", skip_mgr.ensure(0, 1, BLK, 0));
    check("skip load E2", skip_mgr.ensure(0, 2, BLK, 0));
    check("skip load E3", skip_mgr.ensure(0, 3, BLK, 0));
    check("pin skip E0", skip_mgr.pin(0, 0));
    check("pin skip E1", skip_mgr.pin(0, 1));
    check("skip ensure 2-block load", skip_mgr.ensure(0, 4, BLK * 2, 0));
    check("two unpinned victims evicted", skip_mgr.stats().evictions == 2);
    check("E2 evicted", !skip_mgr.is_resident(0, 2));
    check("E3 evicted", !skip_mgr.is_resident(0, 3));
    check("E4 resident", skip_mgr.is_resident(0, 4));
    check("pinned pair survived", skip_mgr.is_resident(0, 0) && skip_mgr.is_resident(0, 1));
    check("each pinned block counted once per ensure",
          skip_mgr.stats().pinned_blocks_skipped == 2);
    // A second multi-iteration ensure counts each pinned block once MORE
    // (per-ensure semantics), not per scan iteration.
    check("skip ensure another 2-block load", skip_mgr.ensure(0, 5, BLK * 2, 0));
    check("per-ensure pinned counting", skip_mgr.stats().pinned_blocks_skipped == 4);
    check("E4 evicted by E5 load", !skip_mgr.is_resident(0, 4));

    // --- clear ---
    mgr.clear();
    check("clear empties cache", mgr.resident_count() == 0);
    check("clear frees budget", mgr.used_bytes() == 0);
    check("clear leaves invariants valid", mgr.validate_invariants());

    printf("=== %s ===\n", g_fail == 0 ? "ALL PASS" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
