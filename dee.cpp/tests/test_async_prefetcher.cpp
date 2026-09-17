// tests/test_async_prefetcher.cpp
//
// Step 6 — AsyncPrefetcher CPU-mock test (no CUDA toolkit needed).
//
// Wires the REAL Step 4 (WeightMmap) + Step 5 (VramCacheManager) + Step 6
// (AsyncPrefetcher) on the mock backend: open the synthetic Ornith shard,
// resolve an expert's BF16 weights, prefetch them into the VRAM cache on the
// decoupled mock stream, wait(), and confirm the bytes arrived intact.
//
// Also validates: (a) independent transfers don't force blocking on each other
// — the compute stream only waits on the expert it needs; (b) sync_fallback
// accounting when compute reaches an unstaged / unfinished expert.
//
// Build (no cmake):
//   g++ -std=c++17 -I../include test_async_prefetcher.cpp \
//       ../src/weight_mmap.cpp ../src/json_min.cpp ../src/vram_cache.cpp \
//       ../src/async_prefetcher.cpp -o test_async_prefetcher

#include "dee/weight_mmap.h"
#include "dee/vram_cache.h"
#include "dee/async_prefetcher.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

static int g_fail = 0;
static void check(const char* what, bool cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_fail;
}

static dee::Arena::Backend host_backend() {
    dee::Arena::Backend be;
    be.kind = "host";
    be.alloc = [](size_t n) -> void* { return n ? malloc(n) : nullptr; };
    be.free  = [](void* p) { if (p) free(p); };
    return be;
}

// BF16 -> float (must match weight_mmap.cpp)
static float bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16; float f; std::memcpy(&f, &u, 4); return f;
}

// Capability probe: the parallel B0b workstream adds
// VramCacheManager::evict_key.  The submit-failure assertions below adapt to
// whichever side of that landing this build sees.
template <typename C>
static auto has_evict_key_impl(int)
    -> decltype(std::declval<C&>().evict_key(0, 0), std::true_type{});
template <typename C>
static std::false_type has_evict_key_impl(...);
static constexpr bool has_evict_key =
    decltype(has_evict_key_impl<dee::VramCacheManager>(0))::value;

int main() {
    printf("=== dee.cpp Step 6 AsyncPrefetcher (CPU mock) test ===\n");

    dee::WeightMmap mmap;
    check("open synthetic shard",
          mmap.open("tests/data/layer0_shard.safetensors"));
    dee::TensorResolver resolver;
    resolver.register_shard(&mmap);

    // VRAM budget: 2 experts worth (256 B each) so we exercise eviction too.
    const size_t BLK = 256;
    dee::VramCacheManager cache;
    check("init cache", cache.init(BLK * 2, host_backend()));

    dee::AsyncPrefetcher prefetcher(cache);
    check("init mock stream", prefetcher.init(/*use_cuda=*/false));
    check("mock mode (no cuda)", !prefetcher.using_cuda());

    // --- Prefetch Layer0/Expert0 gate_proj (BF16, 256 B) ---
    dee::TensorView v = resolver.resolve_expert(0, 0, dee::TensorResolver::GATE_PROJ);
    check("resolved expert0 gate_proj", v.ok());
    prefetcher.begin_batch();
    long id = prefetcher.prefetch(0, 0, v.data, v.nbytes, /*priority=*/0);
    check("prefetch issued id>=0", id >= 0);
    long duplicate_id = prefetcher.prefetch(0, 0, v.data, v.nbytes, /*priority=*/0);
    check("duplicate in-flight request reuses transfer", duplicate_id == id);
    check("in_flight == 1", prefetcher.in_flight() == 1);
    check("not yet resident-done (mock: copy is lazy)", cache.is_resident(0, 0));

    // wait() should drain the mock stream and copy the bytes in.
    check("wait() returns ready", prefetcher.wait(0, 0));
    check("expert0 now resident", cache.is_resident(0, 0));

    // Verify the bytes arrived intact: read back BF16 and compare to known gate values.
    const float exp_gate[5] = {0.5f, -1.25f, 2.0f, -3.75f, 4.25f};
    const uint16_t* back = (const uint16_t*)cache.data(0, 0);
    bool gate_ok = true;
    printf("    expert0 gate_proj first5: ");
    for (int i = 0; i < 5; ++i) {
        float f = bf16_to_f32(back[i]);
        printf("%.4f ", f);
        if (std::fabs(f - exp_gate[i]) > 1e-2f) gate_ok = false;
    }
    printf("\n");
    check("prefetched weights match source (end-to-end mmap->cache)", gate_ok);

    // --- Decoupling: issue expert1 without waiting on it, then a DIFFERENT
    //     expert's wait() should not require expert1 to be "needed" by compute.
    //     Here we just confirm issuing more transfers doesn't mutate expert0. ---
    dee::TensorView v1 = resolver.resolve_expert(0, 1, dee::TensorResolver::UP_PROJ);
    prefetcher.begin_batch();
    long id1 = prefetcher.prefetch(0, 1, v1.data, v1.nbytes, 0);
    check("prefetch expert1 issued", id1 >= 0);
    check("expert0 data still intact after expert1 issued",
          bf16_to_f32(((const uint16_t*)cache.data(0,0))[0]) == 0.5f);

    // wait on expert1 only
    check("wait expert1 ready", prefetcher.wait(0, 1));
    const float exp_up[5] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    const uint16_t* up = (const uint16_t*)cache.data(0, 1);
    bool up_ok = true;
    for (int i = 0; i < 5; ++i) if (std::fabs(bf16_to_f32(up[i]) - exp_up[i]) > 1e-2f) up_ok = false;
    check("expert1 up_proj matches source", up_ok);

    // --- Cache-hit lifetime: re-request resident expert0, then stage a cold
    //     expert2 in the same batch.  The hit must be pinned until wait(0,0),
    //     or the cold allocation could evict it before compute consumes it. ---
    dee::TensorView v2 = resolver.resolve_expert(0, 2, dee::TensorResolver::DOWN_PROJ);
    check("resolved expert2 down_proj", v2.ok());
    prefetcher.begin_batch();
    const uint64_t hits_before = cache.stats().hits;
    long id0_hit = prefetcher.prefetch(0, 0, v.data, v.nbytes, 1);
    check("resident expert0 request reuses transfer", id0_hit == id);
    check("resident expert0 request increments cache hit accounting", cache.stats().hits == hits_before + 1);
    long id2 = prefetcher.prefetch(0, 2, v2.data, v2.nbytes, 0);
    check("prefetch expert2 issued", id2 >= 0);
    check("resident hit survives later cold staging", cache.is_resident(0, 0));
    check("wait resident hit releases its staging pin", prefetcher.wait(0, 0));
    check("wait expert2 ready", prefetcher.wait(0, 2));
    check("resident count capped at 2", cache.resident_count() <= 2);
    check("evictions happened (budget)", cache.stats().evictions >= 1);

    // --- sync_fallback accounting: reaching an expert we never issued. ---
    size_t fb0 = prefetcher.stats().fallbacks;
    bool r = prefetcher.wait(5, 99);  // never staged
    check("wait on unstaged expert reports not-ready", !r);
    check("fallback counted", prefetcher.stats().fallbacks == fb0 + 1);

    // --- synchronize_all drains everything; cache sanity. ---
    prefetcher.synchronize_all();
    check("after sync, expert2 resident", cache.is_resident(0, 2));
    const dee::AsyncPrefetcher::Stats& stats = prefetcher.stats();
    check("request accounting invariant", prefetcher.accounting_valid());
    check("request classifications total five", stats.requests == 5);
    check("resident hit classified", stats.resident_hits == 1);
    check("in-flight hit classified", stats.inflight_hits == 1);
    check("cold loads classified", stats.cold_loads == 3);
    check("same-batch duplicate classified", stats.duplicate_requests == 1);

    // A controlled full-resident profile preloads the cache and then resets
    // transfer/event state before measurement. Re-requesting such a resident
    // block must create a fresh completed transfer record and hold a pin until
    // wait(), without issuing another copy.
    prefetcher.reset();
    prefetcher.reset_stats();
    cache.reset_stats();
    prefetcher.begin_batch();
    long preloaded_hit = prefetcher.prefetch(0, 2, v2.data, v2.nbytes, 0);
    check("preloaded resident is reusable after transfer reset", preloaded_hit >= 0);
    check("preloaded resident wait succeeds", prefetcher.wait(0, 2));
    check("preloaded request classified as resident hit",
          prefetcher.stats().requests == 1 && prefetcher.stats().resident_hits == 1 &&
          prefetcher.stats().cold_loads == 0);
    check("preloaded resident does not reload", cache.stats().hits == 1 && cache.stats().loads == 0);

    // --- Transfer-ledger lifecycle foundation. ---
    // A one-block cache makes the eviction ordering deterministic. The first
    // transfer is consumed before eviction; the second is deliberately not.
    dee::StageProfiler ledger_profiler;
    ledger_profiler.configure(true, true, BLK, 3);
    dee::VramCacheManager ledger_cache;
    check("init ledger cache", ledger_cache.init(BLK, host_backend()));
    ledger_cache.set_debug_validation(true);
    ledger_cache.set_profiler(&ledger_profiler);
    dee::AsyncPrefetcher ledger_prefetcher(ledger_cache);
    check("init ledger prefetcher", ledger_prefetcher.init(false));
    ledger_prefetcher.set_profiler(&ledger_profiler);

    ledger_prefetcher.begin_batch();
    check("ledger cold transfer E0",
          ledger_prefetcher.prefetch(0, 0, v.data, v.nbytes, 0, 1, 7) >= 0);
    const uint64_t ledger_e0_generation = ledger_cache.generation_of(0, 0);
    check("ledger generation nonzero", ledger_e0_generation != 0);
    check("ledger wait E0", ledger_prefetcher.wait(0, 0));
    ledger_prefetcher.mark_consumed(0, 0);

    ledger_prefetcher.begin_batch();
    check("ledger cold transfer E1",
          ledger_prefetcher.prefetch(0, 1, v1.data, v1.nbytes, 0, 2, 7) >= 0);
    check("ledger wait E1", ledger_prefetcher.wait(0, 1));
    ledger_prefetcher.begin_batch();
    check("ledger cold transfer E2",
          ledger_prefetcher.prefetch(0, 2, v2.data, v2.nbytes, 0, 3, 7) >= 0);
    check("ledger wait E2", ledger_prefetcher.wait(0, 2));
    std::string ledger_invariant_error;
    check("ledger cache and prefetch metadata remain consistent after eviction",
          ledger_cache.validate_invariants(&ledger_invariant_error) &&
          ledger_prefetcher.validate_invariants(&ledger_invariant_error));

    const auto& ledger_stats = ledger_prefetcher.stats();
    dee::StageProfile ledger_profile = ledger_profiler.finish(
        0.0, ledger_stats.resident_hits, ledger_stats.inflight_hits,
        ledger_stats.cold_loads, ledger_stats.duplicate_requests,
        ledger_cache.stats().evictions,
        ledger_cache.stats().pinned_blocks_skipped);
    check("ledger trace has three requests", ledger_profile.trace.size() == 3);
    if (ledger_profile.trace.size() == 3) {
        const auto& first = ledger_profile.trace[0];
        const auto& second = ledger_profile.trace[1];
        check("cold trace exposes before/after cache entries",
              first.cache_entries_before == 0 && first.cache_entries_after == 1);
        check("cold trace exposes generation and held pin",
              first.generation == ledger_e0_generation && first.pin_count == 1);
        check("cold trace marks a launched transfer", first.transfer_launched);
        check("consumed transfer remains consumed after eviction",
              first.consumed && !first.evicted_before_use);
        check("unused transfer is marked evicted before use",
              second.transfer_launched && !second.consumed && second.evicted_before_use);
        check("reload generations are monotonic",
              first.generation < second.generation &&
              second.generation < ledger_profile.trace[2].generation);
    }
    const std::string ledger_json = dee::stage_profile_json(ledger_profile, true);
    check("ledger JSON exposes generation and lifecycle fields",
          ledger_json.find("\"cache_bytes_after\":") != std::string::npos &&
          ledger_json.find("\"cache_entries_after\":") != std::string::npos &&
          ledger_json.find("\"generation\":") != std::string::npos &&
          ledger_json.find("\"pin_count\":") != std::string::npos &&
          ledger_json.find("\"transfer_launched\":true") != std::string::npos &&
          ledger_json.find("\"consumed\":true") != std::string::npos &&
          ledger_json.find("\"evicted_before_use\":true") != std::string::npos);
    ledger_prefetcher.set_profiler(nullptr);
    ledger_cache.set_profiler(nullptr);

    // ================= Phase-4 repair gates =================
    dee::VramCacheManager p4_cache;
    check("init p4 cache", p4_cache.init(BLK * 4, host_backend()));
    p4_cache.set_debug_validation(true);
    dee::AsyncPrefetcher p4(p4_cache);
    check("init p4 prefetcher", p4.init(false));

    std::vector<uint8_t> srcA(BLK, 0xA5), srcB(BLK, 0x3C), srcC(BLK, 0x5A);

    // --- (P4-2) LLP64 cross-layer key collision: (0,5) and (1,5) share the
    //     low 32 bits; a 32-bit `long` key truncates them into one transfer. ---
    p4.begin_batch();
    long id_l0 = p4.prefetch(0, 5, srcA.data(), srcA.size(), 0);
    long id_l1 = p4.prefetch(1, 5, srcB.data(), srcB.size(), 0);
    check("cross-layer same-expert transfers are distinct",
          id_l0 >= 0 && id_l1 >= 0 && id_l0 != id_l1);
    check("cross-layer same-expert both in flight", p4.in_flight() == 2);
    check("cross-layer same-expert not a duplicate",
          p4.stats().duplicate_requests == 0);

    // --- (P4-1) staging-slot lifetime: a pending transfer owns its slot until
    //     its completion is observed (mock: drain is the observation point;
    //     CUDA: retire_completed() gates it on cudaEventQuery). ---
    check("pending transfers hold staging slots", p4.staging_slots_in_use() == 2);
    check("layer0 wait", p4.wait(0, 5));
    check("layer0 bytes are its own (no cross-layer alias)",
          p4_cache.is_resident(0, 5) &&
          std::memcmp(p4_cache.data(0, 5), srcA.data(), srcA.size()) == 0);
    check("sibling slot still held until its completion is observed",
          p4.staging_slots_in_use() == 1);
    check("layer1 wait_on_stream (mock degrades to drain)",
          p4.wait_on_stream(1, 5, nullptr));
    check("slot released once completion observed",
          p4.staging_slots_in_use() == 0);
    check("layer1 bytes are its own",
          p4_cache.is_resident(1, 5) &&
          std::memcmp(p4_cache.data(1, 5), srcB.data(), srcB.size()) == 0);

    // --- (P4-3) multi-victim ensure: one 2-block request into a full 2-block
    //     cache evicts BOTH residents; every victim key must leave the map. ---
    dee::VramCacheManager ev_cache;
    check("init multi-evict cache", ev_cache.init(BLK * 2, host_backend()));
    ev_cache.set_debug_validation(true);
    dee::AsyncPrefetcher evp(ev_cache);
    check("init multi-evict prefetcher", evp.init(false));
    std::vector<uint8_t> s1(BLK, 0x11), s2(BLK, 0x22), sbig(BLK * 2, 0x33);
    evp.begin_batch();
    check("victim A issued", evp.prefetch(0, 10, s1.data(), s1.size(), 0) >= 0);
    check("victim A waited", evp.wait(0, 10));
    check("victim B issued", evp.prefetch(0, 11, s2.data(), s2.size(), 0) >= 0);
    check("victim B waited", evp.wait(0, 11));
    check("both victims resident",
          ev_cache.is_resident(0, 10) && ev_cache.is_resident(0, 11));
    evp.begin_batch();
    check("oversize request issued", evp.prefetch(0, 12, sbig.data(), sbig.size(), 0) >= 0);
    check("multi-victim ensure evicted both",
          !ev_cache.is_resident(0, 10) && !ev_cache.is_resident(0, 11));
    check("two evictions counted", ev_cache.stats().evictions >= 2);
    std::string ev_err;
    check("invariants clean after multi-victim ensure",
          evp.validate_invariants(&ev_err) && ev_cache.validate_invariants(&ev_err));
    if (!ev_err.empty()) printf("    invariant: %s\n", ev_err.c_str());
    check("oversize wait", evp.wait(0, 12));
    // Re-requesting an evicted victim must be a fresh cold load — a stale
    // key_to_idx_ entry would surface as an in-flight/resident reuse instead.
    const uint64_t ev_reqs = evp.stats().requests;
    const uint64_t ev_colds = evp.stats().cold_loads;
    long t10b = evp.prefetch(0, 10, s1.data(), s1.size(), 0);
    check("victim re-prefetch is a fresh cold load",
          t10b >= 0 && evp.stats().requests == ev_reqs + 1 &&
          evp.stats().cold_loads == ev_colds + 1);
    check("victim re-prefetch verifies",
          evp.wait(0, 10) &&
          std::memcmp(ev_cache.data(0, 10), s1.data(), s1.size()) == 0);
    check("invariants still clean", evp.validate_invariants(&ev_err));

    // --- (P4-4) submit failure must invalidate the resident garbage block. ---
    p4.debug_fail_next_submit();
    const uint64_t sf0 = p4.stats().submit_failures;
    long bad = p4.prefetch(0, 7, srcC.data(), srcC.size(), 0);
    check("injected submit failure returns -1", bad == -1);
    check("submit failure counted", p4.stats().submit_failures == sf0 + 1);
    check("failed transfer abandoned", p4.stats().transfers_abandoned >= 1);
    if (has_evict_key) {
        check("uninitialized block invalidated", !p4_cache.is_resident(0, 7));
        check("no un-invalidated resident",
              p4.stats().submit_failed_residents == 0);
        // The next request for the key must cold-load cleanly, not hit the
        // abandoned block.
        const uint64_t rf_colds = p4.stats().cold_loads;
        long good = p4.prefetch(0, 7, srcC.data(), srcC.size(), 0);
        check("re-prefetch cold-loads after invalidation",
              good >= 0 && p4.stats().cold_loads == rf_colds + 1);
        check("re-prefetch verifies bytes",
              p4.wait(0, 7) &&
              std::memcmp(p4_cache.data(0, 7), srcC.data(), srcC.size()) == 0);
    } else {
        printf("    (evict_key not landed: abandoned block left resident + flagged)\n");
        check("un-invalidated resident flagged",
              p4.stats().submit_failed_residents >= 1);
    }

    // --- (P4-5) wait/wait_on_stream counter parity. ---
    const uint64_t p4_w0 = p4.stats().waited, p4_fb0 = p4.stats().fallbacks;
    check("parity prefetch issued", p4.prefetch(0, 8, srcC.data(), srcC.size(), 0) >= 0);
    check("wait_on_stream on pending transfer", p4.wait_on_stream(0, 8, nullptr));
    check("wait_on_stream counted as a wait", p4.stats().waited == p4_w0 + 1);
    check("satisfied wait is not a fallback", p4.stats().fallbacks == p4_fb0);
    check("never-staged wait_on_stream returns false", !p4.wait_on_stream(9, 9, nullptr));
    check("never-staged wait_on_stream counts a fallback",
          p4.stats().fallbacks == p4_fb0 + 1 && p4.stats().waited == p4_w0 + 2);
    std::string p4_err;
    check("p4 invariants clean at end", p4.validate_invariants(&p4_err));
    if (!p4_err.empty()) printf("    invariant: %s\n", p4_err.c_str());

    printf("=== %s ===\n", g_fail == 0 ? "ALL PASS" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
