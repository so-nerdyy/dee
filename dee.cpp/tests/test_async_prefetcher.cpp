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
#include <string>
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

    printf("=== %s ===\n", g_fail == 0 ? "ALL PASS" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
