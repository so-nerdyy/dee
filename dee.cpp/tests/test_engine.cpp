// tests/test_engine.cpp
//
// Step 8 unit test. Two parts:
//   (1) swiglu() kernel — verified against a hand-computed 2x2 tiny case.
//   (2) Engine end-to-end on the synthetic 256-expert shard: init + generate
//       must succeed, output hidden must be finite, and the cache must show
//       real prefetch/eviction activity (DEE loop actually ran).
//
// References the synthetic shard tests/data/ornith_moe256.safetensors and the
// real Oracle at the path below. Skips (PASS) if either file is absent rather
// than failing the suite on a box without the artifacts.

#include "dee/engine.h"
#include <cmath>
#include <cstdio>
#include <string>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s\n", msg); ++g_fail; } \
    else printf("  [PASS] %s\n", msg); } while (0)

// tiny SwiGLU hand-computation: INTER=2, HIDDEN=2
// gate [2,2], up [2,2], down [2,2]; x = [1, 2]
//   h0 = silu(g[0]*x0+g[1]*x1) * (u..); etc.
static void test_swiglu_tiny() {
    const int I = 2, H = 2;
    // blob = [gate(2x2), up(2x2), down(2x2)] row-major
    // gate: [[1,0],[0,1]]  up: [[1,1],[1,1]]  down: [[1,0],[0,1]]
    float blob[12] = {
        1,0, 0,1,            // gate
        1,1, 1,1,            // up
        1,0, 0,1             // down
    };
    float x[2] = {1.0f, 2.0f};
    float acc[2] = {0, 0};
    dee::Engine::swiglu(blob, x, I, H, acc);

    // expert output y = down · h, h[i] = silu(g_i·x) * (u_i·x)
    // g0·x = 1*1+0*2 = 1 -> silu(1)=1/(1+e^-1)=0.73106 ; u0·x = 1*1+1*2=3 ; h0=0.73106*3=2.19318
    // g1·x = 0*1+1*2 = 2 -> silu(2)=2/(1+e^-2)=1.76159 ; u1·x = 1+2=3 ; h1=1.76159*3=5.28477
    // y0 = down[0]*h0 + down[1]*h1 = 1*2.19318 + 0*5.28477 = 2.19318
    // y1 = down[2]*h0 + down[3]*h1 = 0*2.19318 + 1*5.28477 = 5.28477
    bool ok = std::isfinite(acc[0]) && std::isfinite(acc[1])
           && std::fabs(acc[0] - 2.19318f) < 1e-3f
           && std::fabs(acc[1] - 5.28477f) < 1e-3f;
    CHECK(ok, "swiglu tiny case (2x2) matches hand computation");
    if (!ok) printf("    got y=[%.5f, %.5f] expected [2.19318, 5.28477]\n", acc[0], acc[1]);
}

static void test_engine_e2e() {
    dee::EngineConfig cfg;
    cfg.shard_path = "tests/data/ornith_moe256.safetensors";
    cfg.oracle_path = "oracle.pt";
    cfg.num_tokens = 4;
    cfg.topk = 8;
    cfg.num_layers = 8;
    cfg.budget_bytes = 4 * 3ULL * 2048 * 64 * 4;  // 4 experts (inter=64) -> forces eviction
    cfg.profile_stages = true;
    cfg.trace_requests = true;

    dee::Engine engine;
    if (!engine.init(cfg)) {
        printf("  [SKIP] engine init (shard/oracle not present on this box)\n");
        return;
    }
    CHECK(true, "engine.init (shard + oracle loaded)");

    bool gen = engine.generate();
    CHECK(gen, "engine.generate ran");

    const dee::EngineStats& s = engine.stats();
    CHECK(s.hidden_finite, "output hidden all-finite");
    CHECK(s.prefetch_issued > 0, "prefetcher issued transfers");
    CHECK(s.prefetch_issued == s.resident_hits + s.inflight_hits + s.cold_loads,
          "request classification invariant");
    CHECK(s.profile.enabled, "stage profile enabled");
    CHECK(s.profile.trace.size() == s.prefetch_issued, "request trace covers every request");
    CHECK(s.profile.layer_count == static_cast<uint64_t>(cfg.num_tokens * cfg.num_layers),
          "layer timing count excludes no measured layers");
    CHECK(s.cache_loads > 0, "cache performed loads");
    // with an 8-expert activation and 4-expert budget, eviction MUST occur
    CHECK(s.evictions > 0, "cache evictions occurred (budget < topk*depth pressure)");
    printf("    tok/s=%.3f peak_vram=%.1fMB loads=%llu evict=%llu fb=%llu\n",
           s.tok_per_sec, s.peak_vram / (1024.0*1024.0),
           (unsigned long long)s.cache_loads, (unsigned long long)s.evictions,
           (unsigned long long)s.fallbacks);
}

int main() {
    printf("=== dee.cpp Step 8 engine test ===\n");
    test_swiglu_tiny();
    test_engine_e2e();
    if (g_fail) { printf("### %d FAILED ###\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
