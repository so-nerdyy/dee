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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
    CHECK(s.profile.oracle_calls == static_cast<uint64_t>(cfg.num_tokens * cfg.num_layers),
          "Oracle internal profiler counts every measured prediction");
    CHECK(s.profile.oracle_ms[static_cast<size_t>(dee::OracleStage::Linear0)] > 0.0,
          "Oracle internal profiler measures first matrix operation");
    CHECK(s.cache_loads > 0, "cache performed loads");
    // with an 8-expert activation and 4-expert budget, eviction MUST occur
    CHECK(s.evictions > 0, "cache evictions occurred (budget < topk*depth pressure)");
    CHECK(engine.reset_external_profile(),
          "external profiler resets counters without rebuilding the engine");
    engine.set_external_token(7);
    std::vector<float> external_input(2048, 0.01f);
    std::vector<float> external_output(2048, 0.0f);
    engine.forward_layer(0, external_input.data(), external_output.data());
    const std::string external_profile = engine.external_profile_json(1.0);
    CHECK(external_profile.find("\"enabled\":true") != std::string::npos,
          "external profile JSON is enabled");
    CHECK(external_profile.find("\"token\":7") != std::string::npos,
          "external profile preserves caller token context");
    CHECK(external_profile.find("\"host_tensor_preparation\"") != std::string::npos,
          "external profile exposes host tensor preparation category");
    const dee::EngineStats memory_stats = engine.runtime_stats();
    CHECK(memory_stats.device_expert_cache_reserved_bytes > 0,
          "runtime stats expose the reserved expert-cache arena");
    CHECK(memory_stats.host_hidden_buffer_bytes > 0,
          "runtime stats expose native host work buffers");
    printf("    tok/s=%.3f peak_vram=%.1fMB loads=%llu evict=%llu fb=%llu\n",
           s.tok_per_sec, s.peak_vram / (1024.0*1024.0),
           (unsigned long long)s.cache_loads, (unsigned long long)s.evictions,
           (unsigned long long)s.fallbacks);
}

// Placement policies must not change a single output bit: the same
// caller-chosen experts run whatever the cache evicts and whatever is hinted.
// Scope, stated plainly: without CUDA the external MoE path reads the mmap
// directly, so on a CPU box this covers the config plumbing, the layer clock
// and the hint API (which must be a safe no-op here), not eviction itself.
// Eviction and hint byte-identity are covered at the cache level
// (test_host_pack_cache, test_vram_cache); the GPU path is gated per run by
// `aws/phase6/score.py --placement-exact`.
static bool run_policy(const char* eviction, const char* host_eviction,
                       size_t hint_lanes, const char* hint_source,
                       std::vector<float>* outputs) {
    dee::EngineConfig cfg;
    cfg.shard_path = "tests/data/ornith_moe256.safetensors";
    cfg.oracle_path = "";                       // caller-owned routing
    cfg.topk = 8;
    cfg.num_layers = 8;
    cfg.num_experts = 256;
    cfg.hidden = 2048;
    cfg.inter = 64;
    cfg.budget_bytes = 4 * 3ULL * 2048 * 64 * 4;  // 4 experts: forces eviction
    cfg.host_pack_cache_bytes = 64ull << 20;
    cfg.eviction_policy = eviction;
    cfg.host_eviction_policy = host_eviction;
    cfg.prefetch_hint_lanes = hint_lanes;
    cfg.prefetch_hint_source = hint_source;
    dee::Engine engine;
    if (!engine.init(cfg)) return false;
    std::vector<float> x(2048);
    for (int j = 0; j < 2048; ++j) x[j] = std::sin(0.37f * (float)(j + 1)) * 0.5f;
    std::vector<float> out(8 * 2048);
    outputs->clear();
    for (int step = 0; step < 4; ++step) {
        for (int layer = 0; layer < 8; ++layer) {
            const std::vector<int> experts = {
                3 + layer, 5, 7 + step, 9, 11, 200, 201 + (step & 1), 240 - layer};
            int hinted[3] = {5, 9, 250};         // right, right, wrong
            engine.prefetch_hint((layer + 1) % 8, hinted, 3);
            if (!engine.moe_forward_experts(layer, x.data(), out.data(), experts)) {
                return false;
            }
            outputs->insert(outputs->end(), out.begin(), out.end());
        }
    }
    return true;
}

static void test_placement_policies_are_exact() {
    std::vector<float> legacy, lru, next_use, hinted;
    if (!run_policy("rank_priority", "lru", 0, "off", &legacy)) {
        printf("  [SKIP] placement exactness (synthetic shard not present)\n");
        return;
    }
    CHECK(run_policy("lru", "lru", 0, "off", &lru), "lru engine runs");
    CHECK(run_policy("next_use", "next_use", 0, "off", &next_use),
          "next_use engine runs");
    CHECK(run_policy("next_use", "next_use", 2, "prev_step", &hinted),
          "next_use engine with prefetch hints runs");
    const auto same = [&](const std::vector<float>& other) {
        return other.size() == legacy.size() &&
               std::memcmp(other.data(), legacy.data(),
                           legacy.size() * sizeof(float)) == 0;
    };
    CHECK(!legacy.empty() && same(lru), "lru outputs are bit-identical to legacy");
    CHECK(same(next_use), "next_use outputs are bit-identical to legacy");
    CHECK(same(hinted), "outputs with prefetch hints are bit-identical to legacy");
    dee::EngineConfig bad;
    bad.shard_path = "tests/data/ornith_moe256.safetensors";
    bad.oracle_path = "";
    bad.host_eviction_policy = "belady";
    dee::Engine refused;
    CHECK(!refused.init(bad), "unknown host eviction policy is refused");
}

int main() {
    printf("=== dee.cpp Step 8 engine test ===\n");
    test_swiglu_tiny();
    test_engine_e2e();
    test_placement_policies_are_exact();
    if (g_fail) { printf("### %d FAILED ###\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
