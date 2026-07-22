#include "dee/engine.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int failures = 0;
#define CHECK(condition, message) do { \
    if (condition) std::printf("  [PASS] %s\n", message); \
    else { std::printf("  [FAIL] %s\n", message); ++failures; } \
} while (0)

int main() {
    dee::EngineConfig cfg;
    cfg.shard_paths = {
        "tests/data/split-router/split-a.safetensors",
        "tests/data/split-router/split-b.safetensors",
    };
    cfg.oracle_path = "";
    cfg.num_tokens = 1;
    cfg.num_layers = 6;
    cfg.base_layer = 5;
    cfg.hidden = 4;
    cfg.inter = 2;
    cfg.num_experts = 3;
    cfg.topk = 2;
    cfg.budget_bytes = 2 * 3ULL * 2 * 4 * sizeof(float);

    dee::Engine engine;
    CHECK(engine.init(cfg), "layer-local engine opens both shards");
    if (failures) return 1;

    const float hidden[4] = {1, 2, 3, 4};
    float logits[3]{};
    float weights[2]{};
    int experts[2]{};
    CHECK(engine.route_topk(5, hidden, logits, weights, experts),
          "native router executes");
    CHECK(std::fabs(logits[0] - 1.0f) < 1e-6f &&
          std::fabs(logits[1] - 2.0f) < 1e-6f &&
          std::fabs(logits[2] - 3.0f) < 1e-6f,
          "native router logits use genuine layer tensor");
    CHECK(experts[0] == 2 && experts[1] == 1,
          "native router preserves descending top-k order");
    CHECK(std::fabs(weights[0] - 0.7310586f) < 1e-5f &&
          std::fabs(weights[1] - 0.2689414f) < 1e-5f,
          "native router renormalizes selected probabilities");

    float output[4]{};
    CHECK(engine.moe_forward_experts(5, hidden, output, std::vector<int>{2}),
          "expert located only in second shard executes");
    bool finite_nonzero = false;
    for (float value : output) {
        finite_nonzero = finite_nonzero || (std::isfinite(value) && value != 0.0f);
    }
    CHECK(finite_nonzero, "split-shard expert output is finite and nonzero");

    const dee::EngineStats stats = engine.runtime_stats();
    CHECK(stats.cache_loads == 0 && stats.current_vram == 0,
          "direct CPU parity path does not invent GPU-cache traffic");

    if (failures) {
        std::printf("### %d FAILED ###\n", failures);
        return 1;
    }
    std::printf("ALL PASS\n");
    return 0;
}

