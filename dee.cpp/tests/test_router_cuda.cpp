#include "dee/engine.h"

#include <cmath>
#include <cstdio>

namespace {
int failures = 0;
#define CHECK(condition, message) do { \
    if (condition) std::printf("  [PASS] %s\n", message); \
    else { std::printf("  [FAIL] %s\n", message); ++failures; } \
} while (0)
}

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
    cfg.use_cuda = true;
    cfg.cache_dtype = dee::DeviceCacheDType::Fp16;
    cfg.transfer_dtype = dee::WeightTransferDType::Bf16;
    cfg.budget_bytes = 2 * 3ULL * 2 * 4 * sizeof(uint16_t);

    dee::Engine engine;
    CHECK(engine.init(cfg), "CUDA router engine initializes");
    if (failures) return 1;

    const float hidden[8] = {1, 2, 3, 4, -1, 4, 2, 0};
    float logits[6]{};
    float weights[4]{};
    int experts[4]{};
    CHECK(engine.route_topk_batch(5, hidden, 2, logits, weights, experts),
          "batched CUDA router executes");
    CHECK(logits[0] == 1.0f && logits[1] == 2.0f && logits[2] == 3.0f &&
          logits[3] == -1.0f && logits[4] == 4.0f && logits[5] == 2.0f,
          "FP16 CUDA router logits match exact linear outputs");
    CHECK(experts[0] == 2 && experts[1] == 1 &&
          experts[2] == 1 && experts[3] == 2,
          "batched CUDA router preserves per-token top-k order");
    CHECK(std::fabs(weights[0] - 0.7310586f) < 1e-5f &&
          std::fabs(weights[1] - 0.2689414f) < 1e-5f &&
          std::fabs(weights[2] - 0.8807971f) < 1e-5f &&
          std::fabs(weights[3] - 0.1192029f) < 1e-5f,
          "batched CUDA router renormalizes selected FP32 probabilities");

    if (failures) std::printf("### %d FAILED ###\n", failures);
    else std::printf("ALL PASS\n");
    return failures ? 1 : 0;
}
