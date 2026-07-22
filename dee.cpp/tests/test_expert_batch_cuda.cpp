#include "dee/engine.h"

#include <cmath>
#include <cstdio>
#include <vector>

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
    CHECK(engine.init(cfg), "CUDA batched MoE engine initializes");
    if (failures) return 1;

    const float hidden[8] = {1, 2, 3, 4, -1, 4, 2, 0};
    const int expert_ids[4] = {2, 1, 1, 2};
    float batch[16]{};
    CHECK(engine.moe_forward_batch(5, hidden, 2, expert_ids, 2, batch),
          "per-expert batched CUDA SwiGLU executes");

    float individual[16]{};
    CHECK(engine.moe_forward_experts(5, hidden, individual, std::vector<int>{2, 1}) &&
          engine.moe_forward_experts(5, hidden + 4, individual + 8,
                                     std::vector<int>{1, 2}),
          "single-token CUDA expert calls provide numerical reference");
    bool close = true;
    for (int i = 0; i < 16; ++i) {
        if (std::fabs(batch[i] - individual[i]) > 2e-3f) close = false;
    }
    CHECK(close, "batched CUDA output preserves token/rank layout and FP16 math");

    if (failures) std::printf("### %d FAILED ###\n", failures);
    else std::printf("ALL PASS\n");
    return failures ? 1 : 0;
}
