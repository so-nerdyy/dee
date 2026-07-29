#include "dee/engine.h"
#include "dee/swiglu_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int failures = 0;
#define CHECK(condition, message) do { \
    if (condition) std::printf("  [PASS] %s\n", message); \
    else { std::printf("  [FAIL] %s\n", message); ++failures; } \
} while (0)

uint16_t half_bits(__half value) {
    uint16_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
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

    float single_batch[8]{};
    CHECK(engine.moe_forward_batch(5, hidden, 1, expert_ids, 2, single_batch),
          "single-row batched CUDA path executes");
    bool single_exact = true;
    for (int i = 0; i < 8; ++i) {
        if (std::fabs(single_batch[i] - individual[i]) > 1e-6f) single_exact = false;
    }
    CHECK(single_exact, "single-row batched output is exact to the proven expert path");

    cudaStream_t external_stream = nullptr;
    __half* d_hidden = nullptr;
    int64_t* d_expert_ids = nullptr;
    float* d_weights = nullptr;
    __half* d_combined = nullptr;
    float* d_raw_trace = nullptr;
    const int64_t combined_ids[4] = {2, 1, 1, 1};
    const float combined_weights_f32[4] = {0.25f, 0.75f, 0.60f, 0.40f};
    __half hidden_f16[8];
    float combined_weights[4];
    for (int i = 0; i < 8; ++i) hidden_f16[i] = __float2half(hidden[i]);
    for (int i = 0; i < 4; ++i) {
        combined_weights[i] = combined_weights_f32[i];
    }
    bool cuda_ok =
        cudaStreamCreateWithFlags(
            &external_stream, cudaStreamNonBlocking) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void**>(&d_hidden),
                   sizeof(hidden_f16)) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void**>(&d_expert_ids),
                   sizeof(combined_ids)) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void**>(&d_weights),
                   sizeof(combined_weights)) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void**>(&d_combined),
                   8 * sizeof(uint16_t)) == cudaSuccess &&
        cudaMalloc(reinterpret_cast<void**>(&d_raw_trace),
                   16 * sizeof(float)) == cudaSuccess;
    CHECK(cuda_ok, "combined-path CUDA inputs allocate");

    std::vector<uint16_t> combined_bits(8);
    std::vector<float> combined_raw(16);
    if (cuda_ok) {
        cuda_ok =
            cudaMemcpyAsync(
                d_hidden, hidden_f16, sizeof(hidden_f16),
                cudaMemcpyHostToDevice, external_stream) == cudaSuccess &&
            cudaMemcpyAsync(
                d_expert_ids, combined_ids, sizeof(combined_ids),
                cudaMemcpyHostToDevice, external_stream) == cudaSuccess &&
            cudaMemcpyAsync(
                d_weights, combined_weights, sizeof(combined_weights),
                cudaMemcpyHostToDevice, external_stream) == cudaSuccess &&
            engine.moe_forward_combined_device(
                5, d_hidden, 2, d_expert_ids, 2, d_weights,
                d_combined, d_raw_trace, external_stream) &&
            cudaMemcpyAsync(
                combined_bits.data(), d_combined,
                combined_bits.size() * sizeof(uint16_t),
                cudaMemcpyDeviceToHost, external_stream) == cudaSuccess &&
            cudaMemcpyAsync(
                combined_raw.data(), d_raw_trace,
                combined_raw.size() * sizeof(float),
                cudaMemcpyDeviceToHost, external_stream) == cudaSuccess &&
            cudaStreamSynchronize(external_stream) == cudaSuccess;
    }
    CHECK(cuda_ok, "combined path hands completion to the external stream");

    const int duplicate_batch_ids[4] = {2, 1, 1, 1};
    float duplicate_batch[16]{};
    CHECK(engine.moe_forward_batch(
              5, hidden, 2, duplicate_batch_ids, 2, duplicate_batch),
          "duplicate-ID raw reference executes");
    bool raw_exact = cuda_ok;
    for (int i = 0; i < 16 && raw_exact; ++i) {
        if (combined_raw[i] != duplicate_batch[i]) raw_exact = false;
    }
    CHECK(raw_exact, "combined raw trace is bitwise exact to raw device output");

    bool combine_exact = cuda_ok;
    for (int token = 0; token < 2 && combine_exact; ++token) {
        const int first = token == 0 ? 1 : 0;
        const int second = token == 0 ? 0 : 1;
        for (int feature = 0; feature < 4; ++feature) {
            __half expected = __float2half(0.0f);
            const int positions[2] = {first, second};
            for (int rank = 0; rank < 2; ++rank) {
                const int position = positions[rank];
                const int selection = token * 2 + position;
                const __half raw = __float2half(
                    duplicate_batch[selection * 4 + feature]);
                const __half weighted = __float2half(
                    __half2float(raw) * combined_weights[selection]);
                expected = __float2half(
                    __half2float(expected)
                    + __half2float(weighted));
            }
            if (combined_bits[token * 4 + feature] != half_bits(expected)) {
                combine_exact = false;
                break;
            }
        }
    }
    CHECK(
        combine_exact,
        "combined output preserves stable expert order and two-stage FP16 rounding");

    std::vector<uint16_t> default_stream_bits(8);
    bool default_stream_ok =
        cudaMemcpyAsync(
            d_hidden, hidden_f16, sizeof(hidden_f16),
            cudaMemcpyHostToDevice, nullptr) == cudaSuccess &&
        cudaMemcpyAsync(
            d_expert_ids, combined_ids, sizeof(combined_ids),
            cudaMemcpyHostToDevice, nullptr) == cudaSuccess &&
        cudaMemcpyAsync(
            d_weights, combined_weights, sizeof(combined_weights),
            cudaMemcpyHostToDevice, nullptr) == cudaSuccess &&
        engine.moe_forward_combined_device(
            5, d_hidden, 2, d_expert_ids, 2, d_weights,
            d_combined, nullptr, nullptr) &&
        cudaMemcpyAsync(
            default_stream_bits.data(), d_combined,
            default_stream_bits.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost, nullptr) == cudaSuccess &&
        cudaStreamSynchronize(nullptr) == cudaSuccess;
    CHECK(
        default_stream_ok && default_stream_bits == combined_bits,
        "combined path accepts CUDA's null default-stream handle");

    // This concrete pair distinguishes eager's two FP16 rounding boundaries
    // from a tempting one-round float multiply-add implementation.
    const float adversarial_raw[2] = {-3.041015625f, 3.09765625f};
    const float adversarial_weights[2] = {1.0f, 0.891144335269928f};
    const int64_t adversarial_ids[2] = {0, 1};
    uint16_t adversarial_bits = 0;
    bool adversarial_ok =
        cudaMemcpyAsync(
            d_raw_trace, adversarial_raw, sizeof(adversarial_raw),
            cudaMemcpyHostToDevice, external_stream) == cudaSuccess &&
        cudaMemcpyAsync(
            d_weights, adversarial_weights, sizeof(adversarial_weights),
            cudaMemcpyHostToDevice, external_stream) == cudaSuccess &&
        cudaMemcpyAsync(
            d_expert_ids, adversarial_ids, sizeof(adversarial_ids),
            cudaMemcpyHostToDevice, external_stream) == cudaSuccess &&
        dee::weighted_combine_fp16_cuda(
            d_raw_trace, d_weights, d_expert_ids, d_combined,
            1, 2, 1, external_stream) &&
        cudaMemcpyAsync(
            &adversarial_bits, d_combined, sizeof(adversarial_bits),
            cudaMemcpyDeviceToHost, external_stream) == cudaSuccess &&
        cudaStreamSynchronize(external_stream) == cudaSuccess;
    const __half adversarial_product = __float2half(
        __half2float(__float2half(adversarial_raw[1]))
        * adversarial_weights[1]);
    const __half adversarial_expected = __float2half(
        __half2float(__float2half(adversarial_raw[0]))
        + __half2float(adversarial_product));
    CHECK(
        adversarial_ok &&
            adversarial_bits == half_bits(adversarial_expected) &&
            adversarial_bits == static_cast<uint16_t>(0xb480),
        "combine preserves the adversarial eager multiply-then-add rounding boundary");

    if (d_hidden) cudaFree(d_hidden);
    if (d_expert_ids) cudaFree(d_expert_ids);
    if (d_weights) cudaFree(d_weights);
    if (d_combined) cudaFree(d_combined);
    if (d_raw_trace) cudaFree(d_raw_trace);
    if (external_stream) cudaStreamDestroy(external_stream);

    if (failures) std::printf("### %d FAILED ###\n", failures);
    else std::printf("ALL PASS\n");
    return failures ? 1 : 0;
}
