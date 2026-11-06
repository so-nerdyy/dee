#include "dee/rmsnorm_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition, message) do { \
    if (condition) std::printf("  [PASS] %s\n", message); \
    else { std::printf("  [FAIL] %s\n", message); ++failures; } \
} while (0)

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(
        stderr, "%s: %s (%s)\n", operation,
        cudaGetErrorName(status), cudaGetErrorString(status));
    return false;
}

uint16_t half_bits(__half value) {
    uint16_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool exact_half_vectors(
        const std::vector<__half>& actual,
        const std::vector<float>& expected,
        const char* label) {
    if (actual.size() != expected.size()) return false;
    for (size_t index = 0; index < actual.size(); ++index) {
        const __half expected_half = __float2half(expected[index]);
        if (half_bits(actual[index]) != half_bits(expected_half)) {
            std::fprintf(
                stderr,
                "%s mismatch at %zu: got=%f (0x%04x), expected=%f (0x%04x)\n",
                label,
                index,
                __half2float(actual[index]),
                half_bits(actual[index]),
                __half2float(expected_half),
                half_bits(expected_half));
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    constexpr int rows = 2;
    constexpr int dim = 4;
    const std::vector<__half> input = {
        __float2half(2.0f), __float2half(2.0f),
        __float2half(2.0f), __float2half(2.0f),
        __float2half(-2.0f), __float2half(-2.0f),
        __float2half(-2.0f), __float2half(-2.0f),
    };
    const std::vector<__half> regular_weight = {
        __float2half(0.0f),
        __float2half(1.0f),
        __float2half(-0.5f),
        __float2half(-1.5f),
    };
    const std::vector<float> regular_expected = {
        1.0f, 2.0f, 0.5f, -0.5f,
        -1.0f, -2.0f, -0.5f, 0.5f,
    };
    const std::vector<__half> gated_weight = {
        __float2half(1.0f),
        __float2half(2.0f),
        __float2half(0.5f),
        __float2half(-1.0f),
    };
    const std::vector<__half> gate = {
        __float2half(0.0f), __float2half(0.0f),
        __float2half(0.0f), __float2half(0.0f),
        __float2half(16.0f), __float2half(16.0f),
        __float2half(16.0f), __float2half(16.0f),
    };
    const std::vector<float> gated_expected = {
        // Negative weighted input times SiLU(+0) preserves IEEE -0.
        0.0f, 0.0f, 0.0f, -0.0f,
        -16.0f, -32.0f, -8.0f, 16.0f,
    };

    __half* d_input = nullptr;
    __half* d_weight = nullptr;
    __half* d_gate = nullptr;
    __half* d_output = nullptr;
    cudaStream_t stream = nullptr;
    bool ok =
        cuda_ok(cudaSetDevice(0), "cudaSetDevice") &&
        cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate") &&
        cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&d_input),
                input.size() * sizeof(__half)),
            "cudaMalloc(input)") &&
        cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&d_weight),
                regular_weight.size() * sizeof(__half)),
            "cudaMalloc(weight)") &&
        cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&d_gate),
                gate.size() * sizeof(__half)),
            "cudaMalloc(gate)") &&
        cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&d_output),
                input.size() * sizeof(__half)),
            "cudaMalloc(output)") &&
        cuda_ok(
            cudaMemcpyAsync(
                d_input, input.data(), input.size() * sizeof(__half),
                cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(input)") &&
        cuda_ok(
            cudaMemcpyAsync(
                d_weight, regular_weight.data(),
                regular_weight.size() * sizeof(__half),
                cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(regular weight)") &&
        dee::qwen_rms_norm_fp16_cuda(
            d_input, d_weight, d_output, rows, dim, 0.0f, stream);
    std::vector<__half> regular_actual(input.size());
    if (ok) {
        ok =
            cuda_ok(
                cudaMemcpyAsync(
                    regular_actual.data(), d_output,
                    regular_actual.size() * sizeof(__half),
                    cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync(regular output)") &&
            cuda_ok(
                cudaStreamSynchronize(stream),
                "cudaStreamSynchronize(regular)");
    }
    CHECK(
        ok && exact_half_vectors(
            regular_actual, regular_expected, "regular"),
        "Qwen RMSNorm preserves exact FP16 boundary on a nondefault stream");

    ok =
        cuda_ok(
            cudaMemcpy(
                d_weight, gated_weight.data(),
                gated_weight.size() * sizeof(__half),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(gated weight)") &&
        cuda_ok(
            cudaMemcpy(
                d_gate, gate.data(), gate.size() * sizeof(__half),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(gate)") &&
        dee::qwen_rms_norm_gated_fp16_cuda(
            d_input, d_weight, d_gate, d_output,
            rows, dim, 0.0f, nullptr);
    std::vector<__half> gated_actual(input.size());
    if (ok) {
        ok =
            cuda_ok(
                cudaMemcpy(
                    gated_actual.data(), d_output,
                    gated_actual.size() * sizeof(__half),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(gated output)") &&
            cuda_ok(
                cudaStreamSynchronize(nullptr),
                "cudaStreamSynchronize(default)");
    }
    CHECK(
        ok && exact_half_vectors(
            gated_actual, gated_expected, "gated"),
        "Qwen gated RMSNorm preserves eager FP16 boundaries on default stream");
    CHECK(
        !dee::qwen_rms_norm_fp16_cuda(
            nullptr, d_weight, d_output, rows, dim, 0.0f, stream),
        "Qwen RMSNorm rejects a null input pointer");
    CHECK(
        !dee::qwen_rms_norm_gated_fp16_cuda(
            d_input, d_weight, d_gate, d_output,
            rows, 4097, 0.0f, stream),
        "Qwen gated RMSNorm rejects dimensions beyond its launch contract");
    CHECK(
        !dee::qwen_rms_norm_fp16_cuda(
            d_input, d_weight, d_output, rows, dim,
            std::numeric_limits<float>::quiet_NaN(), stream),
        "Qwen RMSNorm rejects a NaN epsilon");

    if (d_output) cudaFree(d_output);
    if (d_gate) cudaFree(d_gate);
    if (d_weight) cudaFree(d_weight);
    if (d_input) cudaFree(d_input);
    if (stream) cudaStreamDestroy(stream);
    std::printf("CUDA Qwen RMSNorm contract: %s\n",
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
