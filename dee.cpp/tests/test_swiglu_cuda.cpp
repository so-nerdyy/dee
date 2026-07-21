#include "dee/engine.h"
#include "dee/cuda_convert.h"
#include "dee/swiglu_cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s (%s)\n", operation, cudaGetErrorName(status), cudaGetErrorString(status));
    return false;
}

bool check_cublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "%s: cuBLAS status %d\n", operation, static_cast<int>(status));
    return false;
}

}  // namespace

int main() {
    // Verify the transfer-path expansion separately from SwiGLU. Conversion
    // must preserve the exact FP32 value represented by each BF16 bit pattern.
    const std::vector<float> conversion_source = {0.0f, 1.0f, -2.5f, 3.125f, 0.0078125f};
    std::vector<uint16_t> bf16(conversion_source.size());
    std::vector<float> conversion_expected(conversion_source.size());
    for (size_t i = 0; i < conversion_source.size(); ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &conversion_source[i], sizeof(bits));
        bf16[i] = static_cast<uint16_t>(bits >> 16);
        bits = static_cast<uint32_t>(bf16[i]) << 16;
        std::memcpy(&conversion_expected[i], &bits, sizeof(bits));
    }
    uint16_t* d_bf16 = nullptr;
    float* d_converted = nullptr;
    cudaStream_t conversion_stream = nullptr;
    bool conversion_ok = check(cudaStreamCreate(&conversion_stream), "cudaStreamCreate(conversion)") &&
        check(cudaMalloc(reinterpret_cast<void**>(&d_bf16), bf16.size() * sizeof(uint16_t)),
              "cudaMalloc(BF16 conversion source)") &&
        check(cudaMalloc(reinterpret_cast<void**>(&d_converted), bf16.size() * sizeof(float)),
              "cudaMalloc(BF16 conversion output)") &&
        check(cudaMemcpyAsync(d_bf16, bf16.data(), bf16.size() * sizeof(uint16_t),
                              cudaMemcpyHostToDevice, conversion_stream),
              "cudaMemcpyAsync(BF16 conversion source)") &&
        dee::bf16_to_f32_cuda(d_bf16, d_converted, bf16.size(), conversion_stream);
    std::vector<float> converted(bf16.size());
    if (conversion_ok) {
        conversion_ok = check(cudaMemcpyAsync(converted.data(), d_converted,
                                               converted.size() * sizeof(float),
                                               cudaMemcpyDeviceToHost, conversion_stream),
                              "cudaMemcpyAsync(BF16 conversion output)") &&
            check(cudaStreamSynchronize(conversion_stream),
                  "cudaStreamSynchronize(BF16 conversion)");
    }
    for (size_t i = 0; conversion_ok && i < converted.size(); ++i) {
        if (converted[i] != conversion_expected[i]) conversion_ok = false;
    }
    if (d_bf16) cudaFree(d_bf16);
    if (d_converted) cudaFree(d_converted);
    if (conversion_stream) cudaStreamDestroy(conversion_stream);
    std::printf("CUDA BF16 expansion comparison: %s\n", conversion_ok ? "PASS" : "FAIL");
    if (!conversion_ok) return 1;

    constexpr int inter = 3;
    constexpr int hidden = 4;
    std::vector<float> weights(3 * inter * hidden);
    std::vector<float> input(hidden);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>((int(i) % 7) - 3) * 0.125f;
    for (int i = 0; i < hidden; ++i) input[i] = static_cast<float>(i - 1) * 0.25f;

    std::vector<float> cpu(hidden, 0.0f);
    dee::Engine::swiglu(weights.data(), input.data(), inter, hidden, cpu.data());

    float *d_weights = nullptr, *d_input = nullptr, *d_gate = nullptr, *d_up = nullptr, *d_output = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    bool ok = check(cudaStreamCreate(&stream), "cudaStreamCreate") &&
              check_cublas(cublasCreate(&handle), "cublasCreate") &&
              check_cublas(cublasSetStream(handle, stream), "cublasSetStream") &&
              check(cudaMalloc(reinterpret_cast<void**>(&d_weights), weights.size() * sizeof(float)), "cudaMalloc(weights)") &&
              check(cudaMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(float)), "cudaMalloc(input)") &&
              check(cudaMalloc(reinterpret_cast<void**>(&d_gate), inter * sizeof(float)), "cudaMalloc(gate)") &&
              check(cudaMalloc(reinterpret_cast<void**>(&d_up), inter * sizeof(float)), "cudaMalloc(up)") &&
              check(cudaMalloc(reinterpret_cast<void**>(&d_output), hidden * sizeof(float)), "cudaMalloc(output)");
    if (ok) ok = check(cudaMemcpyAsync(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(weights)") &&
                 check(cudaMemcpyAsync(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(input)") &&
                 dee::swiglu_expert_cuda(handle, d_weights, d_input, d_gate, d_up, d_output, inter, hidden, stream);

    std::vector<float> gpu(hidden);
    if (ok) ok = check(cudaMemcpyAsync(gpu.data(), d_output, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(output)") &&
                 check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    for (int i = 0; ok && i < hidden; ++i) {
        if (std::fabs(cpu[i] - gpu[i]) > 1e-5f) {
            std::fprintf(stderr, "SwiGLU mismatch at %d: cpu=%f cuda=%f\n", i, cpu[i], gpu[i]);
            ok = false;
        }
    }

    // Validate the FP16 device-cache path against the same deterministic CPU
    // reference. These values are exactly representable in BF16 and FP16, so
    // the remaining tolerance covers FP16 activation rounding and GEMM order.
    std::vector<uint16_t> weights_bf16(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &weights[i], sizeof(bits));
        weights_bf16[i] = static_cast<uint16_t>(bits >> 16);
    }
    uint16_t* d_weights_bf16 = nullptr;
    void* d_weights_half = nullptr;
    void* d_input_half = nullptr;
    void* d_activation_half = nullptr;
    if (ok) ok = check(cudaMalloc(reinterpret_cast<void**>(&d_weights_bf16),
                                   weights_bf16.size() * sizeof(uint16_t)),
                       "cudaMalloc(FP16 path BF16 weights)") &&
                 check(cudaMalloc(&d_weights_half, weights.size() * sizeof(uint16_t)),
                       "cudaMalloc(FP16 weights)") &&
                 check(cudaMalloc(&d_input_half, input.size() * sizeof(uint16_t)),
                       "cudaMalloc(FP16 input)") &&
                 check(cudaMalloc(&d_activation_half, inter * sizeof(uint16_t)),
                       "cudaMalloc(FP16 activation)");
    if (ok) ok = check(cudaMemcpyAsync(d_weights_bf16, weights_bf16.data(),
                                        weights_bf16.size() * sizeof(uint16_t),
                                        cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync(FP16 path BF16 weights)") &&
                 dee::bf16_to_f16_cuda(d_weights_bf16, d_weights_half,
                                        weights.size(), stream) &&
                 dee::f32_to_f16_cuda(d_input, d_input_half, input.size(), stream) &&
                 dee::swiglu_expert_fp16_cuda(handle, d_weights_half, d_input_half,
                                              d_gate, d_up, d_activation_half,
                                              d_output, inter, hidden, stream);
    std::vector<float> gpu_fp16(hidden);
    if (ok) ok = check(cudaMemcpyAsync(gpu_fp16.data(), d_output,
                                       gpu_fp16.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync(FP16 output)") &&
                 check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(FP16 path)");
    for (int i = 0; ok && i < hidden; ++i) {
        const float error = std::fabs(cpu[i] - gpu_fp16[i]);
        if (error > 2e-3f) {
            std::fprintf(stderr, "FP16 SwiGLU mismatch at %d: cpu=%f cuda=%f abs=%f\n",
                         i, cpu[i], gpu_fp16[i], error);
            ok = false;
        }
    }
    if (d_weights_bf16) cudaFree(d_weights_bf16);
    if (d_weights_half) cudaFree(d_weights_half);
    if (d_input_half) cudaFree(d_input_half);
    if (d_activation_half) cudaFree(d_activation_half);

    if (d_weights) cudaFree(d_weights);
    if (d_input) cudaFree(d_input);
    if (d_gate) cudaFree(d_gate);
    if (d_up) cudaFree(d_up);
    if (d_output) cudaFree(d_output);
    if (handle) cublasDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    std::printf("CUDA FP32/FP16 SwiGLU CPU comparison: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
