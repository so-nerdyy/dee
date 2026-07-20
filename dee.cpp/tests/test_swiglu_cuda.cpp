#include "dee/engine.h"
#include "dee/swiglu_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s (%s)\n", operation, cudaGetErrorName(status), cudaGetErrorString(status));
    return false;
}

}  // namespace

int main() {
    constexpr int inter = 3;
    constexpr int hidden = 4;
    std::vector<float> weights(3 * inter * hidden);
    std::vector<float> input(hidden);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>((int(i) % 7) - 3) * 0.125f;
    for (int i = 0; i < hidden; ++i) input[i] = static_cast<float>(i - 1) * 0.25f;

    std::vector<float> cpu(hidden, 0.0f);
    dee::Engine::swiglu(weights.data(), input.data(), inter, hidden, cpu.data());

    float *d_weights = nullptr, *d_input = nullptr, *d_intermediate = nullptr, *d_output = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = check(cudaStreamCreate(&stream), "cudaStreamCreate") &&
              check(cudaMalloc(reinterpret_cast<void**>(&d_weights), weights.size() * sizeof(float)), "cudaMalloc(weights)") &&
              check(cudaMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(float)), "cudaMalloc(input)") &&
              check(cudaMalloc(reinterpret_cast<void**>(&d_intermediate), inter * sizeof(float)), "cudaMalloc(intermediate)") &&
              check(cudaMalloc(reinterpret_cast<void**>(&d_output), hidden * sizeof(float)), "cudaMalloc(output)");
    if (ok) ok = check(cudaMemcpyAsync(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(weights)") &&
                 check(cudaMemcpyAsync(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(input)") &&
                 dee::swiglu_expert_cuda(d_weights, d_input, d_intermediate, d_output, inter, hidden, stream);

    std::vector<float> gpu(hidden);
    if (ok) ok = check(cudaMemcpyAsync(gpu.data(), d_output, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(output)") &&
                 check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    for (int i = 0; ok && i < hidden; ++i) {
        if (std::fabs(cpu[i] - gpu[i]) > 1e-5f) {
            std::fprintf(stderr, "SwiGLU mismatch at %d: cpu=%f cuda=%f\n", i, cpu[i], gpu[i]);
            ok = false;
        }
    }

    if (d_weights) cudaFree(d_weights);
    if (d_input) cudaFree(d_input);
    if (d_intermediate) cudaFree(d_intermediate);
    if (d_output) cudaFree(d_output);
    if (stream) cudaStreamDestroy(stream);
    std::printf("CUDA SwiGLU CPU comparison: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
