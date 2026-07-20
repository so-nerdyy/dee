#include "dee/swiglu_cuda.h"

#include "dee/cuda_check.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace dee {
namespace {

constexpr int kThreads = 256;

__global__ void swiglu_h_kernel(const float* W, const float* x, float* h, int inter, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inter) return;
    const float* gate = W + static_cast<size_t>(i) * hidden;
    const float* up = W + static_cast<size_t>(inter + i) * hidden;
    float g = 0.0f;
    float u = 0.0f;
    for (int j = 0; j < hidden; ++j) {
        g += gate[j] * x[j];
        u += up[j] * x[j];
    }
    h[i] = (g / (1.0f + expf(-g))) * u;
}

__global__ void swiglu_y_kernel(const float* W, const float* h, float* y, int inter, int hidden) {
    const int output = blockIdx.x * blockDim.x + threadIdx.x;
    if (output >= hidden) return;
    const float* down = W + 2 * static_cast<size_t>(inter) * hidden;
    float sum = 0.0f;
    for (int i = 0; i < inter; ++i) sum += down[static_cast<size_t>(output) * inter + i] * h[i];
    y[output] = sum;
}

__global__ void combine_kernel(const float* ybuf, float* output, int experts, int hidden) {
    const int value = blockIdx.x * blockDim.x + threadIdx.x;
    if (value >= hidden) return;
    float sum = 0.0f;
    for (int expert = 0; expert < experts; ++expert) sum += ybuf[static_cast<size_t>(expert) * hidden + value];
    output[value] = sum / static_cast<float>(experts);
}

int grid_for(int count) { return (count + kThreads - 1) / kThreads; }

}  // namespace

bool swiglu_expert_cuda(const float* d_weights, const float* d_x, float* d_intermediate,
                        float* d_y, int inter, int hidden, cudaStream_t stream) {
    if (!d_weights || !d_x || !d_intermediate || !d_y || !stream || inter <= 0 || hidden <= 0) {
        std::fprintf(stderr, "[cuda] invalid SwiGLU launch arguments (inter=%d hidden=%d)\n", inter, hidden);
        return false;
    }
    swiglu_h_kernel<<<grid_for(inter), kThreads, 0, stream>>>(d_weights, d_x, d_intermediate, inter, hidden);
    if (!DEE_CUDA_CHECK_LAUNCH("swiglu_h_kernel launch")) return false;
    swiglu_y_kernel<<<grid_for(hidden), kThreads, 0, stream>>>(d_weights, d_intermediate, d_y, inter, hidden);
    if (!DEE_CUDA_CHECK_LAUNCH("swiglu_y_kernel launch")) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SwiGLU validation)");
#else
    return true;
#endif
}

bool combine_cuda(const float* d_ybuf, float* d_output, int experts, int hidden, cudaStream_t stream) {
    if (!d_ybuf || !d_output || !stream || experts <= 0 || hidden <= 0) {
        std::fprintf(stderr, "[cuda] invalid combine launch arguments (experts=%d hidden=%d)\n", experts, hidden);
        return false;
    }
    combine_kernel<<<grid_for(hidden), kThreads, 0, stream>>>(d_ybuf, d_output, experts, hidden);
    if (!DEE_CUDA_CHECK_LAUNCH("combine_kernel launch")) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream), "cudaStreamSynchronize(combine validation)");
#else
    return true;
#endif
}

}  // namespace dee
