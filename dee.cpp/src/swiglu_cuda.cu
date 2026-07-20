#include "dee/swiglu_cuda.h"

#include "dee/cuda_check.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace dee {
namespace {

constexpr int kThreads = 256;

__global__ void swiglu_activation_kernel(float* gate, const float* up, int inter) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inter) return;
    const float g = gate[i];
    gate[i] = (g / (1.0f + expf(-g))) * up[i];
}

__global__ void combine_kernel(const float* ybuf, float* output, int experts, int hidden) {
    const int value = blockIdx.x * blockDim.x + threadIdx.x;
    if (value >= hidden) return;
    float sum = 0.0f;
    for (int expert = 0; expert < experts; ++expert) sum += ybuf[static_cast<size_t>(expert) * hidden + value];
    output[value] = sum / static_cast<float>(experts);
}

int grid_for(int count) { return (count + kThreads - 1) / kThreads; }

bool gemv_row_major(cublasHandle_t handle, const float* matrix, int rows, int cols,
                    const float* input, float* output) {
    // Row-major [rows, cols] is the storage-equivalent of column-major
    // [cols, rows]; transposed SGEMV therefore computes W * input.
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    return DEE_CUBLAS_CHECK_NAMED(
        cublasSgemv(handle, CUBLAS_OP_T, cols, rows, &alpha, matrix, cols,
                    input, 1, &beta, output, 1),
        "cublasSgemv(row-major expert projection)");
}

}  // namespace

bool swiglu_expert_cuda(cublasHandle_t handle, const float* d_weights, const float* d_x,
                        float* d_gate, float* d_up, float* d_y,
                        int inter, int hidden, cudaStream_t stream) {
    if (!handle || !d_weights || !d_x || !d_gate || !d_up || !d_y || !stream || inter <= 0 || hidden <= 0) {
        std::fprintf(stderr, "[cuda] invalid cuBLAS SwiGLU arguments (inter=%d hidden=%d)\n", inter, hidden);
        return false;
    }
    const size_t projection = static_cast<size_t>(inter) * hidden;
    if (!gemv_row_major(handle, d_weights, inter, hidden, d_x, d_gate) ||
        !gemv_row_major(handle, d_weights + projection, inter, hidden, d_x, d_up)) return false;
    swiglu_activation_kernel<<<grid_for(inter), kThreads, 0, stream>>>(d_gate, d_up, inter);
    if (!DEE_CUDA_CHECK_LAUNCH("swiglu_activation_kernel launch")) return false;
    if (!gemv_row_major(handle, d_weights + 2 * projection, hidden, inter, d_gate, d_y)) return false;
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
