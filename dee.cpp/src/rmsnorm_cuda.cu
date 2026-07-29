#include "dee/rmsnorm_cuda.h"

#include "dee/cuda_check.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace dee {
namespace {

constexpr int kThreads = 256;

__device__ float block_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    __shared__ float warp_sums[kThreads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < kThreads / 32 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    return value;
}

__global__ void qwen_rms_norm_fp16_kernel(
        const __half* input, const __half* weight, __half* output,
        int dim, float epsilon) {
    const int row = blockIdx.x;
    const size_t row_offset = static_cast<size_t>(row) * dim;
    float square_sum = 0.0f;
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        const float value = __half2float(input[row_offset + index]);
        const float square = __fmul_rn(value, value);
        square_sum = __fadd_rn(square_sum, square);
    }
    square_sum = block_sum(square_sum);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(dim) + epsilon);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        const float normalized = __fmul_rn(
            __half2float(input[row_offset + index]), inverse_rms);
        const float scale = __fadd_rn(
            1.0f, __half2float(weight[index]));
        output[row_offset + index] = __float2half_rn(
            __fmul_rn(normalized, scale));
    }
}

__global__ void qwen_rms_norm_gated_fp16_kernel(
        const __half* input, const __half* weight, const __half* gate,
        __half* output, int dim, float epsilon) {
    const int row = blockIdx.x;
    const size_t row_offset = static_cast<size_t>(row) * dim;
    float square_sum = 0.0f;
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        const float value = __half2float(input[row_offset + index]);
        const float square = __fmul_rn(value, value);
        square_sum = __fadd_rn(square_sum, square);
    }
    square_sum = block_sum(square_sum);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / static_cast<float>(dim) + epsilon);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        const __half normalized = __float2half_rn(
            __fmul_rn(
                __half2float(input[row_offset + index]), inverse_rms));
        const __half weighted = __hmul(weight[index], normalized);
        const float gate_value = __half2float(gate[row_offset + index]);
        const float silu = gate_value / (1.0f + expf(-gate_value));
        output[row_offset + index] = __float2half_rn(
            __fmul_rn(__half2float(weighted), silu));
    }
}

}  // namespace

bool qwen_rms_norm_fp16_cuda(
        const void* d_input_f16, const void* d_weight_f16,
        void* d_output_f16, int rows, int dim, float epsilon,
        cudaStream_t stream) {
    if (!d_input_f16 || !d_weight_f16 || !d_output_f16 ||
        rows <= 0 || dim <= 0 || dim > 4096 ||
        !std::isfinite(epsilon) || epsilon < 0.0f) {
        return false;
    }
    qwen_rms_norm_fp16_kernel<<<rows, kThreads, 0, stream>>>(
        static_cast<const __half*>(d_input_f16),
        static_cast<const __half*>(d_weight_f16),
        static_cast<__half*>(d_output_f16),
        dim,
        epsilon);
    return DEE_CUDA_CHECK_NAMED(
        cudaPeekAtLastError(), "qwen_rms_norm_fp16_kernel");
}

bool qwen_rms_norm_gated_fp16_cuda(
        const void* d_input_f16, const void* d_weight_f16,
        const void* d_gate_f16, void* d_output_f16,
        int rows, int dim, float epsilon, cudaStream_t stream) {
    if (!d_input_f16 || !d_weight_f16 || !d_gate_f16 || !d_output_f16 ||
        rows <= 0 || dim <= 0 || dim > 4096 ||
        !std::isfinite(epsilon) || epsilon < 0.0f) {
        return false;
    }
    qwen_rms_norm_gated_fp16_kernel<<<rows, kThreads, 0, stream>>>(
        static_cast<const __half*>(d_input_f16),
        static_cast<const __half*>(d_weight_f16),
        static_cast<const __half*>(d_gate_f16),
        static_cast<__half*>(d_output_f16),
        dim,
        epsilon);
    return DEE_CUDA_CHECK_NAMED(
        cudaPeekAtLastError(), "qwen_rms_norm_gated_fp16_kernel");
}

}  // namespace dee
