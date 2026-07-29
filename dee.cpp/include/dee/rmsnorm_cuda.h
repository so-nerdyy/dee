// Exact-shape CUDA normalization kernels used by the Ornith decode adapter.
#pragma once

#ifdef DEE_CUDA
#include <cuda_runtime.h>

namespace dee {

// Qwen3.5 RMSNorm:
//   y = (x.float * rsqrt(mean(x.float^2) + eps))
//       * (1 + weight.float)
//   return y.to(float16)
bool qwen_rms_norm_fp16_cuda(
    const void* d_input_f16, const void* d_weight_f16,
    void* d_output_f16, int rows, int dim, float epsilon,
    cudaStream_t stream);

// Qwen3.5 linear-attention gated RMSNorm. The FP16 boundary after
// weight * normalized_input is intentional and matches the eager module:
//   normalized_half = normalized_float.to(float16)
//   weighted_half = weight_half * normalized_half
//   y = weighted_half.float * silu(gate.float)
//   return y.to(float16)
bool qwen_rms_norm_gated_fp16_cuda(
    const void* d_input_f16, const void* d_weight_f16,
    const void* d_gate_f16, void* d_output_f16,
    int rows, int dim, float epsilon, cudaStream_t stream);

}  // namespace dee
#endif
