// Exact-shape CUDA normalization kernels used by the Ornith decode adapter.
#pragma once

#ifdef DEE_CUDA
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace dee {

// Bounded, opt-in diagnostic outputs for the regular norm kernel. All pointers
// are device pointers owned by the caller. The selector addresses flattened
// rows and an element range within each row; invalid requests fail closed.
struct QwenRmsNormDiagnosticBuffers {
    int row_start = 0;
    int row_count = 0;
    int element_start = 0;
    int element_count = 0;
    size_t input_snapshot_row_stride_bytes = 0;
    size_t normalized_row_stride_bytes = 0;
    size_t output_snapshot_row_stride_bytes = 0;
    size_t weight_snapshot_row_stride_bytes = 0;
    size_t scalar_stride_bytes = 0;
    float* d_input_snapshot_f32 = nullptr;
    float* d_sum_squares_f32 = nullptr;
    float* d_denominator_f32 = nullptr;
    float* d_reciprocal_rms_f32 = nullptr;
    float* d_weight_snapshot_f32 = nullptr;
    float* d_normalized_f32 = nullptr;
    float* d_output_snapshot_f32 = nullptr;
};

// The diagnostic launch uses the same regular-norm arithmetic and output
// boundary as qwen_rms_norm_fp16_cuda, with bounded stores guarded by the
// selector. It records completion on completion_event but does not synchronize
// the host; the caller must wait on that event before reading diagnostic data.
bool qwen_rms_norm_fp16_diagnostic_cuda(
    const void* d_input_f16, const void* d_weight_f16,
    void* d_output_f16, int rows, int dim, float epsilon,
    cudaStream_t stream, const QwenRmsNormDiagnosticBuffers& diagnostics,
    cudaEvent_t completion_event);

// Diagnostic-only reference probe. This never replaces the sealed PyTorch
// control path: the caller must compare its FP16 output and normalized values
// bit-for-bit with the untouched reference module before treating the probe's
// scalar intermediates as control evidence.
bool qwen_rms_norm_fp16_reference_diagnostic_cuda(
    const void* d_input_f16, const void* d_weight_f16,
    void* d_output_f16, int rows, int dim, float epsilon,
    cudaStream_t stream, const QwenRmsNormDiagnosticBuffers& diagnostics,
    cudaEvent_t completion_event);

bool qwen_rms_norm_fp16_diagnostic_validate(
    int rows, int dim, const QwenRmsNormDiagnosticBuffers& diagnostics);

uint64_t qwen_rms_norm_diagnostic_max_bytes();

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
