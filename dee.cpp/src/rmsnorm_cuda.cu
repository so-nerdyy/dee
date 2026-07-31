#include "dee/rmsnorm_cuda.h"

#include "dee/cuda_check.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace dee {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxDiagnosticRows = 64;
constexpr int kMaxDiagnosticElements = 4096;
constexpr uint64_t kDiagnosticMaxBytes = 1ULL << 20;

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

// One kernel body is used for both production and diagnostic launches. The
// diagnostic template parameter removes all diagnostic branches/stores from
// the production instantiation while ensuring the observed arithmetic is the
// candidate kernel's actual reduction/output sequence, not a host reconstruction.
template <bool Diagnostic>
__global__ void qwen_rms_norm_fp16_kernel(
        const __half* input, const __half* weight, __half* output,
        int dim, float epsilon, QwenRmsNormDiagnosticBuffers diagnostics) {
    const int row = blockIdx.x;
    const bool selected = Diagnostic &&
        row >= diagnostics.row_start &&
        row < diagnostics.row_start + diagnostics.row_count;
    const int selected_row = row - diagnostics.row_start;
    const size_t row_offset = static_cast<size_t>(row) * dim;
    float square_sum = 0.0f;
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        const float value = __half2float(input[row_offset + index]);
        const float square = __fmul_rn(value, value);
        square_sum = __fadd_rn(square_sum, square);
    }
    square_sum = block_sum(square_sum);
    __shared__ float inverse_rms;
    __shared__ float denominator;
    if (threadIdx.x == 0) {
        if constexpr (Diagnostic) {
            denominator = square_sum / static_cast<float>(dim) + epsilon;
            inverse_rms = rsqrtf(denominator);
            if (selected) {
                char* scalar_base = reinterpret_cast<char*>(diagnostics.d_sum_squares_f32)
                    + static_cast<size_t>(selected_row) * diagnostics.scalar_stride_bytes;
                *reinterpret_cast<float*>(scalar_base) = square_sum;
                reinterpret_cast<float*>(reinterpret_cast<char*>(diagnostics.d_denominator_f32)
                    + static_cast<size_t>(selected_row) * diagnostics.scalar_stride_bytes)[0] = denominator;
                reinterpret_cast<float*>(reinterpret_cast<char*>(diagnostics.d_reciprocal_rms_f32)
                    + static_cast<size_t>(selected_row) * diagnostics.scalar_stride_bytes)[0] = inverse_rms;
            }
        } else {
            inverse_rms = rsqrtf(square_sum / static_cast<float>(dim) + epsilon);
        }
    }
    __syncthreads();
    for (int index = threadIdx.x; index < dim; index += blockDim.x) {
        const float normalized = __fmul_rn(
            __half2float(input[row_offset + index]), inverse_rms);
        const float scale = __fadd_rn(1.0f, __half2float(weight[index]));
        const __half output_value = __float2half_rn(
            __fmul_rn(normalized, scale));
        output[row_offset + index] = output_value;
        if constexpr (Diagnostic) {
            if (!selected || index < diagnostics.element_start ||
                index >= diagnostics.element_start + diagnostics.element_count) {
                continue;
            }
            const int selected_index = index - diagnostics.element_start;
            char* input_row = reinterpret_cast<char*>(diagnostics.d_input_snapshot_f32)
                + static_cast<size_t>(selected_row) * diagnostics.input_snapshot_row_stride_bytes;
            char* normalized_row = reinterpret_cast<char*>(diagnostics.d_normalized_f32)
                + static_cast<size_t>(selected_row) * diagnostics.normalized_row_stride_bytes;
            char* output_row = reinterpret_cast<char*>(diagnostics.d_output_snapshot_f32)
                + static_cast<size_t>(selected_row) * diagnostics.output_snapshot_row_stride_bytes;
            char* weight_row = reinterpret_cast<char*>(diagnostics.d_weight_snapshot_f32)
                + static_cast<size_t>(selected_row) * diagnostics.weight_snapshot_row_stride_bytes;
            reinterpret_cast<float*>(input_row)[selected_index] = __half2float(input[row_offset + index]);
            reinterpret_cast<float*>(normalized_row)[selected_index] = normalized;
            reinterpret_cast<float*>(output_row)[selected_index] = __half2float(output_value);
            reinterpret_cast<float*>(weight_row)[selected_index] = __half2float(weight[index]);
        }
    }
}

bool valid_diagnostic_range(int rows, int dim,
                           const QwenRmsNormDiagnosticBuffers& diagnostics) {
    if (rows <= 0 || dim <= 0 || dim > 4096 ||
        diagnostics.row_start < 0 || diagnostics.row_count <= 0 ||
        diagnostics.row_count > kMaxDiagnosticRows ||
        diagnostics.row_start > rows - diagnostics.row_count ||
        diagnostics.element_start < 0 || diagnostics.element_count <= 0 ||
        diagnostics.element_count > kMaxDiagnosticElements ||
        diagnostics.element_start > dim - diagnostics.element_count) {
        return false;
    }
    const size_t row_bytes = static_cast<size_t>(diagnostics.element_count) * sizeof(float);
    if (diagnostics.input_snapshot_row_stride_bytes < row_bytes ||
        diagnostics.normalized_row_stride_bytes < row_bytes ||
        diagnostics.output_snapshot_row_stride_bytes < row_bytes ||
        diagnostics.weight_snapshot_row_stride_bytes < row_bytes ||
        diagnostics.scalar_stride_bytes < sizeof(float) ||
        !diagnostics.d_input_snapshot_f32 || !diagnostics.d_sum_squares_f32 ||
        !diagnostics.d_denominator_f32 || !diagnostics.d_reciprocal_rms_f32 ||
        !diagnostics.d_weight_snapshot_f32 || !diagnostics.d_normalized_f32 ||
        !diagnostics.d_output_snapshot_f32) {
        return false;
    }
    const auto checked_span = [](int count, size_t stride, size_t row_size,
                                 uint64_t* result) {
        if (count <= 0 || stride < row_size) return false;
        const uint64_t max_value = std::numeric_limits<uint64_t>::max();
        const uint64_t row_count = static_cast<uint64_t>(count);
        const uint64_t stride_u64 = static_cast<uint64_t>(stride);
        const uint64_t row_size_u64 = static_cast<uint64_t>(row_size);
        if (row_count - 1 > max_value / stride_u64) return false;
        const uint64_t last_offset = (row_count - 1) * stride_u64;
        if (last_offset > max_value - row_size_u64) return false;
        *result = last_offset + row_size_u64;
        return true;
    };
    uint64_t input_bytes = 0;
    uint64_t normalized_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t sum_squares_bytes = 0;
    uint64_t denominator_bytes = 0;
    uint64_t reciprocal_bytes = 0;
    if (!checked_span(diagnostics.row_count,
                      diagnostics.input_snapshot_row_stride_bytes,
                      row_bytes, &input_bytes) ||
        !checked_span(diagnostics.row_count,
                      diagnostics.normalized_row_stride_bytes,
                      row_bytes, &normalized_bytes) ||
        !checked_span(diagnostics.row_count,
                      diagnostics.output_snapshot_row_stride_bytes,
                      row_bytes, &output_bytes) ||
        !checked_span(diagnostics.row_count,
                      diagnostics.weight_snapshot_row_stride_bytes,
                      row_bytes, &weight_bytes) ||
        !checked_span(diagnostics.row_count, diagnostics.scalar_stride_bytes,
                      sizeof(float), &sum_squares_bytes) ||
        !checked_span(diagnostics.row_count, diagnostics.scalar_stride_bytes,
                      sizeof(float), &denominator_bytes) ||
        !checked_span(diagnostics.row_count, diagnostics.scalar_stride_bytes,
                      sizeof(float), &reciprocal_bytes)) {
        return false;
    }
    const uint64_t parts[] = {
        input_bytes, normalized_bytes, output_bytes, weight_bytes,
        sum_squares_bytes, denominator_bytes, reciprocal_bytes,
    };
    uint64_t total_bytes = 0;
    for (uint64_t part : parts) {
        if (part > std::numeric_limits<uint64_t>::max() - total_bytes) {
            return false;
        }
        total_bytes += part;
    }
    return total_bytes <= kDiagnosticMaxBytes;
}

}  // namespace

bool qwen_rms_norm_fp16_diagnostic_validate(
        int rows, int dim, const QwenRmsNormDiagnosticBuffers& diagnostics) {
    return valid_diagnostic_range(rows, dim, diagnostics);
}

uint64_t qwen_rms_norm_diagnostic_max_bytes() {
    return kDiagnosticMaxBytes;
}

bool qwen_rms_norm_fp16_diagnostic_cuda(
        const void* d_input_f16, const void* d_weight_f16,
        void* d_output_f16, int rows, int dim, float epsilon,
        cudaStream_t stream, const QwenRmsNormDiagnosticBuffers& diagnostics,
        cudaEvent_t completion_event) {
    if (!d_input_f16 || !d_weight_f16 || !d_output_f16 ||
        !std::isfinite(epsilon) || epsilon < 0.0f ||
        !valid_diagnostic_range(rows, dim, diagnostics) || !completion_event) {
        return false;
    }
    qwen_rms_norm_fp16_kernel<true><<<rows, kThreads, 0, stream>>>(
        static_cast<const __half*>(d_input_f16),
        static_cast<const __half*>(d_weight_f16),
        static_cast<__half*>(d_output_f16),
        dim,
        epsilon,
        diagnostics);
    if (!DEE_CUDA_CHECK_NAMED(
            cudaPeekAtLastError(), "qwen_rms_norm_fp16_diagnostic_kernel")) {
        return false;
    }
    return DEE_CUDA_CHECK_NAMED(
        cudaEventRecord(completion_event, stream),
        "cudaEventRecord(qwen RMSNorm diagnostic completion)");
}

bool qwen_rms_norm_fp16_reference_diagnostic_cuda(
        const void* d_input_f16, const void* d_weight_f16,
        void* d_output_f16, int rows, int dim, float epsilon,
        cudaStream_t stream, const QwenRmsNormDiagnosticBuffers& diagnostics,
        cudaEvent_t completion_event) {
    if (!d_input_f16 || !d_weight_f16 || !d_output_f16 ||
        !std::isfinite(epsilon) || epsilon < 0.0f ||
        !valid_diagnostic_range(rows, dim, diagnostics) || !completion_event) {
        return false;
    }
    qwen_rms_norm_fp16_kernel<true><<<rows, kThreads, 0, stream>>>(
        static_cast<const __half*>(d_input_f16),
        static_cast<const __half*>(d_weight_f16),
        static_cast<__half*>(d_output_f16),
        dim,
        epsilon,
        diagnostics);
    if (!DEE_CUDA_CHECK_NAMED(
            cudaPeekAtLastError(), "qwen_rms_norm_fp16_reference_diagnostic_kernel")) {
        return false;
    }
    return DEE_CUDA_CHECK_NAMED(
        cudaEventRecord(completion_event, stream),
        "cudaEventRecord(qwen RMSNorm reference diagnostic completion)");
}

bool qwen_rms_norm_fp16_cuda(
        const void* d_input_f16, const void* d_weight_f16,
        void* d_output_f16, int rows, int dim, float epsilon,
        cudaStream_t stream) {
    if (!d_input_f16 || !d_weight_f16 || !d_output_f16 ||
        rows <= 0 || dim <= 0 || dim > 4096 ||
        !std::isfinite(epsilon) || epsilon < 0.0f) {
        return false;
    }
    qwen_rms_norm_fp16_kernel<false><<<rows, kThreads, 0, stream>>>(
        static_cast<const __half*>(d_input_f16),
        static_cast<const __half*>(d_weight_f16),
        static_cast<__half*>(d_output_f16),
        dim,
        epsilon,
        QwenRmsNormDiagnosticBuffers{});
    return DEE_CUDA_CHECK_NAMED(
        cudaPeekAtLastError(), "qwen_rms_norm_fp16_kernel");
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
