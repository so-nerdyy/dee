#include "dee/swiglu_cuda.h"

#include "dee/cuda_check.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
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

__global__ void swiglu_activation_fp16_kernel(const __half* gate, const __half* up,
                                               __half* activation, int inter) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inter) return;
    const float g = __half2float(gate[i]);
    // Match torch FP16 eager semantics: SiLU is rounded to FP16 before the
    // elementwise multiply, whose result is rounded to FP16 again.
    const __half silu = __float2half_rn(g / (1.0f + expf(-g)));
    activation[i] = __hmul(silu, up[i]);
}

__global__ void swiglu_activation_batch_fp16_kernel(
        const __half* gate, const __half* up, __half* activation, int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const float g = __half2float(gate[i]);
    const __half silu = __float2half_rn(g / (1.0f + expf(-g)));
    activation[i] = __hmul(silu, up[i]);
}

__global__ void combine_kernel(const float* ybuf, float* output, int experts, int hidden) {
    const int value = blockIdx.x * blockDim.x + threadIdx.x;
    if (value >= hidden) return;
    float sum = 0.0f;
    for (int expert = 0; expert < experts; ++expert) sum += ybuf[static_cast<size_t>(expert) * hidden + value];
    output[value] = sum / static_cast<float>(experts);
}

__device__ int stable_position_for_rank(
        const int64_t* expert_ids, int topk, int rank) {
    for (int position = 0; position < topk; ++position) {
        int stable_rank = 0;
        const int64_t expert = expert_ids[position];
        for (int other = 0; other < topk; ++other) {
            const int64_t other_expert = expert_ids[other];
            if (other_expert < expert ||
                (other_expert == expert && other < position)) {
                ++stable_rank;
            }
        }
        if (stable_rank == rank) return position;
    }
    return -1;
}

__global__ void weighted_combine_fp16_kernel(
        const float* raw_f32, const float* weights,
        const int64_t* expert_ids, __half* output,
        int tokens, int topk, int hidden) {
    const int value = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = tokens * hidden;
    if (value >= elements) return;
    const int token = value / hidden;
    const int feature = value - token * hidden;
    const size_t token_selection = static_cast<size_t>(token) * topk;
    const int64_t* token_experts = expert_ids + token_selection;
    __half accumulator = __float2half_rn(0.0f);
    for (int rank = 0; rank < topk; ++rank) {
        const int position = stable_position_for_rank(
            token_experts, topk, rank);
        if (position < 0) return;
        const size_t selection = token_selection +
            static_cast<size_t>(position);
        const __half raw = __float2half_rn(
            raw_f32[selection * hidden + feature]);
        // Eager PyTorch materializes the mixed FP16-vector/FP32-scalar
        // multiplication as FP16 before output.add_ performs a second FP16
        // rounding. Keep both boundaries explicit; one fused float expression
        // is observably different for adversarial values.
        const __half weighted = __float2half_rn(
            __half2float(raw) * weights[selection]);
        accumulator = __float2half_rn(
            __half2float(accumulator) + __half2float(weighted));
    }
    output[value] = accumulator;
}

int grid_for(int count) { return (count + kThreads - 1) / kThreads; }

bool gemm_fp16_row_major(cublasHandle_t handle, const __half* matrix,
                         int rows, int cols, const __half* input, float* output,
                         cudaStream_t stream, GpuStage stage,
                         StageProfiler* profiler) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(stage, static_cast<void*>(stream)) : static_cast<size_t>(-1);
    if (profiler && profiler->enabled()) profiler->note_cublas_call();
    const auto dispatch_begin = profiler && profiler->timeline_enabled()
        ? StageProfiler::now() : StageProfiler::TimePoint{};
    const bool ok = DEE_CUBLAS_CHECK_NAMED(
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     rows, 1, cols, &alpha,
                     matrix, CUDA_R_16F, cols,
                     input, CUDA_R_16F, cols,
                     &beta, output, CUDA_R_32F, rows,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx(FP16 row-major expert projection)");
    if (profiler && profiler->timeline_enabled()) {
        profiler->note_cpu_timeline(CpuTimelineKind::CublasDispatch, dispatch_begin);
    }
    if (!ok) return false;
    return !profiler || !profiler->enabled() ||
           profiler->cuda_end(ticket, static_cast<void*>(stream));
}

bool gemm_fp16_row_major_to_fp16(cublasHandle_t handle, const __half* matrix,
                                 int rows, int cols, const __half* input,
                                 __half* output, cudaStream_t stream,
                                 GpuStage stage, StageProfiler* profiler) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(stage, static_cast<void*>(stream)) : static_cast<size_t>(-1);
    if (profiler && profiler->enabled()) profiler->note_cublas_call();
    const bool ok = DEE_CUBLAS_CHECK_NAMED(
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     rows, 1, cols, &alpha,
                     matrix, CUDA_R_16F, cols,
                     input, CUDA_R_16F, cols,
                     &beta, output, CUDA_R_16F, rows,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx(FP16 row-major expert projection to FP16)");
    if (!ok) return false;
    return !profiler || !profiler->enabled() ||
           profiler->cuda_end(ticket, static_cast<void*>(stream));
}

bool gemm_fp16_row_major_batch_to_fp16(
        cublasHandle_t handle, const __half* matrix, int rows, int cols,
        const __half* input, int tokens, __half* output, cudaStream_t stream,
        GpuStage stage, StageProfiler* profiler) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(stage, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    if (profiler && profiler->enabled()) profiler->note_cublas_call();
    // Row-major W[rows,cols] and X[tokens,cols] are storage-equivalent to
    // column-major W^T[cols,rows] and X^T[cols,tokens].
    const bool ok = DEE_CUBLAS_CHECK_NAMED(
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     rows, tokens, cols, &alpha,
                     matrix, CUDA_R_16F, cols,
                     input, CUDA_R_16F, cols,
                     &beta, output, CUDA_R_16F, rows,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx(FP16 row-major expert batch)");
    if (!ok) return false;
    return !profiler || !profiler->enabled() ||
           profiler->cuda_end(ticket, static_cast<void*>(stream));
}

bool gemm_fp16_row_major_batch_to_fp32(
        cublasHandle_t handle, const __half* matrix, int rows, int cols,
        const __half* input, int tokens, float* output, cudaStream_t stream,
        GpuStage stage, StageProfiler* profiler) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(stage, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    if (profiler && profiler->enabled()) profiler->note_cublas_call();
    const bool ok = DEE_CUBLAS_CHECK_NAMED(
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     rows, tokens, cols, &alpha,
                     matrix, CUDA_R_16F, cols,
                     input, CUDA_R_16F, cols,
                     &beta, output, CUDA_R_32F, rows,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx(FP16 row-major expert batch to FP32)");
    if (!ok) return false;
    return !profiler || !profiler->enabled() ||
           profiler->cuda_end(ticket, static_cast<void*>(stream));
}

bool gemm_fp16_pointer_batch(
        cublasHandle_t handle, const void* d_matrix_ptrs,
        int rows, int cols, const void* d_input_ptrs,
        void* d_output_ptrs, cudaDataType_t output_type, int batch,
        cudaStream_t stream, GpuStage stage, StageProfiler* profiler) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(stage, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    if (profiler && profiler->enabled()) profiler->note_cublas_call();
    const bool ok = DEE_CUBLAS_CHECK_NAMED(
        cublasGemmBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            rows, 1, cols, &alpha,
            reinterpret_cast<const void* const*>(d_matrix_ptrs),
            CUDA_R_16F, cols,
            reinterpret_cast<const void* const*>(d_input_ptrs),
            CUDA_R_16F, cols,
            &beta,
            reinterpret_cast<void* const*>(d_output_ptrs),
            output_type, rows,
            batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmBatchedEx(FP16 pointer-batched experts)");
    if (!ok) return false;
    return !profiler || !profiler->enabled() ||
           profiler->cuda_end(ticket, static_cast<void*>(stream));
}

bool gemv_row_major(cublasHandle_t handle, const float* matrix, int rows, int cols,
                    const float* input, float* output, cudaStream_t stream,
                    GpuStage stage, StageProfiler* profiler) {
    // Row-major [rows, cols] is the storage-equivalent of column-major
    // [cols, rows]; transposed SGEMV therefore computes W * input.
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(stage, static_cast<void*>(stream)) : static_cast<size_t>(-1);
    if (profiler && profiler->enabled()) profiler->note_cublas_call();
    const auto dispatch_begin = profiler && profiler->timeline_enabled()
        ? StageProfiler::now() : StageProfiler::TimePoint{};
    const bool ok = DEE_CUBLAS_CHECK_NAMED(
        cublasSgemv(handle, CUBLAS_OP_T, cols, rows, &alpha, matrix, cols,
                    input, 1, &beta, output, 1),
        "cublasSgemv(row-major expert projection)");
    if (profiler && profiler->timeline_enabled()) {
        profiler->note_cpu_timeline(CpuTimelineKind::CublasDispatch, dispatch_begin);
    }
    if (!ok) return false;
    return !profiler || !profiler->enabled() || profiler->cuda_end(ticket, static_cast<void*>(stream));
}

}  // namespace

bool swiglu_expert_cuda(cublasHandle_t handle, const float* d_weights, const float* d_x,
                        float* d_gate, float* d_up, float* d_y,
                        int inter, int hidden, cudaStream_t stream, StageProfiler* profiler) {
    if (!handle || !d_weights || !d_x || !d_gate || !d_up || !d_y || !stream || inter <= 0 || hidden <= 0) {
        std::fprintf(stderr, "[cuda] invalid cuBLAS SwiGLU arguments (inter=%d hidden=%d)\n", inter, hidden);
        return false;
    }
    const size_t projection = static_cast<size_t>(inter) * hidden;
    if (!gemv_row_major(handle, d_weights, inter, hidden, d_x, d_gate, stream,
                        GpuStage::GateProjection, profiler) ||
        !gemv_row_major(handle, d_weights + projection, inter, hidden, d_x, d_up, stream,
                        GpuStage::UpProjection, profiler)) return false;
    const size_t activation_ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::SiluMultiply, static_cast<void*>(stream)) : static_cast<size_t>(-1);
    swiglu_activation_kernel<<<grid_for(inter), kThreads, 0, stream>>>(d_gate, d_up, inter);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("swiglu_activation_kernel launch")) return false;
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(activation_ticket, static_cast<void*>(stream))) return false;
    if (!gemv_row_major(handle, d_weights + 2 * projection, hidden, inter, d_gate, d_y, stream,
                        GpuStage::DownProjection, profiler)) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SwiGLU validation)");
#else
    return true;
#endif
}

bool swiglu_expert_fp16_cuda(cublasHandle_t handle, const void* d_weights,
                             const void* d_x, float* d_gate, float* d_up,
                             void* d_activation, float* d_y,
                             int inter, int hidden, cudaStream_t stream,
                             StageProfiler* profiler) {
    if (!handle || !d_weights || !d_x || !d_gate || !d_up || !d_activation ||
        !d_y || !stream || inter <= 0 || hidden <= 0) {
        std::fprintf(stderr, "[cuda] invalid FP16 cuBLAS SwiGLU arguments (inter=%d hidden=%d)\n",
                     inter, hidden);
        return false;
    }
    const auto* weights = static_cast<const __half*>(d_weights);
    const auto* input = static_cast<const __half*>(d_x);
    auto* activation = static_cast<__half*>(d_activation);
    auto* gate = reinterpret_cast<__half*>(d_gate);
    auto* up = reinterpret_cast<__half*>(d_up);
    const size_t projection = static_cast<size_t>(inter) * hidden;
    if (!gemm_fp16_row_major_to_fp16(handle, weights, inter, hidden, input, gate,
                                     stream, GpuStage::GateProjection, profiler) ||
        !gemm_fp16_row_major_to_fp16(handle, weights + projection, inter, hidden, input, up,
                                     stream, GpuStage::UpProjection, profiler)) return false;
    const size_t activation_ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::SiluMultiply, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    swiglu_activation_fp16_kernel<<<grid_for(inter), kThreads, 0, stream>>>(
        gate, up, activation, inter);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("swiglu_activation_fp16_kernel launch")) return false;
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(activation_ticket, static_cast<void*>(stream))) return false;
    if (!gemm_fp16_row_major(handle, weights + 2 * projection, hidden, inter,
                             activation, d_y, stream,
                             GpuStage::DownProjection, profiler)) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize(FP16 SwiGLU validation)");
#else
    return true;
#endif
}

bool swiglu_expert_batch_fp16_cuda(
        cublasHandle_t handle, const void* d_weights, const void* d_x,
        void* d_gate, void* d_up, void* d_activation, float* d_y,
        int tokens, int inter, int hidden, cudaStream_t stream,
        StageProfiler* profiler) {
    if (!handle || !d_weights || !d_x || !d_gate || !d_up || !d_activation ||
        !d_y || !stream || tokens <= 0 || inter <= 0 || hidden <= 0) {
        std::fprintf(stderr,
            "[cuda] invalid FP16 batched SwiGLU arguments (tokens=%d inter=%d hidden=%d)\n",
            tokens, inter, hidden);
        return false;
    }
    const auto* weights = static_cast<const __half*>(d_weights);
    const auto* input = static_cast<const __half*>(d_x);
    auto* gate = static_cast<__half*>(d_gate);
    auto* up = static_cast<__half*>(d_up);
    auto* activation = static_cast<__half*>(d_activation);
    const size_t projection = static_cast<size_t>(inter) * hidden;
    if (!gemm_fp16_row_major_batch_to_fp16(
            handle, weights, inter, hidden, input, tokens, gate, stream,
            GpuStage::GateProjection, profiler) ||
        !gemm_fp16_row_major_batch_to_fp16(
            handle, weights + projection, inter, hidden, input, tokens, up, stream,
            GpuStage::UpProjection, profiler)) return false;
    const int activation_elements = tokens * inter;
    const size_t activation_ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::SiluMultiply, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    swiglu_activation_batch_fp16_kernel<<<
        grid_for(activation_elements), kThreads, 0, stream>>>(
            gate, up, activation, activation_elements);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("swiglu_activation_batch_fp16_kernel launch")) return false;
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(activation_ticket, static_cast<void*>(stream))) return false;
    if (!gemm_fp16_row_major_batch_to_fp32(
            handle, weights + 2 * projection, hidden, inter, activation, tokens,
            d_y, stream, GpuStage::DownProjection, profiler)) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize(FP16 batched SwiGLU validation)");
#else
    return true;
#endif
}

bool swiglu_expert_pointer_batch_fp16_cuda(
        cublasHandle_t handle,
        const void* d_gate_weight_ptrs, const void* d_up_weight_ptrs,
        const void* d_down_weight_ptrs, const void* d_input_ptrs,
        const void* d_gate_output_ptrs, const void* d_up_output_ptrs,
        const void* d_activation_ptrs, const void* d_raw_output_ptrs,
        void* d_gate, void* d_up, void* d_activation,
        int experts, int inter, int hidden, cudaStream_t stream,
        StageProfiler* profiler) {
    if (!handle || !d_gate_weight_ptrs || !d_up_weight_ptrs ||
        !d_down_weight_ptrs || !d_input_ptrs || !d_gate_output_ptrs ||
        !d_up_output_ptrs || !d_activation_ptrs || !d_raw_output_ptrs ||
        !d_gate || !d_up || !d_activation || experts <= 0 ||
        inter <= 0 || hidden <= 0 || !stream) {
        std::fprintf(
            stderr,
            "[cuda] invalid pointer-batched SwiGLU arguments "
            "(experts=%d inter=%d hidden=%d)\n",
            experts, inter, hidden);
        return false;
    }
    if (!gemm_fp16_pointer_batch(
            handle, d_gate_weight_ptrs, inter, hidden, d_input_ptrs,
            const_cast<void*>(d_gate_output_ptrs), CUDA_R_16F, experts,
            stream, GpuStage::GateProjection, profiler) ||
        !gemm_fp16_pointer_batch(
            handle, d_up_weight_ptrs, inter, hidden, d_input_ptrs,
            const_cast<void*>(d_up_output_ptrs), CUDA_R_16F, experts,
            stream, GpuStage::UpProjection, profiler)) {
        return false;
    }
    const int activation_elements = experts * inter;
    const size_t activation_ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(
            GpuStage::SiluMultiply, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    swiglu_activation_batch_fp16_kernel<<<
        grid_for(activation_elements), kThreads, 0, stream>>>(
            static_cast<const __half*>(d_gate),
            static_cast<const __half*>(d_up),
            static_cast<__half*>(d_activation),
            activation_elements);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH(
            "swiglu_activation_pointer_batch_fp16_kernel launch")) {
        return false;
    }
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(
            activation_ticket, static_cast<void*>(stream))) {
        return false;
    }
    if (!gemm_fp16_pointer_batch(
            handle, d_down_weight_ptrs, hidden, inter, d_activation_ptrs,
            const_cast<void*>(d_raw_output_ptrs), CUDA_R_32F, experts,
            stream, GpuStage::DownProjection, profiler)) {
        return false;
    }
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize(pointer-batched FP16 SwiGLU validation)");
#else
    return true;
#endif
}

bool router_logits_fp16_cuda(cublasHandle_t handle, const void* d_weights,
                             const void* d_hidden, void* d_logits,
                             int tokens, int experts, int hidden,
                             cudaStream_t stream, StageProfiler* profiler) {
    if (!handle || !d_weights || !d_hidden || !d_logits || !stream ||
        tokens <= 0 || experts <= 0 || hidden <= 0) {
        std::fprintf(stderr,
            "[cuda] invalid FP16 router arguments (tokens=%d experts=%d hidden=%d)\n",
            tokens, experts, hidden);
        return false;
    }
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::GateProjection, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    if (profiler && profiler->enabled()) profiler->note_cublas_call();
    // Row-major W[experts,hidden] is column-major W^T[hidden,experts], and
    // row-major X[tokens,hidden] is column-major X^T[hidden,tokens]. The
    // column-major result [experts,tokens] is the same storage as row-major
    // [tokens,experts].
    const bool ok = DEE_CUBLAS_CHECK_NAMED(
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     experts, tokens, hidden, &alpha,
                     d_weights, CUDA_R_16F, hidden,
                     d_hidden, CUDA_R_16F, hidden,
                     &beta, d_logits, CUDA_R_16F, experts,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx(FP16 batched router)");
    if (!ok) return false;
    return !profiler || !profiler->enabled() ||
           profiler->cuda_end(ticket, static_cast<void*>(stream));
}

bool combine_cuda(const float* d_ybuf, float* d_output, int experts, int hidden,
                  cudaStream_t stream, StageProfiler* profiler) {
    if (!d_ybuf || !d_output || !stream || experts <= 0 || hidden <= 0) {
        std::fprintf(stderr, "[cuda] invalid combine launch arguments (experts=%d hidden=%d)\n", experts, hidden);
        return false;
    }
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::Combine, static_cast<void*>(stream)) : static_cast<size_t>(-1);
    combine_kernel<<<grid_for(hidden), kThreads, 0, stream>>>(d_ybuf, d_output, experts, hidden);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("combine_kernel launch")) return false;
    if (profiler && profiler->enabled() && !profiler->cuda_end(ticket, static_cast<void*>(stream))) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream), "cudaStreamSynchronize(combine validation)");
#else
    return true;
#endif
}

bool weighted_combine_fp16_cuda(
        const float* d_raw_f32, const float* d_weights_f32,
        const int64_t* d_expert_ids_i64, void* d_output_f16,
        int tokens, int topk, int hidden, cudaStream_t stream,
        StageProfiler* profiler) {
    if (!d_raw_f32 || !d_weights_f32 || !d_expert_ids_i64 ||
        !d_output_f16 || tokens <= 0 || topk <= 0 ||
        hidden <= 0) {
        std::fprintf(
            stderr,
            "[cuda] invalid weighted combine arguments "
            "(tokens=%d topk=%d hidden=%d)\n",
            tokens, topk, hidden);
        return false;
    }
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(
            GpuStage::Combine, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    weighted_combine_fp16_kernel<<<
        grid_for(tokens * hidden), kThreads, 0, stream>>>(
            d_raw_f32,
            d_weights_f32,
            d_expert_ids_i64,
            static_cast<__half*>(d_output_f16),
            tokens,
            topk,
            hidden);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH(
            "weighted_combine_fp16_kernel launch")) return false;
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(ticket, static_cast<void*>(stream))) {
        return false;
    }
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize(weighted combine validation)");
#else
    return true;
#endif
}

}  // namespace dee
