#include "dee/swiglu_cuda.h"

#include "dee/cuda_check.h"

#include <cuda_fp16.h>
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

__global__ void swiglu_activation_fp16_kernel(float* gate, const float* up,
                                              __half* activation, int inter) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inter) return;
    const float g = gate[i];
    activation[i] = __float2half_rn((g / (1.0f + expf(-g))) * up[i]);
}

__global__ void combine_kernel(const float* ybuf, float* output, int experts, int hidden) {
    const int value = blockIdx.x * blockDim.x + threadIdx.x;
    if (value >= hidden) return;
    float sum = 0.0f;
    for (int expert = 0; expert < experts; ++expert) sum += ybuf[static_cast<size_t>(expert) * hidden + value];
    output[value] = sum / static_cast<float>(experts);
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
    const size_t projection = static_cast<size_t>(inter) * hidden;
    if (!gemm_fp16_row_major(handle, weights, inter, hidden, input, d_gate,
                             stream, GpuStage::GateProjection, profiler) ||
        !gemm_fp16_row_major(handle, weights + projection, inter, hidden, input, d_up,
                             stream, GpuStage::UpProjection, profiler)) return false;
    const size_t activation_ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::SiluMultiply, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    swiglu_activation_fp16_kernel<<<grid_for(inter), kThreads, 0, stream>>>(
        d_gate, d_up, activation, inter);
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

}  // namespace dee
