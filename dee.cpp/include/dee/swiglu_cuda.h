// dee/swiglu_cuda.h
//
// Step 9 — CUDA SwiGLU kernel launch interface (DEE_CUDA build only).
//
// Declared only when DEE_CUDA is defined, so the CPU-only build (this WSL box)
// never sees these symbols and the Engine's #ifdef DEE_CUDA branch is the only
// caller. The implementations live in src/swiglu_cuda.cu (compiled by nvcc).

#pragma once

#ifdef DEE_CUDA
#include "dee/profiling.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace dee {

// One expert's SwiGLU, fully on-GPU:
//   h[i] = silu(Wg[i,:]·x) * (Wu[i,:]·x)              (phase 1)
//   y[o] = Σ_i Wd[o*inter + i] * h[i]                 (phase 2)
// W layout: [gate: inter*hidden][up: inter*hidden][down: hidden*inter].
// d_hbuf is a caller-owned inter-float scratch buffer (reused across experts).
// ``swiglu_limit`` (>0) applies the DeepSeek-V4-Flash activation clamp:
// gate = clamp(gate, max=limit), up = clamp(up, -limit..limit), before SiLU.
// 0 (default) preserves the Ornith (no-clamp) semantics.
bool swiglu_expert_cuda(cublasHandle_t handle, const float* d_W, const float* d_x,
                        float* d_gate, float* d_up, float* d_y,
                        int inter, int hidden, cudaStream_t stream,
                        StageProfiler* profiler = nullptr,
                        float swiglu_limit = 0.0f);

// FP16 cache path: weights and input/activation remain FP16 while cuBLAS
// accumulates each projection into FP32 outputs.
bool swiglu_expert_fp16_cuda(cublasHandle_t handle, const void* d_W,
                             const void* d_x, float* d_gate, float* d_up,
                             void* d_activation, float* d_y,
                             int inter, int hidden, cudaStream_t stream,
                             StageProfiler* profiler = nullptr,
                             float swiglu_limit = 0.0f);

// Batched equivalent of the eager Transformers expert path. Every projection
// consumes [tokens, features] FP16 rows and produces FP16 rows so GEMM shape
// and rounding match torch.nn.functional.linear during prompt prefill.
bool swiglu_expert_batch_fp16_cuda(
    cublasHandle_t handle, const void* d_W, const void* d_x,
    void* d_gate, void* d_up, void* d_activation, float* d_y,
    int tokens, int inter, int hidden, cudaStream_t stream,
    StageProfiler* profiler = nullptr, float swiglu_limit = 0.0f);

// M5F token-1 pointer-batched path. Each pointer-array argument is a
// device-resident array of `experts` pointers. It executes all selected
// experts with three cublasGemmBatchedEx calls and one contiguous activation
// launch while preserving the established FP16 gate/up/activation and FP32
// down-projection output boundaries.
bool swiglu_expert_pointer_batch_fp16_cuda(
    cublasHandle_t handle,
    const void* d_gate_weight_ptrs, const void* d_up_weight_ptrs,
    const void* d_down_weight_ptrs, const void* d_input_ptrs,
    const void* d_gate_output_ptrs, const void* d_up_output_ptrs,
    const void* d_activation_ptrs, const void* d_raw_output_ptrs,
    void* d_gate, void* d_up, void* d_activation,
    int experts, int inter, int hidden, cudaStream_t stream,
    StageProfiler* profiler = nullptr, float swiglu_limit = 0.0f);

// Match torch.nn.functional.linear for an FP16 [tokens, hidden] input and
// FP16 [experts, hidden] router matrix. Output is row-major [tokens, experts].
bool router_logits_fp16_cuda(cublasHandle_t handle, const void* d_weights,
                             const void* d_hidden, void* d_logits,
                             int tokens, int experts, int hidden,
                             cudaStream_t stream,
                             StageProfiler* profiler = nullptr);

// Combine K expert outputs (each hidden floats, packed in d_ybuf) -> mean.
bool combine_cuda(const float* d_ybuf, float* d_out, int K, int hidden,
                  cudaStream_t stream, StageProfiler* profiler = nullptr);

// Exact eager-compatible weighted combine for routed expert rows. Raw FP32
// down-projection outputs are rounded to FP16; each multiplication by an FP32
// router scalar is rounded to FP16; then every in-place addition is separately
// rounded to FP16. Expert positions are visited in stable ascending expert-ID
// order. A null stream handle denotes CUDA's valid default stream.
bool weighted_combine_fp16_cuda(
    const float* d_raw_f32, const float* d_weights_f32,
    const int64_t* d_expert_ids_i64, void* d_output_f16,
    int tokens, int topk, int hidden, cudaStream_t stream,
    StageProfiler* profiler = nullptr);

} // namespace dee
#endif
