// dee/swiglu_cuda.h
//
// Step 9 — CUDA SwiGLU kernel launch interface (DEE_CUDA build only).
//
// Declared only when DEE_CUDA is defined, so the CPU-only build (this WSL box)
// never sees these symbols and the Engine's #ifdef DEE_CUDA branch is the only
// caller. The implementations live in src/swiglu_cuda.cu (compiled by nvcc).

#pragma once

#ifdef DEE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace dee {

// One expert's SwiGLU, fully on-GPU:
//   h[i] = silu(Wg[i,:]·x) * (Wu[i,:]·x)              (phase 1)
//   y[o] = Σ_i Wd[o*inter + i] * h[i]                 (phase 2)
// W layout: [gate: inter*hidden][up: inter*hidden][down: hidden*inter].
// d_hbuf is a caller-owned inter-float scratch buffer (reused across experts).
bool swiglu_expert_cuda(cublasHandle_t handle, const float* d_W, const float* d_x,
                        float* d_gate, float* d_up, float* d_y,
                        int inter, int hidden, cudaStream_t stream);

// Combine K expert outputs (each hidden floats, packed in d_ybuf) -> mean.
bool combine_cuda(const float* d_ybuf, float* d_out, int K, int hidden,
                  cudaStream_t stream);

} // namespace dee
#endif
