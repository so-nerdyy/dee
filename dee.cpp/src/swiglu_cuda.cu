// dee/swiglu_cuda.cu
//
// Step 9 — CUDA SwiGLU kernels (DEE_CUDA build only; compiled by nvcc).
//
// Each expert's forward is two phases (no shared-memory coordination needed;
// the intermediate h[inter] lives in a caller-owned device scratch buffer):
//   phase 1 (inter threads): h[i] = silu(Wg[i,:]·x) * (Wu[i,:]·x)
//   phase 2 (hidden threads): y[o] = Σ_i Wd[o*inter + i] * h[i]
// Then a combine kernel mean-pools the K expert outputs into h_out.

#include "dee/swiglu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

namespace dee {

__global__ void swiglu_h_kernel(const float* W, const float* x, float* h,
                                int inter, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inter) return;
    const float* Wg = W;
    const float* Wu = W + (size_t)inter * hidden;
    float g = 0.0f, f = 0.0f;
    const float* rg = Wg + (size_t)i * hidden;
    const float* ru = Wu + (size_t)i * hidden;
    for (int j = 0; j < hidden; ++j) { g += rg[j] * x[j]; f += ru[j] * x[j]; }
    float s = g / (1.0f + expf(g));   // SiLU
    h[i] = s * f;
}

__global__ void swiglu_y_kernel(const float* W, const float* h, float* y,
                                int inter, int hidden) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= hidden) return;
    const float* Wd = W + 2 * (size_t)inter * hidden;
    float acc = 0.0f;
    for (int i = 0; i < inter; ++i) acc += Wd[(size_t)o * inter + i] * h[i];
    y[o] = acc;
}

__global__ void combine_kernel(const float* ybuf, float* out, int K, int hidden) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= hidden) return;
    float s = 0.0f;
    for (int e = 0; e < K; ++e) s += ybuf[(size_t)e * hidden + o];
    out[o] = s / (float)K;
}

static inline int grid(int n) { return (n + 255) / 256; }

void swiglu_expert_cuda(const float* d_W, const float* d_x, float* d_hbuf,
                        float* d_y, int inter, int hidden, cudaStream_t stream) {
    swiglu_h_kernel<<<grid(inter), 256, 0, stream>>>(d_W, d_x, d_hbuf, inter, hidden);
    swiglu_y_kernel<<<grid(hidden), 256, 0, stream>>>(d_W, d_hbuf, d_y, inter, hidden);
}

void combine_cuda(const float* d_ybuf, float* d_out, int K, int hidden,
                  cudaStream_t stream) {
    combine_kernel<<<grid(hidden), 256, 0, stream>>>(d_ybuf, d_out, K, hidden);
}

} // namespace dee
