#pragma once

#ifdef DEE_CUDA

#include "dee/profiling.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace dee {

// Expand raw BF16 bit patterns to FP32 without changing their represented
// values. The conversion is ordered on `stream` after the preceding H2D copy.
bool bf16_to_f32_cuda(const uint16_t* source, float* destination, size_t elements,
                      cudaStream_t stream, StageProfiler* profiler = nullptr);

bool bf16_to_f16_cuda(const uint16_t* source, void* destination, size_t elements,
                      cudaStream_t stream, StageProfiler* profiler = nullptr);

bool f32_to_f16_cuda(const float* source, void* destination, size_t elements,
                     cudaStream_t stream, StageProfiler* profiler = nullptr);

bool int8_to_f16_cuda(const int8_t* source, void* destination, size_t elements,
                      size_t projection_elements, const float scales[3],
                      cudaStream_t stream, StageProfiler* profiler = nullptr);

bool int4_to_f16_cuda(const uint8_t* source, void* destination, size_t elements,
                      size_t projection_elements, const float scales[3],
                      cudaStream_t stream, StageProfiler* profiler = nullptr);

// DeepSeek-V4-Flash-0731 FP4 (e2m1fn) expert-weight dequantize to FP16.
// Decodes the official packed layout on the transfer stream:
//   - packed I8 [out, in//2]: low nibble -> col 2i, high nibble -> col 2i+1,
//     decoded through the 16-entry e2m1fn table (matches weight_mmap.cpp).
//   - scale F8_E8M0 [out, in//32]: one 2^(bits-127) scale per 32 input cols.
// destination receives out*in __half values.  Unlike int4_to_f16_cuda (three
// scalar per-projection scales), the scale here is a per-block tensor, which
// is why this kernel takes its own signature rather than reusing scales[3].
bool fp4_e2m1_to_f16_cuda(const uint8_t* packed, const uint8_t* scale,
                          void* destination, size_t out, size_t in,
                          cudaStream_t stream, StageProfiler* profiler = nullptr);

// Small ReLU kernel for GPU Oracle activations (n <= 256).
void oracle_relu_cuda(float* data, int n, cudaStream_t stream);

}  // namespace dee

#endif
