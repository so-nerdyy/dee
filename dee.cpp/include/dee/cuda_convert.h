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

}  // namespace dee

#endif
