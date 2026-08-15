#include "dee/cuda_convert.h"

#include "dee/cuda_check.h"

#include <cstdio>
#include <cuda_fp16.h>
#include <limits>

namespace dee {
namespace {

constexpr int kConversionThreads = 256;

__global__ void bf16_to_f32_kernel(const uint16_t* source, float* destination,
                                   size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    destination[index] = __uint_as_float(static_cast<unsigned int>(source[index]) << 16);
}

__global__ void bf16_to_f16_kernel(const uint16_t* source, __half* destination,
                                   size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float value = __uint_as_float(static_cast<unsigned int>(source[index]) << 16);
    destination[index] = __float2half_rn(value);
}

__global__ void f32_to_f16_kernel(const float* source, __half* destination,
                                  size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    destination[index] = __float2half_rn(source[index]);
}

__global__ void int8_to_f16_kernel(const int8_t* source, __half* destination,
                                   size_t elements, size_t projection_elements,
                                   float gate_scale, float up_scale,
                                   float down_scale) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float scale = index < projection_elements ? gate_scale
        : (index < 2 * projection_elements ? up_scale : down_scale);
    destination[index] = __float2half_rn(static_cast<float>(source[index]) * scale);
}

__global__ void int4_to_f16_kernel(const uint8_t* source, __half* destination,
                                   size_t elements, size_t projection_elements,
                                   float gate_scale, float up_scale,
                                   float down_scale) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const uint8_t packed = source[index / 2];
    const uint8_t nibble = index & 1 ? packed >> 4 : packed & 0x0f;
    const int value = nibble >= 8 ? static_cast<int>(nibble) - 16 : nibble;
    const float scale = index < projection_elements ? gate_scale
        : (index < 2 * projection_elements ? up_scale : down_scale);
    destination[index] = __float2half_rn(static_cast<float>(value) * scale);
}

bool valid_conversion(const void* source, const void* destination, size_t elements,
                      cudaStream_t stream, const char* name) {
    if (!source || !destination || !stream || elements == 0) {
        std::fprintf(stderr, "[cuda] invalid %s arguments (elements=%zu)\n", name, elements);
        return false;
    }
    return true;
}

unsigned int conversion_blocks(size_t elements, const char* name) {
    const size_t blocks = (elements + kConversionThreads - 1) / kConversionThreads;
    if (blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        std::fprintf(stderr, "[cuda] %s grid is too large: %zu blocks\n", name, blocks);
        return 0;
    }
    return static_cast<unsigned int>(blocks);
}

}  // namespace

bool bf16_to_f32_cuda(const uint16_t* source, float* destination, size_t elements,
                      cudaStream_t stream, StageProfiler* profiler) {
    if (!source || !destination || !stream || elements == 0) {
        std::fprintf(stderr, "[cuda] invalid BF16 conversion arguments (elements=%zu)\n", elements);
        return false;
    }
    const size_t blocks = (elements + kConversionThreads - 1) / kConversionThreads;
    if (blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        std::fprintf(stderr, "[cuda] BF16 conversion grid is too large: %zu blocks\n", blocks);
        return false;
    }
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::WeightConversion, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    bf16_to_f32_kernel<<<static_cast<unsigned int>(blocks), kConversionThreads, 0, stream>>>(
        source, destination, elements);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("bf16_to_f32_kernel launch")) return false;
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(ticket, static_cast<void*>(stream))) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize(BF16 conversion validation)");
#else
    return true;
#endif
}

bool bf16_to_f16_cuda(const uint16_t* source, void* destination, size_t elements,
                      cudaStream_t stream, StageProfiler* profiler) {
    if (!valid_conversion(source, destination, elements, stream, "BF16-to-FP16 conversion")) return false;
    const unsigned int blocks = conversion_blocks(elements, "BF16-to-FP16 conversion");
    if (!blocks) return false;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::WeightConversion, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    bf16_to_f16_kernel<<<blocks, kConversionThreads, 0, stream>>>(
        source, static_cast<__half*>(destination), elements);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("bf16_to_f16_kernel launch")) return false;
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(ticket, static_cast<void*>(stream))) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize(BF16-to-FP16 validation)");
#else
    return true;
#endif
}

bool f32_to_f16_cuda(const float* source, void* destination, size_t elements,
                     cudaStream_t stream, StageProfiler* profiler) {
    if (!valid_conversion(source, destination, elements, stream, "FP32-to-FP16 conversion")) return false;
    const unsigned int blocks = conversion_blocks(elements, "FP32-to-FP16 conversion");
    if (!blocks) return false;
    f32_to_f16_kernel<<<blocks, kConversionThreads, 0, stream>>>(
        source, static_cast<__half*>(destination), elements);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("f32_to_f16_kernel launch")) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize(FP32-to-FP16 validation)");
#else
    return true;
#endif
}

bool int8_to_f16_cuda(const int8_t* source, void* destination, size_t elements,
                      size_t projection_elements, const float scales[3],
                      cudaStream_t stream, StageProfiler* profiler) {
    if (!valid_conversion(source, destination, elements, stream,
                          "INT8-to-FP16 conversion") || !scales ||
        projection_elements == 0 || elements != 3 * projection_elements) {
        std::fprintf(stderr, "[cuda] invalid INT8 projection layout (elements=%zu projection=%zu)\n",
                     elements, projection_elements);
        return false;
    }
    const unsigned int blocks = conversion_blocks(elements, "INT8-to-FP16 conversion");
    if (!blocks) return false;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::WeightConversion, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    int8_to_f16_kernel<<<blocks, kConversionThreads, 0, stream>>>(
        source, static_cast<__half*>(destination), elements, projection_elements,
        scales[0], scales[1], scales[2]);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("int8_to_f16_kernel launch")) return false;
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(ticket, static_cast<void*>(stream))) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize(INT8-to-FP16 validation)");
#else
    return true;
#endif
}

bool int4_to_f16_cuda(const uint8_t* source, void* destination, size_t elements,
                      size_t projection_elements, const float scales[3],
                      cudaStream_t stream, StageProfiler* profiler) {
    if (!valid_conversion(source, destination, elements, stream,
                          "INT4-to-FP16 conversion") || !scales ||
        projection_elements == 0 || elements != 3 * projection_elements) {
        std::fprintf(stderr, "[cuda] invalid INT4 projection layout (elements=%zu projection=%zu)\n",
                     elements, projection_elements);
        return false;
    }
    const unsigned int blocks = conversion_blocks(elements, "INT4-to-FP16 conversion");
    if (!blocks) return false;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::WeightConversion, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    int4_to_f16_kernel<<<blocks, kConversionThreads, 0, stream>>>(
        source, static_cast<__half*>(destination), elements, projection_elements,
        scales[0], scales[1], scales[2]);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("int4_to_f16_kernel launch")) return false;
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(ticket, static_cast<void*>(stream))) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize(INT4-to-FP16 validation)");
#else
    return true;
#endif
}

// Official 16-entry e2m1fn lookup table (convert.py).  Held in constant
// memory and copied once per block into shared memory: the per-element lookup
// index differs across threads, which serializes on a __constant__ broadcast,
// so the kernel reads the table from shared memory instead.
__constant__ float kFp4E2m1Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float fp4_scale_value(uint8_t bits) {
    // ue8m0: 2^(bits - 127).  Realistic bytes 0x7d..0x82; clamp extremes.
    const int exponent = static_cast<int>(bits) - 127;
    if (exponent >= 127) return __int_as_float(0x7F7FFFFFu);
    if (exponent <= -127) return 0.0f;
    return __uint_as_float(static_cast<unsigned int>(exponent + 127) << 23);
}

// Decode packed I8 [out, in//2] + e8m0 scale [out, in//32] -> FP16 [out, in].
// One thread per 32-element scale block (2D grid: blockIdx.y = row, blockIdx.x
// = scale-block tile).  Each thread loads its scale byte once, computes the
// e8m0 value once, then decodes 16 packed bytes (32 nibbles) with contiguous
// 16-byte reads / 64-byte writes.  The per-element version re-read and
// re-computed the scale for every element (32x redundant work).
__global__ void fp4_e2m1_to_f16_kernel(const uint8_t* packed, const uint8_t* scale,
                                       __half* destination, size_t out, size_t in) {
    __shared__ float s_table[16];
    if (threadIdx.x < 16) s_table[threadIdx.x] = kFp4E2m1Table[threadIdx.x];
    __syncthreads();
    const size_t scale_stride = in >> 5;   // 32-element scale blocks per row
    const size_t half_in = in >> 1;        // packed bytes per row
    const size_t o = blockIdx.y;           // output row
    const size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= scale_stride) return;
    const float s = fp4_scale_value(scale[o * scale_stride + b]);
    const uint8_t* p = packed + o * half_in + (b << 4);   // 16 packed bytes
    __half* d = destination + o * in + (b << 5);          // 32 output halves
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint8_t byte = p[j];
        d[j * 2]     = __float2half_rn(s_table[byte & 0x0F] * s);
        d[j * 2 + 1] = __float2half_rn(s_table[byte >> 4] * s);
    }
}

__global__ void oracle_relu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] = 0.0f;
}

bool fp4_e2m1_to_f16_cuda(const uint8_t* packed, const uint8_t* scale,
                          void* destination, size_t out, size_t in,
                          cudaStream_t stream, StageProfiler* profiler) {
    const size_t elements = out * in;
    if (!packed || !scale || !destination || !stream || elements == 0) {
        std::fprintf(stderr, "[cuda] invalid FP4-e2m1 conversion arguments (out=%zu in=%zu)\n",
                     out, in);
        return false;
    }
    if ((in % 64) != 0) {
        // in must be a multiple of 64 for the packed (in/2) and block (in/32)
        // layouts to be whole-byte aligned (real model: 2048/4096).
        std::fprintf(stderr, "[cuda] FP4-e2m1 requires in multiple of 64, got %zu\n", in);
        return false;
    }
    // 2D grid: blockIdx.y walks rows (out), blockIdx.x walks scale-block tiles
    // (in/32) of the row. No per-thread 64-bit division; reads/writes coalesced.
    if (out > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) return false;
    const size_t scale_stride = in >> 5;
    const dim3 grid(static_cast<unsigned int>(
                        (scale_stride + kConversionThreads - 1) / kConversionThreads),
                    static_cast<unsigned int>(out));
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::WeightConversion, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    fp4_e2m1_to_f16_kernel<<<grid, kConversionThreads, 0, stream>>>(
        packed, scale, static_cast<__half*>(destination), out, in);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("fp4_e2m1_to_f16_kernel launch")) return false;
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(ticket, static_cast<void*>(stream))) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize(FP4-e2m1 conversion validation)");
#else
    return true;
#endif
}

void oracle_relu_cuda(float* data, int n, cudaStream_t stream) {
    oracle_relu_kernel<<<1, 256, 0, stream>>>(data, n);
    DEE_CUDA_CHECK_LAUNCH("oracle_relu_kernel");
}

}  // namespace dee
