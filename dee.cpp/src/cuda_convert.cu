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

__global__ void mixed_int4b_to_f16_kernel(
    const uint8_t* source, __half* destination,
    unsigned int elements_per_proj, unsigned int region,
    unsigned int projection_elements, unsigned int bulk_bytes,
    float int4_scale, float int8_scale, uint64_t outlier_mask,
    uint32_t int8_offset, unsigned int header_bytes,
    unsigned int chunk_elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements_per_proj) return;
    const size_t global_idx = static_cast<size_t>(region) * projection_elements + index;
    const unsigned int chunk = static_cast<unsigned int>(index / chunk_elements);
    const bool is_outlier = (outlier_mask >> chunk) & 1ULL;
    float value = 0.0f;
    if (is_outlier) {
        const uint32_t chunk_popcount = static_cast<uint32_t>(
            __popcll(outlier_mask & ((1ULL << chunk) - 1)));
        const uint32_t outlier_idx = int8_offset + chunk_popcount * chunk_elements
            + static_cast<uint32_t>(index % chunk_elements);
        const int8_t int8_val = static_cast<int8_t>(source[header_bytes + bulk_bytes + outlier_idx]);
        value = static_cast<float>(int8_val) * int8_scale;
    } else {
        const unsigned int packed_idx = static_cast<unsigned int>(
            static_cast<size_t>(region) * projection_elements / 2 + index / 2);
        const uint8_t packed = source[header_bytes + packed_idx];
        const uint8_t nibble = index & 1 ? packed >> 4 : packed & 0x0f;
        const int int_val = nibble >= 8 ? static_cast<int>(nibble) - 16 : nibble;
        value = static_cast<float>(int_val) * int4_scale;
    }
    destination[global_idx] = __float2half_rn(value);
}

bool mixed_int4b_to_f16_cuda(const uint8_t* source, void* destination,
                             size_t total_elements, size_t projection_elements,
                             const float int4_scales[3], const float int8_scales[3],
                             const uint64_t outlier_masks[3],
                             cudaStream_t stream, StageProfiler* profiler) {
    if (!source || !destination || !stream || total_elements == 0 ||
        !int4_scales || !int8_scales || !outlier_masks ||
        projection_elements == 0 || total_elements != 3 * projection_elements) {
        std::fprintf(stderr, "[cuda] invalid mixed-int4b conversion arguments\n");
        return false;
    }
    constexpr size_t kHeaderBytes = 36;
    constexpr size_t kChunkElements = 2048;
    const size_t bulk_bytes = (total_elements + 1) / 2;
    const unsigned int elements_per_proj = static_cast<unsigned int>(projection_elements);
    const unsigned int blocks = (elements_per_proj + kConversionThreads - 1) / kConversionThreads;
    if (!blocks) return false;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::WeightConversion, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    // Scan outlier masks on host to precompute per-projection INT8 offsets
    uint32_t int8_offsets[3] = {0, 0, 0};
    for (size_t region = 1; region < 3; ++region) {
        int8_offsets[region] = int8_offsets[region - 1]
            + static_cast<uint32_t>(__builtin_popcountll(outlier_masks[region - 1]))
            * static_cast<uint32_t>(kChunkElements);
    }
    // Launch per-projection dequantization
    for (size_t region = 0; region < 3; ++region) {
        mixed_int4b_to_f16_kernel<<<blocks, kConversionThreads, 0, stream>>>(
            source, static_cast<__half*>(destination),
            elements_per_proj, static_cast<unsigned int>(region),
            static_cast<unsigned int>(projection_elements),
            static_cast<unsigned int>(bulk_bytes),
            int4_scales[region], int8_scales[region],
            outlier_masks[region], int8_offsets[region],
            static_cast<unsigned int>(kHeaderBytes), static_cast<unsigned int>(kChunkElements));
        if (profiler && profiler->enabled()) profiler->note_kernel_launch();
        if (!DEE_CUDA_CHECK_LAUNCH("mixed_int4b_to_f16_kernel launch")) return false;
    }
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(ticket, static_cast<void*>(stream))) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize(mixed-int4b validation)");
#else
    return true;
#endif
}

}  // namespace dee
