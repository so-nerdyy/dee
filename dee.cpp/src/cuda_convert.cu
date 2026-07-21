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

__global__ void mixed_int4_to_f16_kernel(__half* destination, size_t projection_elements,
                                         MixedInt4Args args) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = 3 * projection_elements;
    if (index >= elements) return;

    // Determine which projection this element belongs to.
    int projection = 0;
    size_t offset_in_projection = index;
    if (index >= 2 * projection_elements) {
        projection = 2;
        offset_in_projection = index - 2 * projection_elements;
    } else if (index >= projection_elements) {
        projection = 1;
        offset_in_projection = index - projection_elements;
    }

    const int row_width = args.row_width[projection];
    const int row_count = args.row_count[projection];
    if (row_width <= 0 || row_count <= 0) {
        destination[index] = __float2half_rn(0.0f);
        return;
    }

    const int row = static_cast<int>(offset_in_projection / row_width);
    const int col = static_cast<int>(offset_in_projection % row_width);

    const uint32_t outlier_info = args.outlier_row_offsets[projection][row];
    if (outlier_info != 0xFFFFFFFFu) {
        const unsigned int outlier_bits = static_cast<unsigned int>(
            args.outlier_values[projection][outlier_info + col]) << 16;
        const float outlier_f32 = __uint_as_float(outlier_bits);
        destination[index] = __float2half_rn(outlier_f32);
        return;
    }

    const int group_size = args.group_size;
    const int groups_per_row = (row_width + group_size - 1) / group_size;
    const int group_idx = col / group_size;
    const float scale = args.group_scales[projection][row * groups_per_row + group_idx];

    const size_t bulk_idx = static_cast<size_t>(row) * row_width + col;
    const uint8_t packed = args.bulk[projection][bulk_idx / 2];
    const uint8_t nibble = (bulk_idx & 1) ? (packed >> 4) : (packed & 0x0f);
    const int value = nibble >= 8 ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
    destination[index] = __float2half_rn(static_cast<float>(value) * scale);
}

bool mixed_int4_to_f16_cuda(const uint8_t* source, void* destination,
                            size_t projection_elements, const MixedInt4Args& args,
                            cudaStream_t stream, StageProfiler* profiler) {
    if (!destination || !stream || projection_elements == 0 ||
        args.group_size <= 0) {
        std::fprintf(stderr, "[cuda] invalid mixed-INT4 arguments\n");
        return false;
    }
    const size_t elements = 3 * projection_elements;
    const unsigned int blocks = conversion_blocks(elements, "mixed-INT4-to-FP16 conversion");
    if (!blocks) return false;
    const size_t ticket = profiler && profiler->enabled()
        ? profiler->cuda_begin(GpuStage::WeightConversion, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    mixed_int4_to_f16_kernel<<<blocks, kConversionThreads, 0, stream>>>(
        static_cast<__half*>(destination), projection_elements, args);
    if (profiler && profiler->enabled()) profiler->note_kernel_launch();
    if (!DEE_CUDA_CHECK_LAUNCH("mixed_int4_to_f16_kernel launch")) return false;
    if (profiler && profiler->enabled() &&
        !profiler->cuda_end(ticket, static_cast<void*>(stream))) return false;
#ifdef DEE_CUDA_VALIDATE
    return DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(stream),
                                "cudaStreamSynchronize(mixed-INT4-to-FP16 validation)");
#else
    return true;
#endif
}

}  // namespace dee
