#pragma once

#include <cstddef>
#include <cstdint>

namespace dee {

// Layout descriptor for a mixed-INT4 expert weight blob.  The host packs
// (bulk INT4 nibbles, per-group FP32 scales, per-row outlier offsets,
// FP16 outlier values) into a single transferable buffer.  The device
// dequantization kernel uses these offsets to reconstruct FP16 weights.
struct MixedInt4Args {
    int group_size = 0;
    size_t projection_elements = 0;
    int row_width[3] = {};
    int row_count[3] = {};
    size_t num_groups[3] = {};

    // Offsets (in bytes) within the transferred source blob.
    size_t bulk_offset[3] = {};
    size_t scale_offset[3] = {};
    size_t outlier_row_offset_offset[3] = {};
    size_t outlier_values_offset[3] = {};

    // Device pointers into the transferred source blob (filled by prefetcher).
    const uint8_t* bulk[3] = {};
    const float* group_scales[3] = {};
    const uint32_t* outlier_row_offsets[3] = {};
    const uint16_t* outlier_values[3] = {};
};

} // namespace dee
