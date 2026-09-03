// kt_bridge/packed_expert_view.hpp
//
// Borrowed views into dee's bounded expert record. No ownership.
//
// Part of the isolated KT CPU-bridge prototype
// (dee.cpp/experiments/kt_cpu_bridge). NOT wired into the production build.
// Apache-2.0 (prototype code in this directory is original to dee; any
// KTransformers-derived regions are documented in THIRD_PARTY_KTRANSFORMERS.md
// and are NOT copied here — this header is interface-only).
#pragma once

#include <cstddef>
#include <cstdint>

namespace dee {
namespace ktbridge {

// One projection: packed E2M1 weights + E8M0 scales, row-major [out, in].
// For DeepSeek-V4-Flash: GATE=w1 [inter,hidden], UP=w3 [inter,hidden],
// DOWN=w2 [hidden,inter]. packed is I8 [out, in/2] (low nibble -> col 2i,
// high nibble -> col 2i+1); scale is F8_E8M0 [out, in/32], value 2^(bits-127).
struct PackedProjection {
    const uint8_t* packed = nullptr;  // borrowed, out*in/2 bytes
    const uint8_t* scale = nullptr;   // borrowed, out*in/32 bytes
    size_t out = 0;
    size_t in = 0;
    size_t packed_nbytes = 0;  // must equal out*in/2
    size_t scale_nbytes = 0;   // must equal out*in/32
};

// Bounded expert record supplied by dee (HostPackCache entry / mmap views).
// Borrowed for the duration of one execute() call only.
struct PackedExpertView {
    PackedProjection gate;  // w1
    PackedProjection up;    // w3
    PackedProjection down;  // w2
};

}  // namespace ktbridge
}  // namespace dee
