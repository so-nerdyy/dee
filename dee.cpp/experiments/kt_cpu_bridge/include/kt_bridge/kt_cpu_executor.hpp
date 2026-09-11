// kt_bridge/kt_cpu_executor.hpp
//
// KT-faithful executor: same codec, but emulates KTransformers' numeric path
// (BF16 round-trips at gate/up/down boundaries, late routing-weight placement)
// so Phase D can quantify the BF16-boundary delta without AVX512 hardware.
//
// Future AMX port replaces the portable inner dots with
// mxfp4_to_bf16_32[_natural] + _mm512_dpbf16_ps behind this same interface.
#pragma once

#include <cstdint>
#include <vector>

#include "kt_bridge/cpu_executor.hpp"

namespace dee {
namespace ktbridge {

class KTransformersCpuExecutor : public CpuExpertExecutor {
 public:
    ExecuteError execute(int layer, int expert_id,
                         const PackedExpertView& packed, const float* hidden,
                         size_t hidden_dim, float routing_weight,
                         const ExecuteConfig& config, float* out,
                         size_t out_dim, ExecuteStats* stats = nullptr) override;
    const char* name() const override { return "KTransformersCpuExecutor"; }

    // Scale pipeline mirroring KT BufferB: ue8m0 -> bf16 -> fp32 -> compact e8.
    // Exposed for unit tests + host-cache (cache the compacted bytes to skip
    // torch conversion on hit). Returns false on rejected byte (0xFF).
    static bool compact_scales(const uint8_t* scale_u8, size_t n, float* fp32_out,
                               uint8_t* e8_out);
    // Expand compacted e8 back to fp32 via (e << 23), mirroring expand_e8_scales.
    static void expand_scales(const uint8_t* e8, size_t n, float* fp32_out);

 private:
    // Flat per-call blob: [packed weights copy][compacted e8], proving the
    // KT-resident form is memcpy-evictable and bounded (no pointers).
    std::vector<uint8_t> blob_;
};

}  // namespace ktbridge
}  // namespace dee
