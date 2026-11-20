// KTransformersCpuExecutor: KT-faithful numeric path (portable, no intrinsics).
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "kt_bridge/cpu_executor.hpp"
#include "kt_bridge/kt_cpu_executor.hpp"

namespace dee {
namespace ktbridge {
namespace {

// Round fp32 -> bf16 -> fp32 (round-to-nearest-even via bias trick on the
// truncated word). Mirrors KT's bf16 wire format without intrinsics.
inline float round_via_bf16(float x) {
    if (!std::isfinite(x)) return x;
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    // Round to nearest even at bit 16: add 0x7FFF + LSB-of-kept-half.
    uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
    u += bias;
    u &= 0xFFFF0000u;
    float y = 0.0f;
    std::memcpy(&y, &u, sizeof(y));
    return y;
}

inline float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }

const float kFp4TableKt[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

inline float e8_to_f32(uint8_t e) {
    if (e == 0x00) return 5.8774717541114375e-39f;
    uint32_t bits = static_cast<uint32_t>(e) << 23;
    float s = 0.0f;
    std::memcpy(&s, &bits, sizeof(s));
    return s;
}

}  // namespace

bool KTransformersCpuExecutor::compact_scales(const uint8_t* scale_u8,
                                              size_t n, float* fp32_out,
                                              uint8_t* e8_out) {
    if (!scale_u8 || !fp32_out || !e8_out) return false;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t b = scale_u8[i];
        if (b == 0xFF) return false;
        // ue8m0 -> (u8<<7).view(bf16) -> fp32, then validate positive-pow2
        // and compact back to the exponent byte (KT finalize_scale_e8).
        const uint16_t bf16bits = static_cast<uint16_t>(b) << 7;
        // bf16 -> fp32: exponent rebias (bf16 bias 127 == fp32 bias 127,
        // mantissa 0), so value = 2^(b-127) exactly for b not in {0,255}.
        float f = e8_to_f32(b);
        // Validate: positive finite pow2 (mirrors KT sign/mant/exp check).
        uint32_t fb = 0;
        std::memcpy(&fb, &f, sizeof(fb));
        const uint32_t exp = (fb >> 23) & 0xFFu;
        const bool ok = ((fb & 0x80000000u) == 0) && ((fb & 0x007FFFFFu) == 0) &&
                        (b != 0) && (exp != 0) && (exp != 0xFFu);
        // b==0 decodes to 2^-127 (subnormal-adjacent); KT compacts the byte
        // itself (value-preserving at the byte level) — allow it through.
        if (b != 0 && !ok) return false;
        (void)bf16bits;
        fp32_out[i] = f;
        e8_out[i] = b;  // lossless: compacted form equals source byte
    }
    return true;
}

void KTransformersCpuExecutor::expand_scales(const uint8_t* e8, size_t n,
                                             float* fp32_out) {
    for (size_t i = 0; i < n; ++i) fp32_out[i] = e8_to_f32(e8[i]);
}

ExecuteError KTransformersCpuExecutor::execute(
    int /*layer*/, int /*expert_id*/, const PackedExpertView& packed,
    const float* hidden, size_t hidden_dim, float routing_weight,
    const ExecuteConfig& config, float* out, size_t out_dim,
    ExecuteStats* stats) {
    const auto t0 = std::chrono::steady_clock::now();
    ExecuteError err =
        validate_execute_args(packed, hidden, hidden_dim, routing_weight,
                              config, out, out_dim);
    if (err != ExecuteError::kOk) return err;
    const size_t inter = packed.gate.out;
    const float limit = config.swiglu_limit;

    // Build the flat KT-style blob: [packed copies][compacted e8], proving
    // the resident form is memcpy-evictable. (Per-call in Phase 1; a bounded
    // host cache would retain blob_ across calls.)
    const size_t g_pb = packed.gate.packed_nbytes, g_sb = packed.gate.scale_nbytes;
    const size_t u_pb = packed.up.packed_nbytes, u_sb = packed.up.scale_nbytes;
    const size_t d_pb = packed.down.packed_nbytes, d_sb = packed.down.scale_nbytes;
    blob_.resize(g_pb + u_pb + d_pb + g_sb + u_sb + d_sb);
    uint8_t* dst = blob_.data();
    std::memcpy(dst, packed.gate.packed, g_pb); dst += g_pb;
    std::memcpy(dst, packed.up.packed, u_pb); dst += u_pb;
    std::memcpy(dst, packed.down.packed, d_pb); dst += d_pb;
    uint8_t* e8_base = dst;
    std::vector<float> scale_f32(g_sb + u_sb + d_sb);
    if (!compact_scales(packed.gate.scale, g_sb, scale_f32.data(), e8_base))
        return ExecuteError::kScale;
    if (!compact_scales(packed.up.scale, u_sb, scale_f32.data() + g_sb,
                        e8_base + g_sb))
        return ExecuteError::kScale;
    if (!compact_scales(packed.down.scale, d_sb,
                        scale_f32.data() + g_sb + u_sb, e8_base + g_sb + u_sb))
        return ExecuteError::kScale;
    // Re-expand through the KT vector path (e<<23) for compute.
    std::vector<float> g_scale(g_sb), u_scale(u_sb), d_scale(d_sb);
    expand_scales(e8_base, g_sb, g_scale.data());
    expand_scales(e8_base + g_sb, u_sb, u_scale.data());
    expand_scales(e8_base + g_sb + u_sb, d_sb, d_scale.data());

    const uint8_t* g_packed = blob_.data();
    const uint8_t* u_packed = g_packed + g_pb;
    const uint8_t* d_packed = u_packed + u_pb;

    auto dequant = [&](const uint8_t* P, const float* S, size_t in, size_t o,
                       size_t i) -> float {
        const uint8_t byte = P[o * (in / 2) + i / 2];
        const uint8_t nib = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        return kFp4TableKt[nib] * S[o * (in / 32) + i / 32];
    };

    // KT wire: input row rounded to bf16 once.
    std::vector<float> x(hidden_dim);
    for (size_t i = 0; i < hidden_dim; ++i) x[i] = round_via_bf16(hidden[i]);

    const auto t1 = std::chrono::steady_clock::now();
    // gate/up GEMM in fp32, outputs rounded back to bf16 (KT BufferC->bf16).
    std::vector<float> gate(inter), up(inter);
    for (size_t o = 0; o < inter; ++o) {
        float g = 0.0f, u = 0.0f;
        for (size_t i = 0; i < hidden_dim; ++i) {
            g += dequant(g_packed, g_scale.data(), hidden_dim, o, i) * x[i];
            u += dequant(u_packed, u_scale.data(), hidden_dim, o, i) * x[i];
        }
        gate[o] = round_via_bf16(g);
        up[o] = round_via_bf16(u);
    }
    // SwiGLU in fp32, result rounded to bf16 for the down-projection input
    // (KT down_ba_ is bf16; natural path permutes but preserves values).
    std::vector<float> h(inter);
    for (size_t o = 0; o < inter; ++o) {
        float g = gate[o], u = up[o];
        if (limit > 0.0f) {
            if (g > limit) g = limit;
            if (u > limit) u = limit;
            if (u < -limit) u = -limit;
        }
        h[o] = round_via_bf16(silu_f32(g) * u);
    }
    // Down projection in fp32; routing weight applied AFTER w2 (KT placement),
    // then merged fp32 -> bf16 (KT merge_results) -> fp32 out.
    for (size_t o = 0; o < hidden_dim; ++o) {
        float y = 0.0f;
        for (size_t i = 0; i < inter; ++i)
            y += dequant(d_packed, d_scale.data(), inter, o, i) * h[i];
        y *= routing_weight;  // KT late weighting (algebraically = dee placement)
        out[o] = round_via_bf16(y);
    }
    const auto t2 = std::chrono::steady_clock::now();
    if (stats) {
        stats->dequant_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->gemm_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
        stats->total_ms =
            std::chrono::duration<double, std::milli>(t2 - t0).count();
    }
    return ExecuteError::kOk;
}

}  // namespace ktbridge
}  // namespace dee
