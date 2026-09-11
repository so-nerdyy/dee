// kt_bridge CPU executor common + ReferenceCpuExecutor (portable FP32).
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "kt_bridge/cpu_executor.hpp"
#include "kt_bridge/reference_cpu_executor.hpp"

namespace dee {
namespace ktbridge {
namespace {

const float kFp4Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

bool valid_projection(const PackedProjection& p) {
    if (p.packed == nullptr || p.scale == nullptr) return false;
    if (p.out == 0 || p.in == 0) return false;
    if (p.in % 2 != 0 || p.in % 32 != 0) return false;
    if (p.packed_nbytes != p.out * p.in / 2) return false;
    if (p.scale_nbytes != p.out * p.in / 32) return false;
    return true;
}

inline float dequant_weight(const PackedProjection& p, size_t o, size_t i) {
    const size_t pi = o * (p.in / 2) + i / 2;
    const uint8_t byte = p.packed[pi];
    const uint8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    const float v = kFp4Table[nibble];
    const size_t si = o * (p.in / 32) + i / 32;
    const uint8_t sb = p.scale[si];
    float s = 0.0f;
    // 0xFF fails closed at validation; here decode by the 2^(e-127) law.
    if (sb == 0xFF) return 0.0f;  // unreachable after validation
    if (sb == 0x00) {
        // 2^-127: below BF16 range; reference keeps the fp32 value.
        s = 5.8774717541114375e-39f;  // 2^-127
    } else {
        // Exponent field = sb, mantissa 0, sign 0.
        uint32_t bits = static_cast<uint32_t>(sb) << 23;
        std::memcpy(&s, &bits, sizeof(s));
    }
    return v * s;
}

inline float silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

}  // namespace

const float* kt_bridge_fp4_table() { return kFp4Table; }

bool kt_bridge_e8m0_to_f32(uint8_t bits, float* value) {
    if (value == nullptr) return false;
    if (bits == 0xFF) return false;  // +inf divergence: fail closed
    if (bits == 0x00) {
        *value = 5.8774717541114375e-39f;  // 2^-127
        return true;
    }
    uint32_t u = static_cast<uint32_t>(bits) << 23;
    std::memcpy(value, &u, sizeof(*value));
    return true;
}

ExecuteError validate_execute_args(const PackedExpertView& packed,
                                   const float* hidden, size_t hidden_dim,
                                   float routing_weight,
                                   const ExecuteConfig& config,
                                   const float* out, size_t out_dim) {
    if (hidden == nullptr || out == nullptr) return ExecuteError::kNull;
    if (hidden_dim == 0 || out_dim == 0) return ExecuteError::kShape;
    if (!valid_projection(packed.gate) || !valid_projection(packed.up) ||
        !valid_projection(packed.down))
        return ExecuteError::kShape;
    // gate/up: [inter, hidden]; down: [hidden, inter]; inter must agree.
    const size_t inter = packed.gate.out;
    if (packed.up.out != inter) return ExecuteError::kShape;
    if (packed.gate.in != hidden_dim || packed.up.in != hidden_dim)
        return ExecuteError::kShape;
    if (packed.down.out != hidden_dim || packed.down.in != inter)
        return ExecuteError::kShape;
    if (out_dim != hidden_dim) return ExecuteError::kShape;
    if (config.swiglu_alpha != 0.0f) return ExecuteError::kConfig;
    if (!std::isfinite(routing_weight)) return ExecuteError::kNonFinite;
    for (size_t i = 0; i < hidden_dim; ++i) {
        if (!std::isfinite(hidden[i])) return ExecuteError::kNonFinite;
    }
    // Reject 0xFF scale bytes up front (KT +inf vs dee clamp divergence).
    const PackedProjection* projs[3] = {&packed.gate, &packed.up, &packed.down};
    for (const PackedProjection* p : projs) {
        for (size_t i = 0; i < p->scale_nbytes; ++i) {
            if (p->scale[i] == 0xFF) return ExecuteError::kScale;
        }
    }
    return ExecuteError::kOk;
}

ExecuteError ReferenceCpuExecutor::execute(
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

    // gate = x @ W1^T ; up = x @ W3^T  (fp32, row-major)
    // Use caller-stack-friendly heap via static thread_local scratch growth.
    // Phase 1 favors clarity over arena reuse (bounded: inter <= 8192 typical).
    std::vector<float> gate(inter), up(inter), h(inter);
    const auto t1 = std::chrono::steady_clock::now();
    for (size_t o = 0; o < inter; ++o) {
        float g = 0.0f, u = 0.0f;
        for (size_t i = 0; i < hidden_dim; ++i) {
            g += dequant_weight(packed.gate, o, i) * hidden[i];
            u += dequant_weight(packed.up, o, i) * hidden[i];
        }
        if (limit > 0.0f) {
            if (g > limit) g = limit;  // gate: max-only (official DeepSeek)
            if (u > limit) u = limit;
            if (u < -limit) u = -limit;
        }
        gate[o] = g;
        up[o] = u;
        h[o] = silu_f32(g) * u * routing_weight;  // dee placement: before w2
    }
    for (size_t o = 0; o < hidden_dim; ++o) {
        float y = 0.0f;
        for (size_t i = 0; i < inter; ++i) {
            y += dequant_weight(packed.down, o, i) * h[i];
        }
        out[o] = y;
    }
    const auto t2 = std::chrono::steady_clock::now();
    if (stats) {
        stats->dequant_ms = 0.0;  // fused dequant+gemm in reference; split in KT port
        stats->gemm_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
        stats->total_ms =
            std::chrono::duration<double, std::milli>(t2 - t0).count();
    }
    return ExecuteError::kOk;
}

}  // namespace ktbridge
}  // namespace dee
