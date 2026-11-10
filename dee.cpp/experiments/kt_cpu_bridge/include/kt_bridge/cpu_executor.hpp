// kt_bridge/cpu_executor.hpp
//
// Model-neutral CPU expert execution interface. No KTransformers types leak
// past this boundary.
//
// Isolated prototype: dee.cpp/experiments/kt_cpu_bridge. NOT in production build.
#pragma once

#include <cstddef>

#include "kt_bridge/packed_expert_view.hpp"

namespace dee {
namespace ktbridge {

enum class ExecuteError {
    kOk = 0,
    kNull = 1,       // null pointer argument
    kShape = 2,      // dimension / byte-size mismatch
    kConfig = 3,     // unsupported config (e.g. swiglu_alpha != 0)
    kScale = 4,      // rejected scale byte (0xFF) or non-pow2 scale
    kNonFinite = 5,  // non-finite hidden input
};

struct ExecuteConfig {
    // DeepSeek-V4-Flash: 10.0. 0 disables clamping (Ornith compat).
    float swiglu_limit = 10.0f;
    // MiniMax swigluoai path. Only 0.0 supported in Phase 1.
    float swiglu_alpha = 0.0f;
};

struct ExecuteStats {
    double dequant_ms = 0.0;
    double gemm_ms = 0.0;
    double total_ms = 0.0;
};

inline const char* execute_error_name(ExecuteError e) {
    switch (e) {
        case ExecuteError::kOk: return "ok";
        case ExecuteError::kNull: return "null";
        case ExecuteError::kShape: return "shape";
        case ExecuteError::kConfig: return "config";
        case ExecuteError::kScale: return "scale";
        case ExecuteError::kNonFinite: return "nonfinite";
    }
    return "unknown";
}

// Abstract per-expert CPU executor.
//
// execute() runs ONE routed expert:
//   gate = clamp(x@W1^T); up = clamp(x@W3^T); h = silu(gate)*up;
//   h *= routing_weight; y = h@W2^T
// and writes y (length out_dim == hidden_dim) to `out`.
// Output INCLUDES the routing weight. Caller combines experts.
// Deterministic: same inputs -> bitwise identical outputs (Phase 1: no threads).
class CpuExpertExecutor {
 public:
    virtual ~CpuExpertExecutor() = default;
    virtual ExecuteError execute(int layer, int expert_id,
                                 const PackedExpertView& packed,
                                 const float* hidden, size_t hidden_dim,
                                 float routing_weight,
                                 const ExecuteConfig& config, float* out,
                                 size_t out_dim,
                                 ExecuteStats* stats = nullptr) = 0;
    virtual const char* name() const = 0;
};

// Shared validation used by all executors. Returns kOk or the failure.
ExecuteError validate_execute_args(const PackedExpertView& packed,
                                   const float* hidden, size_t hidden_dim,
                                   float routing_weight,
                                   const ExecuteConfig& config,
                                   const float* out, size_t out_dim);

// E2M1 decode table shared with dee's weight_mmap semantics:
// indices 0..7 positive, 8..15 negative, index 8 is -0.0.
const float* kt_bridge_fp4_table();

// Decode one F8_E8M0 byte to FP32: 2^(bits-127). Returns false for 0xFF
// (KT +inf vs dee clamp divergence; fail closed).
bool kt_bridge_e8m0_to_f32(uint8_t bits, float* value);

}  // namespace ktbridge
}  // namespace dee
