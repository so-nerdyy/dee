#pragma once
// dee.cpp - Step 6: MoE expert MLP forward (computational core).
//
// Per-expert SwiGLU MLP for a single token. Weights are FP32 (already BF16-
// decoded). Row-major layout:
//   gate_up : [2*inter, hidden]   (top `inter` rows = gate, next `inter` = up)
//   down    : [hidden,   inter]
// Caller provides scratch `gate`, `up`, `act` (each `inter` floats) and `out`
// (`hidden` floats).
//
// This is the computational node the MoE layer graph is built from. When ggml is
// vendored, these matvecs become GGML_OP_MUL_MAT nodes; the structure (gate/up
// -> SiLU*gate -> down) is the graph.

#include <cstddef>

namespace dee {

void expert_mlp(const float* x, int hidden, int inter, const float* gate_up,
                const float* down, float* out, float* gate, float* up, float* act);

}  // namespace dee
