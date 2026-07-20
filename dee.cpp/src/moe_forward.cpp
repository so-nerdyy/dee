// dee.cpp - Step 6: MoE expert MLP forward (computational core).

#include "dee/moe_forward.h"

#include <cmath>

namespace dee {

static inline float silu(float z) { return z / (1.0f + std::exp(-z)); }

// y = W * x   (W is [rows, cols] row-major)
static inline void matvec(const float* W, const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float s = 0.f;
        const float* wr = W + static_cast<size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) s += wr[c] * x[c];
        y[r] = s;
    }
}

void expert_mlp(const float* x, int hidden, int inter, const float* gate_up,
                const float* down, float* out, float* gate, float* up, float* act) {
    matvec(gate_up, x, gate, inter, hidden);                     // gate = W_gate * x
    matvec(gate_up + static_cast<size_t>(inter) * hidden, x, up, inter, hidden);  // up = W_up * x
    for (int i = 0; i < inter; ++i) act[i] = silu(gate[i]) * up[i];  // act = SiLU(gate) * up
    matvec(down, act, out, hidden, inter);                       // out = W_down * act
}

}  // namespace dee
