#pragma once
// dee.cpp - Step 4/5/6: expert weight source interface.
//
// Production implementation is WeightMmap (mmap'd safetensors + BF16 decode),
// the ported Step 4 data layer. A real MoE expert's weights live in TWO
// projections (gate_up_proj and down_proj), typically in two separate shards,
// so the interface is projection-aware: expert_tensors() returns one view per
// projection, each with the host (mmap) pointer and the offset where it should
// land inside the expert's VRAM slot.

#include <cstddef>
#include <string>

namespace dee {

// MoE MLP projections an expert is made of.
enum class ExpertTensor { GateUp = 0, Down = 1 };
static constexpr int kNumExpertTensors = 2;

struct ExpertTensorView {
    const void* data = nullptr;   // host (mmap) pointer to this projection's slice
    size_t nbytes = 0;            // bytes of this projection for the expert
    size_t slot_offset = 0;       // offset within the expert's VRAM slot
    ExpertTensor kind = ExpertTensor::GateUp;
};

class WeightSource {
public:
    virtual ~WeightSource() = default;

    virtual int num_layers() const = 0;
    virtual int experts_per_layer() const = 0;
    virtual int num_tensors_per_expert() const = 0;  // kNumExpertTensors

    // Map a 0-based layer position (the Generator's layer-loop index) to the
    // actual layer index in the weight files. Identity by default (synthetic /
    // single-block sources); WeightMmap overrides this because its shards carry
    // real layer numbers (e.g. model.layers.10), not 0-based positions.
    virtual int layer_index(int pos) const { return pos; }

    // Fill `views` (size = num_tensors_per_expert()) with the host pointers +
    // slot offsets for (layer, expert_id). The prefetcher copies each view into
    // the expert's VRAM slot (at view.slot_offset).
    virtual void expert_tensors(int layer, int expert_id, ExpertTensorView* views) = 0;

    // Total bytes an expert occupies in VRAM (sum of its projections).
    virtual size_t expert_total_bytes(int layer, int expert_id) const = 0;

    // Canonical tensor name for debugging / observability.
    virtual std::string tensor_name(int layer, int expert_id, ExpertTensor t) const = 0;

    // Decode BF16 expert weights (any projection) to FP32. `n` = number of
    // bf16 elements; `out` must hold `n` floats.
    virtual void decode_bf16_to_fp32(const void* bf16, float* out, size_t n) const = 0;
};

// BF16 (bfloat16) -> FP32. Standalone helper (bf16 is the upper 16 bits of FP32).
void bf16_to_fp32(const void* bf16, float* out, size_t n);

}  // namespace dee
