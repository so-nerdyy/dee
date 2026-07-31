// dee/weight_mmap.h
//
// Data layer for Dynamic Expert Eviction (Step 4).
//
//  - WeightMmap:    memory-maps a safetensors shard (read-only, POSIX mmap +
//                   posix_madvise(POSIX_MADV_RANDOM), mirroring llama.cpp's
//                   llama-mmap.cpp access pattern for random weight slicing).
//  - TensorResolver: maps (layer, expert, kind) -> a raw pointer + byte range
//                   inside the mmap'd region, porting the Python prototype's
//                   _resolve_key / get_fused_expert logic to C++.
//
// No ggml / CUDA dependency here: this layer only touches POSIX + the
// safetensors on-disk format, so it is unit-testable on any machine.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dee {

// safetensors scalar dtypes we care about for the MoE weights.
enum class DType { F32, F16, BF16, F8, I8, I64, UNKNOWN };

// A resolved view into the mmap'd region. `data` points directly at the
// tensor's first element (no copy); `nbytes` is the tensor byte length.
struct TensorView {
    const uint8_t* data   = nullptr;  // absolute pointer into the mmap
    size_t         nbytes = 0;        // total bytes
    DType          dtype  = DType::UNKNOWN;
    std::vector<int64_t> shape;
    bool ok() const { return data != nullptr && nbytes > 0; }
};

// Per-tensor entry parsed from a safetensors file header.
struct TensorMeta {
    size_t data_offset = 0;  // offset *within the data section* (after 8+header)
    size_t nbytes      = 0;
    DType  dtype       = DType::UNKNOWN;
    std::vector<int64_t> shape;
};

// ---------------------------------------------------------------------------
// WeightMmap: maps ONE safetensors shard file.
// ---------------------------------------------------------------------------
class WeightMmap {
public:
    WeightMmap();
    ~WeightMmap();

    // Open + mmap `path`. Returns false on failure (logged to stderr).
    bool open(const std::string& path);

    // Point this mapper at an already-parsed header (used when the global
    // index.json told us which shard a tensor lives in).
    bool open(const std::string& path, const std::string& json_header);

    void close();

    bool is_open() const { return fd_ >= 0; }
    size_t file_size() const { return size_; }

    // Base address of the mmap'd region (valid only while open).
    const uint8_t* base() const { return base_; }

    // Parse the safetensors header (called automatically by open()).
    bool parse_header();

    // Direct lookup by exact tensor name. Returns a view, or !ok() if missing.
    TensorView lookup(const std::string& tensor_name) const;

    // Raw header map (tensor name -> meta). Exposed for debugging/tests.
    const std::unordered_map<std::string, TensorMeta>& tensors() const { return tensors_; }

private:
    int            fd_   = -1;
    void*          mapping_handle_ = nullptr;  // Windows file mapping handle
    uint8_t*       base_ = nullptr;
    size_t         size_ = 0;
    std::string    header_json_;
    std::unordered_map<std::string, TensorMeta> tensors_;

    bool map_file(const std::string& path);
    bool parse_header_json(const std::string& json);
};

// ---------------------------------------------------------------------------
// TensorResolver: the C++ port of the Python prototype's _resolve_key +
// get_fused_expert. Knows the Ornith/Qwen3.5-MoE tensor naming and can
// resolve a single expert's gate/up/down projection by (layer, expert).
//
// Also supports the deepseek-ai/DeepSeek-V4-Flash-0731 layout when set_model()
// selects DEEPSEEK_V4:
//   - routed experts:  layers.<L>.ffn.experts.<E>.w1|w2|w3.weight  (packed FP4,
//     dtype I8, two e2m1fn values per byte) + matching .scale (F8_E8M0)
//   - shared experts:  layers.<L>.ffn.shared_experts.w1|w2|w3.weight
//   - hash/index:      layers.<L>.hc_* , layers.<L>.attn.indexer.*
//   - DSpark:          mtp.<I>.*  (layers 40-42)
// The Kind mapping for DEEPSEEK_V4 is GATE_PROJ->w1, UP_PROJ->w3, DOWN_PROJ->w2.
// The default remains ORNITH so existing engine.cpp callers are unchanged.
// ---------------------------------------------------------------------------
class TensorResolver {
public:
    // Expert weight kind. For DEEPSEEK_V4: GATE_PROJ=w1, UP_PROJ=w3, DOWN_PROJ=w2.
    enum Kind { GATE_PROJ, UP_PROJ, DOWN_PROJ };

    // Model naming dialect the resolver should use.
    enum class Model { ORNITH, DEEPSEEK_V4 };

    // The model is loaded one shard at a time. Call register_shard() for each
    // WeightMmap you open; the resolver routes lookups to the right shard by
    // tensor name. (For the single-shard smoke test, register just one.)
    void register_shard(WeightMmap* mmap);

    // Select the checkpoint naming dialect. Defaults to ORNITH.
    void set_model(Model model) { model_ = model; }
    Model model() const { return model_; }

    // Resolve one expert weight tensor by Kind. ORNITH: returns the
    // gate/up/down_proj view (gate and up are separate tensors).
    // DEEPSEEK_V4: GATE_PROJ->w1, UP_PROJ->w3, DOWN_PROJ->w2.
    TensorView resolve_expert(int layer, int expert, Kind kind) const;

    // Resolve the quantization scale tensor paired with an expert weight
    // (DEEPSEEK_V4 only; returns !ok() for ORNITH, which has no scales).
    TensorView resolve_expert_scale(int layer, int expert, Kind kind) const;

    // Resolve any exact checkpoint tensor name across the registered shards.
    TensorView resolve_tensor(const std::string& tensor_name) const;

    // Build the canonical tensor name for the Ornith/Qwen3.5-MoE architecture.
    static std::string expert_tensor_name(int layer, int expert, Kind kind);

    // DEEPSEEK_V4 naming helpers (independent of the instance model setting,
    // so callers can build names without a resolver).
    static std::string v4_expert_tensor_name(int layer, int expert, Kind kind);
    static std::string v4_expert_scale_name(int layer, int expert, Kind kind);
    static std::string v4_shared_expert_tensor_name(int layer, Kind kind);

private:
    std::vector<WeightMmap*> shards_;
    Model model_ = Model::ORNITH;
};

// ---------------------------------------------------------------------------
// dtype + conversion helpers
// ---------------------------------------------------------------------------
DType dtype_from_string(const std::string& s);
const char* dtype_to_string(DType d);

// BF16 -> float32 (BF16 is the top 16 bits of a float32).
float bf16_to_f32(uint16_t h);
// F16 -> float32 (IEEE half).
float f16_to_f32(uint16_t h);

} // namespace dee
