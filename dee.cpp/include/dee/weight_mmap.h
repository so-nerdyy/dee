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
enum class DType { F32, F16, BF16, UNKNOWN };

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
// ---------------------------------------------------------------------------
class TensorResolver {
public:
    // Expert weight kind.
    enum Kind { GATE_PROJ, UP_PROJ, DOWN_PROJ };

    // The model is loaded one shard at a time. Call register_shard() for each
    // WeightMmap you open; the resolver routes lookups to the right shard by
    // tensor name. (For the single-shard smoke test, register just one.)
    void register_shard(WeightMmap* mmap);

    // Resolve the fused gate_up projection for an expert.
    // Returns the gate_proj view (caller may also fetch up_proj separately).
    TensorView resolve_expert(int layer, int expert, Kind kind) const;

    // Build the canonical tensor name for the Ornith/Qwen3.5-MoE architecture.
    static std::string expert_tensor_name(int layer, int expert, Kind kind);

private:
    std::vector<WeightMmap*> shards_;
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
