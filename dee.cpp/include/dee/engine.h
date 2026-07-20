// dee/engine.h
//
// Step 8 — MoE forward pass + autoregressive generation driver (DEE runtime).
//
// Wires the data layer (WeightMmap + TensorResolver), the Oracle
// (OracleScheduler), the streaming layer (AsyncPrefetcher) and the expert
// cache (VramCacheManager) into a single inference engine that runs the
// Dynamic Expert Eviction loop:
//
//   for each token:
//     hidden = embed(token)            // mock: deterministic init
//     for each layer L in 0..L-1:
//       experts = oracle.predict(L, hidden, topk)   // which experts activate
//       for each expert e: prefetcher.prefetch(L, e) // stream into VRAM
//       for each expert e: prefetcher.wait(L, e)     // sync fallback if needed
//       hidden = moe_swiglu(L, hidden, experts)      // SwiGLU combine
//     emit(hidden)
//
// The expert weights are bundled (gate|up|down) into one F32 staging blob per
// shard-expert and streamed into the VRAM arena as a single block. The forward
// pass slices the three projections out of that block. SwiGLU is plain C++
// (row-major matmuls); the CUDA/ggml path slots in later behind the same API.
//
// On a box without a GPU (this WSL build, DEE_CUDA=OFF) the cache + prefetcher
// run on the host-mock backend: VRAM is a malloc'd arena, copies are synchronous
// on the mock stream. This verifies the END-TO-END control + math logic; the
// throughput/VRAM numbers are mock but the code path is identical to CUDA.

#pragma once

#include "dee/async_prefetcher.h"
#include "dee/oracle.h"
#include "dee/vram_cache.h"
#include "dee/weight_mmap.h"

#include <string>
#include <vector>

namespace dee {

struct EngineConfig {
    std::string shard_path;    // safetensors MoE shard (mapped by WeightMmap)
    std::string oracle_path;   // PyTorch .pt Oracle (read by PtLoader)
    int         num_tokens  = 32;   // autoregressive steps to run
    int         topk        = 8;    // experts activated per layer (top-K)
    int         num_layers  = 40;   // depth (clamped to oracle.num_layers)
    size_t      budget_bytes = 0;   // VRAM budget (0 => 4 experts auto)
    bool        use_cuda    = false;// DEE_CUDA path (only if built WITH cuda)
    int         hidden      = 2048; // model hidden dim (checked vs shard)
    int         inter       = 256;  // expert intermediate dim (checked vs shard)
    bool        verbose     = false;
};

struct EngineStats {
    int    tokens        = 0;
    double elapsed_sec   = 0.0;
    double tok_per_sec   = 0.0;
    size_t peak_vram     = 0;       // bytes (mock arena high-water mark)
    uint64_t cache_hits  = 0;
    uint64_t cache_loads = 0;
    uint64_t evictions   = 0;
    uint64_t fallbacks   = 0;       // sync_fallback stalls
    uint64_t prefetch_issued = 0;
    uint64_t prefetch_fallbacks = 0;
    bool   hidden_finite = true;   // output hidden all-finite at the end
    size_t cuda_total    = 0;       // GPU memory total (cudaMemGetInfo), 0 if N/A
    size_t cuda_free     = 0;       // GPU memory free  (cudaMemGetInfo), 0 if N/A
};

class Engine {
public:
    Engine() : prefetcher_(cache_) {}
    bool init(const EngineConfig& cfg);
    const EngineConfig& config() const { return cfg_; }

    // Run the autoregressive generation loop and fill `stats_`.
    bool generate();

    const EngineStats& stats() const { return stats_; }

    // Expose for tests: run a single layer's MoE on caller-provided hidden,
    // write the new hidden to `h_out` (length hidden_).
    void forward_layer(int layer, const float* h_in, float* h_out);

    // Expose for tests: the naive (no DEE) SwiGLU kernel on a pre-staged blob.
    static void swiglu(const float* blob, const float* x,
                       int inter, int hidden, float* acc);

    int hidden_dim() const { return hidden_; }
    int inter_dim()  const { return inter_; }

    ~Engine();

private:
    EngineConfig cfg_;

    WeightMmap     mmap_;
    TensorResolver resolver_;
    OracleScheduler oracle_;
    VramCacheManager cache_;
    AsyncPrefetcher prefetcher_;

    int    hidden_ = 2048;
    int    inter_  = 256;
    size_t blob_elems_ = 0;   // floats per expert blob (gate|up|down)
    size_t blob_bytes_ = 0;   // bytes per expert blob

#ifdef DEE_CUDA
    // GPU working set (allocated in init when use_cuda)
    cudaStream_t compute_stream_ = nullptr;
    float* d_h_in_   = nullptr;   // [hidden]
    float* d_h_out_  = nullptr;   // [hidden]
    float* d_hbuf_   = nullptr;   // [inter]  scratch for swiglu phase 1
    float* d_ybuf_   = nullptr;   // [topk * hidden] per-expert outputs
    size_t cuda_total_ = 0, cuda_free_ = 0;  // from cudaMemGetInfo
    void cuda_cleanup();
    void forward_layer_cuda(int layer, const float* h_in, float* h_out);
#endif

    // host staging: shard-expert -> F32 blob [gate|up|down]
    // keyed by (shard) expert only; all model layers map onto the same shard
    // layer in a synthetic single-layer shard (see avail_layer()).
    std::unordered_map<int, std::vector<float>> staging_;

    std::vector<float> hidden_buf_[2];  // double buffer for the loop

    EngineStats stats_{};

    // map a model layer to the shard layer that actually exists (synthetic
    // single-layer shards expose only layer 0).
    int avail_layer(int layer) const;

    // ensure the F32 staging blob for `expert` exists (fill from mmap/upcast).
    const float* get_staging(int expert);

    // stream `expert` (shard-expert, model-layer `layer`) into VRAM.
    void stage_expert(int layer, int expert, int priority);
};

} // namespace dee
