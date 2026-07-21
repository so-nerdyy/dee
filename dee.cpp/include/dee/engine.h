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
// CPU mode bundles gate|up|down into an F32 host blob. CUDA keeps a BF16 host
// blob through pinned staging and H2D, then converts it into the configured
// FP16 or FP32 cache block on the prefetch stream. The forward pass slices the
// projections from that block. SwiGLU is plain C++
// (row-major matmuls); the CUDA/ggml path slots in later behind the same API.
//
// On a box without a GPU (this WSL build, DEE_CUDA=OFF) the cache + prefetcher
// run on the host-mock backend: VRAM is a malloc'd arena, copies are synchronous
// on the mock stream. This verifies the END-TO-END control + math logic; the
// throughput/VRAM numbers are mock but the code path is identical to CUDA.

#pragma once

#include "dee/async_prefetcher.h"
#include "dee/oracle.h"
#include "dee/profiling.h"
#include "dee/vram_cache.h"
#include "dee/weight_mmap.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef DEE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace dee {

enum class BenchmarkScenario {
    EndToEnd,
    FullResident,
    ResidentBypass,
    TransferOnly,
    ComputeOnly,
    OracleOnly,
    CacheMetadataOnly
};

const char* benchmark_scenario_name(BenchmarkScenario scenario);

enum class DeviceCacheDType {
    Fp32,
    Fp16
};

const char* device_cache_dtype_name(DeviceCacheDType dtype);

enum class WeightTransferDType {
    Bf16,
    Int8,
    Int4
};

const char* weight_transfer_dtype_name(WeightTransferDType dtype);

struct EngineConfig {
    std::string shard_path;    // safetensors MoE shard (mapped by WeightMmap)
    std::string oracle_path;   // PyTorch .pt Oracle (read by PtLoader)
    int         num_tokens  = 32;   // autoregressive steps to run
    int         topk        = 8;    // experts activated per layer (top-K)
    int         num_layers  = 40;   // depth (clamped to oracle.num_layers)
    size_t      budget_bytes = 0;   // VRAM budget (0 => 4 experts auto)
    size_t      prefetch_depth = 64;// bounded pinned/device staging ring
    DeviceCacheDType cache_dtype = DeviceCacheDType::Fp32;
    WeightTransferDType transfer_dtype = WeightTransferDType::Bf16;
    bool        use_cuda    = false;// DEE_CUDA path (only if built WITH cuda)
    int         hidden      = 2048; // model hidden dim (checked vs shard)
    int         inter       = 256;  // expert intermediate dim (checked vs shard)
    bool        verbose     = false;
    bool        profile_stages = false;
    bool        trace_requests = false;
    bool        profile_timeline = false;
    BenchmarkScenario scenario = BenchmarkScenario::EndToEnd;
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
    uint64_t resident_hits = 0;
    uint64_t inflight_hits = 0;
    uint64_t cold_loads = 0;
    uint64_t duplicate_requests = 0;
    bool   hidden_finite = true;   // output hidden all-finite at the end
    std::vector<float> final_hidden; // final normalized hidden for validation
    size_t cuda_total    = 0;       // GPU memory total (cudaMemGetInfo), 0 if N/A
    size_t cuda_free     = 0;       // GPU memory free  (cudaMemGetInfo), 0 if N/A
    std::string cuda_device_name;
    int cuda_compute_major = 0;
    int cuda_compute_minor = 0;
    int cuda_runtime_version = 0;
    StageProfile profile{};
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
    StageProfiler profiler_;

    int    hidden_ = 2048;
    int    inter_  = 256;
    size_t blob_elems_ = 0;   // floats per expert blob (gate|up|down)
    size_t blob_bytes_ = 0;   // FP32 bytes per expert blob (budget baseline)
    size_t cache_blob_bytes_ = 0; // bytes per expert in selected device format

#ifdef DEE_CUDA
    // GPU working set (allocated in init when use_cuda)
    cudaStream_t compute_stream_ = nullptr;
    float* d_h_in_   = nullptr;   // [hidden]
    float* d_h_out_  = nullptr;   // [hidden]
    float* d_hbuf_   = nullptr;   // [inter]  scratch for swiglu phase 1
    float* d_ubuf_   = nullptr;   // [inter]  up-projection scratch
    float* d_ybuf_   = nullptr;   // [topk * hidden] per-expert outputs
    void* d_h_in_half_ = nullptr; // [hidden] FP16 input for FP16 cache mode
    void* d_activation_half_ = nullptr; // [inter] FP16 SwiGLU activation
    cublasHandle_t cublas_handle_ = nullptr;
    size_t cuda_total_ = 0, cuda_free_ = 0;  // from cudaMemGetInfo
    void cuda_cleanup();
    bool forward_layer_cuda(int layer, const float* h_in, float* h_out);
#endif

    // host staging: resolved shard (layer, expert) -> F32 blob
    // [gate|up|down].  Synthetic single-layer shards intentionally map every
    // model layer to source layer 0; real multi-layer shards stay distinct.
    std::unordered_map<uint64_t, std::vector<float>> staging_;
    std::unordered_map<uint64_t, std::vector<uint16_t>> staging_bf16_;
    std::unordered_map<uint64_t, void*> pinned_staging_bf16_;
    struct QuantizedExpert {
        std::vector<int8_t> host;
        void* pinned = nullptr;
        float scales[3] = {1.0f, 1.0f, 1.0f};
    };
    std::unordered_map<uint64_t, QuantizedExpert> staging_int8_;
    size_t pinned_staging_bytes_ = 0;
    static constexpr size_t kPinnedStagingLimit = 192ULL * 1024 * 1024;

    std::vector<float> hidden_buf_[2];  // double buffer for the loop

    EngineStats stats_{};
    int current_token_ = -1;
    uint64_t scenario_requests_ = 0;
    uint64_t scenario_resident_hits_ = 0;
    uint64_t scenario_cold_loads_ = 0;

    // map a model layer to the shard layer that actually exists (synthetic
    // single-layer shards expose only layer 0).
    int avail_layer(int layer) const;

    // Ensure the F32 staging blob for a resolved shard expert exists (fill
    // from mmap/upcast).
    const float* get_staging(int source_layer, int expert);
    const uint16_t* get_staging_bf16(int source_layer, int expert);
    const QuantizedExpert* get_staging_int8(int source_layer, int expert);
    const QuantizedExpert* get_staging_int4(int source_layer, int expert);

    // Stream `expert` from a resolved shard layer into VRAM.  The cache key
    // must describe the source weights, not merely the logical model layer.
    bool stage_expert(int logical_layer, int source_layer, int expert, int priority);
    bool prepare_profile_scenario();
    bool preload_all_experts();
};

} // namespace dee
