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
#include <memory>
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
    std::vector<std::string> shard_paths; // all shards needed by this layer
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
    int         num_experts = 0;    // total routed experts per layer (real-model mode); 0=fallback
    int         base_layer  = 0;    // real layer represented by a layer-local engine
    int         device_id   = 0;    // CUDA device that owns this engine/cache
    bool        verbose     = false;
    bool        profile_stages = false;
    bool        trace_requests = false;
    bool        profile_timeline = false;
    bool        debug_validate_cache = false;
    bool        prepack_quantized_source = true;
    BenchmarkScenario scenario = BenchmarkScenario::EndToEnd;
    float       oracle_strict_margin = 0.0f; // <=0=raw GPU; >0=boundary fallback
};

// Copy of OracleScheduler::BoundaryStats so EngineStats can hold it without
// pulling in a CUDA-only header dependency on engine consumers.
struct OracleBoundaryStats {
    size_t gpu_calls = 0;
    size_t cpu_fallback_calls = 0;
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
    uint64_t h2d_bytes = 0;
    uint64_t h2d_copies = 0;
    size_t current_vram = 0;
    size_t resident_experts = 0;
    // Measurement-only ownership ledger.  These are live allocation sizes,
    // not RSS/NVML estimates, and deliberately separate reserved cache arenas
    // from the resident payload tracked by current_vram.
    size_t host_pinned_expert_staging_bytes = 0;
    size_t host_pageable_expert_staging_bytes = 0;
    size_t host_router_weight_bytes = 0;
    size_t host_hidden_buffer_bytes = 0;
    size_t host_moe_dispatch_bytes = 0;
    size_t host_prefetch_ring_bytes = 0;
    size_t host_prefetch_ring_slots = 0;
    size_t peak_transient_host_bytes = 0;
    size_t device_expert_cache_reserved_bytes = 0;
    size_t device_prefetch_staging_bytes = 0;
    size_t device_fixed_work_buffer_bytes = 0;
    size_t device_router_weight_bytes = 0;
    size_t device_router_dynamic_bytes = 0;
    size_t device_moe_batch_buffer_bytes = 0;
    size_t device_moe_raw_workspace_bytes = 0;
    size_t device_oracle_scratch_bytes = 0;
    bool   hidden_finite = true;   // output hidden all-finite at the end
    std::vector<float> final_hidden; // final normalized hidden for validation
    size_t cuda_total    = 0;       // GPU memory total (cudaMemGetInfo), 0 if N/A
    size_t cuda_free     = 0;       // GPU memory free  (cudaMemGetInfo), 0 if N/A
    std::string cuda_device_name;
    int cuda_compute_major = 0;
    int cuda_compute_minor = 0;
    int cuda_runtime_version = 0;
    double quantized_prepack_ms = 0.0;
    uint64_t quantized_prepack_experts = 0;
    size_t quantized_prepack_bytes = 0;
    bool quantized_prepack_complete = false;
    OracleBoundaryStats oracle_boundary{};
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
    EngineStats runtime_stats() const;
    bool reset_runtime_cache();
    bool validate_cache_invariants(std::string* error = nullptr) const;

    // Measurement-only controls for the Python-owned full-model path.  They
    // reset counters/timing without evicting resident experts, attach the
    // external decode-step index to cache/transfer records, and expose the
    // existing StageProfiler data after the caller's measured interval.
    bool reset_external_profile();
    void set_external_token(int token) { current_token_ = token; }
    std::string external_profile_json(double total_wall_ms);
    std::string external_timeline_json(double total_wall_ms);

    // Expose for tests: run a single layer's MoE on caller-provided hidden,
    // write the new hidden to `h_out` (length hidden_).
    void forward_layer(int layer, const float* h_in, float* h_out);

    // Expose for tests: the naive (no DEE) SwiGLU kernel on a pre-staged blob.
    static void swiglu(const float* blob, const float* x,
                       int inter, int hidden, float* acc);

    // Real-model integration: caller provides hidden + per-expert chosen
    // indices (from the model's own router). Engine runs SwiGLU for each
    // requested expert and writes per-expert outputs to `experts_out`
    // (length = experts.size() * hidden_, contiguous, expert k in slot
    // [k*hidden_, (k+1)*hidden_)). Caller combines externally
    // (typically: sum(expert_out[k] * routing_weight[k])).
    // Does NOT use the dee.cpp Oracle (python HF model owns the router).
    // Optional MoE combine is intentionally left out: HF reference is the
    // arbiter for the per-token agreement check.
    bool moe_forward_experts(int layer, const float* h_in, float* experts_out,
                             const std::vector<int>& experts);
    bool moe_forward_batch(int layer, const float* h_in, int tokens,
                           const int* expert_ids, int topk, float* experts_out);

    // Milestone 3 fix (defect #6): device-resident MoE forward path.
    // Accepts FP16 device hidden (d_h_in) + host expert IDs (h_expert_ids,
    // small, host-side for grouping only) and writes FP32 per-(token,position)
    // expert outputs to d_experts_out (both device-resident).  Eliminates the
    // measured Python d2h→call→h2d round-trips (router_hidden_gpu_to_cpu,
    // expert_inputs_gpu_to_cpu, expert_outputs_cpu_to_gpu) by keeping the
    // per-layer hidden and MoE outputs on-device throughout.  Only expert_ids
    // cross the host boundary (tokens * topk * sizeof(int) ≈ 32 bytes).
    // Caller casts raw expert outputs to the surrounding dtype and handles
    // the weighted combine on-device in Python (tensor operations, not host
    // orchestration). Requires FP16 device cache (DEE_CUDA).
    bool moe_forward_batch_device(int layer, const void* d_h_in, int tokens,
                                   const int* h_expert_ids, int topk,
                                   void* d_experts_out);

    // M5C: fused device API. Expert IDs and weights remain PyTorch device
    // tensors; the engine performs the small ID dispatch D2H internally,
    // computes raw expert rows into persistent workspace, applies the exact
    // stable FP16 weighted combine, and hands completion back to the caller's
    // CUDA stream. d_raw_trace_out may be null outside parity/profiler runs;
    // a null external_stream denotes CUDA's valid default stream.
    bool moe_forward_combined_device(
        int layer, const void* d_h_in, int tokens,
        const int64_t* d_expert_ids, int topk, const float* d_weights_f32,
        void* d_output_f16, void* d_raw_trace_out,
        void* external_stream);

    // Genuine checkpoint router: logits = W_router * hidden using the active
    // runtime dtype (FP16 CUDA for Ornith), then FP32 softmax, ordered top-K,
    // and top-K probability renormalization.
    bool route_topk(int layer, const float* h_in, float* router_logits,
                    float* routing_weights, int* experts);
    bool route_topk_batch(int layer, const float* h_in, int tokens,
                          float* router_logits, float* routing_weights,
                          int* experts);

    int hidden_dim() const { return hidden_; }
    int inter_dim()  const { return inter_; }
    uintptr_t compute_stream_handle() const;

    ~Engine();

    // Milestone 3 forensic: capture the most recent native error context so
    // the Python binding surfaces precise file/line/state instead of
    // collapsing it to a generic RuntimeError.  Set on every early-return
    // site of moe_forward_experts / moe_forward_batch /
    // moe_forward_batch_device / stage_expert / the cache ensure path.
    // Cleared at entry of each forward call so the value reflects only the
    // most recent attempt.  Used by pydee bindings to disambiguate native
    // root causes in the analyzer.
    void set_last_error(const std::string& m) { last_error_message_ = m; }
    void clear_last_error() { last_error_message_.clear(); }
    const std::string& last_error_message() const { return last_error_message_; }

private:
    EngineConfig cfg_;
    std::string last_error_message_;

    WeightMmap     mmap_;
    std::vector<std::unique_ptr<WeightMmap>> extra_mmaps_;
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
    void* d_router_weight_half_ = nullptr; // [num_experts, hidden]
    float* d_router_input_ = nullptr;       // [router_capacity_tokens, hidden]
    void* d_router_input_half_ = nullptr;   // [router_capacity_tokens, hidden]
    void* d_router_logits_half_ = nullptr;  // [router_capacity_tokens, num_experts]
    size_t router_capacity_tokens_ = 0;
    int router_weight_layer_ = -1;
    float* d_moe_batch_input_ = nullptr;       // [moe_batch_capacity_tokens, hidden]
    void* d_moe_batch_input_half_ = nullptr;   // [moe_batch_capacity_tokens, hidden]
    void* d_moe_batch_gate_half_ = nullptr;    // [moe_batch_capacity_tokens, inter]
    void* d_moe_batch_up_half_ = nullptr;      // [moe_batch_capacity_tokens, inter]
    void* d_moe_batch_activation_half_ = nullptr; // [moe_batch_capacity_tokens, inter]
    float* d_moe_batch_output_ = nullptr;      // [moe_batch_capacity_tokens, hidden]
    size_t moe_batch_capacity_tokens_ = 0;
    float* d_moe_raw_f32_ = nullptr;           // [tokens * topk, hidden]
    size_t moe_raw_capacity_selections_ = 0;
    int64_t* h_moe_expert_ids_i64_ = nullptr;  // pinned [tokens * topk]
    size_t h_moe_expert_ids_capacity_ = 0;
    cudaEvent_t combined_output_ready_event_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    float* d_oracle_scratch_ = nullptr; // GPU Oracle scratch (H_ + E_ floats)
    bool gpu_oracle_ready_ = false;
    size_t cuda_total_ = 0, cuda_free_ = 0;  // from cudaMemGetInfo
    void cuda_cleanup();
    bool forward_layer_cuda(int layer, const float* h_in, float* h_out);
    bool ensure_combined_dispatch_capacity(size_t selections);
    bool ensure_combined_raw_capacity(size_t selections);
#endif
    bool moe_forward_batch_device_impl(
        int layer, const void* d_h_in, int tokens,
        const int* h_expert_ids, int topk, void* d_experts_out,
        bool synchronize_output);

    // host staging: resolved shard (layer, expert) -> F32 blob
    // [gate|up|down].  Synthetic single-layer shards intentionally map every
    // model layer to source layer 0; real multi-layer shards stay distinct.
    std::unordered_map<uint64_t, std::vector<float>> staging_;
    std::unordered_map<uint64_t, std::vector<uint16_t>> staging_bf16_;
    std::unordered_map<uint64_t, void*> pinned_staging_bf16_;
    // Milestone 2.5 fix: persistent pinned source pool.  The first-touch pin
    // path now registers the resolved mmap view in place (cudaHostRegister)
    // instead of allocating a separate pinned buffer and copying into it,
    // eliminating the measured 1225 ms Pinning + 267 ms MmapToPinned costs
    // and the per-call re-pin churn from release_transient_bf16_sources().
    // Entries survive across decode calls and are unregistered at teardown.
    std::unordered_map<uint64_t, void*> registered_mmap_views_bf16_;
    std::unordered_map<int, std::vector<float>> router_weights_;
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
    size_t peak_transient_host_bytes_ = 0;
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
    bool prepack_quantized_sources();
    StageProfile external_profile_snapshot(double total_wall_ms);

    // Stream `expert` from a resolved shard layer into VRAM.  The cache key
    // must describe the source weights, not merely the logical model layer.
    bool stage_expert(int logical_layer, int source_layer, int expert, int priority);
    const float* get_router_weights(int source_layer);
    void release_transient_bf16_sources();
    // Milestone 2.5 fix (defects #3/#5): pinned BF16 staging persists across
    // decode calls so first-touch cudaHostAlloc + mmap->pinned memcpy happens
    // once per unique expert, not every step. Only the unused F32 fallback
    // staging_bf16_ is dropped. Pinning is still bounded by
    // kPinnedStagingLimit in get_staging_bf16.
    void release_transient_f32_sources();
    bool prepare_profile_scenario();
    bool preload_all_experts();
};

// Compiled-library ABI sentinel used by separately built bindings.  Engine's
// layout changes when DEE_CUDA is enabled, so compare this with sizeof(Engine)
// before constructing an instance across that boundary.
size_t engine_abi_size();

} // namespace dee
