#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dee {

enum class CpuStage : size_t {
    Oracle,
    TensorResolution,
    CacheLookup,
    CacheHitPinning,
    EvictionSelection,
    Pinning,
    HostTensorPreparation,
    MmapToPinned,
    TransferSubmission,
    BatchConstruction,
    HostWaiting,
    Synchronization,
    MiscEngine,
    Count
};

enum class GpuStage : size_t {
    H2D,
    WeightConversion,
    GateProjection,
    UpProjection,
    SiluMultiply,
    DownProjection,
    Combine,
    StreamWait,
    ActivationH2D,
    ActivationConversion,
    D2H,
    Count
};

enum class RequestKind : uint8_t { ResidentHit, InflightHit, ColdLoad };

enum class HostWaitReason : size_t {
    CacheReadiness,
    StagingSlot,
    ComputeBatch,
    LayerOutput,
    PrefetchDrain,
    EngineTeardown,
    Count
};

enum class CpuTimelineKind : uint8_t {
    HostScheduling,
    OracleOutput,
    TensorResolution,
    CacheLookup,
    EvictionEligibility,
    FirstTouchQuantization,
    TransferSubmit,
    CublasDispatch,
    ConsumerWait,
    StagingSlotWait,
    ComputeBatchWait,
    LayerOutputWait,
    PrefetchDrain,
    IntentionalNoWork
};

enum class IdleGapCategory : size_t {
    WaitingOracleOutput,
    WaitingHostScheduling,
    WaitingFirstTouchQuantization,
    WaitingCacheLookup,
    WaitingEvictionEligibility,
    WaitingCacheEntryReadiness,
    WaitingLayerDependency,
    WaitingTransferSubmission,
    WaitingCublasDispatch,
    WaitingStreamEventSynchronization,
    IntentionalNoWork,
    Unknown,
    Count
};

enum class ReadinessWaitCategory : size_t {
    CopyNotSubmitted,
    CopyInFlight,
    DequantizationInFlight,
    CachePinConflict,
    EvictionDependency,
    ConsumerReachedEntryTooEarly,
    Count
};

enum class OracleStage : size_t {
    ModelLookup,
    InputFeatures,
    Linear0,
    Relu0,
    Linear1,
    Relu1,
    Linear2,
    TopKSort,
    TopKOutput,
    Allocation,
    TensorConversion,
    Synchronization,
    InvocationOverhead,
    Count
};

struct RequestTraceRecord {
    uint64_t index = 0;
    double request_time_ms = 0.0;
    int token = -1;
    int logical_layer = -1;
    int resolved_layer = -1;
    int expert = -1;
    RequestKind kind = RequestKind::ColdLoad;
    size_t cache_bytes_before = 0;
    size_t cache_entries_before = 0;
    size_t cache_bytes_after = 0;
    size_t cache_entries_after = 0;
    int evicted_layer = -1;
    int evicted_expert = -1;
    uint64_t evicted_generation = 0;
    uint64_t generation = 0;
    uint32_t pin_count = 0;
    int64_t reuse_distance = -1;
    int64_t distinct_reuse_distance = -1;
    size_t theoretical_min_cache_bytes = 0;
    int priority = 0;
    size_t source_bytes = 0;
    size_t destination_bytes = 0;
    uint64_t transfer_id = 0;
    bool source_pinned = false;
    bool transfer_launched = false;
    bool consumed = false;
    bool evicted_before_use = false;
};

struct TimelineRecord {
    bool gpu = false;
    GpuStage gpu_stage = GpuStage::H2D;
    CpuTimelineKind cpu_kind = CpuTimelineKind::TransferSubmit;
    double start_ms = 0.0;
    double end_ms = 0.0;
    int lane = 0;  // 0=host, 1=prefetch stream, 2=compute stream
    int token = -1;
    int logical_layer = -1;
    int expert = -1;
    size_t bytes = 0;
    uint64_t transfer_id = 0;
    size_t queue_depth = 0;
    size_t staging_slot = 0;
};

struct IdleGapRecord {
    double start_ms = 0.0;
    double end_ms = 0.0;
    IdleGapCategory category = IdleGapCategory::Unknown;
    int token = -1;
    int logical_layer = -1;
    int expert = -1;
    uint64_t transfer_id = 0;
};

struct StageProfile {
    bool enabled = false;
    bool trace_enabled = false;
    std::array<double, static_cast<size_t>(CpuStage::Count)> cpu_ms{};
    std::array<double, static_cast<size_t>(GpuStage::Count)> gpu_ms{};
    std::array<uint64_t, static_cast<size_t>(GpuStage::Count)> gpu_samples{};
    std::array<double, static_cast<size_t>(OracleStage::Count)> oracle_ms{};
    uint64_t oracle_calls = 0;
    uint64_t oracle_allocations = 0;
    uint64_t oracle_allocation_bytes = 0;

    double layer_wall_ms = 0.0;
    uint64_t layer_count = 0;
    double token_latency_avg_ms = 0.0;
    double token_latency_median_ms = 0.0;
    double token_latency_p95_ms = 0.0;
    double token_latency_max_ms = 0.0;

    uint64_t expert_requests = 0;
    uint64_t resident_hits = 0;
    uint64_t inflight_hits = 0;
    uint64_t cold_loads = 0;
    uint64_t duplicate_requests = 0;
    uint64_t unique_experts_requested = 0;
    uint64_t unique_experts_loaded = 0;
    uint64_t evictions = 0;
    uint64_t pinned_blocks_skipped = 0;

    uint64_t mmap_to_pinned_bytes = 0;
    uint64_t h2d_bytes = 0;
    uint64_t h2d_copies = 0;
    uint64_t cublas_calls = 0;
    uint64_t kernel_launches = 0;
    uint64_t stream_waits = 0;
    uint64_t host_synchronizations = 0;
    uint64_t timing_events_allocated = 0;
    std::array<double, static_cast<size_t>(HostWaitReason::Count)> host_wait_ms{};
    std::array<uint64_t, static_cast<size_t>(HostWaitReason::Count)> host_wait_count{};

    double average_expert_request_us = 0.0;
    double average_cold_load_us = 0.0;
    double average_h2d_copy_bytes = 0.0;
    double total_gpu_compute_ms = 0.0;
    double total_gpu_transfer_ms = 0.0;
    double total_gpu_weight_conversion_ms = 0.0;
    double average_working_set_per_token = 0.0;
    uint64_t max_working_set_per_token = 0;
    uint64_t repeated_requests = 0;
    double reused_before_eviction_fraction = 0.0;
    double oracle_adjacent_topk_overlap = 0.0;
    double oracle_random_overlap_expectation = 0.0;
    double gpu_timeline_span_ms = 0.0;
    double copy_engine_active_ms = 0.0;
    double compute_engine_active_ms = 0.0;
    double copy_compute_overlap_ms = 0.0;
    double gpu_neither_active_ms = 0.0;
    double copy_engine_utilization = 0.0;
    double compute_engine_utilization = 0.0;
    double copy_compute_overlap_fraction = 0.0;
    double average_transfer_queue_depth = 0.0;
    uint64_t max_transfer_queue_depth = 0;
    std::array<uint64_t, 4> h2d_bucket_copies{};
    std::array<uint64_t, 4> h2d_bucket_bytes{};
    std::array<double, 4> h2d_bucket_ms{};
    std::array<double, static_cast<size_t>(IdleGapCategory::Count)> idle_gap_ms{};
    std::array<uint64_t, static_cast<size_t>(IdleGapCategory::Count)> idle_gap_count{};
    std::array<double, static_cast<size_t>(IdleGapCategory::Count)> idle_gap_avg_ms{};
    std::array<double, static_cast<size_t>(IdleGapCategory::Count)> idle_gap_p95_ms{};
    std::array<double, static_cast<size_t>(IdleGapCategory::Count)> idle_gap_max_ms{};
    double idle_attributed_ms = 0.0;
    double idle_attributed_fraction = 0.0;
    std::array<double, static_cast<size_t>(ReadinessWaitCategory::Count)> readiness_wait_ms{};
    std::array<uint64_t, static_cast<size_t>(ReadinessWaitCategory::Count)> readiness_wait_count{};

    std::vector<RequestTraceRecord> trace;
    std::vector<TimelineRecord> timeline;
    std::vector<IdleGapRecord> top_idle_gaps;
};

class StageProfiler {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    StageProfiler() = default;
    ~StageProfiler();

    StageProfiler(const StageProfiler&) = delete;
    StageProfiler& operator=(const StageProfiler&) = delete;

    void configure(bool enabled, bool trace_enabled, size_t expert_blob_bytes,
                   int oracle_experts, bool timeline_enabled = false);
    bool enabled() const { return enabled_; }
    bool trace_enabled() const { return trace_enabled_; }
    bool timeline_enabled() const { return timeline_enabled_; }

    static TimePoint now() { return Clock::now(); }
    void add_cpu(CpuStage stage, TimePoint begin);
    void add_cpu_ms(CpuStage stage, double milliseconds);
    void add_layer_latency(TimePoint begin);
    void add_token_latency(TimePoint begin);
    void add_oracle(OracleStage stage, TimePoint begin);
    void add_oracle_ms(OracleStage stage, double milliseconds);
    void note_oracle_call() { if (enabled_) ++oracle_calls_; }
    void note_oracle_allocation(size_t bytes) {
        if (enabled_) { ++oracle_allocations_; oracle_allocation_bytes_ += bytes; }
    }

    void note_request(int token, int logical_layer, int resolved_layer, int expert,
                       RequestKind kind, int evicted_layer,
                       int evicted_expert, int priority,
                       size_t cache_bytes_before = 0,
                       size_t cache_entries_before = 0,
                       size_t cache_bytes_after = 0,
                       size_t cache_entries_after = 0,
                       size_t source_bytes = 0,
                       size_t destination_bytes = 0,
                       uint64_t transfer_id = 0,
                       bool source_pinned = false,
                       uint64_t generation = 0,
                       uint32_t pin_count = 0,
                       bool transfer_launched = false,
                       uint64_t evicted_generation = 0);
    void note_generation_evicted(int resolved_layer, int expert, uint64_t generation);
    void note_transfer_consumed(int resolved_layer, int expert, uint64_t generation);
    void note_prediction(int token, int logical_layer, int resolved_layer,
                         const std::vector<int>& experts);
    void note_eviction(uint64_t count = 1) { evictions_ += count; }
    void note_pinned_skip(uint64_t count = 1) { pinned_blocks_skipped_ += count; }
    void note_mmap_copy(size_t bytes) { mmap_to_pinned_bytes_ += bytes; }
    void note_h2d_copy(size_t bytes) { h2d_bytes_ += bytes; ++h2d_copies_; }
    void note_cublas_call(uint64_t count = 1) { cublas_calls_ += count; }
    void note_kernel_launch(uint64_t count = 1) { kernel_launches_ += count; }
    void note_stream_wait(uint64_t count = 1) { stream_waits_ += count; }
    void note_host_synchronization(uint64_t count = 1) { host_synchronizations_ += count; }
    void note_duplicate_request(uint64_t count = 1) { duplicate_requests_ += count; }
    void note_host_wait(HostWaitReason reason, TimePoint begin, TimePoint end,
                        int token = -1,
                        int logical_layer = -1, int expert = -1,
                        uint64_t transfer_id = 0, size_t queue_depth = 0,
                        size_t staging_slot = 0);
    void note_cpu_timeline(CpuTimelineKind kind, TimePoint begin, int token = -1,
                           int logical_layer = -1, int expert = -1,
                           size_t bytes = 0, uint64_t transfer_id = 0,
                           size_t queue_depth = 0, size_t staging_slot = 0);
    void note_cpu_timeline_interval(CpuTimelineKind kind, TimePoint begin, TimePoint end,
                                    int token = -1, int logical_layer = -1,
                                    int expert = -1, size_t bytes = 0,
                                    uint64_t transfer_id = 0,
                                    size_t queue_depth = 0,
                                    size_t staging_slot = 0);

#ifdef DEE_CUDA
    bool begin_cuda_timeline(void* compute_stream, void* transfer_stream);
    void set_cuda_context(int token, int logical_layer, int expert, size_t bytes = 0,
                          uint64_t transfer_id = 0, size_t queue_depth = 0,
                          size_t staging_slot = 0);
    size_t cuda_begin(GpuStage stage, void* stream);
    bool cuda_end(size_t ticket, void* stream);
    bool cuda_collect_ready();
#endif

    StageProfile finish(double total_wall_ms, uint64_t resident_hits,
                        uint64_t inflight_hits, uint64_t cold_loads,
                        uint64_t duplicate_requests, uint64_t evictions,
                        uint64_t pinned_blocks_skipped);

private:
    bool enabled_ = false;
    bool trace_enabled_ = false;
    bool timeline_enabled_ = false;
    size_t expert_blob_bytes_ = 0;
    int oracle_experts_ = 0;
    std::array<double, static_cast<size_t>(CpuStage::Count)> cpu_ms_{};
    std::array<double, static_cast<size_t>(GpuStage::Count)> gpu_ms_{};
    std::array<uint64_t, static_cast<size_t>(GpuStage::Count)> gpu_samples_{};
    std::array<double, static_cast<size_t>(OracleStage::Count)> oracle_ms_{};
    uint64_t oracle_calls_ = 0;
    uint64_t oracle_allocations_ = 0;
    uint64_t oracle_allocation_bytes_ = 0;
    std::vector<double> token_latencies_ms_;
    double layer_wall_ms_ = 0.0;
    uint64_t layer_count_ = 0;

    uint64_t evictions_ = 0;
    uint64_t pinned_blocks_skipped_ = 0;
    uint64_t mmap_to_pinned_bytes_ = 0;
    uint64_t h2d_bytes_ = 0;
    uint64_t h2d_copies_ = 0;
    uint64_t cublas_calls_ = 0;
    uint64_t kernel_launches_ = 0;
    uint64_t stream_waits_ = 0;
    uint64_t host_synchronizations_ = 0;
    uint64_t duplicate_requests_ = 0;
    uint64_t repeated_hits_ = 0;
    std::array<double, static_cast<size_t>(HostWaitReason::Count)> host_wait_ms_{};
    std::array<uint64_t, static_cast<size_t>(HostWaitReason::Count)> host_wait_count_{};
    TimePoint cpu_timeline_origin_{};
    std::vector<TimelineRecord> timeline_;

    uint64_t request_index_ = 0;
    std::unordered_map<uint64_t, uint64_t> last_request_index_;
    std::unordered_set<uint64_t> unique_requested_;
    std::unordered_set<uint64_t> unique_loaded_;
    std::unordered_map<int, std::unordered_set<uint64_t>> token_working_sets_;
    std::vector<RequestTraceRecord> trace_;
    std::vector<std::vector<uint64_t>> predictions_;

#ifdef DEE_CUDA
    struct PendingCudaSample {
        GpuStage stage = GpuStage::H2D;
        void* begin = nullptr;
        void* end = nullptr;
        bool ended = false;
        int lane = 0;
        int token = -1;
        int logical_layer = -1;
        int expert = -1;
        size_t bytes = 0;
        uint64_t transfer_id = 0;
        size_t queue_depth = 0;
        size_t staging_slot = 0;
    };
    struct CudaContext {
        int token = -1;
        int logical_layer = -1;
        int expert = -1;
        size_t bytes = 0;
        uint64_t transfer_id = 0;
        size_t queue_depth = 0;
        size_t staging_slot = 0;
    };
    static constexpr size_t kMaxTimingEvents = 128;
    std::vector<void*> all_cuda_events_;
    std::vector<void*> free_cuda_events_;
    std::vector<PendingCudaSample> pending_cuda_;
    void* timeline_origin_event_ = nullptr;
    void* timeline_compute_stream_ = nullptr;
    void* timeline_transfer_stream_ = nullptr;
    CudaContext cuda_context_{};
    void* acquire_cuda_event();
    void release_cuda_event(void* event);
#endif
};

const char* cpu_stage_name(CpuStage stage);
const char* gpu_stage_name(GpuStage stage);
const char* request_kind_name(RequestKind kind);
const char* oracle_stage_name(OracleStage stage);
const char* host_wait_reason_name(HostWaitReason reason);
const char* cpu_timeline_kind_name(CpuTimelineKind kind);
const char* idle_gap_category_name(IdleGapCategory category);
const char* readiness_wait_category_name(ReadinessWaitCategory category);
std::string stage_profile_json(const StageProfile& profile, bool include_trace);
std::string cuda_timeline_json(const StageProfile& profile);

}  // namespace dee
