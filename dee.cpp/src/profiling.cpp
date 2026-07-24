#include "dee/profiling.h"
#include "dee/trace_alloc.h"  // Milestone 3 v5 teardown-forensics sentinel

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>

#ifdef DEE_CUDA
#include "dee/cuda_check.h"
#include <cuda_runtime.h>
#endif

namespace dee {
namespace {

uint64_t physical_key(int layer, int expert) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
           static_cast<uint32_t>(expert);
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(position);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double weight = position - static_cast<double>(lo);
    return values[lo] * (1.0 - weight) + values[hi] * weight;
}

using Interval = std::pair<double, double>;

std::vector<Interval> merge_intervals(std::vector<Interval> intervals) {
    if (intervals.empty()) return {};
    std::sort(intervals.begin(), intervals.end());
    std::vector<Interval> merged;
    for (const Interval& interval : intervals) {
        if (interval.second <= interval.first) continue;
        if (merged.empty() || interval.first > merged.back().second) {
            merged.push_back(interval);
        } else {
            merged.back().second = std::max(merged.back().second, interval.second);
        }
    }
    return merged;
}

double interval_duration(const std::vector<Interval>& intervals) {
    double total = 0.0;
    for (const Interval& interval : intervals) total += interval.second - interval.first;
    return total;
}

double intersection_duration(const std::vector<Interval>& a,
                             const std::vector<Interval>& b) {
    size_t i = 0, j = 0;
    double total = 0.0;
    while (i < a.size() && j < b.size()) {
        total += std::max(0.0, std::min(a[i].second, b[j].second) -
                              std::max(a[i].first, b[j].first));
        if (a[i].second < b[j].second) ++i;
        else ++j;
    }
    return total;
}

size_t h2d_bucket(size_t bytes) {
    if (bytes <= 16 * 1024) return 0;
    if (bytes <= 256 * 1024) return 1;
    if (bytes <= 1024 * 1024) return 2;
    return 3;
}

std::vector<Interval> complement_intervals(const std::vector<Interval>& busy,
                                           double begin, double end) {
    std::vector<Interval> idle;
    double cursor = begin;
    for (const Interval& interval : busy) {
        const double clipped_begin = std::max(begin, interval.first);
        const double clipped_end = std::min(end, interval.second);
        if (clipped_end <= begin || clipped_begin >= end) continue;
        if (clipped_begin > cursor) idle.emplace_back(cursor, clipped_begin);
        cursor = std::max(cursor, clipped_end);
    }
    if (cursor < end) idle.emplace_back(cursor, end);
    return idle;
}

bool contains_time(const TimelineRecord& record, double time_ms) {
    return record.start_ms <= time_ms && time_ms < record.end_ms;
}

IdleGapCategory idle_category(CpuTimelineKind kind) {
    switch (kind) {
        case CpuTimelineKind::OracleOutput: return IdleGapCategory::WaitingOracleOutput;
        case CpuTimelineKind::FirstTouchQuantization:
            return IdleGapCategory::WaitingFirstTouchQuantization;
        case CpuTimelineKind::CacheLookup: return IdleGapCategory::WaitingCacheLookup;
        case CpuTimelineKind::EvictionEligibility:
            return IdleGapCategory::WaitingEvictionEligibility;
        case CpuTimelineKind::ConsumerWait:
            return IdleGapCategory::WaitingCacheEntryReadiness;
        case CpuTimelineKind::LayerOutputWait:
            return IdleGapCategory::WaitingLayerDependency;
        case CpuTimelineKind::TransferSubmit:
            return IdleGapCategory::WaitingTransferSubmission;
        case CpuTimelineKind::CublasDispatch:
            return IdleGapCategory::WaitingCublasDispatch;
        case CpuTimelineKind::StagingSlotWait:
        case CpuTimelineKind::ComputeBatchWait:
        case CpuTimelineKind::PrefetchDrain:
            return IdleGapCategory::WaitingStreamEventSynchronization;
        case CpuTimelineKind::IntentionalNoWork:
            return IdleGapCategory::IntentionalNoWork;
        case CpuTimelineKind::HostScheduling:
        case CpuTimelineKind::TensorResolution:
            return IdleGapCategory::WaitingHostScheduling;
    }
    return IdleGapCategory::Unknown;
}

int idle_category_priority(IdleGapCategory category) {
    switch (category) {
        case IdleGapCategory::WaitingOracleOutput: return 100;
        case IdleGapCategory::WaitingFirstTouchQuantization: return 95;
        case IdleGapCategory::WaitingCacheLookup: return 90;
        case IdleGapCategory::WaitingEvictionEligibility: return 85;
        case IdleGapCategory::WaitingCacheEntryReadiness: return 80;
        case IdleGapCategory::WaitingLayerDependency: return 75;
        case IdleGapCategory::WaitingTransferSubmission: return 70;
        case IdleGapCategory::WaitingCublasDispatch: return 65;
        case IdleGapCategory::WaitingStreamEventSynchronization: return 60;
        case IdleGapCategory::IntentionalNoWork: return 55;
        case IdleGapCategory::WaitingHostScheduling: return 10;
        case IdleGapCategory::Unknown: return 0;
        case IdleGapCategory::Count: return -1;
    }
    return -1;
}

}  // namespace

StageProfiler::~StageProfiler() {
#ifdef DEE_CUDA
    if (timeline_origin_event_) {
        DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_DESTROY(static_cast<cudaEvent_t>(timeline_origin_event_), "timeline_origin_event_"),
                             "cudaEventDestroy(timeline origin)");
        timeline_origin_event_ = nullptr;
    }
    for (void* event : all_cuda_events_) {
        if (event) {
            DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_DESTROY(static_cast<cudaEvent_t>(event), "event"),
                                 "cudaEventDestroy(stage profiler pool)");
        }
    }
#endif
}

void StageProfiler::configure(bool enabled, bool trace_enabled,
                              size_t expert_blob_bytes, int oracle_experts,
                              bool timeline_enabled) {
    enabled_ = enabled;
    trace_enabled_ = enabled && trace_enabled;
    timeline_enabled_ = enabled && timeline_enabled;
    expert_blob_bytes_ = expert_blob_bytes;
    oracle_experts_ = oracle_experts;
    cpu_ms_.fill(0.0);
    gpu_ms_.fill(0.0);
    gpu_samples_.fill(0);
    oracle_ms_.fill(0.0);
    oracle_calls_ = 0;
    oracle_allocations_ = 0;
    oracle_allocation_bytes_ = 0;
    token_latencies_ms_.clear();
    layer_wall_ms_ = 0.0;
    layer_count_ = 0;
    evictions_ = 0;
    pinned_blocks_skipped_ = 0;
    mmap_to_pinned_bytes_ = 0;
    h2d_bytes_ = 0;
    h2d_copies_ = 0;
    cublas_calls_ = 0;
    kernel_launches_ = 0;
    stream_waits_ = 0;
    host_synchronizations_ = 0;
    duplicate_requests_ = 0;
    repeated_hits_ = 0;
    host_wait_ms_.fill(0.0);
    host_wait_count_.fill(0);
    cpu_timeline_origin_ = Clock::now();
    timeline_.clear();
    request_index_ = 0;
    last_request_index_.clear();
    unique_requested_.clear();
    unique_loaded_.clear();
    token_working_sets_.clear();
    trace_.clear();
    predictions_.clear();
}

void StageProfiler::note_cpu_timeline(CpuTimelineKind kind, TimePoint begin,
                                      int token, int logical_layer, int expert,
                                      size_t bytes, uint64_t transfer_id,
                                      size_t queue_depth, size_t staging_slot) {
    note_cpu_timeline_interval(kind, begin, Clock::now(), token, logical_layer,
                               expert, bytes, transfer_id, queue_depth,
                               staging_slot);
}

void StageProfiler::note_cpu_timeline_interval(CpuTimelineKind kind,
                                               TimePoint begin, TimePoint end,
                                               int token, int logical_layer,
                                               int expert, size_t bytes,
                                               uint64_t transfer_id,
                                               size_t queue_depth,
                                               size_t staging_slot) {
    if (!timeline_enabled_ || end <= begin) return;
    TimelineRecord record;
    record.cpu_kind = kind;
    record.start_ms = std::chrono::duration<double, std::milli>(
        begin - cpu_timeline_origin_).count();
    record.end_ms = std::chrono::duration<double, std::milli>(
        end - cpu_timeline_origin_).count();
    record.token = token;
    record.logical_layer = logical_layer;
    record.expert = expert;
    record.bytes = bytes;
    record.transfer_id = transfer_id;
    record.queue_depth = queue_depth;
    record.staging_slot = staging_slot;
    timeline_.push_back(record);
}

void StageProfiler::note_host_wait(HostWaitReason reason, TimePoint begin, TimePoint end,
                                   int token, int logical_layer, int expert,
                                   uint64_t transfer_id, size_t queue_depth,
                                   size_t staging_slot) {
    if (!enabled_) return;
    const double elapsed = std::chrono::duration<double, std::milli>(end - begin).count();
    host_wait_ms_[static_cast<size_t>(reason)] += elapsed;
    ++host_wait_count_[static_cast<size_t>(reason)];
    CpuTimelineKind kind = CpuTimelineKind::ConsumerWait;
    switch (reason) {
        case HostWaitReason::CacheReadiness: kind = CpuTimelineKind::ConsumerWait; break;
        case HostWaitReason::StagingSlot: kind = CpuTimelineKind::StagingSlotWait; break;
        case HostWaitReason::ComputeBatch: kind = CpuTimelineKind::ComputeBatchWait; break;
        case HostWaitReason::LayerOutput: kind = CpuTimelineKind::LayerOutputWait; break;
        case HostWaitReason::PrefetchDrain:
        case HostWaitReason::EngineTeardown: kind = CpuTimelineKind::PrefetchDrain; break;
        case HostWaitReason::Count: return;
    }
    note_cpu_timeline_interval(kind, begin, end, token, logical_layer, expert,
                               0, transfer_id, queue_depth, staging_slot);
}

void StageProfiler::add_cpu(CpuStage stage, TimePoint begin) {
    if (!enabled_) return;
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    add_cpu_ms(stage, ms);
}

void StageProfiler::add_cpu_ms(CpuStage stage, double milliseconds) {
    if (!enabled_) return;
    cpu_ms_[static_cast<size_t>(stage)] += milliseconds;
}

void StageProfiler::add_layer_latency(TimePoint begin) {
    if (!enabled_) return;
    layer_wall_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    ++layer_count_;
}

void StageProfiler::add_token_latency(TimePoint begin) {
    if (!enabled_) return;
    token_latencies_ms_.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
}

void StageProfiler::add_oracle(OracleStage stage, TimePoint begin) {
    if (!enabled_) return;
    add_oracle_ms(stage,
                  std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
}

void StageProfiler::add_oracle_ms(OracleStage stage, double milliseconds) {
    if (!enabled_) return;
    oracle_ms_[static_cast<size_t>(stage)] += milliseconds;
}

void StageProfiler::note_request(int token, int logical_layer, int resolved_layer,
                                 int expert, RequestKind kind, size_t cache_bytes_used,
                                 int evicted_layer, int evicted_expert, int priority,
                                 size_t cache_bytes_before,
                                 size_t cache_entries_before,
                                 size_t source_bytes,
                                 size_t destination_bytes,
                                 uint64_t transfer_id,
                                 bool source_pinned) {
    if (!enabled_) return;
    const uint64_t key = physical_key(resolved_layer, expert);
    unique_requested_.insert(key);
    if (kind == RequestKind::ColdLoad) unique_loaded_.insert(key);
    if (token >= 0) token_working_sets_[token].insert(key);

    const auto previous = last_request_index_.find(key);
    if (previous != last_request_index_.end() && kind != RequestKind::ColdLoad) {
        ++repeated_hits_;
    }
    const int64_t reuse_distance = previous == last_request_index_.end()
        ? -1
        : static_cast<int64_t>(request_index_ - previous->second - 1);
    last_request_index_[key] = request_index_;

    if (trace_enabled_) {
        RequestTraceRecord record;
        record.index = request_index_;
        record.request_time_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - cpu_timeline_origin_).count();
        record.token = token;
        record.logical_layer = logical_layer;
        record.resolved_layer = resolved_layer;
        record.expert = expert;
        record.kind = kind;
        record.cache_bytes_before = cache_bytes_before;
        record.cache_entries_before = cache_entries_before;
        record.cache_bytes_used = cache_bytes_used;
        record.evicted_layer = evicted_layer;
        record.evicted_expert = evicted_expert;
        record.reuse_distance = reuse_distance;
        record.priority = priority;
        record.source_bytes = source_bytes;
        record.destination_bytes = destination_bytes;
        record.transfer_id = transfer_id;
        record.source_pinned = source_pinned;
        trace_.push_back(record);
    }
    ++request_index_;
}

void StageProfiler::note_prediction(int, int, int resolved_layer,
                                    const std::vector<int>& experts) {
    if (!enabled_) return;
    std::vector<uint64_t> keys;
    keys.reserve(experts.size());
    for (int expert : experts) keys.push_back(physical_key(resolved_layer, expert));
    predictions_.push_back(std::move(keys));
}

#ifdef DEE_CUDA
bool StageProfiler::begin_cuda_timeline(void* compute_stream, void* transfer_stream) {
    if (!timeline_enabled_) return true;
    if (!compute_stream || !transfer_stream) return false;
    if (timeline_origin_event_) {
        if (!DEE_CUDA_CHECK_NAMED(
                DEE_TA_EVENT_DESTROY(static_cast<cudaEvent_t>(timeline_origin_event_), "timeline_origin_event_"),
                "cudaEventDestroy(previous timeline origin)")) return false;
        timeline_origin_event_ = nullptr;
    }
    cudaEvent_t origin = nullptr;
    if (!DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_CREATE(&origin, "origin"),
                              "cudaEventCreate(timeline origin)")) return false;
    timeline_origin_event_ = static_cast<void*>(origin);
    timeline_compute_stream_ = compute_stream;
    timeline_transfer_stream_ = transfer_stream;
    if (!DEE_CUDA_CHECK_NAMED(
            cudaEventRecord(origin, static_cast<cudaStream_t>(compute_stream)),
            "cudaEventRecord(timeline origin)")) return false;
    // Calibration is outside the measured interval. Both non-blocking streams
    // subsequently share the same completed timing origin.
    if (!DEE_CUDA_CHECK_NAMED(cudaEventSynchronize(origin),
                              "cudaEventSynchronize(timeline calibration)")) return false;
    if (!DEE_CUDA_CHECK_NAMED(
            cudaStreamWaitEvent(static_cast<cudaStream_t>(transfer_stream), origin, 0),
            "cudaStreamWaitEvent(timeline calibration)")) return false;
    cpu_timeline_origin_ = Clock::now();
    return true;
}

void StageProfiler::set_cuda_context(int token, int logical_layer, int expert,
                                     size_t bytes, uint64_t transfer_id,
                                     size_t queue_depth, size_t staging_slot) {
    if (!enabled_) return;
    cuda_context_ = CudaContext{token, logical_layer, expert, bytes, transfer_id,
                                queue_depth, staging_slot};
}

void* StageProfiler::acquire_cuda_event() {
    if (!free_cuda_events_.empty()) {
        void* event = free_cuda_events_.back();
        free_cuda_events_.pop_back();
        return event;
    }
    if (all_cuda_events_.size() >= kMaxTimingEvents) {
        std::fprintf(stderr, "[profile] CUDA timing event pool exhausted (%zu events)\n",
                     kMaxTimingEvents);
        return nullptr;
    }
    cudaEvent_t event = nullptr;
    if (!DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_CREATE(&event, "event"),
                              "cudaEventCreate(stage profiler timing)")) return nullptr;
    all_cuda_events_.push_back(static_cast<void*>(event));
    return static_cast<void*>(event);
}

void StageProfiler::release_cuda_event(void* event) {
    if (event) free_cuda_events_.push_back(event);
}

size_t StageProfiler::cuda_begin(GpuStage stage, void* stream) {
    if (!enabled_ || !stream) return static_cast<size_t>(-1);
    void* begin = acquire_cuda_event();
    void* end = acquire_cuda_event();
    if (!begin || !end) {
        release_cuda_event(begin);
        release_cuda_event(end);
        return static_cast<size_t>(-1);
    }
    if (!DEE_CUDA_CHECK_NAMED(
            cudaEventRecord(static_cast<cudaEvent_t>(begin), static_cast<cudaStream_t>(stream)),
            "cudaEventRecord(stage profiler begin)")) {
        release_cuda_event(begin);
        release_cuda_event(end);
        return static_cast<size_t>(-1);
    }
    PendingCudaSample sample;
    sample.stage = stage;
    sample.begin = begin;
    sample.end = end;
    sample.lane = stream == timeline_transfer_stream_ ? 1 :
                  (stream == timeline_compute_stream_ ? 2 : 0);
    sample.token = cuda_context_.token;
    sample.logical_layer = cuda_context_.logical_layer;
    sample.expert = cuda_context_.expert;
    sample.bytes = cuda_context_.bytes;
    sample.transfer_id = cuda_context_.transfer_id;
    sample.queue_depth = cuda_context_.queue_depth;
    sample.staging_slot = cuda_context_.staging_slot;
    pending_cuda_.push_back(sample);
    return pending_cuda_.size() - 1;
}

bool StageProfiler::cuda_end(size_t ticket, void* stream) {
    if (!enabled_ || ticket == static_cast<size_t>(-1)) return true;
    if (!stream || ticket >= pending_cuda_.size()) return false;
    PendingCudaSample& sample = pending_cuda_[ticket];
    if (!DEE_CUDA_CHECK_NAMED(
            cudaEventRecord(static_cast<cudaEvent_t>(sample.end), static_cast<cudaStream_t>(stream)),
            "cudaEventRecord(stage profiler end)")) return false;
    sample.ended = true;
    return true;
}

bool StageProfiler::cuda_collect_ready() {
    if (!enabled_) return true;
    std::vector<PendingCudaSample> remaining;
    remaining.reserve(pending_cuda_.size());
    for (const PendingCudaSample& sample : pending_cuda_) {
        if (!sample.ended) {
            remaining.push_back(sample);
            continue;
        }
        const cudaError_t query = cudaEventQuery(static_cast<cudaEvent_t>(sample.end));
        if (query == cudaErrorNotReady) {
            remaining.push_back(sample);
            continue;
        }
        if (!DEE_CUDA_CHECK_NAMED(query, "cudaEventQuery(stage profiler end)")) return false;
        float elapsed_ms = 0.0f;
        if (!DEE_CUDA_CHECK_NAMED(
                cudaEventElapsedTime(&elapsed_ms, static_cast<cudaEvent_t>(sample.begin),
                                     static_cast<cudaEvent_t>(sample.end)),
                "cudaEventElapsedTime(stage profiler)")) return false;
        const size_t stage = static_cast<size_t>(sample.stage);
        gpu_ms_[stage] += elapsed_ms;
        ++gpu_samples_[stage];
        if (timeline_enabled_ && timeline_origin_event_) {
            float start_ms = 0.0f;
            float end_ms = 0.0f;
            if (!DEE_CUDA_CHECK_NAMED(
                    cudaEventElapsedTime(&start_ms,
                        static_cast<cudaEvent_t>(timeline_origin_event_),
                        static_cast<cudaEvent_t>(sample.begin)),
                    "cudaEventElapsedTime(timeline start)") ||
                !DEE_CUDA_CHECK_NAMED(
                    cudaEventElapsedTime(&end_ms,
                        static_cast<cudaEvent_t>(timeline_origin_event_),
                        static_cast<cudaEvent_t>(sample.end)),
                    "cudaEventElapsedTime(timeline end)")) return false;
            TimelineRecord record;
            record.gpu = true;
            record.gpu_stage = sample.stage;
            record.start_ms = start_ms;
            record.end_ms = end_ms;
            record.lane = sample.lane;
            record.token = sample.token;
            record.logical_layer = sample.logical_layer;
            record.expert = sample.expert;
            record.bytes = sample.bytes;
            record.transfer_id = sample.transfer_id;
            record.queue_depth = sample.queue_depth;
            record.staging_slot = sample.staging_slot;
            timeline_.push_back(record);
        }
        release_cuda_event(sample.begin);
        release_cuda_event(sample.end);
    }
    pending_cuda_.swap(remaining);
    return true;
}
#endif

StageProfile StageProfiler::finish(double total_wall_ms, uint64_t resident_hits,
                                   uint64_t inflight_hits, uint64_t cold_loads,
                                   uint64_t duplicate_requests, uint64_t evictions,
                                   uint64_t pinned_blocks_skipped) {
    StageProfile result;
    result.enabled = enabled_;
    result.trace_enabled = trace_enabled_;
    if (!enabled_) return result;

    result.cpu_ms = cpu_ms_;
    result.gpu_ms = gpu_ms_;
    result.gpu_samples = gpu_samples_;
    result.oracle_ms = oracle_ms_;
    result.oracle_calls = oracle_calls_;
    result.oracle_allocations = oracle_allocations_;
    result.oracle_allocation_bytes = oracle_allocation_bytes_;
    double classified_oracle_ms = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(OracleStage::InvocationOverhead); ++i) {
        classified_oracle_ms += result.oracle_ms[i];
    }
    result.oracle_ms[static_cast<size_t>(OracleStage::InvocationOverhead)] =
        std::max(0.0, result.cpu_ms[static_cast<size_t>(CpuStage::Oracle)] - classified_oracle_ms);
    result.layer_wall_ms = layer_wall_ms_;
    result.layer_count = layer_count_;
    result.expert_requests = resident_hits + inflight_hits + cold_loads;
    result.resident_hits = resident_hits;
    result.inflight_hits = inflight_hits;
    result.cold_loads = cold_loads;
    result.duplicate_requests = duplicate_requests;
    result.unique_experts_requested = unique_requested_.size();
    result.unique_experts_loaded = unique_loaded_.size();
    result.evictions = evictions;
    result.pinned_blocks_skipped = pinned_blocks_skipped;
    result.mmap_to_pinned_bytes = mmap_to_pinned_bytes_;
    result.h2d_bytes = h2d_bytes_;
    result.h2d_copies = h2d_copies_;
    result.cublas_calls = cublas_calls_;
    result.kernel_launches = kernel_launches_;
    result.stream_waits = stream_waits_;
    result.host_synchronizations = host_synchronizations_;
    result.host_wait_ms = host_wait_ms_;
    result.host_wait_count = host_wait_count_;
#ifdef DEE_CUDA
    result.timing_events_allocated = all_cuda_events_.size() +
        (timeline_origin_event_ ? 1 : 0);
#endif
    result.average_h2d_copy_bytes = h2d_copies_ ? static_cast<double>(h2d_bytes_) / h2d_copies_ : 0.0;

    if (!token_latencies_ms_.empty()) {
        double sum = 0.0;
        for (double latency : token_latencies_ms_) sum += latency;
        result.token_latency_avg_ms = sum / token_latencies_ms_.size();
        result.token_latency_median_ms = percentile(token_latencies_ms_, 0.50);
        result.token_latency_p95_ms = percentile(token_latencies_ms_, 0.95);
        result.token_latency_max_ms = *std::max_element(token_latencies_ms_.begin(), token_latencies_ms_.end());
    }

    const double request_cpu_ms =
        cpu_ms_[static_cast<size_t>(CpuStage::TensorResolution)] +
        cpu_ms_[static_cast<size_t>(CpuStage::CacheLookup)] +
        cpu_ms_[static_cast<size_t>(CpuStage::CacheHitPinning)] +
        cpu_ms_[static_cast<size_t>(CpuStage::EvictionSelection)] +
        cpu_ms_[static_cast<size_t>(CpuStage::Pinning)] +
        cpu_ms_[static_cast<size_t>(CpuStage::HostTensorPreparation)] +
        cpu_ms_[static_cast<size_t>(CpuStage::MmapToPinned)] +
        cpu_ms_[static_cast<size_t>(CpuStage::TransferSubmission)] +
        cpu_ms_[static_cast<size_t>(CpuStage::HostWaiting)];
    result.average_expert_request_us = result.expert_requests
        ? request_cpu_ms * 1000.0 / result.expert_requests : 0.0;
    result.average_cold_load_us = cold_loads
        ? (cpu_ms_[static_cast<size_t>(CpuStage::TensorResolution)] +
           cpu_ms_[static_cast<size_t>(CpuStage::CacheLookup)] +
           cpu_ms_[static_cast<size_t>(CpuStage::EvictionSelection)] +
           cpu_ms_[static_cast<size_t>(CpuStage::Pinning)] +
           cpu_ms_[static_cast<size_t>(CpuStage::HostTensorPreparation)] +
           cpu_ms_[static_cast<size_t>(CpuStage::MmapToPinned)] +
           cpu_ms_[static_cast<size_t>(CpuStage::TransferSubmission)] +
           cpu_ms_[static_cast<size_t>(CpuStage::HostWaiting)]) * 1000.0 / cold_loads
        : 0.0;

    result.total_gpu_transfer_ms =
        gpu_ms_[static_cast<size_t>(GpuStage::H2D)] +
        gpu_ms_[static_cast<size_t>(GpuStage::ActivationH2D)] +
        gpu_ms_[static_cast<size_t>(GpuStage::D2H)];
    result.total_gpu_weight_conversion_ms =
        gpu_ms_[static_cast<size_t>(GpuStage::WeightConversion)];
    result.total_gpu_compute_ms =
        gpu_ms_[static_cast<size_t>(GpuStage::GateProjection)] +
        gpu_ms_[static_cast<size_t>(GpuStage::UpProjection)] +
        gpu_ms_[static_cast<size_t>(GpuStage::SiluMultiply)] +
        gpu_ms_[static_cast<size_t>(GpuStage::DownProjection)] +
        gpu_ms_[static_cast<size_t>(GpuStage::Combine)];

    uint64_t working_sum = 0;
    for (const auto& item : token_working_sets_) {
        working_sum += item.second.size();
        result.max_working_set_per_token = std::max<uint64_t>(result.max_working_set_per_token,
                                                               item.second.size());
    }
    result.average_working_set_per_token = token_working_sets_.empty()
        ? 0.0 : static_cast<double>(working_sum) / token_working_sets_.size();

    const uint64_t repeated = result.expert_requests >= result.unique_experts_requested
        ? result.expert_requests - result.unique_experts_requested : 0;
    result.repeated_requests = repeated;
    result.reused_before_eviction_fraction = repeated
        ? static_cast<double>(repeated_hits_) / repeated : 0.0;

    double overlap_sum = 0.0;
    uint64_t overlap_pairs = 0;
    for (size_t i = 1; i < predictions_.size(); ++i) {
        std::unordered_set<uint64_t> previous(predictions_[i - 1].begin(), predictions_[i - 1].end());
        uint64_t intersection = 0;
        for (uint64_t key : predictions_[i]) if (previous.count(key)) ++intersection;
        const size_t denominator = std::max(predictions_[i - 1].size(), predictions_[i].size());
        if (denominator) {
            overlap_sum += static_cast<double>(intersection) / denominator;
            ++overlap_pairs;
        }
    }
    result.oracle_adjacent_topk_overlap = overlap_pairs ? overlap_sum / overlap_pairs : 0.0;
    const size_t observed_topk = predictions_.empty() ? 0 : predictions_.front().size();
    result.oracle_random_overlap_expectation = oracle_experts_ > 0
        ? static_cast<double>(observed_topk) / oracle_experts_ : 0.0;

    if (timeline_enabled_) {
        result.timeline = timeline_;
        std::vector<Interval> copy_intervals;
        std::vector<Interval> compute_intervals;
        double first_gpu = std::numeric_limits<double>::max();
        double last_gpu = 0.0;
        double queue_sum = 0.0;
        uint64_t queue_samples = 0;
        for (const TimelineRecord& record : timeline_) {
            if (!record.gpu) continue;
            first_gpu = std::min(first_gpu, record.start_ms);
            last_gpu = std::max(last_gpu, record.end_ms);
            if (record.gpu_stage == GpuStage::H2D ||
                record.gpu_stage == GpuStage::ActivationH2D ||
                record.gpu_stage == GpuStage::D2H) {
                copy_intervals.emplace_back(record.start_ms, record.end_ms);
                if (record.gpu_stage == GpuStage::H2D) {
                    const size_t bucket = h2d_bucket(record.bytes);
                    ++result.h2d_bucket_copies[bucket];
                    result.h2d_bucket_bytes[bucket] += record.bytes;
                    result.h2d_bucket_ms[bucket] += record.end_ms - record.start_ms;
                }
                if (record.gpu_stage == GpuStage::H2D && record.expert >= 0) {
                    queue_sum += static_cast<double>(record.queue_depth);
                    ++queue_samples;
                    result.max_transfer_queue_depth = std::max<uint64_t>(
                        result.max_transfer_queue_depth, record.queue_depth);
                }
            } else if (record.gpu_stage != GpuStage::StreamWait) {
                compute_intervals.emplace_back(record.start_ms, record.end_ms);
            }
        }
        copy_intervals = merge_intervals(std::move(copy_intervals));
        compute_intervals = merge_intervals(std::move(compute_intervals));
        result.copy_engine_active_ms = interval_duration(copy_intervals);
        result.compute_engine_active_ms = interval_duration(compute_intervals);
        result.copy_compute_overlap_ms = intersection_duration(copy_intervals,
                                                               compute_intervals);
        if (first_gpu != std::numeric_limits<double>::max() && last_gpu > first_gpu) {
            result.gpu_timeline_span_ms = last_gpu - first_gpu;
            result.gpu_neither_active_ms = std::max(
                0.0, result.gpu_timeline_span_ms - result.copy_engine_active_ms -
                     result.compute_engine_active_ms + result.copy_compute_overlap_ms);
            result.copy_engine_utilization = result.copy_engine_active_ms /
                                             result.gpu_timeline_span_ms;
            result.compute_engine_utilization = result.compute_engine_active_ms /
                                                result.gpu_timeline_span_ms;
            result.copy_compute_overlap_fraction = result.copy_compute_overlap_ms /
                                                   result.gpu_timeline_span_ms;

            std::vector<Interval> all_busy = copy_intervals;
            all_busy.insert(all_busy.end(), compute_intervals.begin(),
                            compute_intervals.end());
            all_busy = merge_intervals(std::move(all_busy));
            const std::vector<Interval> idle_intervals =
                complement_intervals(all_busy, first_gpu, last_gpu);
            std::vector<IdleGapRecord> classified_segments;
            std::array<std::vector<double>,
                       static_cast<size_t>(IdleGapCategory::Count)> idle_samples;

            for (const Interval& idle : idle_intervals) {
                std::vector<double> boundaries{idle.first, idle.second};
                for (const TimelineRecord& record : timeline_) {
                    if (record.gpu || record.end_ms <= idle.first ||
                        record.start_ms >= idle.second) continue;
                    boundaries.push_back(std::max(idle.first, record.start_ms));
                    boundaries.push_back(std::min(idle.second, record.end_ms));
                }
                std::sort(boundaries.begin(), boundaries.end());
                boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                                 boundaries.end());
                for (size_t boundary = 1; boundary < boundaries.size(); ++boundary) {
                    const double segment_begin = boundaries[boundary - 1];
                    const double segment_end = boundaries[boundary];
                    if (segment_end <= segment_begin) continue;
                    const double midpoint = (segment_begin + segment_end) * 0.5;
                    IdleGapCategory category = IdleGapCategory::Unknown;
                    int best_priority = 0;
                    const TimelineRecord* best_record = nullptr;
                    for (const TimelineRecord& record : timeline_) {
                        if (record.gpu || !contains_time(record, midpoint)) continue;
                        const IdleGapCategory candidate = idle_category(record.cpu_kind);
                        const int priority = idle_category_priority(candidate);
                        if (priority > best_priority) {
                            category = candidate;
                            best_priority = priority;
                            best_record = &record;
                        }
                    }
                    IdleGapRecord segment;
                    segment.start_ms = segment_begin;
                    segment.end_ms = segment_end;
                    segment.category = category;
                    if (best_record) {
                        segment.token = best_record->token;
                        segment.logical_layer = best_record->logical_layer;
                        segment.expert = best_record->expert;
                        segment.transfer_id = best_record->transfer_id;
                    }
                    if (!classified_segments.empty()) {
                        IdleGapRecord& previous = classified_segments.back();
                        if (previous.category == segment.category &&
                            previous.token == segment.token &&
                            previous.logical_layer == segment.logical_layer &&
                            previous.expert == segment.expert &&
                            previous.transfer_id == segment.transfer_id &&
                            std::fabs(previous.end_ms - segment.start_ms) < 1e-6) {
                            previous.end_ms = segment.end_ms;
                            continue;
                        }
                    }
                    classified_segments.push_back(segment);
                }
            }

            for (const IdleGapRecord& gap : classified_segments) {
                const size_t category = static_cast<size_t>(gap.category);
                const double duration = gap.end_ms - gap.start_ms;
                result.idle_gap_ms[category] += duration;
                ++result.idle_gap_count[category];
                result.idle_gap_max_ms[category] =
                    std::max(result.idle_gap_max_ms[category], duration);
                idle_samples[category].push_back(duration);
            }
            for (size_t category = 0;
                 category < static_cast<size_t>(IdleGapCategory::Count);
                 ++category) {
                result.idle_gap_avg_ms[category] = result.idle_gap_count[category]
                    ? result.idle_gap_ms[category] /
                          static_cast<double>(result.idle_gap_count[category])
                    : 0.0;
                result.idle_gap_p95_ms[category] =
                    percentile(idle_samples[category], 0.95);
            }
            const size_t unknown = static_cast<size_t>(IdleGapCategory::Unknown);
            result.idle_attributed_ms = std::max(
                0.0, result.gpu_neither_active_ms - result.idle_gap_ms[unknown]);
            result.idle_attributed_fraction = result.gpu_neither_active_ms > 0.0
                ? result.idle_attributed_ms / result.gpu_neither_active_ms : 0.0;
            std::sort(classified_segments.begin(), classified_segments.end(),
                      [](const IdleGapRecord& lhs, const IdleGapRecord& rhs) {
                          return lhs.end_ms - lhs.start_ms > rhs.end_ms - rhs.start_ms;
                      });
            if (classified_segments.size() > 20) classified_segments.resize(20);
            result.top_idle_gaps = std::move(classified_segments);

            // Attribute each cache-readiness host wait against the matching
            // transfer's common-origin GPU records. Residual queued/event time
            // is classified as the consumer reaching the entry before it was
            // ready, rather than being silently folded into synchronization.
            for (const TimelineRecord& wait : timeline_) {
                if (wait.gpu || wait.cpu_kind != CpuTimelineKind::ConsumerWait ||
                    wait.end_ms <= wait.start_ms) continue;
                std::vector<double> boundaries{wait.start_ms, wait.end_ms};
                const TimelineRecord* matching_h2d = nullptr;
                const TimelineRecord* matching_conversion = nullptr;
                const TimelineRecord* matching_submit = nullptr;
                for (const TimelineRecord& record : timeline_) {
                    if (record.transfer_id != wait.transfer_id) continue;
                    if (record.gpu && record.gpu_stage == GpuStage::H2D) {
                        matching_h2d = &record;
                    } else if (record.gpu &&
                               record.gpu_stage == GpuStage::WeightConversion) {
                        matching_conversion = &record;
                    } else if (!record.gpu &&
                               record.cpu_kind == CpuTimelineKind::TransferSubmit) {
                        matching_submit = &record;
                    }
                }
                if (matching_h2d) {
                    const double overlap_begin = std::max(wait.start_ms,
                                                          matching_h2d->start_ms);
                    const double overlap_end = std::min(wait.end_ms,
                                                        matching_h2d->end_ms);
                    if (overlap_end > overlap_begin) {
                        boundaries.push_back(overlap_begin);
                        boundaries.push_back(overlap_end);
                    }
                }
                if (matching_conversion) {
                    const double overlap_begin = std::max(
                        wait.start_ms, matching_conversion->start_ms);
                    const double overlap_end = std::min(
                        wait.end_ms, matching_conversion->end_ms);
                    if (overlap_end > overlap_begin) {
                        boundaries.push_back(overlap_begin);
                        boundaries.push_back(overlap_end);
                    }
                }
                std::sort(boundaries.begin(), boundaries.end());
                boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                                 boundaries.end());
                std::array<bool,
                           static_cast<size_t>(ReadinessWaitCategory::Count)> seen{};
                for (size_t boundary = 1; boundary < boundaries.size(); ++boundary) {
                    const double segment_begin = boundaries[boundary - 1];
                    const double segment_end = boundaries[boundary];
                    if (segment_end <= segment_begin) continue;
                    const double midpoint = (segment_begin + segment_end) * 0.5;
                    ReadinessWaitCategory category =
                        ReadinessWaitCategory::ConsumerReachedEntryTooEarly;
                    if (matching_h2d && contains_time(*matching_h2d, midpoint)) {
                        category = ReadinessWaitCategory::CopyInFlight;
                    } else if (matching_conversion &&
                               contains_time(*matching_conversion, midpoint)) {
                        category = ReadinessWaitCategory::DequantizationInFlight;
                    } else if (!matching_submit ||
                               midpoint < matching_submit->end_ms) {
                        category = ReadinessWaitCategory::CopyNotSubmitted;
                    }
                    const size_t index = static_cast<size_t>(category);
                    result.readiness_wait_ms[index] += segment_end - segment_begin;
                    seen[index] = true;
                }
                for (size_t category = 0;
                     category < static_cast<size_t>(ReadinessWaitCategory::Count);
                     ++category) {
                    if (seen[category]) ++result.readiness_wait_count[category];
                }
            }
        }
        result.average_transfer_queue_depth = queue_samples ? queue_sum / queue_samples : 0.0;
    }

    if (trace_enabled_) {
        std::unordered_map<uint64_t, size_t> previous_position;
        for (size_t i = 0; i < trace_.size(); ++i) {
            RequestTraceRecord& record = trace_[i];
            const uint64_t key = physical_key(record.resolved_layer, record.expert);
            const auto previous = previous_position.find(key);
            if (previous != previous_position.end()) {
                std::unordered_set<uint64_t> distinct;
                for (size_t j = previous->second + 1; j < i; ++j) {
                    distinct.insert(physical_key(trace_[j].resolved_layer, trace_[j].expert));
                }
                record.distinct_reuse_distance = static_cast<int64_t>(distinct.size());
                record.theoretical_min_cache_bytes = (distinct.size() + 1) * expert_blob_bytes_;
            }
            previous_position[key] = i;
        }
        result.trace = trace_;
    }

    double categorized_cpu_ms = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(CpuStage::MiscEngine); ++i) {
        categorized_cpu_ms += cpu_ms_[i];
    }
    result.cpu_ms[static_cast<size_t>(CpuStage::MiscEngine)] =
        std::max(0.0, total_wall_ms - categorized_cpu_ms);
    return result;
}

const char* cpu_stage_name(CpuStage stage) {
    static const char* names[] = {
        "oracle", "tensor_resolution", "cache_lookup", "cache_hit_pinning",
        "eviction_selection", "pinning", "host_tensor_preparation",
        "mmap_to_pinned", "transfer_submission",
        "batch_construction", "host_waiting", "synchronization", "everything_else"
    };
    return names[static_cast<size_t>(stage)];
}

const char* gpu_stage_name(GpuStage stage) {
    static const char* names[] = {
        "h2d", "weight_conversion", "gate_projection", "up_projection", "silu_multiply",
        "down_projection", "combine", "stream_wait", "activation_h2d",
        "activation_conversion", "d2h"
    };
    return names[static_cast<size_t>(stage)];
}

const char* request_kind_name(RequestKind kind) {
    switch (kind) {
        case RequestKind::ResidentHit: return "resident";
        case RequestKind::InflightHit: return "inflight";
        case RequestKind::ColdLoad: return "cold";
    }
    return "unknown";
}

const char* oracle_stage_name(OracleStage stage) {
    static const char* names[] = {
        "model_lookup", "input_features", "linear0", "relu0", "linear1",
        "relu1", "linear2", "topk_sort", "topk_output", "allocation",
        "tensor_conversion", "synchronization", "invocation_overhead"
    };
    return names[static_cast<size_t>(stage)];
}

const char* host_wait_reason_name(HostWaitReason reason) {
    static const char* names[] = {
        "cache_readiness", "staging_slot", "compute_batch", "layer_output",
        "prefetch_drain", "engine_teardown"
    };
    return names[static_cast<size_t>(reason)];
}

const char* cpu_timeline_kind_name(CpuTimelineKind kind) {
    switch (kind) {
        case CpuTimelineKind::HostScheduling: return "host_scheduling";
        case CpuTimelineKind::OracleOutput: return "oracle_output";
        case CpuTimelineKind::TensorResolution: return "tensor_resolution";
        case CpuTimelineKind::CacheLookup: return "cache_lookup";
        case CpuTimelineKind::EvictionEligibility: return "eviction_eligibility";
        case CpuTimelineKind::FirstTouchQuantization: return "first_touch_quantization";
        case CpuTimelineKind::TransferSubmit: return "transfer_submit";
        case CpuTimelineKind::CublasDispatch: return "cublas_dispatch";
        case CpuTimelineKind::ConsumerWait: return "consumer_wait";
        case CpuTimelineKind::StagingSlotWait: return "staging_slot_wait";
        case CpuTimelineKind::ComputeBatchWait: return "compute_batch_wait";
        case CpuTimelineKind::LayerOutputWait: return "layer_output_wait";
        case CpuTimelineKind::PrefetchDrain: return "prefetch_drain";
        case CpuTimelineKind::IntentionalNoWork: return "intentional_no_work";
    }
    return "unknown";
}

const char* idle_gap_category_name(IdleGapCategory category) {
    static const char* names[] = {
        "waiting_oracle_output", "waiting_host_scheduling",
        "waiting_first_touch_quantization", "waiting_cache_lookup",
        "waiting_eviction_eligibility", "waiting_cache_entry_readiness",
        "waiting_layer_dependency", "waiting_transfer_submission",
        "waiting_cublas_dispatch", "waiting_stream_event_synchronization",
        "intentional_no_work", "unknown"
    };
    return names[static_cast<size_t>(category)];
}

const char* readiness_wait_category_name(ReadinessWaitCategory category) {
    static const char* names[] = {
        "copy_not_submitted", "copy_in_flight", "dequantization_in_flight",
        "cache_pin_conflict", "eviction_dependency",
        "consumer_reached_entry_too_early"
    };
    return names[static_cast<size_t>(category)];
}

std::string stage_profile_json(const StageProfile& profile, bool include_trace) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\"enabled\":" << (profile.enabled ? "true" : "false");
    out << ",\"cpu_ms\":{";
    for (size_t i = 0; i < static_cast<size_t>(CpuStage::Count); ++i) {
        if (i) out << ',';
        out << '\"' << cpu_stage_name(static_cast<CpuStage>(i)) << "\":" << profile.cpu_ms[i];
    }
    out << "},\"gpu_ms\":{";
    for (size_t i = 0; i < static_cast<size_t>(GpuStage::Count); ++i) {
        if (i) out << ',';
        out << '\"' << gpu_stage_name(static_cast<GpuStage>(i)) << "\":" << profile.gpu_ms[i];
    }
    out << "},\"gpu_samples\":{";
    for (size_t i = 0; i < static_cast<size_t>(GpuStage::Count); ++i) {
        if (i) out << ',';
        out << '\"' << gpu_stage_name(static_cast<GpuStage>(i)) << "\":" << profile.gpu_samples[i];
    }
    out << "},\"oracle_internal_ms\":{";
    for (size_t i = 0; i < static_cast<size_t>(OracleStage::Count); ++i) {
        if (i) out << ',';
        out << '\"' << oracle_stage_name(static_cast<OracleStage>(i)) << "\":"
            << profile.oracle_ms[i];
    }
    out << "},\"oracle_operations\":{\"calls\":" << profile.oracle_calls
        << ",\"allocations\":" << profile.oracle_allocations
        << ",\"allocation_bytes\":" << profile.oracle_allocation_bytes;
    out << "},\"layer_wall_ms\":" << profile.layer_wall_ms;
    out << ",\"layer_count\":" << profile.layer_count;
    out << ",\"token_latency_ms\":{\"average\":" << profile.token_latency_avg_ms
        << ",\"median\":" << profile.token_latency_median_ms
        << ",\"p95\":" << profile.token_latency_p95_ms
        << ",\"max\":" << profile.token_latency_max_ms << '}';
    out << ",\"requests\":{\"total\":" << profile.expert_requests
        << ",\"resident_hits\":" << profile.resident_hits
        << ",\"inflight_hits\":" << profile.inflight_hits
        << ",\"cold_loads\":" << profile.cold_loads
        << ",\"duplicates\":" << profile.duplicate_requests
        << ",\"unique_requested\":" << profile.unique_experts_requested
        << ",\"unique_loaded\":" << profile.unique_experts_loaded << '}';
    out << ",\"cache\":{\"evictions\":" << profile.evictions
        << ",\"pinned_blocks_skipped\":" << profile.pinned_blocks_skipped << '}';
    out << ",\"transfers\":{\"mmap_to_pinned_bytes\":" << profile.mmap_to_pinned_bytes
        << ",\"h2d_bytes\":" << profile.h2d_bytes
        << ",\"h2d_copies\":" << profile.h2d_copies
        << ",\"average_h2d_copy_bytes\":" << profile.average_h2d_copy_bytes << '}';
    out << ",\"operations\":{\"cublas_calls\":" << profile.cublas_calls
        << ",\"kernel_launches\":" << profile.kernel_launches
        << ",\"stream_waits\":" << profile.stream_waits
        << ",\"host_synchronizations\":" << profile.host_synchronizations
        << ",\"timing_events_allocated\":" << profile.timing_events_allocated << '}';
    out << ",\"host_waits\":{";
    for (size_t i = 0; i < static_cast<size_t>(HostWaitReason::Count); ++i) {
        if (i) out << ',';
        out << '\"' << host_wait_reason_name(static_cast<HostWaitReason>(i)) << "\":{"
            << "\"milliseconds\":" << profile.host_wait_ms[i]
            << ",\"count\":" << profile.host_wait_count[i] << '}';
    }
    out << '}';
    out << ",\"idle_gap_attribution\":{\"attributed_ms\":"
        << profile.idle_attributed_ms
        << ",\"attributed_fraction\":" << profile.idle_attributed_fraction
        << ",\"categories\":{";
    for (size_t i = 0; i < static_cast<size_t>(IdleGapCategory::Count); ++i) {
        if (i) out << ',';
        out << '\"' << idle_gap_category_name(static_cast<IdleGapCategory>(i))
            << "\":{\"milliseconds\":" << profile.idle_gap_ms[i]
            << ",\"count\":" << profile.idle_gap_count[i]
            << ",\"average_ms\":" << profile.idle_gap_avg_ms[i]
            << ",\"p95_ms\":" << profile.idle_gap_p95_ms[i]
            << ",\"max_ms\":" << profile.idle_gap_max_ms[i] << '}';
    }
    out << "},\"top_gaps\":[";
    for (size_t i = 0; i < profile.top_idle_gaps.size(); ++i) {
        if (i) out << ',';
        const IdleGapRecord& gap = profile.top_idle_gaps[i];
        out << "{\"start_ms\":" << gap.start_ms
            << ",\"end_ms\":" << gap.end_ms
            << ",\"duration_ms\":" << gap.end_ms - gap.start_ms
            << ",\"category\":\"" << idle_gap_category_name(gap.category) << '\"'
            << ",\"token\":" << gap.token
            << ",\"logical_layer\":" << gap.logical_layer
            << ",\"expert\":" << gap.expert
            << ",\"transfer_id\":" << gap.transfer_id << '}';
    }
    out << "]}";
    out << ",\"cache_readiness_attribution\":{";
    for (size_t i = 0; i < static_cast<size_t>(ReadinessWaitCategory::Count); ++i) {
        if (i) out << ',';
        out << '\"' << readiness_wait_category_name(
            static_cast<ReadinessWaitCategory>(i)) << "\":{\"milliseconds\":"
            << profile.readiness_wait_ms[i] << ",\"count\":"
            << profile.readiness_wait_count[i] << '}';
    }
    out << '}';
    out << ",\"derived\":{\"average_expert_request_us\":" << profile.average_expert_request_us
        << ",\"average_cold_load_us\":" << profile.average_cold_load_us
        << ",\"total_gpu_compute_ms\":" << profile.total_gpu_compute_ms
        << ",\"total_gpu_transfer_ms\":" << profile.total_gpu_transfer_ms
        << ",\"total_gpu_weight_conversion_ms\":" << profile.total_gpu_weight_conversion_ms
        << ",\"average_working_set_per_token\":" << profile.average_working_set_per_token
        << ",\"max_working_set_per_token\":" << profile.max_working_set_per_token
        << ",\"repeated_requests\":" << profile.repeated_requests
        << ",\"reused_before_eviction_fraction\":" << profile.reused_before_eviction_fraction
        << ",\"oracle_adjacent_topk_overlap\":" << profile.oracle_adjacent_topk_overlap
        << ",\"oracle_random_overlap_expectation\":" << profile.oracle_random_overlap_expectation
        << ",\"gpu_timeline_span_ms\":" << profile.gpu_timeline_span_ms
        << ",\"copy_engine_active_ms\":" << profile.copy_engine_active_ms
        << ",\"compute_engine_active_ms\":" << profile.compute_engine_active_ms
        << ",\"copy_compute_overlap_ms\":" << profile.copy_compute_overlap_ms
        << ",\"gpu_neither_active_ms\":" << profile.gpu_neither_active_ms
        << ",\"copy_engine_utilization\":" << profile.copy_engine_utilization
        << ",\"compute_engine_utilization\":" << profile.compute_engine_utilization
        << ",\"copy_compute_overlap_fraction\":" << profile.copy_compute_overlap_fraction
        << ",\"average_transfer_queue_depth\":" << profile.average_transfer_queue_depth
        << ",\"max_transfer_queue_depth\":" << profile.max_transfer_queue_depth << '}';
    out << ",\"h2d_size_buckets\":[";
    static const char* bucket_names[] = {"0-16KiB", "16-256KiB", "256KiB-1MiB", ">1MiB"};
    for (size_t i = 0; i < 4; ++i) {
        if (i) out << ',';
        out << "{\"range\":\"" << bucket_names[i] << "\",\"copies\":"
            << profile.h2d_bucket_copies[i] << ",\"bytes\":" << profile.h2d_bucket_bytes[i]
            << ",\"milliseconds\":" << profile.h2d_bucket_ms[i] << '}';
    }
    out << ']';
    if (include_trace) {
        out << ",\"trace\":[";
        for (size_t i = 0; i < profile.trace.size(); ++i) {
            if (i) out << ',';
            const RequestTraceRecord& record = profile.trace[i];
            out << "{\"index\":" << record.index
                << ",\"request_time_ms\":" << record.request_time_ms
                << ",\"token\":" << record.token
                << ",\"logical_layer\":" << record.logical_layer
                << ",\"resolved_shard_layer\":" << record.resolved_layer
                << ",\"expert\":" << record.expert
                << ",\"kind\":\"" << request_kind_name(record.kind) << '\"'
                << ",\"cache_bytes_before\":" << record.cache_bytes_before
                << ",\"cache_entries_before\":" << record.cache_entries_before
                << ",\"cache_bytes_used\":" << record.cache_bytes_used
                << ",\"evicted_layer\":" << record.evicted_layer
                << ",\"evicted_expert\":" << record.evicted_expert
                << ",\"reuse_distance\":" << record.reuse_distance
                << ",\"distinct_reuse_distance\":" << record.distinct_reuse_distance
                << ",\"theoretical_min_cache_bytes\":" << record.theoretical_min_cache_bytes
                << ",\"priority\":" << record.priority
                << ",\"source_bytes\":" << record.source_bytes
                << ",\"destination_bytes\":" << record.destination_bytes
                << ",\"transfer_id\":" << record.transfer_id
                << ",\"source_pinned\":" << (record.source_pinned ? "true" : "false")
                << ",\"eviction_reason\":"
                << (record.evicted_expert >= 0 ? "\"capacity_lru\"" : "null")
                << '}';
        }
        out << ']';
    }
    out << '}';
    return out.str();
}

std::string cuda_timeline_json(const StageProfile& profile) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[";
    bool first = true;
    auto emit_metadata = [&](int tid, const char* name) {
        if (!first) out << ',';
        first = false;
        out << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":"
            << tid << ",\"args\":{\"name\":\"" << name << "\"}}";
    };
    emit_metadata(0, "host");
    emit_metadata(1, "prefetch_stream");
    emit_metadata(2, "compute_stream");
    emit_metadata(3, "top_gpu_idle_gaps");
    for (const TimelineRecord& record : profile.timeline) {
        if (!first) out << ',';
        first = false;
        const char* name = record.gpu ? gpu_stage_name(record.gpu_stage)
                                      : cpu_timeline_kind_name(record.cpu_kind);
        out << "{\"name\":\"" << name << "\",\"cat\":\""
            << (record.gpu ? "cuda" : "host") << "\",\"ph\":\"X\",\"pid\":1,\"tid\":"
            << record.lane << ",\"ts\":" << record.start_ms * 1000.0
            << ",\"dur\":" << std::max(0.0, record.end_ms - record.start_ms) * 1000.0
            << ",\"args\":{\"token\":" << record.token
            << ",\"logical_layer\":" << record.logical_layer
            << ",\"expert\":" << record.expert
            << ",\"bytes\":" << record.bytes
            << ",\"transfer_id\":" << record.transfer_id
            << ",\"queue_depth\":" << record.queue_depth
            << ",\"staging_slot\":" << record.staging_slot << "}}";
    }
    for (const IdleGapRecord& gap : profile.top_idle_gaps) {
        if (!first) out << ',';
        first = false;
        out << "{\"name\":\"" << idle_gap_category_name(gap.category)
            << "\",\"cat\":\"gpu_idle_attribution\",\"ph\":\"X\",\"pid\":1,\"tid\":3"
            << ",\"ts\":" << gap.start_ms * 1000.0
            << ",\"dur\":" << (gap.end_ms - gap.start_ms) * 1000.0
            << ",\"args\":{\"token\":" << gap.token
            << ",\"logical_layer\":" << gap.logical_layer
            << ",\"expert\":" << gap.expert
            << ",\"transfer_id\":" << gap.transfer_id << "}}";
    }
    out << "],\"summary\":{\"gpu_span_ms\":" << profile.gpu_timeline_span_ms
        << ",\"copy_active_ms\":" << profile.copy_engine_active_ms
        << ",\"compute_active_ms\":" << profile.compute_engine_active_ms
        << ",\"overlap_ms\":" << profile.copy_compute_overlap_ms
        << ",\"neither_ms\":" << profile.gpu_neither_active_ms
        << ",\"copy_utilization\":" << profile.copy_engine_utilization
        << ",\"compute_utilization\":" << profile.compute_engine_utilization
        << ",\"overlap_fraction\":" << profile.copy_compute_overlap_fraction
        << ",\"idle_attributed_ms\":" << profile.idle_attributed_ms
        << ",\"idle_attributed_fraction\":" << profile.idle_attributed_fraction
        << ",\"idle_categories\":{";
    for (size_t i = 0; i < static_cast<size_t>(IdleGapCategory::Count); ++i) {
        if (i) out << ',';
        out << '\"' << idle_gap_category_name(static_cast<IdleGapCategory>(i))
            << "\":" << profile.idle_gap_ms[i];
    }
    out << "},\"cache_readiness_categories\":{";
    for (size_t i = 0; i < static_cast<size_t>(ReadinessWaitCategory::Count); ++i) {
        if (i) out << ',';
        out << '\"' << readiness_wait_category_name(
            static_cast<ReadinessWaitCategory>(i)) << "\":"
            << profile.readiness_wait_ms[i];
    }
    out << "}}}";
    return out.str();
}

}  // namespace dee
