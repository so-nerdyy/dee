#include "dee/profiling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
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

}  // namespace

StageProfiler::~StageProfiler() {
#ifdef DEE_CUDA
    for (void* event : all_cuda_events_) {
        if (event) {
            DEE_CUDA_CHECK_NAMED(cudaEventDestroy(static_cast<cudaEvent_t>(event)),
                                 "cudaEventDestroy(stage profiler pool)");
        }
    }
#endif
}

void StageProfiler::configure(bool enabled, bool trace_enabled,
                              size_t expert_blob_bytes, int oracle_experts) {
    enabled_ = enabled;
    trace_enabled_ = enabled && trace_enabled;
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
    request_index_ = 0;
    last_request_index_.clear();
    unique_requested_.clear();
    unique_loaded_.clear();
    token_working_sets_.clear();
    trace_.clear();
    predictions_.clear();
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
                                 int evicted_layer, int evicted_expert, int priority) {
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
        record.token = token;
        record.logical_layer = logical_layer;
        record.resolved_layer = resolved_layer;
        record.expert = expert;
        record.kind = kind;
        record.cache_bytes_used = cache_bytes_used;
        record.evicted_layer = evicted_layer;
        record.evicted_expert = evicted_expert;
        record.reuse_distance = reuse_distance;
        record.priority = priority;
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
    if (!DEE_CUDA_CHECK_NAMED(cudaEventCreate(&event),
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
    pending_cuda_.push_back(PendingCudaSample{stage, begin, end, false});
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
#ifdef DEE_CUDA
    result.timing_events_allocated = all_cuda_events_.size();
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
        cpu_ms_[static_cast<size_t>(CpuStage::MmapToPinned)] +
        cpu_ms_[static_cast<size_t>(CpuStage::TransferSubmission)] +
        cpu_ms_[static_cast<size_t>(CpuStage::HostWaiting)];
    result.average_expert_request_us = result.expert_requests
        ? request_cpu_ms * 1000.0 / result.expert_requests : 0.0;
    result.average_cold_load_us = cold_loads
        ? (cpu_ms_[static_cast<size_t>(CpuStage::TensorResolution)] +
           cpu_ms_[static_cast<size_t>(CpuStage::CacheLookup)] +
           cpu_ms_[static_cast<size_t>(CpuStage::EvictionSelection)] +
           cpu_ms_[static_cast<size_t>(CpuStage::MmapToPinned)] +
           cpu_ms_[static_cast<size_t>(CpuStage::TransferSubmission)] +
           cpu_ms_[static_cast<size_t>(CpuStage::HostWaiting)]) * 1000.0 / cold_loads
        : 0.0;

    result.total_gpu_transfer_ms = gpu_ms_[static_cast<size_t>(GpuStage::H2D)];
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
        "eviction_selection", "mmap_to_pinned", "transfer_submission",
        "batch_construction", "host_waiting", "synchronization", "everything_else"
    };
    return names[static_cast<size_t>(stage)];
}

const char* gpu_stage_name(GpuStage stage) {
    static const char* names[] = {
        "h2d", "weight_conversion", "gate_projection", "up_projection", "silu_multiply",
        "down_projection", "combine", "stream_wait"
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
        << ",\"oracle_random_overlap_expectation\":" << profile.oracle_random_overlap_expectation << '}';
    if (include_trace) {
        out << ",\"trace\":[";
        for (size_t i = 0; i < profile.trace.size(); ++i) {
            if (i) out << ',';
            const RequestTraceRecord& record = profile.trace[i];
            out << "{\"index\":" << record.index
                << ",\"token\":" << record.token
                << ",\"logical_layer\":" << record.logical_layer
                << ",\"resolved_shard_layer\":" << record.resolved_layer
                << ",\"expert\":" << record.expert
                << ",\"kind\":\"" << request_kind_name(record.kind) << '\"'
                << ",\"cache_bytes_used\":" << record.cache_bytes_used
                << ",\"evicted_layer\":" << record.evicted_layer
                << ",\"evicted_expert\":" << record.evicted_expert
                << ",\"reuse_distance\":" << record.reuse_distance
                << ",\"distinct_reuse_distance\":" << record.distinct_reuse_distance
                << ",\"theoretical_min_cache_bytes\":" << record.theoretical_min_cache_bytes
                << ",\"priority\":" << record.priority << '}';
        }
        out << ']';
    }
    out << '}';
    return out.str();
}

}  // namespace dee
