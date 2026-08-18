// dee/engine.cpp
#include "dee/engine.h"
#include "dee/trace_alloc.h"  // Milestone 3 v5 teardown-forensics sentinel

#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef DEE_CUDA
#include <cuda_runtime.h>
#include "dee/cuda_check.h"
#include "dee/cuda_convert.h"
#include "dee/rmsnorm_cuda.h"
#include "dee/swiglu_cuda.h"
#endif

namespace dee {

bool Engine::validate_cache_invariants(std::string* error) const {
    std::string detail;
    if (!cache_.validate_invariants(&detail)) {
        if (error) *error = "cache: " + detail;
        return false;
    }
    if (!prefetcher_.validate_invariants(&detail)) {
        if (error) *error = "prefetcher: " + detail;
        return false;
    }
    if (error) error->clear();
    return true;
}
namespace {

uint64_t staging_key(int layer, int expert) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
           static_cast<uint32_t>(expert);
}

float quantize_bf16_projection_scalar(const uint16_t* source, int8_t* destination,
                                      size_t elements) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < elements; ++i) {
        max_abs = std::max(max_abs, std::fabs(bf16_to_f32(source[i])));
    }
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    for (size_t i = 0; i < elements; ++i) {
        const long value = std::lrint(bf16_to_f32(source[i]) / scale);
        destination[i] = static_cast<int8_t>(std::max(-127L, std::min(127L, value)));
    }
    return scale;
}

float quantize_bf16_projection_int4_scalar(const uint16_t* source, uint8_t* destination,
                                           size_t elements) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < elements; ++i) {
        max_abs = std::max(max_abs, std::fabs(bf16_to_f32(source[i])));
    }
    const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    for (size_t i = 0; i < elements; i += 2) {
        const long low_value = std::lrint(bf16_to_f32(source[i]) / scale);
        const int low = static_cast<int>(std::max(-7L, std::min(7L, low_value)));
        int high = 0;
        if (i + 1 < elements) {
            const long high_value = std::lrint(bf16_to_f32(source[i + 1]) / scale);
            high = static_cast<int>(std::max(-7L, std::min(7L, high_value)));
        }
        destination[i / 2] = static_cast<uint8_t>((low & 0x0f) | ((high & 0x0f) << 4));
    }
    return scale;
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
float quantize_bf16_projection_avx2(const uint16_t* source, int8_t* destination,
                                    size_t elements) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 maximum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= elements; i += 8) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i));
        const __m256 values = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        maximum = _mm256_max_ps(maximum, _mm256_and_ps(values, abs_mask));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, maximum);
    float max_abs = 0.0f;
    for (float lane : lanes) max_abs = std::max(max_abs, lane);
    for (; i < elements; ++i) max_abs = std::max(max_abs, std::fabs(bf16_to_f32(source[i])));
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    const __m256 inverse_scale = _mm256_set1_ps(1.0f / scale);
    alignas(32) int32_t integers[8];
    i = 0;
    for (; i + 8 <= elements; i += 8) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i));
        const __m256 values = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        _mm256_store_si256(reinterpret_cast<__m256i*>(integers),
                           _mm256_cvtps_epi32(_mm256_mul_ps(values, inverse_scale)));
        for (size_t lane = 0; lane < 8; ++lane) {
            destination[i + lane] = static_cast<int8_t>(
                std::max(-127, std::min(127, integers[lane])));
        }
    }
    for (; i < elements; ++i) {
        const long value = std::lrint(bf16_to_f32(source[i]) / scale);
        destination[i] = static_cast<int8_t>(std::max(-127L, std::min(127L, value)));
    }
    return scale;
}

__attribute__((target("avx2")))
float quantize_bf16_projection_int4_avx2(const uint16_t* source, uint8_t* destination,
                                         size_t elements) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 maximum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= elements; i += 8) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i));
        const __m256 values = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        maximum = _mm256_max_ps(maximum, _mm256_and_ps(values, abs_mask));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, maximum);
    float max_abs = 0.0f;
    for (float lane : lanes) max_abs = std::max(max_abs, lane);
    for (; i < elements; ++i) max_abs = std::max(max_abs, std::fabs(bf16_to_f32(source[i])));
    const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    const __m256 inverse_scale = _mm256_set1_ps(1.0f / scale);
    alignas(32) int32_t integers[8];
    i = 0;
    for (; i + 8 <= elements; i += 8) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i));
        const __m256 values = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        _mm256_store_si256(reinterpret_cast<__m256i*>(integers),
                           _mm256_cvtps_epi32(_mm256_mul_ps(values, inverse_scale)));
        for (size_t lane = 0; lane < 8; lane += 2) {
            const int low = std::max(-7, std::min(7, integers[lane]));
            const int high = std::max(-7, std::min(7, integers[lane + 1]));
            destination[(i + lane) / 2] = static_cast<uint8_t>(
                (low & 0x0f) | ((high & 0x0f) << 4));
        }
    }
    for (; i < elements; i += 2) {
        const long low_value = std::lrint(bf16_to_f32(source[i]) / scale);
        const int low = static_cast<int>(std::max(-7L, std::min(7L, low_value)));
        int high = 0;
        if (i + 1 < elements) {
            const long high_value = std::lrint(bf16_to_f32(source[i + 1]) / scale);
            high = static_cast<int>(std::max(-7L, std::min(7L, high_value)));
        }
        destination[i / 2] = static_cast<uint8_t>((low & 0x0f) | ((high & 0x0f) << 4));
    }
    return scale;
}
#endif

float quantize_bf16_projection(const uint16_t* source, int8_t* destination,
                               size_t elements) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("avx2")) {
        return quantize_bf16_projection_avx2(source, destination, elements);
    }
#endif
    return quantize_bf16_projection_scalar(source, destination, elements);
}

float quantize_bf16_projection_int4(const uint16_t* source, uint8_t* destination,
                                    size_t elements) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("avx2")) {
        return quantize_bf16_projection_int4_avx2(source, destination, elements);
    }
#endif
    return quantize_bf16_projection_int4_scalar(source, destination, elements);
}

}  // namespace

const char* benchmark_scenario_name(BenchmarkScenario scenario) {
    switch (scenario) {
        case BenchmarkScenario::EndToEnd: return "end-to-end";
        case BenchmarkScenario::FullResident: return "full-resident";
        case BenchmarkScenario::ResidentBypass: return "resident-bypass";
        case BenchmarkScenario::TransferOnly: return "transfer-only";
        case BenchmarkScenario::ComputeOnly: return "compute-only";
        case BenchmarkScenario::OracleOnly: return "oracle-only";
        case BenchmarkScenario::CacheMetadataOnly: return "cache-metadata-only";
    }
    return "unknown";
}

const char* device_cache_dtype_name(DeviceCacheDType dtype) {
    switch (dtype) {
        case DeviceCacheDType::Fp32: return "fp32";
        case DeviceCacheDType::Fp16: return "fp16";
    }
    return "unknown";
}

const char* weight_transfer_dtype_name(WeightTransferDType dtype) {
    switch (dtype) {
        case WeightTransferDType::Bf16: return "bf16";
        case WeightTransferDType::Int8: return "int8";
        case WeightTransferDType::Int4: return "int4";
        case WeightTransferDType::Fp4E2m1: return "fp4";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// static SwiGLU kernel (raw C++; CUDA/ggml slots in later)
// blob layout: [gate: INTER*HIDDEN][up: INTER*HIDDEN][down: HIDDEN*INTER]
//   gate/up are [INTER, HIDDEN] (row-major); down is [HIDDEN, INTER].
// x is the input hidden (length HIDDEN). acc is accumulated (length HIDDEN).
// ---------------------------------------------------------------------------
void Engine::swiglu(const float* blob, const float* x,
                    int inter, int hidden, float* acc) {
    const float* Wg = blob;
    const float* Wu = blob + (size_t)inter * hidden;
    const float* Wd = blob + 2 * (size_t)inter * hidden;
    for (int i = 0; i < inter; ++i) {
        float g = 0.0f, f = 0.0f;
        const float* rg = Wg + (size_t)i * hidden;
        const float* ru = Wu + (size_t)i * hidden;
        for (int j = 0; j < hidden; ++j) { g += rg[j] * x[j]; f += ru[j] * x[j]; }
        float s = g / (1.0f + std::exp(-g));   // SiLU
        float h = s * f;
        for (int o = 0; o < hidden; ++o) acc[o] += Wd[(size_t)o * inter + i] * h;
    }
}

// Real-model integration: caller-owned routing + combine. Engine runs SwiGLU
// per requested expert and writes per-expert FP32 outputs to a contiguous
// buffer. The Python adapter (pydee.adapter) is responsible for the standard
// gate-weighted sum combine so HF reference parity is preserved.
bool Engine::moe_forward_experts(int layer, const float* h_in, float* experts_out,
                                  const std::vector<int>& experts) {
    if (experts.empty() || experts_out == nullptr || h_in == nullptr) return false;
    const int K = (int)experts.size();
    if (K > cfg_.topk) {
        std::fprintf(stderr, "[engine] requested %d experts, configured topk is %d\n", K, cfg_.topk);
        return false;
    }
    for (int expert : experts) {
        if (expert < 0 || expert >= cfg_.num_experts) {
            std::fprintf(stderr, "[engine] expert index %d is outside [0,%d)\n",
                         expert, cfg_.num_experts);
            return false;
        }
    }
    if ((size_t)K * (size_t)hidden_ > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
    std::memset(experts_out, 0, (size_t)K * (size_t)hidden_ * sizeof(float));
    const int source_layer = avail_layer(layer);
#ifdef DEE_CUDA
    if (cfg_.use_cuda) {
        if (!DEE_CUDA_CHECK_NAMED(cudaSetDevice(cfg_.device_id), "cudaSetDevice(external MoE)")) return false;
        if (!DEE_CUDA_CHECK_NAMED(
                cudaMemcpyAsync(d_h_in_, h_in, (size_t)hidden_ * sizeof(float),
                                cudaMemcpyHostToDevice, compute_stream_),
                "cudaMemcpyAsync(external hidden host to device)")) return false;
        if (cfg_.cache_dtype == DeviceCacheDType::Fp16 &&
            !f32_to_f16_cuda(d_h_in_, d_h_in_half_, static_cast<size_t>(hidden_),
                             compute_stream_, nullptr)) return false;

        const int batch_size = std::max(
            1, static_cast<int>(cache_.budget_bytes() / cache_blob_bytes_));
        for (int first = 0; first < K; first += batch_size) {
            const int last = std::min(K, first + batch_size);
            prefetcher_.begin_batch();
            for (int k = first; k < last; ++k) {
                if (!stage_expert(layer, source_layer, experts[k], K - k)) return false;
            }
            std::vector<int> pinned;
            pinned.reserve(static_cast<size_t>(last - first));
            const int batch_experts = last - first;
            const bool use_batched =
                cfg_.use_batched_experts && batch_experts > 1 &&
                cfg_.cache_dtype == DeviceCacheDType::Fp16;
            if (!use_batched) {
                // Per-expert path (single expert, Fp32 cache, or opt-out).
                for (int k = first; k < last; ++k) {
                    const int expert = experts[k];
                    // Milestone 2.5 fix (defect #2): device-side wait so the host
                    // can keep issuing expert cache pin / GEMM.
                    if (!prefetcher_.wait_on_stream(source_layer, expert, compute_stream_) &&
                        !prefetcher_.wait(source_layer, expert)) return false;
                    if (!cache_.pin(source_layer, expert)) return false;
                    pinned.push_back(expert);
                    const void* d_blob = cache_.data(source_layer, expert);
                    StageProfiler* prof = profiler_.enabled() ? &profiler_ : nullptr;
                    const bool ok = d_blob && (cfg_.cache_dtype == DeviceCacheDType::Fp16
                        ? swiglu_expert_fp16_cuda(
                              cublas_handle_, d_blob, d_h_in_half_, d_hbuf_, d_ubuf_,
                              d_activation_half_, d_ybuf_ + (size_t)k * hidden_,
                              inter_, hidden_, compute_stream_, prof, cfg_.swiglu_limit)
                        : swiglu_expert_cuda(
                              cublas_handle_, static_cast<const float*>(d_blob), d_h_in_,
                              d_hbuf_, d_ubuf_, d_ybuf_ + (size_t)k * hidden_,
                               inter_, hidden_, compute_stream_, prof, cfg_.swiglu_limit));
                    if (!ok) return false;
                    prefetcher_.mark_consumed(source_layer, expert);
                }
            } else {
                // Stage 2: one strided/pointer-batched SwiGLU launch per
                // projection for all experts of this layer-token (shared
                // activation input; per-expert outputs in d_ybuf_).  Uses the
                // existing M5F pointer-batched kernel + pointer table.
                const size_t selections = static_cast<size_t>(batch_experts);
                if (!ensure_pointer_batch_capacity(selections)) return false;
                void* gate_out = d_moe_pointer_batch_gate_half_;
                void* up_out = d_moe_pointer_batch_up_half_;
                void* act_out = d_moe_pointer_batch_activation_half_;
                const size_t projection_elems =
                    static_cast<size_t>(inter_) * hidden_;
                for (int k = first; k < last; ++k) {
                    const int expert = experts[k];
                    if (!prefetcher_.wait_on_stream(source_layer, expert, compute_stream_) &&
                        !prefetcher_.wait(source_layer, expert)) return false;
                    if (!cache_.pin(source_layer, expert)) return false;
                    pinned.push_back(expert);
                    auto* blob = static_cast<uint16_t*>(
                        cache_.data(source_layer, expert));
                    if (!blob) {
                        set_last_error(
                            "batched expert cache pointer is null");
                        return false;
                    }
                    const size_t position_index =
                        static_cast<size_t>(k - first);
                    h_moe_pointer_table_[0 * selections + position_index] = blob;
                    h_moe_pointer_table_[1 * selections + position_index] =
                        blob + projection_elems;
                    h_moe_pointer_table_[2 * selections + position_index] =
                        blob + 2 * projection_elems;
                    h_moe_pointer_table_[3 * selections + position_index] =
                        const_cast<void*>(d_h_in_half_);
                    h_moe_pointer_table_[4 * selections + position_index] =
                        static_cast<uint16_t*>(gate_out) +
                        position_index * static_cast<size_t>(inter_);
                    h_moe_pointer_table_[5 * selections + position_index] =
                        static_cast<uint16_t*>(up_out) +
                        position_index * static_cast<size_t>(inter_);
                    h_moe_pointer_table_[6 * selections + position_index] =
                        static_cast<uint16_t*>(act_out) +
                        position_index * static_cast<size_t>(inter_);
                    h_moe_pointer_table_[7 * selections + position_index] =
                        d_ybuf_ + position_index * static_cast<size_t>(hidden_);
                }
                constexpr size_t kPointerArrays = 8;
                if (!DEE_CUDA_CHECK_NAMED(
                        cudaMemcpyAsync(
                            d_moe_pointer_table_, h_moe_pointer_table_,
                            kPointerArrays * selections * sizeof(void*),
                            cudaMemcpyHostToDevice, compute_stream_),
                        "cudaMemcpyAsync(batched expert pointer table)")) {
                    set_last_error("batched expert pointer upload failed");
                    return false;
                }
                void** table = d_moe_pointer_table_;
                StageProfiler* prof = profiler_.enabled() ? &profiler_ : nullptr;
                const bool computed = swiglu_expert_pointer_batch_fp16_cuda(
                    cublas_handle_,
                    table + 0 * selections, table + 1 * selections,
                    table + 2 * selections, table + 3 * selections,
                    table + 4 * selections, table + 5 * selections,
                    table + 6 * selections, table + 7 * selections,
                    gate_out, up_out, act_out,
                    batch_experts, inter_, hidden_, compute_stream_,
                    prof, cfg_.swiglu_limit);
                if (!computed) {
                    set_last_error(
                        "pointer-batched SwiGLU failed (use_batched_experts)");
                    return false;
                }
                for (int k = first; k < last; ++k) {
                    prefetcher_.mark_consumed(source_layer, experts[k]);
                }
            }
            if (!DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_),
                                      "cudaStreamSynchronize(external expert batch)")) return false;
            for (int expert : pinned) cache_.unpin(source_layer, expert);
        }
        if (!DEE_CUDA_CHECK_NAMED(
                cudaMemcpyAsync(experts_out, d_ybuf_,
                                (size_t)K * (size_t)hidden_ * sizeof(float),
                                cudaMemcpyDeviceToHost, compute_stream_),
                "cudaMemcpyAsync(external expert output device to host)") ||
            !DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_),
                                  "cudaStreamSynchronize(external expert output)")) return false;
        release_transient_f32_sources();
        stats_.peak_vram = std::max(stats_.peak_vram, cache_.used_bytes());
        return true;
    }
#endif
    std::vector<float> per_expert(hidden_, 0.0f);
    for (int k = 0; k < K; ++k) {
        const int e = experts[k];
        const float* blob = get_staging(source_layer, e);
        if (!blob) {
            std::fprintf(stderr, "[engine] moe_forward_experts: missing expert weights for layer=%d expert=%d\n",
                         layer, e);
            return false;
        }
        std::memset(per_expert.data(), 0, (size_t)hidden_ * sizeof(float));
        swiglu(blob, h_in, inter_, hidden_, per_expert.data());
        std::memcpy(experts_out + (size_t)k * (size_t)hidden_, per_expert.data(),
                    (size_t)hidden_ * sizeof(float));
        cache_.touch(source_layer, e);
    }
    return true;
}

bool Engine::moe_forward_batch(int layer, const float* h_in, int tokens,
                               const int* expert_ids, int topk,
                               float* experts_out) {
    if (!h_in || !expert_ids || !experts_out || tokens <= 0 || topk <= 0 ||
        topk > cfg_.topk) return false;
    const size_t selections = static_cast<size_t>(tokens) * topk;
    const auto external_layer_begin = profiler_.enabled()
        ? StageProfiler::now() : StageProfiler::TimePoint{};
    if (selections > std::numeric_limits<size_t>::max() /
                         (static_cast<size_t>(hidden_) * sizeof(float))) return false;
    std::memset(experts_out, 0, selections * static_cast<size_t>(hidden_) * sizeof(float));
    for (size_t i = 0; i < selections; ++i) {
        if (expert_ids[i] < 0 || expert_ids[i] >= cfg_.num_experts) {
            std::fprintf(stderr, "[engine] batched expert index %d is outside [0,%d)\n",
                         expert_ids[i], cfg_.num_experts);
            return false;
        }
    }

#ifdef DEE_CUDA
    if (cfg_.use_cuda && cfg_.cache_dtype == DeviceCacheDType::Fp16) {
        if (!DEE_CUDA_CHECK_NAMED(cudaSetDevice(cfg_.device_id),
                                  "cudaSetDevice(external batched MoE)")) return false;
        const auto batch_begin = profiler_.enabled()
            ? StageProfiler::now() : StageProfiler::TimePoint{};
        std::vector<std::vector<size_t>> groups(static_cast<size_t>(cfg_.num_experts));
        size_t max_group_tokens = 0;
        // Match torch.where(expert_mask[expert]) traversal: top-k rank first,
        // then token index. The resulting batch shape is exactly the eager
        // Transformers F.linear shape for this expert.
        for (int expert = 0; expert < cfg_.num_experts; ++expert) {
            auto& positions = groups[static_cast<size_t>(expert)];
            for (int rank = 0; rank < topk; ++rank) {
                for (int token = 0; token < tokens; ++token) {
                    const size_t position = static_cast<size_t>(token) * topk + rank;
                    if (expert_ids[position] == expert) positions.push_back(position);
                }
            }
            max_group_tokens = std::max(max_group_tokens, positions.size());
        }
        if (max_group_tokens == 0) return false;
        if (moe_batch_capacity_tokens_ < max_group_tokens) {
            if (d_moe_batch_input_) {
                if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_input_, "d_moe_batch_input_"),
                                          "cudaFree(MoE batch input)")) return false;
                d_moe_batch_input_ = nullptr;
            }
            if (d_moe_batch_input_half_) {
                if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_input_half_, "d_moe_batch_input_half_"),
                                          "cudaFree(MoE batch FP16 input)")) return false;
                d_moe_batch_input_half_ = nullptr;
            }
            if (d_moe_batch_gate_half_) {
                if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_gate_half_, "d_moe_batch_gate_half_"),
                                          "cudaFree(MoE batch gate)")) return false;
                d_moe_batch_gate_half_ = nullptr;
            }
            if (d_moe_batch_up_half_) {
                if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_up_half_, "d_moe_batch_up_half_"),
                                          "cudaFree(MoE batch up)")) return false;
                d_moe_batch_up_half_ = nullptr;
            }
            if (d_moe_batch_activation_half_) {
                if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_activation_half_, "d_moe_batch_activation_half_"),
                                          "cudaFree(MoE batch activation)")) return false;
                d_moe_batch_activation_half_ = nullptr;
            }
            if (d_moe_batch_output_) {
                if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_output_, "d_moe_batch_output_"),
                                          "cudaFree(MoE batch output)")) return false;
                d_moe_batch_output_ = nullptr;
            }
            const size_t input_elements = max_group_tokens * static_cast<size_t>(hidden_);
            const size_t inter_elements = max_group_tokens * static_cast<size_t>(inter_);
            if (!DEE_CUDA_CHECK_NAMED(
                    DEE_TA_MALLOC(reinterpret_cast<void**>(&d_moe_batch_input_),
                               input_elements * sizeof(float), "d_moe_batch_input_"),
                    "cudaMalloc(MoE batch input)") ||
                !DEE_CUDA_CHECK_NAMED(
                    DEE_TA_MALLOC(&d_moe_batch_input_half_, input_elements * sizeof(uint16_t), "d_moe_batch_input_half_"),
                    "cudaMalloc(MoE batch FP16 input)") ||
                !DEE_CUDA_CHECK_NAMED(
                    DEE_TA_MALLOC(&d_moe_batch_gate_half_, inter_elements * sizeof(uint16_t), "d_moe_batch_gate_half_"),
                    "cudaMalloc(MoE batch gate)") ||
                !DEE_CUDA_CHECK_NAMED(
                    DEE_TA_MALLOC(&d_moe_batch_up_half_, inter_elements * sizeof(uint16_t), "d_moe_batch_up_half_"),
                    "cudaMalloc(MoE batch up)") ||
                !DEE_CUDA_CHECK_NAMED(
                    DEE_TA_MALLOC(&d_moe_batch_activation_half_,
                               inter_elements * sizeof(uint16_t), "d_moe_batch_activation_half_"),
                    "cudaMalloc(MoE batch activation)") ||
                !DEE_CUDA_CHECK_NAMED(
                    DEE_TA_MALLOC(reinterpret_cast<void**>(&d_moe_batch_output_),
                               input_elements * sizeof(float), "d_moe_batch_output_"),
                    "cudaMalloc(MoE batch output)")) return false;
            moe_batch_capacity_tokens_ = max_group_tokens;
        }

        std::vector<int> active_experts;
        for (int expert = 0; expert < cfg_.num_experts; ++expert) {
            if (!groups[static_cast<size_t>(expert)].empty()) active_experts.push_back(expert);
        }
        if (profiler_.enabled()) {
            profiler_.add_cpu(CpuStage::BatchConstruction, batch_begin);
            profiler_.note_prediction(current_token_, layer, avail_layer(layer),
                                      active_experts);
        }
        std::vector<float> host_input(max_group_tokens * static_cast<size_t>(hidden_));
        std::vector<float> host_output(max_group_tokens * static_cast<size_t>(hidden_));
        peak_transient_host_bytes_ = std::max(
            peak_transient_host_bytes_,
            (host_input.size() + host_output.size()) * sizeof(float));
        const int source_layer = avail_layer(layer);
        const int cache_batch = std::max(
            1, static_cast<int>(cache_.budget_bytes() / cache_blob_bytes_));
        for (size_t first = 0; first < active_experts.size();
             first += static_cast<size_t>(cache_batch)) {
            const size_t last = std::min(
                active_experts.size(), first + static_cast<size_t>(cache_batch));
            prefetcher_.begin_batch();
            for (size_t i = first; i < last; ++i) {
                if (!stage_expert(layer, source_layer, active_experts[i],
                                  static_cast<int>(active_experts.size() - i))) return false;
            }
            for (size_t i = first; i < last; ++i) {
                const int expert = active_experts[i];
                const auto& positions = groups[static_cast<size_t>(expert)];
                const size_t group_tokens = positions.size();
                const size_t input_elements = group_tokens * static_cast<size_t>(hidden_);
                const auto host_prepare_begin = profiler_.enabled()
                    ? StageProfiler::now() : StageProfiler::TimePoint{};
                for (size_t row = 0; row < group_tokens; ++row) {
                    const size_t token = positions[row] / static_cast<size_t>(topk);
                    std::memcpy(host_input.data() + row * static_cast<size_t>(hidden_),
                                h_in + token * static_cast<size_t>(hidden_),
                                static_cast<size_t>(hidden_) * sizeof(float));
                }
                if (profiler_.enabled()) {
                    profiler_.add_cpu(CpuStage::HostTensorPreparation,
                                      host_prepare_begin);
                    profiler_.set_cuda_context(
                        current_token_, layer, expert,
                        input_elements * sizeof(float));
                }
                const size_t activation_h2d = profiler_.enabled()
                    ? profiler_.cuda_begin(GpuStage::ActivationH2D,
                                           static_cast<void*>(compute_stream_))
                    : static_cast<size_t>(-1);
                if (!DEE_CUDA_CHECK_NAMED(
                        cudaMemcpyAsync(d_moe_batch_input_, host_input.data(),
                                        input_elements * sizeof(float),
                                        cudaMemcpyHostToDevice, compute_stream_),
                        "cudaMemcpyAsync(MoE batch input)")) return false;
                if (profiler_.enabled() &&
                    !profiler_.cuda_end(activation_h2d,
                                        static_cast<void*>(compute_stream_))) return false;

                if (profiler_.enabled()) {
                    profiler_.set_cuda_context(
                        current_token_, layer, expert,
                        input_elements * sizeof(uint16_t));
                }
                const size_t activation_convert = profiler_.enabled()
                    ? profiler_.cuda_begin(GpuStage::ActivationConversion,
                                           static_cast<void*>(compute_stream_))
                    : static_cast<size_t>(-1);
                if (!f32_to_f16_cuda(d_moe_batch_input_, d_moe_batch_input_half_,
                                     input_elements, compute_stream_, nullptr)) return false;
                if (profiler_.enabled() &&
                    !profiler_.cuda_end(activation_convert,
                                        static_cast<void*>(compute_stream_))) return false;
                // Milestone 2.5 fix (defect #2): arm a device-side wait on the
                // compute stream for this expert's prefetch completion instead
                // of blocking the host via cudaEventSynchronize. Lets the host
                // keeps issuing the next expert's input H2D / cache stage while
                // cuBLAS waits on-device for the weight transfer to land.
                if (!prefetcher_.wait_on_stream(source_layer, expert, compute_stream_) &&
                    !prefetcher_.wait(source_layer, expert)) return false;
                if (!cache_.pin(source_layer, expert)) return false;
                const void* d_blob = cache_.data(source_layer, expert);
                if (profiler_.enabled()) {
                    profiler_.set_cuda_context(current_token_, layer, expert);
                }
                const bool computed = d_blob && swiglu_expert_batch_fp16_cuda(
                    cublas_handle_, d_blob, d_moe_batch_input_half_,
                    d_moe_batch_gate_half_, d_moe_batch_up_half_,
                    d_moe_batch_activation_half_, d_moe_batch_output_,
                    static_cast<int>(group_tokens), inter_, hidden_, compute_stream_,
                    profiler_.enabled() ? &profiler_ : nullptr);
                if (!computed) {
                    cache_.unpin(source_layer, expert);
                    return false;
                }
                prefetcher_.mark_consumed(source_layer, expert);
                if (profiler_.enabled()) {
                    profiler_.set_cuda_context(
                        current_token_, layer, expert,
                        input_elements * sizeof(float));
                }
                const size_t d2h = profiler_.enabled()
                    ? profiler_.cuda_begin(GpuStage::D2H,
                                           static_cast<void*>(compute_stream_))
                    : static_cast<size_t>(-1);
                if (!DEE_CUDA_CHECK_NAMED(
                        cudaMemcpyAsync(host_output.data(), d_moe_batch_output_,
                                        input_elements * sizeof(float),
                                        cudaMemcpyDeviceToHost, compute_stream_),
                        "cudaMemcpyAsync(MoE batch output)")) {
                    cache_.unpin(source_layer, expert);
                    return false;
                }
                if (profiler_.enabled() &&
                    !profiler_.cuda_end(d2h, static_cast<void*>(compute_stream_))) {
                    cache_.unpin(source_layer, expert);
                    return false;
                }
                const auto output_wait_begin = profiler_.enabled()
                    ? StageProfiler::now() : StageProfiler::TimePoint{};
                if (!DEE_CUDA_CHECK_NAMED(
                        cudaStreamSynchronize(compute_stream_),
                        "cudaStreamSynchronize(MoE batch output)")) {
                    cache_.unpin(source_layer, expert);
                    return false;
                }
                if (profiler_.enabled()) {
                    const auto output_wait_end = StageProfiler::now();
                    profiler_.add_cpu_ms(
                        CpuStage::Synchronization,
                        std::chrono::duration<double, std::milli>(
                            output_wait_end - output_wait_begin).count());
                    profiler_.note_host_wait(
                        HostWaitReason::LayerOutput, output_wait_begin,
                        output_wait_end, current_token_, layer, expert);
                    profiler_.note_host_synchronization();
                    if (!profiler_.cuda_collect_ready()) {
                        cache_.unpin(source_layer, expert);
                        return false;
                    }
                }
                cache_.unpin(source_layer, expert);
                const auto host_output_begin = profiler_.enabled()
                    ? StageProfiler::now() : StageProfiler::TimePoint{};
                for (size_t row = 0; row < group_tokens; ++row) {
                    float* destination = experts_out +
                        positions[row] * static_cast<size_t>(hidden_);
                    const float* source = host_output.data() +
                        row * static_cast<size_t>(hidden_);
                    std::memcpy(destination, source,
                                static_cast<size_t>(hidden_) * sizeof(float));
                }
                if (profiler_.enabled()) {
                    profiler_.add_cpu(CpuStage::HostTensorPreparation,
                                      host_output_begin);
                }
            }
        }
        release_transient_f32_sources();
        stats_.peak_vram = std::max(stats_.peak_vram, cache_.used_bytes());
        if (profiler_.enabled()) profiler_.add_layer_latency(external_layer_begin);
        return true;
    }
#endif

    for (int token = 0; token < tokens; ++token) {
        std::vector<int> row(static_cast<size_t>(topk));
        for (int rank = 0; rank < topk; ++rank) {
            row[static_cast<size_t>(rank)] =
                expert_ids[static_cast<size_t>(token) * topk + rank];
        }
        if (!moe_forward_experts(
                layer, h_in + static_cast<size_t>(token) * hidden_,
                experts_out + static_cast<size_t>(token) * topk * hidden_, row)) return false;
    }
    if (profiler_.enabled()) profiler_.add_layer_latency(external_layer_begin);
    return true;
}

// Milestone 3 fix (defect #6): device-resident MoE forward path.
// Accepts FP16 device hidden (d_h_in) + host expert IDs (h_expert_ids,
// small, host-side only for grouping) and writes FP32 per-(token,position)
// expert outputs to d_experts_out (both device-resident).  Eliminates the
// measured Python d2h->call->h2d round-trips (router_hidden_gpu_to_cpu,
// expert_inputs_gpu_to_cpu, expert_outputs_cpu_to_gpu) by keeping the
// per-layer hidden and MoE outputs on-device throughout.  Only expert_ids
// cross the host boundary (tokens * topk * sizeof(int) ≈ 32 bytes).
// Caller still handles the weighted combine on-device in Python.
// Requires FP16 device cache (DEE_CUDA); returns false if unavailable.
bool Engine::moe_forward_batch_device(int layer, const void* d_h_in, int tokens,
                                       const int* h_expert_ids, int topk,
                                       void* d_experts_out) {
    return moe_forward_batch_device_impl(
        layer, d_h_in, tokens, h_expert_ids, topk, d_experts_out,
        true, false);
}

bool Engine::moe_forward_batch_device_impl(
        int layer, const void* d_h_in, int tokens,
        const int* h_expert_ids, int topk, void* d_experts_out,
        bool synchronize_output, bool direct_single_row_io) {
    if (!d_h_in || !h_expert_ids || !d_experts_out || tokens <= 0 || topk <= 0 ||
        topk > cfg_.topk) return false;

#ifdef DEE_CUDA
    if (!cfg_.use_cuda || cfg_.cache_dtype != DeviceCacheDType::Fp16) return false;
    if (!DEE_CUDA_CHECK_NAMED(cudaSetDevice(cfg_.device_id),
                              "cudaSetDevice(external device MoE)")) return false;

    const size_t selections = static_cast<size_t>(tokens) * topk;
    const auto external_layer_begin = profiler_.enabled()
        ? StageProfiler::now() : StageProfiler::TimePoint{};

    // Validate expert IDs on host (no D2H of hidden needed).
    for (size_t i = 0; i < selections; ++i) {
        if (h_expert_ids[i] < 0 || h_expert_ids[i] >= cfg_.num_experts) {
            std::fprintf(stderr, "[engine] device expert index %d is outside [0,%d)\n",
                         h_expert_ids[i], cfg_.num_experts);
            return false;
        }
    }

    const auto batch_begin = profiler_.enabled()
        ? StageProfiler::now() : StageProfiler::TimePoint{};
    std::vector<std::vector<size_t>> groups(static_cast<size_t>(cfg_.num_experts));
    size_t max_group_tokens = 0;
    for (int expert = 0; expert < cfg_.num_experts; ++expert) {
        auto& positions = groups[static_cast<size_t>(expert)];
        for (int rank = 0; rank < topk; ++rank) {
            for (int token = 0; token < tokens; ++token) {
                const size_t position = static_cast<size_t>(token) * topk + rank;
                if (h_expert_ids[position] == expert) positions.push_back(position);
            }
        }
        max_group_tokens = std::max(max_group_tokens, positions.size());
    }
    if (max_group_tokens == 0) return false;

    // Ensure batch buffers are sized for the largest expert group.
    if (moe_batch_capacity_tokens_ < max_group_tokens) {
        if (d_moe_batch_input_) {
            if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_input_, "d_moe_batch_input_"),
                                      "cudaFree(MoE batch input)")) return false;
            d_moe_batch_input_ = nullptr;
        }
        if (d_moe_batch_input_half_) {
            if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_input_half_, "d_moe_batch_input_half_"),
                                      "cudaFree(MoE batch FP16 input)")) return false;
            d_moe_batch_input_half_ = nullptr;
        }
        if (d_moe_batch_gate_half_) {
            if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_gate_half_, "d_moe_batch_gate_half_"),
                                      "cudaFree(MoE batch gate)")) return false;
            d_moe_batch_gate_half_ = nullptr;
        }
        if (d_moe_batch_up_half_) {
            if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_up_half_, "d_moe_batch_up_half_"),
                                      "cudaFree(MoE batch up)")) return false;
            d_moe_batch_up_half_ = nullptr;
        }
        if (d_moe_batch_activation_half_) {
            if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_activation_half_, "d_moe_batch_activation_half_"),
                                      "cudaFree(MoE batch activation)")) return false;
            d_moe_batch_activation_half_ = nullptr;
        }
        if (d_moe_batch_output_) {
            if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_output_, "d_moe_batch_output_"),
                                      "cudaFree(MoE batch output)")) return false;
            d_moe_batch_output_ = nullptr;
        }
        const size_t input_elements = max_group_tokens * static_cast<size_t>(hidden_);
        const size_t inter_elements = max_group_tokens * static_cast<size_t>(inter_);
        if (!DEE_CUDA_CHECK_NAMED(
                DEE_TA_MALLOC(reinterpret_cast<void**>(&d_moe_batch_input_),
                           input_elements * sizeof(float), "d_moe_batch_input_"),
                "cudaMalloc(MoE batch input)") ||
            !DEE_CUDA_CHECK_NAMED(
                DEE_TA_MALLOC(&d_moe_batch_input_half_, input_elements * sizeof(uint16_t), "d_moe_batch_input_half_"),
                "cudaMalloc(MoE batch FP16 input)") ||
            !DEE_CUDA_CHECK_NAMED(
                DEE_TA_MALLOC(&d_moe_batch_gate_half_, inter_elements * sizeof(uint16_t), "d_moe_batch_gate_half_"),
                "cudaMalloc(MoE batch gate)") ||
            !DEE_CUDA_CHECK_NAMED(
                DEE_TA_MALLOC(&d_moe_batch_up_half_, inter_elements * sizeof(uint16_t), "d_moe_batch_up_half_"),
                "cudaMalloc(MoE batch up)") ||
            !DEE_CUDA_CHECK_NAMED(
                DEE_TA_MALLOC(&d_moe_batch_activation_half_,
                           inter_elements * sizeof(uint16_t), "d_moe_batch_activation_half_"),
                "cudaMalloc(MoE batch activation)") ||
            !DEE_CUDA_CHECK_NAMED(
                DEE_TA_MALLOC(reinterpret_cast<void**>(&d_moe_batch_output_),
                           input_elements * sizeof(float), "d_moe_batch_output_"),
                "cudaMalloc(MoE batch output)")) return false;
        moe_batch_capacity_tokens_ = max_group_tokens;
    }

    std::vector<int> active_experts;
    for (int expert = 0; expert < cfg_.num_experts; ++expert) {
        if (!groups[static_cast<size_t>(expert)].empty()) active_experts.push_back(expert);
    }
    if (profiler_.enabled()) {
        profiler_.add_cpu(CpuStage::BatchConstruction, batch_begin);
        profiler_.note_prediction(current_token_, layer, avail_layer(layer),
                                  active_experts);
    }
    const int source_layer = avail_layer(layer);
    const int cache_batch = std::max(
        1, static_cast<int>(cache_.budget_bytes() / cache_blob_bytes_));

    const auto* d_h_in_half = static_cast<const uint16_t*>(d_h_in);
    auto* d_experts_out_f32 = static_cast<float*>(d_experts_out);
    const size_t hidden_half = static_cast<size_t>(hidden_) * sizeof(uint16_t);
    const size_t hidden_float = static_cast<size_t>(hidden_) * sizeof(float);

    for (size_t first = 0; first < active_experts.size();
         first += static_cast<size_t>(cache_batch)) {
        const size_t last = std::min(
            active_experts.size(), first + static_cast<size_t>(cache_batch));
        prefetcher_.begin_batch();
        for (size_t i = first; i < last; ++i) {
            if (!stage_expert(layer, source_layer, active_experts[i],
                              static_cast<int>(active_experts.size() - i))) return false;
        }
        for (size_t i = first; i < last; ++i) {
            const int expert = active_experts[i];
            const auto& positions = groups[static_cast<size_t>(expert)];
            const size_t group_tokens = positions.size();
            const bool direct_single_row =
                direct_single_row_io && group_tokens == 1;
            const size_t direct_position =
                direct_single_row ? positions[0] : 0;
            const size_t direct_token =
                direct_single_row
                ? direct_position / static_cast<size_t>(topk)
                : 0;
            const void* expert_input_half = d_moe_batch_input_half_;

            // Device-to-device gather: pull each token's hidden row from
            // d_h_in_half into d_moe_batch_input_half_.  No host staging.
            if (direct_single_row) {
                expert_input_half =
                    d_h_in_half + direct_token * static_cast<size_t>(hidden_);
                ++stats_.direct_row_gather_bypasses;
            } else {
                if (profiler_.enabled()) {
                    profiler_.set_cuda_context(
                        current_token_, layer, expert,
                        group_tokens * hidden_half);
                }
                const size_t gather_ticket = profiler_.enabled()
                    ? profiler_.cuda_begin(
                        GpuStage::D2DGather,
                        static_cast<void*>(compute_stream_))
                    : static_cast<size_t>(-1);
                for (size_t row = 0; row < group_tokens; ++row) {
                    const size_t token =
                        positions[row] / static_cast<size_t>(topk);
                    if (!DEE_CUDA_CHECK_NAMED(
                            cudaMemcpyAsync(
                                static_cast<uint16_t*>(
                                    d_moe_batch_input_half_) +
                                    row * static_cast<size_t>(hidden_),
                                d_h_in_half +
                                    token * static_cast<size_t>(hidden_),
                                hidden_half,
                                cudaMemcpyDeviceToDevice,
                                compute_stream_),
                            "cudaMemcpyAsync(MoE batch D2D gather)")) {
                        return false;
                    }
                    ++stats_.d2d_gather_copies;
                    stats_.d2d_gather_bytes += hidden_half;
                    if (profiler_.enabled()) {
                        profiler_.note_d2d_gather_copy(hidden_half);
                    }
                }
                if (profiler_.enabled() &&
                    !profiler_.cuda_end(
                        gather_ticket,
                        static_cast<void*>(compute_stream_))) {
                    return false;
                }
            }

            // Arm device-side wait for expert weight transfer.
            if (!prefetcher_.wait_on_stream(source_layer, expert, compute_stream_) &&
                !prefetcher_.wait(source_layer, expert)) return false;
            if (!cache_.pin(source_layer, expert)) return false;

            const void* d_blob = cache_.data(source_layer, expert);
            if (profiler_.enabled()) {
                profiler_.set_cuda_context(current_token_, layer, expert);
            }
            float* expert_output = direct_single_row
                ? d_experts_out_f32 +
                    direct_position * static_cast<size_t>(hidden_)
                : d_moe_batch_output_;
            const bool computed = d_blob && swiglu_expert_batch_fp16_cuda(
                cublas_handle_, d_blob, expert_input_half,
                d_moe_batch_gate_half_, d_moe_batch_up_half_,
                d_moe_batch_activation_half_, expert_output,
                static_cast<int>(group_tokens), inter_, hidden_, compute_stream_,
                profiler_.enabled() ? &profiler_ : nullptr);
            if (!computed) {
                cache_.unpin(source_layer, expert);
                return false;
            }
            prefetcher_.mark_consumed(source_layer, expert);

            // Device-to-device scatter: write each row's FP32 result from
            // d_moe_batch_output_ to d_experts_out position.  FP32 output
            // matches the host-path moe_forward_batch contract; Python caller
            // converts to FP16 on-device via .to(dtype=hidden_states.dtype).
            if (direct_single_row) {
                ++stats_.direct_row_scatter_bypasses;
            } else {
                if (profiler_.enabled()) {
                    profiler_.set_cuda_context(
                        current_token_, layer, expert,
                        group_tokens * hidden_float);
                }
                const size_t scatter_ticket = profiler_.enabled()
                    ? profiler_.cuda_begin(
                        GpuStage::D2DScatter,
                        static_cast<void*>(compute_stream_))
                    : static_cast<size_t>(-1);
                for (size_t row = 0; row < group_tokens; ++row) {
                    const size_t position = positions[row];
                    if (!DEE_CUDA_CHECK_NAMED(
                            cudaMemcpyAsync(
                                d_experts_out_f32 +
                                    position * static_cast<size_t>(hidden_),
                                static_cast<float*>(d_moe_batch_output_) +
                                    row * static_cast<size_t>(hidden_),
                                hidden_float,
                                cudaMemcpyDeviceToDevice,
                                compute_stream_),
                            "cudaMemcpyAsync(MoE batch D2D scatter)")) {
                        cache_.unpin(source_layer, expert);
                        return false;
                    }
                    ++stats_.d2d_scatter_copies;
                    stats_.d2d_scatter_bytes += hidden_float;
                    if (profiler_.enabled()) {
                        profiler_.note_d2d_scatter_copy(hidden_float);
                    }
                }
                if (profiler_.enabled() &&
                    !profiler_.cuda_end(
                        scatter_ticket,
                        static_cast<void*>(compute_stream_))) {
                    cache_.unpin(source_layer, expert);
                    return false;
                }
            }
            cache_.unpin(source_layer, expert);
        }
    }

    // No D2H, no host output scatter.  Output is already on-device.
    // Synchronize the compute stream so the Python caller can safely read
    // d_experts_out without needing access to the engine's internal stream.
    if (synchronize_output) {
        const auto output_sync_begin = profiler_.enabled()
            ? StageProfiler::now() : StageProfiler::TimePoint{};
        if (!DEE_CUDA_CHECK_NAMED(
                cudaStreamSynchronize(compute_stream_),
                "cudaStreamSynchronize(device MoE batch)")) return false;
        if (profiler_.enabled()) {
            const auto output_sync_end = StageProfiler::now();
            profiler_.add_cpu_ms(
                CpuStage::Synchronization,
                std::chrono::duration<double, std::milli>(
                    output_sync_end - output_sync_begin).count());
            profiler_.note_host_synchronization();
            profiler_.note_host_wait(
                HostWaitReason::LayerOutput, output_sync_begin,
                output_sync_end, current_token_, layer);
        }
    }
    release_transient_f32_sources();
    stats_.peak_vram = std::max(stats_.peak_vram, cache_.used_bytes());
    if (profiler_.enabled() && synchronize_output) {
        profiler_.add_layer_latency(external_layer_begin);
    }
    return true;
#else
    (void)layer;
    (void)synchronize_output;
    (void)direct_single_row_io;
    return false;
#endif
}

#ifdef DEE_CUDA
bool Engine::ensure_combined_dispatch_capacity(size_t selections) {
    if (!combined_output_ready_event_) {
        if (!DEE_CUDA_CHECK_NAMED(
                DEE_TA_EVENT_CREATE_FLAGS(
                    &combined_output_ready_event_,
                    cudaEventDisableTiming,
                    "combined_output_ready_event_"),
                "cudaEventCreate(combined output ready)")) {
            combined_output_ready_event_ = nullptr;
            return false;
        }
    }
    if (h_moe_expert_ids_capacity_ >= selections) return true;
    if (h_moe_expert_ids_i64_) {
        if (!DEE_CUDA_CHECK_NAMED(
                DEE_TA_FREE_HOST(
                    h_moe_expert_ids_i64_, "h_moe_expert_ids_i64_"),
                "cudaFreeHost(combined expert IDs)")) {
            return false;
        }
        h_moe_expert_ids_i64_ = nullptr;
        h_moe_expert_ids_capacity_ = 0;
    }
    void* allocation = nullptr;
    if (!DEE_CUDA_CHECK_NAMED(
            DEE_TA_HOST_ALLOC(
                &allocation,
                selections * sizeof(int64_t),
                cudaHostAllocDefault,
                "h_moe_expert_ids_i64_"),
            "cudaHostAlloc(combined expert IDs)")) {
        return false;
    }
    h_moe_expert_ids_i64_ = static_cast<int64_t*>(allocation);
    h_moe_expert_ids_capacity_ = selections;
    return true;
}

bool Engine::ensure_combined_raw_capacity(size_t selections) {
    if (moe_raw_capacity_selections_ >= selections) return true;
    if (d_moe_raw_f32_) {
        if (!DEE_CUDA_CHECK_NAMED(
                DEE_TA_FREE(d_moe_raw_f32_, "d_moe_raw_f32_"),
                "cudaFree(combined raw workspace)")) {
            return false;
        }
        d_moe_raw_f32_ = nullptr;
        moe_raw_capacity_selections_ = 0;
    }
    if (!DEE_CUDA_CHECK_NAMED(
            DEE_TA_MALLOC(
                reinterpret_cast<void**>(&d_moe_raw_f32_),
                selections * static_cast<size_t>(hidden_) * sizeof(float),
                "d_moe_raw_f32_"),
            "cudaMalloc(combined raw workspace)")) {
        return false;
    }
    moe_raw_capacity_selections_ = selections;
    return true;
}

bool Engine::ensure_pointer_batch_capacity(size_t selections) {
    constexpr size_t kPointerArrays = 8;
    if (moe_pointer_batch_capacity_selections_ >= selections) {
        return d_moe_pointer_batch_gate_half_ &&
            d_moe_pointer_batch_up_half_ &&
            d_moe_pointer_batch_activation_half_ &&
            d_moe_pointer_table_ &&
            h_moe_pointer_table_ &&
            pointer_batch_complete_event_;
    }
    // top-k is immutable after init, so allocate the maximum selection count
    // once. This avoids resizing a workspace that may still be referenced by
    // the internal stream and makes the multi-allocation update transactional.
    if (moe_pointer_batch_capacity_selections_ != 0 ||
        d_moe_pointer_batch_gate_half_ ||
        d_moe_pointer_batch_up_half_ ||
        d_moe_pointer_batch_activation_half_ ||
        d_moe_pointer_table_ ||
        h_moe_pointer_table_) {
        return false;
    }

    const size_t capacity = std::max(
        selections, static_cast<size_t>(cfg_.topk));
    const size_t inter_elements =
        capacity * static_cast<size_t>(inter_);
    const size_t pointer_bytes =
        kPointerArrays * capacity * sizeof(void*);
    void* new_gate = nullptr;
    void* new_up = nullptr;
    void* new_activation = nullptr;
    void** new_device_table = nullptr;
    void* new_host_table = nullptr;
    const auto rollback = [&]() {
        if (new_gate) {
            DEE_CUDA_CHECK_NAMED(
                DEE_TA_FREE(new_gate, "d_moe_pointer_batch_gate_half_"),
                "cudaFree(pointer batch gate rollback)");
            new_gate = nullptr;
        }
        if (new_up) {
            DEE_CUDA_CHECK_NAMED(
                DEE_TA_FREE(new_up, "d_moe_pointer_batch_up_half_"),
                "cudaFree(pointer batch up rollback)");
            new_up = nullptr;
        }
        if (new_activation) {
            DEE_CUDA_CHECK_NAMED(
                DEE_TA_FREE(
                    new_activation,
                    "d_moe_pointer_batch_activation_half_"),
                "cudaFree(pointer batch activation rollback)");
            new_activation = nullptr;
        }
        if (new_device_table) {
            DEE_CUDA_CHECK_NAMED(
                DEE_TA_FREE(new_device_table, "d_moe_pointer_table_"),
                "cudaFree(pointer batch table rollback)");
            new_device_table = nullptr;
        }
        if (new_host_table) {
            DEE_CUDA_CHECK_NAMED(
                DEE_TA_FREE_HOST(new_host_table, "h_moe_pointer_table_"),
                "cudaFreeHost(pointer batch table rollback)");
            new_host_table = nullptr;
        }
    };
    if (!DEE_CUDA_CHECK_NAMED(
            DEE_TA_MALLOC(
                &new_gate,
                inter_elements * sizeof(uint16_t),
                "d_moe_pointer_batch_gate_half_"),
            "cudaMalloc(pointer batch gate)")) {
        rollback();
        return false;
    }
    if (!DEE_CUDA_CHECK_NAMED(
            DEE_TA_MALLOC(
                &new_up,
                inter_elements * sizeof(uint16_t),
                "d_moe_pointer_batch_up_half_"),
            "cudaMalloc(pointer batch up)")) {
        rollback();
        return false;
    }
    if (!DEE_CUDA_CHECK_NAMED(
            DEE_TA_MALLOC(
                &new_activation,
                inter_elements * sizeof(uint16_t),
                "d_moe_pointer_batch_activation_half_"),
            "cudaMalloc(pointer batch activation)")) {
        rollback();
        return false;
    }
    if (!DEE_CUDA_CHECK_NAMED(
            DEE_TA_MALLOC(
                reinterpret_cast<void**>(&new_device_table),
                pointer_bytes,
                "d_moe_pointer_table_"),
            "cudaMalloc(pointer batch table)")) {
        rollback();
        return false;
    }
    if (!DEE_CUDA_CHECK_NAMED(
            DEE_TA_HOST_ALLOC(
                &new_host_table, pointer_bytes, cudaHostAllocDefault,
                "h_moe_pointer_table_"),
            "cudaHostAlloc(pointer batch table)")) {
        rollback();
        return false;
    }
    if (!pointer_batch_complete_event_ &&
        !DEE_CUDA_CHECK_NAMED(
            DEE_TA_EVENT_CREATE_FLAGS(
                &pointer_batch_complete_event_,
                cudaEventDisableTiming,
                "pointer_batch_complete_event_"),
            "cudaEventCreate(pointer batch complete)")) {
        pointer_batch_complete_event_ = nullptr;
        rollback();
        return false;
    }

    d_moe_pointer_batch_gate_half_ = new_gate;
    d_moe_pointer_batch_up_half_ = new_up;
    d_moe_pointer_batch_activation_half_ = new_activation;
    d_moe_pointer_table_ = new_device_table;
    h_moe_pointer_table_ = static_cast<void**>(new_host_table);
    moe_pointer_batch_capacity_selections_ = capacity;
    return true;
}

bool Engine::drain_pointer_batch_pending(bool compute_stream_synchronized) {
    if (pointer_batch_retirement_poisoned_ &&
        !compute_stream_synchronized) {
        return false;
    }
    if (pointer_batch_pending_pins_.empty()) {
        if (compute_stream_synchronized) {
            pointer_batch_completion_event_valid_ = false;
            pointer_batch_retirement_poisoned_ = false;
        }
        return !pointer_batch_retirement_poisoned_;
    }
    if (!compute_stream_synchronized) {
        if (!pointer_batch_complete_event_ ||
            !pointer_batch_completion_event_valid_ ||
            !DEE_CUDA_CHECK_NAMED(
                cudaEventSynchronize(pointer_batch_complete_event_),
                "cudaEventSynchronize(pointer batch complete)")) {
            pointer_batch_retirement_poisoned_ = true;
            return false;
        }
    }
    bool unpinned = true;
    for (int expert : pointer_batch_pending_pins_) {
        if (!cache_.unpin(pointer_batch_pending_source_layer_, expert)) {
            unpinned = false;
        }
    }
    pointer_batch_pending_pins_.clear();
    pointer_batch_pending_source_layer_ = -1;
    pointer_batch_completion_event_valid_ = false;
    pointer_batch_retirement_poisoned_ = !unpinned;
    return unpinned;
}
#endif

bool Engine::moe_forward_combined_device(
        int layer, const void* d_h_in, int tokens,
        const int64_t* d_expert_ids, int topk,
        const float* d_weights_f32, void* d_output_f16,
        void* d_raw_trace_out, void* external_stream) {
    return moe_forward_combined_device_impl(
        layer, d_h_in, tokens, d_expert_ids, topk, d_weights_f32,
        d_output_f16, d_raw_trace_out, external_stream, false, false);
}

bool Engine::moe_forward_combined_direct_device(
        int layer, const void* d_h_in, int tokens,
        const int64_t* d_expert_ids, int topk,
        const float* d_weights_f32, void* d_output_f16,
        void* d_raw_trace_out, void* external_stream) {
    return moe_forward_combined_device_impl(
        layer, d_h_in, tokens, d_expert_ids, topk, d_weights_f32,
        d_output_f16, d_raw_trace_out, external_stream, true, false);
}

bool Engine::moe_forward_combined_pointer_batched_device(
        int layer, const void* d_h_in, int tokens,
        const int64_t* d_expert_ids, int topk,
        const float* d_weights_f32, void* d_output_f16,
        void* d_raw_trace_out, void* external_stream) {
    return moe_forward_combined_device_impl(
        layer, d_h_in, tokens, d_expert_ids, topk, d_weights_f32,
        d_output_f16, d_raw_trace_out, external_stream, true, true);
}

bool Engine::qwen_rms_norm_device(
        const void* d_input_f16, const void* d_weight_f16,
        void* d_output_f16, int rows, int dim, float epsilon,
        void* external_stream) {
    clear_last_error();
    if (!d_input_f16 || !d_weight_f16 || !d_output_f16 ||
        rows <= 0 || dim <= 0 || dim > 4096 ||
        !std::isfinite(epsilon) || epsilon < 0.0f) {
        set_last_error("invalid qwen_rms_norm_device arguments");
        return false;
    }
#ifdef DEE_CUDA
    if (!cfg_.use_cuda) {
        set_last_error("Qwen RMSNorm device path requires CUDA");
        return false;
    }
    if (!DEE_CUDA_CHECK_NAMED(
            cudaSetDevice(cfg_.device_id),
            "cudaSetDevice(Qwen RMSNorm)")) {
        set_last_error("cudaSetDevice failed for Qwen RMSNorm");
        return false;
    }
    if (!qwen_rms_norm_fp16_cuda(
            d_input_f16, d_weight_f16, d_output_f16,
            rows, dim, epsilon,
            static_cast<cudaStream_t>(external_stream))) {
        set_last_error("Qwen RMSNorm CUDA launch failed");
        return false;
    }
    return true;
#else
    (void)external_stream;
    set_last_error("Qwen RMSNorm device path unavailable without CUDA");
    return false;
#endif
}

uint64_t Engine::qwen_rms_norm_device_diagnostic_impl(
        const void* d_input_f16, const void* d_weight_f16,
        void* d_output_f16, int rows, int dim, float epsilon,
        int row_start, int row_count, int element_start, int element_count,
        size_t input_snapshot_row_stride_bytes,
        size_t normalized_row_stride_bytes,
        size_t output_snapshot_row_stride_bytes,
        size_t weight_snapshot_row_stride_bytes,
        size_t scalar_stride_bytes,
        void* d_input_snapshot_f32,
        void* d_sum_squares_f32,
        void* d_denominator_f32,
        void* d_reciprocal_rms_f32,
        void* d_weight_snapshot_f32,
        void* d_normalized_f32,
        void* d_output_snapshot_f32,
        void* external_stream,
        bool reference_kernel) {
    clear_last_error();
#ifdef DEE_CUDA
    if (!cfg_.use_cuda) {
        set_last_error("Qwen RMSNorm diagnostic requires CUDA");
        return 0;
    }
    if (!DEE_CUDA_CHECK_NAMED(
            cudaSetDevice(cfg_.device_id),
            "cudaSetDevice(Qwen RMSNorm diagnostic)")) {
        set_last_error("cudaSetDevice failed for Qwen RMSNorm diagnostic");
        return 0;
    }
    QwenRmsNormDiagnosticBuffers diagnostics;
    diagnostics.row_start = row_start;
    diagnostics.row_count = row_count;
    diagnostics.element_start = element_start;
    diagnostics.element_count = element_count;
    diagnostics.input_snapshot_row_stride_bytes = input_snapshot_row_stride_bytes;
    diagnostics.normalized_row_stride_bytes = normalized_row_stride_bytes;
    diagnostics.output_snapshot_row_stride_bytes = output_snapshot_row_stride_bytes;
    diagnostics.weight_snapshot_row_stride_bytes = weight_snapshot_row_stride_bytes;
    diagnostics.scalar_stride_bytes = scalar_stride_bytes;
    diagnostics.d_input_snapshot_f32 = static_cast<float*>(d_input_snapshot_f32);
    diagnostics.d_sum_squares_f32 = static_cast<float*>(d_sum_squares_f32);
    diagnostics.d_denominator_f32 = static_cast<float*>(d_denominator_f32);
    diagnostics.d_reciprocal_rms_f32 = static_cast<float*>(d_reciprocal_rms_f32);
    diagnostics.d_weight_snapshot_f32 = static_cast<float*>(d_weight_snapshot_f32);
    diagnostics.d_normalized_f32 = static_cast<float*>(d_normalized_f32);
    diagnostics.d_output_snapshot_f32 = static_cast<float*>(d_output_snapshot_f32);
    if (!d_input_f16 || !d_weight_f16 || !d_output_f16 ||
        !std::isfinite(epsilon) || epsilon < 0.0f || rows <= 0 || dim <= 0 ||
        !qwen_rms_norm_fp16_diagnostic_validate(rows, dim, diagnostics)) {
        set_last_error("invalid Qwen RMSNorm diagnostic arguments or bounded buffers");
        return 0;
    }
    const auto checked_bytes = [](size_t count, size_t stride, size_t* result) {
        if (stride != 0 && count > std::numeric_limits<size_t>::max() / stride) {
            return false;
        }
        *result = count * stride;
        return true;
    };
    const auto ranges_overlap = [](const void* left, size_t left_bytes,
                                   const void* right, size_t right_bytes) {
        if (!left || !right || left_bytes == 0 || right_bytes == 0) return false;
        const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
        const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);
        if (left_begin > std::numeric_limits<uintptr_t>::max() - left_bytes ||
            right_begin > std::numeric_limits<uintptr_t>::max() - right_bytes) {
            return true;
        }
        const uintptr_t left_end = left_begin + left_bytes;
        const uintptr_t right_end = right_begin + right_bytes;
        return left_begin < right_end && right_begin < left_end;
    };
    size_t input_bytes = 0;
    size_t weight_bytes = 0;
    size_t output_bytes = 0;
    size_t input_snapshot_bytes = 0;
    size_t normalized_bytes = 0;
    size_t output_snapshot_bytes = 0;
    size_t weight_snapshot_bytes = 0;
    size_t scalar_bytes = 0;
    if (!checked_bytes(static_cast<size_t>(rows), static_cast<size_t>(dim) * sizeof(uint16_t), &input_bytes) ||
        !checked_bytes(static_cast<size_t>(dim), sizeof(uint16_t), &weight_bytes) ||
        !checked_bytes(static_cast<size_t>(rows), static_cast<size_t>(dim) * sizeof(uint16_t), &output_bytes) ||
        !checked_bytes(static_cast<size_t>(row_count), input_snapshot_row_stride_bytes, &input_snapshot_bytes) ||
        !checked_bytes(static_cast<size_t>(row_count), normalized_row_stride_bytes, &normalized_bytes) ||
        !checked_bytes(static_cast<size_t>(row_count), output_snapshot_row_stride_bytes, &output_snapshot_bytes) ||
        !checked_bytes(static_cast<size_t>(row_count), weight_snapshot_row_stride_bytes, &weight_snapshot_bytes) ||
        !checked_bytes(static_cast<size_t>(row_count), scalar_stride_bytes, &scalar_bytes)) {
        set_last_error("Qwen RMSNorm diagnostic buffer size overflow");
        return 0;
    }
    const std::pair<const void*, size_t> primary[] = {
        {d_input_f16, input_bytes}, {d_weight_f16, weight_bytes},
        {d_output_f16, output_bytes},
    };
    const std::pair<const void*, size_t> diagnostic[] = {
        {d_input_snapshot_f32, input_snapshot_bytes},
        {d_sum_squares_f32, scalar_bytes},
        {d_denominator_f32, scalar_bytes},
        {d_reciprocal_rms_f32, scalar_bytes},
        {d_weight_snapshot_f32, weight_bytes > 0 ? weight_snapshot_bytes : 0},
        {d_normalized_f32, normalized_bytes},
        {d_output_snapshot_f32, output_snapshot_bytes},
    };
    for (const auto& diagnostic_range : diagnostic) {
        for (const auto& primary_range : primary) {
            if (ranges_overlap(diagnostic_range.first, diagnostic_range.second,
                               primary_range.first, primary_range.second)) {
                set_last_error("Qwen RMSNorm diagnostic buffer aliases a norm input, weight, or output");
                return 0;
            }
        }
    }
    for (size_t left = 0; left < sizeof(diagnostic) / sizeof(diagnostic[0]); ++left) {
        for (size_t right = left + 1; right < sizeof(diagnostic) / sizeof(diagnostic[0]); ++right) {
            if (ranges_overlap(diagnostic[left].first, diagnostic[left].second,
                               diagnostic[right].first, diagnostic[right].second)) {
                set_last_error("Qwen RMSNorm diagnostic buffers overlap each other");
                return 0;
            }
        }
    }
    cudaEvent_t completion_event = nullptr;
    if (!DEE_CUDA_CHECK_NAMED(
            cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming),
            "cudaEventCreateWithFlags(Qwen RMSNorm diagnostic)")) {
        set_last_error("cudaEventCreate failed for Qwen RMSNorm diagnostic");
        return 0;
    }
    const bool launched = reference_kernel
        ? qwen_rms_norm_fp16_reference_diagnostic_cuda(
            d_input_f16, d_weight_f16, d_output_f16, rows, dim, epsilon,
            static_cast<cudaStream_t>(external_stream), diagnostics, completion_event)
        : qwen_rms_norm_fp16_diagnostic_cuda(
            d_input_f16, d_weight_f16, d_output_f16, rows, dim, epsilon,
            static_cast<cudaStream_t>(external_stream), diagnostics, completion_event);
    if (!launched) {
        cudaEventDestroy(completion_event);
        set_last_error("Qwen RMSNorm diagnostic CUDA launch failed");
        return 0;
    }
    const cudaError_t wait_status = cudaEventSynchronize(completion_event);
    const cudaError_t destroy_status = cudaEventDestroy(completion_event);
    if (!DEE_CUDA_CHECK_NAMED(wait_status,
                              "cudaEventSynchronize(Qwen RMSNorm diagnostic)")) {
        set_last_error("Qwen RMSNorm diagnostic completion failed");
        return 0;
    }
    if (!DEE_CUDA_CHECK_NAMED(destroy_status,
                              "cudaEventDestroy(Qwen RMSNorm diagnostic)")) {
        set_last_error("Qwen RMSNorm diagnostic event destruction failed");
        return 0;
    }
    ++diagnostic_sequence_;
    return diagnostic_sequence_;
#else
    (void)d_input_f16;
    (void)d_weight_f16;
    (void)d_output_f16;
    (void)rows;
    (void)dim;
    (void)epsilon;
    (void)row_start;
    (void)row_count;
    (void)element_start;
    (void)element_count;
    (void)input_snapshot_row_stride_bytes;
    (void)normalized_row_stride_bytes;
    (void)output_snapshot_row_stride_bytes;
    (void)weight_snapshot_row_stride_bytes;
    (void)scalar_stride_bytes;
    (void)d_input_snapshot_f32;
    (void)d_sum_squares_f32;
    (void)d_denominator_f32;
    (void)d_reciprocal_rms_f32;
    (void)d_weight_snapshot_f32;
    (void)d_normalized_f32;
    (void)d_output_snapshot_f32;
    (void)external_stream;
    (void)reference_kernel;
    set_last_error("Qwen RMSNorm diagnostic unavailable without CUDA");
    return 0;
#endif
}

uint64_t Engine::qwen_rms_norm_device_diagnostic(
        const void* d_input_f16, const void* d_weight_f16,
        void* d_output_f16, int rows, int dim, float epsilon,
        int row_start, int row_count, int element_start, int element_count,
        size_t input_snapshot_row_stride_bytes,
        size_t normalized_row_stride_bytes,
        size_t output_snapshot_row_stride_bytes,
        size_t weight_snapshot_row_stride_bytes,
        size_t scalar_stride_bytes,
        void* d_input_snapshot_f32,
        void* d_sum_squares_f32,
        void* d_denominator_f32,
        void* d_reciprocal_rms_f32,
        void* d_weight_snapshot_f32,
        void* d_normalized_f32,
        void* d_output_snapshot_f32,
        void* external_stream) {
    return qwen_rms_norm_device_diagnostic_impl(
        d_input_f16, d_weight_f16, d_output_f16, rows, dim, epsilon,
        row_start, row_count, element_start, element_count,
        input_snapshot_row_stride_bytes, normalized_row_stride_bytes,
        output_snapshot_row_stride_bytes, weight_snapshot_row_stride_bytes,
        scalar_stride_bytes, d_input_snapshot_f32, d_sum_squares_f32,
        d_denominator_f32, d_reciprocal_rms_f32, d_weight_snapshot_f32,
        d_normalized_f32, d_output_snapshot_f32, external_stream, false);
}

uint64_t Engine::qwen_rms_norm_reference_diagnostic(
        const void* d_input_f16, const void* d_weight_f16,
        void* d_output_f16, int rows, int dim, float epsilon,
        int row_start, int row_count, int element_start, int element_count,
        size_t input_snapshot_row_stride_bytes,
        size_t normalized_row_stride_bytes,
        size_t output_snapshot_row_stride_bytes,
        size_t weight_snapshot_row_stride_bytes,
        size_t scalar_stride_bytes,
        void* d_input_snapshot_f32,
        void* d_sum_squares_f32,
        void* d_denominator_f32,
        void* d_reciprocal_rms_f32,
        void* d_weight_snapshot_f32,
        void* d_normalized_f32,
        void* d_output_snapshot_f32,
        void* external_stream) {
    return qwen_rms_norm_device_diagnostic_impl(
        d_input_f16, d_weight_f16, d_output_f16, rows, dim, epsilon,
        row_start, row_count, element_start, element_count,
        input_snapshot_row_stride_bytes, normalized_row_stride_bytes,
        output_snapshot_row_stride_bytes, weight_snapshot_row_stride_bytes,
        scalar_stride_bytes, d_input_snapshot_f32, d_sum_squares_f32,
        d_denominator_f32, d_reciprocal_rms_f32, d_weight_snapshot_f32,
        d_normalized_f32, d_output_snapshot_f32, external_stream, true);
}

bool Engine::qwen_rms_norm_gated_device(
        const void* d_input_f16, const void* d_weight_f16,
        const void* d_gate_f16, void* d_output_f16,
        int rows, int dim, float epsilon, void* external_stream) {
    clear_last_error();
    if (!d_input_f16 || !d_weight_f16 || !d_gate_f16 || !d_output_f16 ||
        rows <= 0 || dim <= 0 || dim > 4096 ||
        !std::isfinite(epsilon) || epsilon < 0.0f) {
        set_last_error("invalid qwen_rms_norm_gated_device arguments");
        return false;
    }
#ifdef DEE_CUDA
    if (!cfg_.use_cuda) {
        set_last_error("Qwen gated RMSNorm device path requires CUDA");
        return false;
    }
    if (!DEE_CUDA_CHECK_NAMED(
            cudaSetDevice(cfg_.device_id),
            "cudaSetDevice(Qwen gated RMSNorm)")) {
        set_last_error("cudaSetDevice failed for Qwen gated RMSNorm");
        return false;
    }
    if (!qwen_rms_norm_gated_fp16_cuda(
            d_input_f16, d_weight_f16, d_gate_f16, d_output_f16,
            rows, dim, epsilon,
            static_cast<cudaStream_t>(external_stream))) {
        set_last_error("Qwen gated RMSNorm CUDA launch failed");
        return false;
    }
    return true;
#else
    (void)external_stream;
    set_last_error("Qwen gated RMSNorm device path unavailable without CUDA");
    return false;
#endif
}

#ifdef DEE_CUDA
bool Engine::moe_forward_pointer_batched_device_impl(
        int layer, const void* d_h_in, int tokens,
        const int* h_expert_ids, int topk, float* d_raw_output) {
    constexpr size_t kPointerArrays = 8;
    if (!d_h_in || !h_expert_ids || !d_raw_output ||
        tokens != 1 || topk <= 0 || topk > cfg_.topk) {
        set_last_error(
            "pointer-batched expert path requires one token and valid top-k");
        return false;
    }
    const size_t selections = static_cast<size_t>(topk);
    std::vector<uint8_t> seen(static_cast<size_t>(cfg_.num_experts), 0);
    for (size_t position = 0; position < selections; ++position) {
        const int expert = h_expert_ids[position];
        if (expert < 0 || expert >= cfg_.num_experts ||
            seen[static_cast<size_t>(expert)] != 0) {
            set_last_error(
                "pointer-batched expert path requires unique in-range IDs");
            return false;
        }
        seen[static_cast<size_t>(expert)] = 1;
    }
    if (!ensure_pointer_batch_capacity(selections)) {
        set_last_error("failed to allocate pointer-batched expert workspace");
        return false;
    }

    const int source_layer = avail_layer(layer);
    prefetcher_.begin_batch();
    for (size_t position = 0; position < selections; ++position) {
        if (!stage_expert(
                layer, source_layer, h_expert_ids[position],
                static_cast<int>(selections - position))) {
            set_last_error("pointer-batched expert staging failed");
            return false;
        }
    }

    std::vector<int> pinned;
    pinned.reserve(selections);
    const auto unpin_all = [&]() -> bool {
        bool unpinned = true;
        for (int expert : pinned) {
            if (!cache_.unpin(source_layer, expert)) {
                unpinned = false;
            }
        }
        pinned.clear();
        if (!unpinned) {
            pointer_batch_retirement_poisoned_ = true;
        }
        return unpinned;
    };
    const auto synchronize_and_unpin = [&]() -> bool {
        if (DEE_CUDA_CHECK_NAMED(
                cudaStreamSynchronize(compute_stream_),
                "cudaStreamSynchronize(pointer batch failure cleanup)")) {
            return unpin_all();
        }
        // No current event can prove completion after a failed stream
        // synchronization. Preserve any pins and poison reuse until a later
        // full stream synchronization (reset/teardown) succeeds.
        pointer_batch_completion_event_valid_ = false;
        pointer_batch_retirement_poisoned_ = true;
        pointer_batch_pending_source_layer_ = source_layer;
        pointer_batch_pending_pins_.swap(pinned);
        return false;
    };
    const size_t projection =
        static_cast<size_t>(inter_) * static_cast<size_t>(hidden_);
    auto* gate_output = static_cast<uint16_t*>(
        d_moe_pointer_batch_gate_half_);
    auto* up_output = static_cast<uint16_t*>(
        d_moe_pointer_batch_up_half_);
    auto* activation = static_cast<uint16_t*>(
        d_moe_pointer_batch_activation_half_);
    for (size_t position = 0; position < selections; ++position) {
        const int expert = h_expert_ids[position];
        if (!prefetcher_.wait_on_stream(
                source_layer, expert, compute_stream_) &&
            !prefetcher_.wait(source_layer, expert)) {
            synchronize_and_unpin();
            set_last_error("pointer-batched expert cache wait failed");
            return false;
        }
        if (!cache_.pin(source_layer, expert)) {
            synchronize_and_unpin();
            set_last_error("pointer-batched expert cache pin failed");
            return false;
        }
        pinned.push_back(expert);
        auto* blob = static_cast<uint16_t*>(
            cache_.data(source_layer, expert));
        if (!blob) {
            synchronize_and_unpin();
            set_last_error(
                "pointer-batched expert cache pointer is null");
            return false;
        }
        h_moe_pointer_table_[0 * selections + position] = blob;
        h_moe_pointer_table_[1 * selections + position] =
            blob + projection;
        h_moe_pointer_table_[2 * selections + position] =
            blob + 2 * projection;
        h_moe_pointer_table_[3 * selections + position] =
            const_cast<void*>(d_h_in);
        h_moe_pointer_table_[4 * selections + position] =
            gate_output + position * static_cast<size_t>(inter_);
        h_moe_pointer_table_[5 * selections + position] =
            up_output + position * static_cast<size_t>(inter_);
        h_moe_pointer_table_[6 * selections + position] =
            activation + position * static_cast<size_t>(inter_);
        h_moe_pointer_table_[7 * selections + position] =
            d_raw_output + position * static_cast<size_t>(hidden_);
    }
    const size_t pointer_bytes =
        kPointerArrays * selections * sizeof(void*);
    if (!DEE_CUDA_CHECK_NAMED(
            cudaMemcpyAsync(
                d_moe_pointer_table_, h_moe_pointer_table_,
                pointer_bytes, cudaMemcpyHostToDevice, compute_stream_),
            "cudaMemcpyAsync(pointer-batched expert table)")) {
        synchronize_and_unpin();
        set_last_error("pointer-batched expert pointer upload failed");
        return false;
    }

    void** table = d_moe_pointer_table_;
    const bool computed = swiglu_expert_pointer_batch_fp16_cuda(
        cublas_handle_,
        table + 0 * selections,
        table + 1 * selections,
        table + 2 * selections,
        table + 3 * selections,
        table + 4 * selections,
        table + 5 * selections,
        table + 6 * selections,
        table + 7 * selections,
        d_moe_pointer_batch_gate_half_,
        d_moe_pointer_batch_up_half_,
        d_moe_pointer_batch_activation_half_,
        static_cast<int>(selections), inter_, hidden_, compute_stream_,
        profiler_.enabled() ? &profiler_ : nullptr);
    if (!computed) {
        synchronize_and_unpin();
        set_last_error("pointer-batched expert projection failed");
        return false;
    }
    for (int expert : pinned) {
        prefetcher_.mark_consumed(source_layer, expert);
    }
    if (!DEE_CUDA_CHECK_NAMED(
            cudaEventRecord(pointer_batch_complete_event_, compute_stream_),
            "cudaEventRecord(pointer batch complete)")) {
        synchronize_and_unpin();
        set_last_error("pointer-batched completion event record failed");
        return false;
    }
    pointer_batch_completion_event_valid_ = true;
    pointer_batch_pending_source_layer_ = source_layer;
    pointer_batch_pending_pins_.swap(pinned);
    ++stats_.pointer_batched_expert_calls;
    stats_.pointer_batched_experts += selections;
    release_transient_f32_sources();
    stats_.peak_vram = std::max(stats_.peak_vram, cache_.used_bytes());
    return true;
}
#endif

bool Engine::moe_forward_combined_device_impl(
        int layer, const void* d_h_in, int tokens,
        const int64_t* d_expert_ids, int topk,
        const float* d_weights_f32, void* d_output_f16,
        void* d_raw_trace_out, void* external_stream,
        bool direct_single_row_io, bool pointer_batched_experts) {
    clear_last_error();
    if (!d_h_in || !d_expert_ids || !d_weights_f32 || !d_output_f16 ||
        tokens <= 0 || topk <= 0 || topk > cfg_.topk) {
        set_last_error("invalid moe_forward_combined_device arguments");
        return false;
    }
#ifdef DEE_CUDA
    if (!cfg_.use_cuda || cfg_.cache_dtype != DeviceCacheDType::Fp16) {
        set_last_error("combined device path requires FP16 CUDA cache");
        return false;
    }
    if (!DEE_CUDA_CHECK_NAMED(
            cudaSetDevice(cfg_.device_id),
            "cudaSetDevice(combined device MoE)")) {
        set_last_error("cudaSetDevice failed for combined device MoE");
        return false;
    }
    // A null CUDA stream handle is the valid legacy/default stream. PyTorch
    // may expose it as integer 0, so it must not be rejected as a null pointer.
    auto stream = static_cast<cudaStream_t>(external_stream);
    const size_t selections = static_cast<size_t>(tokens) * topk;
    const size_t id_bytes = selections * sizeof(int64_t);
    const auto combined_layer_begin = profiler_.enabled()
        ? StageProfiler::now() : StageProfiler::TimePoint{};
    if (!ensure_combined_dispatch_capacity(selections)) {
        set_last_error("failed to allocate combined dispatch workspace");
        return false;
    }

    if (profiler_.enabled()) {
        profiler_.set_cuda_context(
            current_token_, layer, -1, id_bytes);
    }
    const size_t ids_ticket = profiler_.enabled()
        ? profiler_.cuda_begin(
            GpuStage::D2H, static_cast<void*>(stream))
        : static_cast<size_t>(-1);
    if (!DEE_CUDA_CHECK_NAMED(
            cudaMemcpyAsync(
                h_moe_expert_ids_i64_,
                d_expert_ids,
                id_bytes,
                cudaMemcpyDeviceToHost,
                stream),
            "cudaMemcpyAsync(combined expert IDs D2H)")) {
        set_last_error("combined expert ID D2H submission failed");
        return false;
    }
    if (profiler_.enabled() &&
        !profiler_.cuda_end(ids_ticket, static_cast<void*>(stream))) {
        set_last_error("combined expert ID timing event failed");
        return false;
    }
    const auto ids_wait_begin = profiler_.enabled()
        ? StageProfiler::now() : StageProfiler::TimePoint{};
    if (!DEE_CUDA_CHECK_NAMED(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(combined expert IDs)")) {
        set_last_error("combined expert ID synchronization failed");
        return false;
    }
    if (profiler_.enabled()) {
        const auto ids_wait_end = StageProfiler::now();
        profiler_.add_cpu_ms(
            CpuStage::Synchronization,
            std::chrono::duration<double, std::milli>(
                ids_wait_end - ids_wait_begin).count());
        profiler_.note_host_synchronization();
        profiler_.note_host_wait(
            HostWaitReason::ComputeBatch,
            ids_wait_begin,
            ids_wait_end,
            current_token_,
            layer);
    }
    // The pointer table is pinned host memory and expert cache blocks are
    // shared across calls. Retire their previous compute-stream use after the
    // current ID copy has had a chance to overlap it, but before any cache
    // staging or pointer-table overwrite.
    if (!drain_pointer_batch_pending()) {
        set_last_error("failed to retire prior pointer-batched expert work");
        return false;
    }

    std::vector<int> host_ids(selections);
    for (size_t index = 0; index < selections; ++index) {
        const int64_t expert = h_moe_expert_ids_i64_[index];
        if (expert < 0 || expert >= cfg_.num_experts) {
            set_last_error(
                "combined device expert ID outside configured range");
            return false;
        }
        host_ids[index] = static_cast<int>(expert);
    }
    // The external-stream synchronization above also proves the previous use
    // of this persistent raw workspace is complete before a capacity growth.
    if (!ensure_combined_raw_capacity(selections)) {
        set_last_error("failed to allocate combined raw workspace");
        return false;
    }
    const bool experts_ok = pointer_batched_experts
        ? moe_forward_pointer_batched_device_impl(
            layer, d_h_in, tokens, host_ids.data(), topk, d_moe_raw_f32_)
        : moe_forward_batch_device_impl(
            layer,
            d_h_in,
            tokens,
            host_ids.data(),
            topk,
            d_moe_raw_f32_,
            false,
            direct_single_row_io);
    if (!experts_ok) {
        if (last_error_message_.empty()) {
            set_last_error("raw expert stage failed inside combined path");
        }
        return false;
    }
    if (!weighted_combine_fp16_cuda(
            d_moe_raw_f32_,
            d_weights_f32,
            d_expert_ids,
            d_output_f16,
            tokens,
            topk,
            hidden_,
            compute_stream_,
            profiler_.enabled() ? &profiler_ : nullptr)) {
        set_last_error("exact FP16 weighted combine launch failed");
        return false;
    }
    if (d_raw_trace_out &&
        !DEE_CUDA_CHECK_NAMED(
            cudaMemcpyAsync(
                d_raw_trace_out,
                d_moe_raw_f32_,
                selections * static_cast<size_t>(hidden_) * sizeof(float),
                cudaMemcpyDeviceToDevice,
                compute_stream_),
            "cudaMemcpyAsync(combined raw trace)")) {
        set_last_error("combined raw trace copy failed");
        return false;
    }
    if (!DEE_CUDA_CHECK_NAMED(
            cudaEventRecord(
                combined_output_ready_event_, compute_stream_),
            "cudaEventRecord(combined output ready)") ||
        !DEE_CUDA_CHECK_NAMED(
            cudaStreamWaitEvent(
                stream, combined_output_ready_event_, 0),
            "cudaStreamWaitEvent(combined output ready)")) {
        set_last_error("combined output stream handoff failed");
        return false;
    }
    if (profiler_.enabled()) {
        profiler_.note_stream_wait();
        profiler_.add_layer_latency(combined_layer_begin);
    }
    return true;
#else
    (void)layer;
    (void)d_raw_trace_out;
    (void)external_stream;
    (void)direct_single_row_io;
    (void)pointer_batched_experts;
    return false;
#endif
}

EngineStats Engine::runtime_stats() const {
    EngineStats result = stats_;
    const auto& cs = cache_.stats();
    const auto& ps = prefetcher_.stats();
    result.cache_hits = cs.hits;
    result.cache_loads = cs.loads;
    result.evictions = cs.evictions;
    result.fallbacks = cs.fallbacks;
    result.prefetch_issued = ps.requests;
    result.prefetch_fallbacks = ps.fallbacks;
    result.resident_hits = ps.resident_hits;
    result.inflight_hits = ps.inflight_hits;
    result.cold_loads = ps.cold_loads;
    result.duplicate_requests = ps.duplicate_requests;
    result.h2d_bytes = ps.h2d_bytes;
    result.h2d_copies = ps.h2d_copies;
    const auto& hp = pack_cache_.stats();
    result.host_pack_hits = hp.hits;
    result.host_pack_misses = hp.misses;
    result.host_pack_evictions = hp.evictions;
    result.host_pack_bytes = hp.bytes;
    result.host_pack_entries = hp.entries;
    result.current_vram = cache_.used_bytes();
    result.resident_experts = cache_.resident_count();
    result.peak_vram = std::max(result.peak_vram, result.current_vram);
    result.host_pinned_expert_staging_bytes = pinned_staging_bytes_;
    for (const auto& entry : staging_) {
        result.host_pageable_expert_staging_bytes +=
            entry.second.size() * sizeof(float);
    }
    for (const auto& entry : staging_bf16_) {
        result.host_pageable_expert_staging_bytes +=
            entry.second.size() * sizeof(uint16_t);
    }
    for (const auto& entry : staging_int8_) {
        result.host_pageable_expert_staging_bytes += entry.second.host.size();
    }
    for (const auto& entry : router_weights_) {
        result.host_router_weight_bytes += entry.second.size() * sizeof(float);
    }
    result.host_hidden_buffer_bytes =
        (hidden_buf_[0].size() + hidden_buf_[1].size()) * sizeof(float);
    result.host_prefetch_ring_bytes = prefetcher_.pinned_staging_bytes();
    result.host_prefetch_ring_slots = prefetcher_.staging_slot_count();
    result.peak_transient_host_bytes = peak_transient_host_bytes_;
    result.device_expert_cache_reserved_bytes = cache_.budget_bytes();
    result.device_prefetch_staging_bytes = prefetcher_.device_staging_bytes();
#ifdef DEE_CUDA
    if (cfg_.use_cuda) {
        result.host_moe_dispatch_bytes = h_moe_expert_ids_i64_
            ? h_moe_expert_ids_capacity_ * sizeof(int64_t) : 0;
        result.host_moe_pointer_table_bytes = h_moe_pointer_table_
            ? 8 * moe_pointer_batch_capacity_selections_ * sizeof(void*)
            : 0;
        if (d_h_in_) result.device_fixed_work_buffer_bytes +=
            static_cast<size_t>(hidden_) * sizeof(float);
        if (d_h_out_) result.device_fixed_work_buffer_bytes +=
            static_cast<size_t>(hidden_) * sizeof(float);
        if (d_hbuf_) result.device_fixed_work_buffer_bytes +=
            static_cast<size_t>(inter_) * sizeof(float);
        if (d_ubuf_) result.device_fixed_work_buffer_bytes +=
            static_cast<size_t>(inter_) * sizeof(float);
        if (d_ybuf_) result.device_fixed_work_buffer_bytes +=
            static_cast<size_t>(cfg_.topk) * hidden_ * sizeof(float);
        if (d_h_in_half_) result.device_fixed_work_buffer_bytes +=
            static_cast<size_t>(hidden_) * sizeof(uint16_t);
        if (d_activation_half_) result.device_fixed_work_buffer_bytes +=
            static_cast<size_t>(inter_) * sizeof(uint16_t);
        if (d_router_weight_half_) result.device_router_weight_bytes =
            static_cast<size_t>(cfg_.num_experts) * hidden_ * sizeof(uint16_t);
        if (router_capacity_tokens_ > 0) {
            if (d_router_input_) result.device_router_dynamic_bytes +=
                router_capacity_tokens_ * hidden_ * sizeof(float);
            if (d_router_input_half_) result.device_router_dynamic_bytes +=
                router_capacity_tokens_ * hidden_ * sizeof(uint16_t);
            if (d_router_logits_half_) result.device_router_dynamic_bytes +=
                router_capacity_tokens_ * static_cast<size_t>(cfg_.num_experts) *
                sizeof(uint16_t);
        }
        if (moe_batch_capacity_tokens_ > 0) {
            const size_t input_elements = moe_batch_capacity_tokens_ *
                static_cast<size_t>(hidden_);
            const size_t inter_elements = moe_batch_capacity_tokens_ *
                static_cast<size_t>(inter_);
            if (d_moe_batch_input_) result.device_moe_batch_buffer_bytes +=
                input_elements * sizeof(float);
            if (d_moe_batch_input_half_) result.device_moe_batch_buffer_bytes +=
                input_elements * sizeof(uint16_t);
            if (d_moe_batch_gate_half_) result.device_moe_batch_buffer_bytes +=
                inter_elements * sizeof(uint16_t);
            if (d_moe_batch_up_half_) result.device_moe_batch_buffer_bytes +=
                inter_elements * sizeof(uint16_t);
            if (d_moe_batch_activation_half_) result.device_moe_batch_buffer_bytes +=
                inter_elements * sizeof(uint16_t);
            if (d_moe_batch_output_) result.device_moe_batch_buffer_bytes +=
                input_elements * sizeof(float);
        }
        if (d_moe_raw_f32_) {
            result.device_moe_raw_workspace_bytes =
                moe_raw_capacity_selections_ *
                static_cast<size_t>(hidden_) * sizeof(float);
        }
        if (moe_pointer_batch_capacity_selections_ > 0) {
            const size_t inter_elements =
                moe_pointer_batch_capacity_selections_ *
                static_cast<size_t>(inter_);
            if (d_moe_pointer_batch_gate_half_) {
                result.device_moe_pointer_batch_workspace_bytes +=
                    inter_elements * sizeof(uint16_t);
            }
            if (d_moe_pointer_batch_up_half_) {
                result.device_moe_pointer_batch_workspace_bytes +=
                    inter_elements * sizeof(uint16_t);
            }
            if (d_moe_pointer_batch_activation_half_) {
                result.device_moe_pointer_batch_workspace_bytes +=
                    inter_elements * sizeof(uint16_t);
            }
            if (d_moe_pointer_table_) {
                result.device_moe_pointer_batch_workspace_bytes +=
                    8 * moe_pointer_batch_capacity_selections_ *
                    sizeof(void*);
            }
        }
        if (d_oracle_scratch_) result.device_oracle_scratch_bytes =
            static_cast<size_t>(256 + 256) * sizeof(float);
    }
#endif
    return result;
}

uintptr_t Engine::compute_stream_handle() const {
#ifdef DEE_CUDA
    return reinterpret_cast<uintptr_t>(compute_stream_);
#else
    return 0;
#endif
}

bool Engine::reset_runtime_cache() {
#ifdef DEE_CUDA
    if (cfg_.use_cuda) {
        if (!DEE_CUDA_CHECK_NAMED(cudaSetDevice(cfg_.device_id),
                                  "cudaSetDevice(reset runtime cache)")) return false;
        prefetcher_.synchronize_all();
        if (compute_stream_ &&
            !DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_),
                                  "cudaStreamSynchronize(reset runtime cache)")) return false;
        if (!drain_pointer_batch_pending(true)) return false;
    }
#endif
    prefetcher_.reset();
    cache_.clear();
    cache_.reset_stats();
    prefetcher_.reset_stats();
    stats_.peak_vram = 0;
    stats_.current_vram = 0;
    stats_.resident_experts = 0;
    return true;
}

bool Engine::reset_external_profile() {
#ifdef DEE_CUDA
    if (cfg_.use_cuda) {
        if (!DEE_CUDA_CHECK_NAMED(cudaSetDevice(cfg_.device_id),
                                  "cudaSetDevice(reset external profile)")) return false;
        prefetcher_.synchronize_all();
        if (compute_stream_ &&
            !DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_),
                                  "cudaStreamSynchronize(reset external profile)")) return false;
        // The previous epoch may have completed CUDA samples that were never
        // snapshotted (for example a warmup immediately followed by a reset).
        // Resolve them against the old timeline origin before configure()
        // clears counters and begin_cuda_timeline() replaces that origin.
        if (profiler_.enabled() &&
            (!profiler_.cuda_collect_ready() ||
             profiler_.pending_cuda_samples() != 0)) {
            std::fprintf(
                stderr,
                "[profile] reset external profile left pending CUDA samples\n");
            return false;
        }
    }
#endif
    cache_.reset_stats();
    prefetcher_.reset_stats();
    stats_.profile = StageProfile{};
    current_token_ = -1;
    profiler_.configure(cfg_.profile_stages, cfg_.trace_requests,
                        cache_blob_bytes_, oracle_.num_experts(),
                        cfg_.profile_timeline);
    cache_.set_profiler(cfg_.profile_stages ? &profiler_ : nullptr);
    prefetcher_.set_profiler(cfg_.profile_stages ? &profiler_ : nullptr);
    oracle_.set_profiler(cfg_.profile_stages ? &profiler_ : nullptr);
#ifdef DEE_CUDA
    if (cfg_.use_cuda && cfg_.profile_timeline &&
        !profiler_.begin_cuda_timeline(static_cast<void*>(compute_stream_),
                                       prefetcher_.cuda_stream())) return false;
#endif
    return true;
}

StageProfile Engine::external_profile_snapshot(double total_wall_ms) {
#ifdef DEE_CUDA
    if (cfg_.use_cuda) {
        DEE_CUDA_CHECK_NAMED(cudaSetDevice(cfg_.device_id),
                             "cudaSetDevice(external profile snapshot)");
        prefetcher_.synchronize_all();
        if (compute_stream_) {
            DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_),
                                 "cudaStreamSynchronize(external profile snapshot)");
        }
        profiler_.cuda_collect_ready();
    }
#endif
    const auto& cache_stats = cache_.stats();
    const auto& prefetch_stats = prefetcher_.stats();
    return profiler_.finish(
        total_wall_ms,
        prefetch_stats.resident_hits,
        prefetch_stats.inflight_hits,
        prefetch_stats.cold_loads,
        prefetch_stats.duplicate_requests,
        cache_stats.evictions,
        cache_stats.pinned_blocks_skipped);
}

std::string Engine::external_profile_json(double total_wall_ms) {
    return stage_profile_json(external_profile_snapshot(total_wall_ms), true);
}

std::string Engine::external_timeline_json(double total_wall_ms) {
    return cuda_timeline_json(external_profile_snapshot(total_wall_ms));
}

const float* Engine::get_router_weights(int source_layer) {
    auto found = router_weights_.find(source_layer);
    if (found != router_weights_.end()) return found->second.data();
    const std::string name = "model.language_model.layers." +
        std::to_string(source_layer) + ".mlp.gate.weight";
    const TensorView view = resolver_.resolve_tensor(name);
    if (!view.ok() || view.shape.size() != 2 ||
        view.shape[0] != cfg_.num_experts || view.shape[1] != hidden_) {
        std::fprintf(stderr, "[engine] router tensor missing or has wrong shape: %s\n",
                     name.c_str());
        return nullptr;
    }
    const size_t elements = static_cast<size_t>(cfg_.num_experts) * hidden_;
    std::vector<float> weights(elements);
    if (view.dtype == DType::BF16 && view.nbytes == elements * sizeof(uint16_t)) {
        const auto* src = reinterpret_cast<const uint16_t*>(view.data);
        for (size_t i = 0; i < elements; ++i) weights[i] = bf16_to_f32(src[i]);
    } else if (view.dtype == DType::F16 && view.nbytes == elements * sizeof(uint16_t)) {
        const auto* src = reinterpret_cast<const uint16_t*>(view.data);
        for (size_t i = 0; i < elements; ++i) weights[i] = f16_to_f32(src[i]);
    } else if (view.dtype == DType::F32 && view.nbytes == elements * sizeof(float)) {
        std::memcpy(weights.data(), view.data, view.nbytes);
    } else {
        std::fprintf(stderr, "[engine] router tensor has unsupported dtype/size: %s\n",
                     name.c_str());
        return nullptr;
    }
    auto inserted = router_weights_.emplace(source_layer, std::move(weights));
    return inserted.first->second.data();
}

bool Engine::route_topk(int layer, const float* h_in, float* router_logits,
                        float* routing_weights, int* experts) {
    return route_topk_batch(layer, h_in, 1, router_logits, routing_weights, experts);
}

bool Engine::route_topk_batch(int layer, const float* h_in, int tokens,
                              float* router_logits, float* routing_weights,
                              int* experts) {
    if (!h_in || !router_logits || !routing_weights || !experts ||
        tokens <= 0 || cfg_.topk <= 0 || cfg_.topk > cfg_.num_experts) return false;
    const int source_layer = avail_layer(layer);

#ifdef DEE_CUDA
    if (cfg_.use_cuda) {
        if (!DEE_CUDA_CHECK_NAMED(cudaSetDevice(cfg_.device_id),
                                  "cudaSetDevice(router)")) return false;
        const TensorView view = resolver_.resolve_tensor(
            "model.language_model.layers." + std::to_string(source_layer) +
            ".mlp.gate.weight");
        const size_t weight_elements = static_cast<size_t>(cfg_.num_experts) * hidden_;
        if (!view.ok() || view.shape.size() != 2 ||
            view.shape[0] != cfg_.num_experts || view.shape[1] != hidden_) {
            std::fprintf(stderr, "[engine] CUDA router tensor missing or has wrong shape\n");
            return false;
        }
        if (router_weight_layer_ != source_layer) {
            if (d_router_weight_half_) {
                if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_router_weight_half_, "d_router_weight_half_"),
                                          "cudaFree(old router weight)")) return false;
                d_router_weight_half_ = nullptr;
            }
            if (!DEE_CUDA_CHECK_NAMED(
                    DEE_TA_MALLOC(&d_router_weight_half_, weight_elements * sizeof(uint16_t), "d_router_weight_half_"),
                    "cudaMalloc(router FP16 weight)")) return false;
            void* source = nullptr;
            if (!DEE_CUDA_CHECK_NAMED(DEE_TA_MALLOC(&source, view.nbytes, "source"),
                                      "cudaMalloc(router source)")) return false;
            bool converted = DEE_CUDA_CHECK_NAMED(
                cudaMemcpyAsync(source, view.data, view.nbytes, cudaMemcpyHostToDevice,
                                compute_stream_),
                "cudaMemcpyAsync(router source)");
            if (converted && view.dtype == DType::BF16 &&
                view.nbytes == weight_elements * sizeof(uint16_t)) {
                converted = bf16_to_f16_cuda(
                    static_cast<const uint16_t*>(source), d_router_weight_half_,
                    weight_elements, compute_stream_, &profiler_);
            } else if (converted && view.dtype == DType::F16 &&
                       view.nbytes == weight_elements * sizeof(uint16_t)) {
                converted = DEE_CUDA_CHECK_NAMED(
                    cudaMemcpyAsync(d_router_weight_half_, source, view.nbytes,
                                    cudaMemcpyDeviceToDevice, compute_stream_),
                    "cudaMemcpyAsync(router FP16 weight)");
            } else if (converted && view.dtype == DType::F32 &&
                       view.nbytes == weight_elements * sizeof(float)) {
                converted = f32_to_f16_cuda(
                    static_cast<const float*>(source), d_router_weight_half_,
                    weight_elements, compute_stream_, &profiler_);
            } else {
                converted = false;
                std::fprintf(stderr, "[engine] CUDA router tensor has unsupported dtype/size\n");
            }
            if (converted) {
                converted = DEE_CUDA_CHECK_NAMED(
                    cudaStreamSynchronize(compute_stream_),
                    "cudaStreamSynchronize(router weight)");
            }
            DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(source, "source"), "cudaFree(router source)");
            if (!converted) return false;
            router_weight_layer_ = source_layer;
        }
        if (router_capacity_tokens_ < static_cast<size_t>(tokens)) {
            if (d_router_input_) {
                if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_router_input_, "d_router_input_"),
                                          "cudaFree(router input)")) return false;
                d_router_input_ = nullptr;
            }
            if (d_router_input_half_) {
                if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_router_input_half_, "d_router_input_half_"),
                                          "cudaFree(router FP16 input)")) return false;
                d_router_input_half_ = nullptr;
            }
            if (d_router_logits_half_) {
                if (!DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_router_logits_half_, "d_router_logits_half_"),
                                          "cudaFree(router FP16 logits)")) return false;
                d_router_logits_half_ = nullptr;
            }
            const size_t input_elements = static_cast<size_t>(tokens) * hidden_;
            const size_t logit_elements = static_cast<size_t>(tokens) * cfg_.num_experts;
            if (!DEE_CUDA_CHECK_NAMED(DEE_TA_MALLOC(reinterpret_cast<void**>(&d_router_input_),
                                                  input_elements * sizeof(float), "d_router_input_"),
                                      "cudaMalloc(router input)") ||
                !DEE_CUDA_CHECK_NAMED(DEE_TA_MALLOC(&d_router_input_half_,
                                                  input_elements * sizeof(uint16_t), "d_router_input_half_"),
                                      "cudaMalloc(router FP16 input)") ||
                !DEE_CUDA_CHECK_NAMED(DEE_TA_MALLOC(&d_router_logits_half_,
                                                  logit_elements * sizeof(uint16_t), "d_router_logits_half_"),
                                      "cudaMalloc(router FP16 logits)")) return false;
            router_capacity_tokens_ = static_cast<size_t>(tokens);
        }
        const size_t input_elements = static_cast<size_t>(tokens) * hidden_;
        const size_t logit_elements = static_cast<size_t>(tokens) * cfg_.num_experts;
        if (!DEE_CUDA_CHECK_NAMED(
                cudaMemcpyAsync(d_router_input_, h_in, input_elements * sizeof(float),
                                cudaMemcpyHostToDevice, compute_stream_),
                "cudaMemcpyAsync(router input)") ||
            !f32_to_f16_cuda(d_router_input_, d_router_input_half_, input_elements,
                             compute_stream_, &profiler_) ||
            !router_logits_fp16_cuda(cublas_handle_, d_router_weight_half_,
                                     d_router_input_half_, d_router_logits_half_,
                                     tokens, cfg_.num_experts, hidden_, compute_stream_,
                                     &profiler_)) return false;
        std::vector<uint16_t> half_logits(logit_elements);
        if (!DEE_CUDA_CHECK_NAMED(
                cudaMemcpyAsync(half_logits.data(), d_router_logits_half_,
                                logit_elements * sizeof(uint16_t),
                                cudaMemcpyDeviceToHost, compute_stream_),
                "cudaMemcpyAsync(router logits)") ||
            !DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_),
                                  "cudaStreamSynchronize(router logits)")) return false;
        for (size_t i = 0; i < logit_elements; ++i) {
            router_logits[i] = f16_to_f32(half_logits[i]);
        }
    } else
#endif
    {
        const float* weights = get_router_weights(source_layer);
        if (!weights) return false;
        for (int token = 0; token < tokens; ++token) {
            const float* input = h_in + static_cast<size_t>(token) * hidden_;
            float* logits = router_logits + static_cast<size_t>(token) * cfg_.num_experts;
            for (int expert = 0; expert < cfg_.num_experts; ++expert) {
                const float* row = weights + static_cast<size_t>(expert) * hidden_;
                float logit = 0.0f;
                for (int h = 0; h < hidden_; ++h) logit += row[h] * input[h];
                logits[expert] = logit;
            }
        }
    }

    std::vector<float> probabilities(static_cast<size_t>(cfg_.num_experts));
    std::vector<int> order(static_cast<size_t>(cfg_.num_experts));
    for (int token = 0; token < tokens; ++token) {
        float* logits = router_logits + static_cast<size_t>(token) * cfg_.num_experts;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int expert = 0; expert < cfg_.num_experts; ++expert)
            maximum = std::max(maximum, logits[expert]);
        double denominator = 0.0;
        for (int expert = 0; expert < cfg_.num_experts; ++expert) {
            probabilities[expert] = std::exp(logits[expert] - maximum);
            denominator += probabilities[expert];
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator)) return false;
        const float inv_denominator = static_cast<float>(1.0 / denominator);
        for (float& probability : probabilities) probability *= inv_denominator;
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + cfg_.topk, order.end(),
            [&](int lhs, int rhs) {
                if (probabilities[lhs] != probabilities[rhs])
                    return probabilities[lhs] > probabilities[rhs];
                return lhs < rhs;
            });
        float selected_sum = 0.0f;
        for (int k = 0; k < cfg_.topk; ++k) selected_sum += probabilities[order[k]];
        if (!(selected_sum > 0.0f)) return false;
        for (int k = 0; k < cfg_.topk; ++k) {
            experts[static_cast<size_t>(token) * cfg_.topk + k] = order[k];
            routing_weights[static_cast<size_t>(token) * cfg_.topk + k] =
                probabilities[order[k]] / selected_sum;
        }
    }
    return true;
}

int Engine::avail_layer(int layer) const {
    // synthetic single-layer shard exposes only layer 0; map everything to it.
    // (Real multi-layer shards return `layer` directly.)
    TensorView requested = resolver_.resolve_expert(layer, 0, TensorResolver::GATE_PROJ);
    return requested.ok() ? layer : cfg_.base_layer;
}

const float* Engine::get_staging(int source_layer, int expert) {
    const auto profile_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    const uint64_t key = staging_key(source_layer, expert);
    auto it = staging_.find(key);
    if (it != staging_.end()) {
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
        return it->second.data();
    }

    TensorView gv = resolver_.resolve_expert(source_layer, expert, TensorResolver::GATE_PROJ);
    TensorView uv = resolver_.resolve_expert(source_layer, expert, TensorResolver::UP_PROJ);
    TensorView dv = resolver_.resolve_expert(source_layer, expert, TensorResolver::DOWN_PROJ);
    const size_t gh = (size_t)inter_ * hidden_;
    const size_t dh = (size_t)hidden_ * inter_;
    if (!gv.ok() || !uv.ok() || !dv.ok() || gv.dtype != DType::BF16 || uv.dtype != DType::BF16 || dv.dtype != DType::BF16 ||
        gv.nbytes != gh * sizeof(uint16_t) || uv.nbytes != gh * sizeof(uint16_t) || dv.nbytes != dh * sizeof(uint16_t)) {
        std::fprintf(stderr, "[engine] expert %d has unsupported or inconsistent BF16 tensor layout\n", expert);
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
        return nullptr;
    }

    std::vector<float> blob(blob_elems_);
    for (size_t i = 0; i < gh; ++i) blob[i] = bf16_to_f32(*(const uint16_t*)(gv.data + i * 2));
    for (size_t i = 0; i < gh; ++i) blob[gh + i] = bf16_to_f32(*(const uint16_t*)(uv.data + i * 2));
    for (size_t i = 0; i < dh; ++i) blob[2 * gh + i] = bf16_to_f32(*(const uint16_t*)(dv.data + i * 2));

    auto res = staging_.emplace(key, std::move(blob));
    if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
    return res.first->second.data();
}

const uint16_t* Engine::get_staging_bf16(int source_layer, int expert) {
    const auto profile_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    const uint64_t key = staging_key(source_layer, expert);
    auto pinned = pinned_staging_bf16_.find(key);
    if (pinned != pinned_staging_bf16_.end()) {
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
        return static_cast<const uint16_t*>(pinned->second);
    }
    auto it = staging_bf16_.find(key);
    if (it != staging_bf16_.end()) {
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
        return it->second.data();
    }

    TensorView gv = resolver_.resolve_expert(source_layer, expert, TensorResolver::GATE_PROJ);
    TensorView uv = resolver_.resolve_expert(source_layer, expert, TensorResolver::UP_PROJ);
    TensorView dv = resolver_.resolve_expert(source_layer, expert, TensorResolver::DOWN_PROJ);
    const size_t projection = static_cast<size_t>(inter_) * hidden_;
    const size_t projection_bytes = projection * sizeof(uint16_t);
    if (!gv.ok() || !uv.ok() || !dv.ok() || gv.dtype != DType::BF16 ||
        uv.dtype != DType::BF16 || dv.dtype != DType::BF16 ||
        gv.nbytes != projection_bytes || uv.nbytes != projection_bytes ||
        dv.nbytes != projection_bytes) {
        std::fprintf(stderr, "[engine] expert %d has unsupported or inconsistent BF16 tensor layout\n",
                     expert);
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
        return nullptr;
    }

#ifdef DEE_CUDA
    const size_t source_bytes = blob_elems_ * sizeof(uint16_t);
    if (cfg_.use_cuda && pinned_staging_bytes_ + source_bytes <= kPinnedStagingLimit) {
        void* allocation = nullptr;
        const auto pin_begin = profiler_.enabled()
            ? StageProfiler::now() : StageProfiler::TimePoint{};
        if (DEE_CUDA_CHECK_NAMED(
                DEE_TA_HOST_ALLOC(&allocation, source_bytes, cudaHostAllocDefault,
                                  "persistent_bf16_expert_source"),
                                 "cudaHostAlloc(persistent BF16 expert source)")) {
            if (profiler_.enabled()) profiler_.add_cpu(CpuStage::Pinning, pin_begin);
            // Tensor lookup, pin allocation, and host copying are distinct
            // non-overlapping profile categories.
            if (profiler_.enabled()) {
                profiler_.add_cpu_ms(
                    CpuStage::TensorResolution,
                    std::chrono::duration<double, std::milli>(
                        pin_begin - profile_begin).count());
            }
            const auto copy_begin = profiler_.enabled()
                ? StageProfiler::now() : StageProfiler::TimePoint{};
            auto* destination = static_cast<uint16_t*>(allocation);
            std::memcpy(destination, gv.data, projection_bytes);
            std::memcpy(destination + projection, uv.data, projection_bytes);
            std::memcpy(destination + 2 * projection, dv.data, projection_bytes);
            if (profiler_.enabled()) {
                profiler_.add_cpu(CpuStage::MmapToPinned, copy_begin);
                profiler_.note_mmap_copy(source_bytes);
            }
            DEE_TA_INSERT("pinned_staging_bf16_", key, allocation, "cudaMallocHost_or_cudaHostAlloc");  // Milestone 3 v5: assert origin tag in post-mortem
            pinned_staging_bf16_.emplace(key, allocation);
            pinned_staging_bytes_ += source_bytes;
            return destination;
        }
        if (profiler_.enabled()) profiler_.add_cpu(CpuStage::Pinning, pin_begin);
    }
#endif

    const auto prepare_begin = profiler_.enabled()
        ? StageProfiler::now() : StageProfiler::TimePoint{};
    std::vector<uint16_t> blob(blob_elems_);
    std::memcpy(blob.data(), gv.data, projection_bytes);
    std::memcpy(blob.data() + projection, uv.data, projection_bytes);
    std::memcpy(blob.data() + 2 * projection, dv.data, projection_bytes);
    auto result = staging_bf16_.emplace(key, std::move(blob));
    if (profiler_.enabled()) {
        profiler_.add_cpu(CpuStage::HostTensorPreparation, prepare_begin);
        profiler_.add_cpu_ms(
            CpuStage::TensorResolution,
            std::chrono::duration<double, std::milli>(
                prepare_begin - profile_begin).count());
    }
    return result.first->second.data();
}

const Engine::QuantizedExpert* Engine::get_staging_int8(int source_layer, int expert) {
    const auto profile_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    const uint64_t key = staging_key(source_layer, expert);
    auto existing = staging_int8_.find(key);
    if (existing != staging_int8_.end()) {
        if (profiler_.enabled()) {
            const auto resolution_end = StageProfiler::now();
            profiler_.add_cpu_ms(CpuStage::TensorResolution,
                std::chrono::duration<double, std::milli>(resolution_end - profile_begin).count());
            profiler_.note_cpu_timeline_interval(CpuTimelineKind::TensorResolution,
                profile_begin, resolution_end, current_token_, source_layer, expert);
        }
        return &existing->second;
    }

    TensorView views[3] = {
        resolver_.resolve_expert(source_layer, expert, TensorResolver::GATE_PROJ),
        resolver_.resolve_expert(source_layer, expert, TensorResolver::UP_PROJ),
        resolver_.resolve_expert(source_layer, expert, TensorResolver::DOWN_PROJ)};
    const size_t projection = static_cast<size_t>(inter_) * hidden_;
    const size_t projection_bytes = projection * sizeof(uint16_t);
    for (const TensorView& view : views) {
        if (!view.ok() || view.dtype != DType::BF16 || view.nbytes != projection_bytes) {
            std::fprintf(stderr, "[engine] expert %d has unsupported INT8 source layout\n", expert);
            if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
            return nullptr;
        }
    }

    QuantizedExpert quantized;
#ifdef DEE_CUDA
    const size_t output_bytes = blob_elems_ * sizeof(int8_t);
    if (cfg_.use_cuda && pinned_staging_bytes_ + output_bytes <= kPinnedStagingLimit) {
        if (DEE_CUDA_CHECK_NAMED(
                DEE_TA_HOST_ALLOC(&quantized.pinned, output_bytes,
                                  cudaHostAllocDefault,
                                  "persistent_int8_expert_source"),
                                 "cudaHostAlloc(persistent INT8 expert source)")) {
            pinned_staging_bytes_ += output_bytes;
        }
    }
#endif
    if (!quantized.pinned) quantized.host.resize(blob_elems_);
    auto* destination = quantized.pinned
        ? static_cast<int8_t*>(quantized.pinned) : quantized.host.data();
    const auto quantize_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    if (profiler_.enabled()) {
        profiler_.add_cpu_ms(CpuStage::TensorResolution,
            std::chrono::duration<double, std::milli>(quantize_begin - profile_begin).count());
        profiler_.note_cpu_timeline_interval(CpuTimelineKind::TensorResolution,
            profile_begin, quantize_begin, current_token_, source_layer, expert);
    }
    for (size_t region = 0; region < 3; ++region) {
        const auto* source = reinterpret_cast<const uint16_t*>(views[region].data);
        int8_t* output = destination + region * projection;
        quantized.scales[region] = quantize_bf16_projection(source, output, projection);
    }
    if (profiler_.enabled()) {
        const auto quantize_end = StageProfiler::now();
        profiler_.add_cpu_ms(CpuStage::MmapToPinned,
            std::chrono::duration<double, std::milli>(quantize_end - quantize_begin).count());
        profiler_.note_cpu_timeline_interval(CpuTimelineKind::FirstTouchQuantization,
            quantize_begin, quantize_end, current_token_, source_layer, expert,
            blob_elems_ * sizeof(uint16_t));
        profiler_.note_mmap_copy(blob_elems_ * sizeof(uint16_t));
    }
    if (quantized.pinned) {
        DEE_TA_INSERT("staging_int8_", key, quantized.pinned, "cudaHostAlloc");
    }
    auto inserted = staging_int8_.emplace(key, std::move(quantized));
    return &inserted.first->second;
}

const Engine::QuantizedExpert* Engine::get_staging_int4(int source_layer, int expert) {
    const auto profile_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    const uint64_t key = staging_key(source_layer, expert);
    auto existing = staging_int8_.find(key);
    if (existing != staging_int8_.end()) {
        if (profiler_.enabled()) {
            const auto resolution_end = StageProfiler::now();
            profiler_.add_cpu_ms(CpuStage::TensorResolution,
                std::chrono::duration<double, std::milli>(resolution_end - profile_begin).count());
            profiler_.note_cpu_timeline_interval(CpuTimelineKind::TensorResolution,
                profile_begin, resolution_end, current_token_, source_layer, expert);
        }
        return &existing->second;
    }
    TensorView views[3] = {
        resolver_.resolve_expert(source_layer, expert, TensorResolver::GATE_PROJ),
        resolver_.resolve_expert(source_layer, expert, TensorResolver::UP_PROJ),
        resolver_.resolve_expert(source_layer, expert, TensorResolver::DOWN_PROJ)};
    const size_t projection = static_cast<size_t>(inter_) * hidden_;
    const size_t projection_bytes = projection * sizeof(uint16_t);
    for (const TensorView& view : views) {
        if (!view.ok() || view.dtype != DType::BF16 || view.nbytes != projection_bytes) {
            std::fprintf(stderr, "[engine] expert %d has unsupported INT4 source layout\n", expert);
            if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
            return nullptr;
        }
    }
    QuantizedExpert quantized;
    const size_t packed_bytes = (blob_elems_ + 1) / 2;
#ifdef DEE_CUDA
    if (cfg_.use_cuda && pinned_staging_bytes_ + packed_bytes <= kPinnedStagingLimit) {
        if (DEE_CUDA_CHECK_NAMED(
                DEE_TA_HOST_ALLOC(&quantized.pinned, packed_bytes,
                                  cudaHostAllocDefault,
                                  "persistent_int4_expert_source"),
                                 "cudaHostAlloc(persistent INT4 expert source)")) {
            pinned_staging_bytes_ += packed_bytes;
        }
    }
#endif
    if (!quantized.pinned) quantized.host.resize(packed_bytes);
    auto* destination = quantized.pinned
        ? static_cast<uint8_t*>(quantized.pinned)
        : reinterpret_cast<uint8_t*>(quantized.host.data());
    const auto quantize_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    if (profiler_.enabled()) {
        profiler_.add_cpu_ms(CpuStage::TensorResolution,
            std::chrono::duration<double, std::milli>(quantize_begin - profile_begin).count());
        profiler_.note_cpu_timeline_interval(CpuTimelineKind::TensorResolution,
            profile_begin, quantize_begin, current_token_, source_layer, expert);
    }
    for (size_t region = 0; region < 3; ++region) {
        const auto* source = reinterpret_cast<const uint16_t*>(views[region].data);
        uint8_t* output = destination + (region * projection) / 2;
        quantized.scales[region] = quantize_bf16_projection_int4(source, output, projection);
    }
    if (profiler_.enabled()) {
        const auto quantize_end = StageProfiler::now();
        profiler_.add_cpu_ms(CpuStage::MmapToPinned,
            std::chrono::duration<double, std::milli>(quantize_end - quantize_begin).count());
        profiler_.note_cpu_timeline_interval(CpuTimelineKind::FirstTouchQuantization,
            quantize_begin, quantize_end, current_token_, source_layer, expert,
            blob_elems_ * sizeof(uint16_t));
        profiler_.note_mmap_copy(blob_elems_ * sizeof(uint16_t));
    }
    if (quantized.pinned) {
        DEE_TA_INSERT("staging_int8_", key, quantized.pinned, "cudaHostAlloc");
    }
    auto inserted = staging_int8_.emplace(key, std::move(quantized));
    return &inserted.first->second;
}

const Engine::QuantizedExpert* Engine::get_staging_fp4(int source_layer, int expert) {
    const auto profile_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    const uint64_t key = staging_key(source_layer, expert);
    // NOTE: both the hit and miss paths below re-consult the host pack cache
    // and refresh the six region pointers before returning, because the pack
    // LRU may have evicted (and re-materialized elsewhere) this expert's
    // buffer between calls.  Never return a cached struct without re-pointing.

    // DEEPSEEK_V4 FP4 (e2m1fn): the weights are already packed on disk
    // (dtype I8 [out, in//2]) with a matching per-block F8_E8M0 scale tensor
    // (dtype F8 [out, in//32]).  No host-side quantization: stage them verbatim
    // into one contiguous [gate_packed][up_packed][down_packed]
    // [gate_scale][up_scale][down_scale] and let the CUDA transfer stream decode.
    TensorView weights[3] = {
        resolver_.resolve_expert(source_layer, expert, TensorResolver::GATE_PROJ),
        resolver_.resolve_expert(source_layer, expert, TensorResolver::UP_PROJ),
        resolver_.resolve_expert(source_layer, expert, TensorResolver::DOWN_PROJ)};
    TensorView scales[3] = {
        resolver_.resolve_expert_scale(source_layer, expert, TensorResolver::GATE_PROJ),
        resolver_.resolve_expert_scale(source_layer, expert, TensorResolver::UP_PROJ),
        resolver_.resolve_expert_scale(source_layer, expert, TensorResolver::DOWN_PROJ)};

    QuantizedExpert quantized;
    size_t packed_total = 0;
    size_t scale_total = 0;
    for (int p = 0; p < 3; ++p) {
        const TensorView& w = weights[p];
        const TensorView& s = scales[p];
        if (!w.ok() || w.dtype != DType::I8 || w.shape.size() < 2 ||
            !s.ok() || s.dtype != DType::F8 || s.shape.size() < 2) {
            std::fprintf(stderr, "[engine] expert %d has unsupported FP4 source layout\n", expert);
            if (profiler_.enabled()) profiler_.add_cpu(CpuStage::TensorResolution, profile_begin);
            return nullptr;
        }
        // Packed I8 [out, in//2]: the stored column count is half the
        // logical input width (two e2m1fn values per byte).  The scale is
        // F8_E8M0 [out, in//32] (block size 32 on the in axis).
        quantized.fp4[p].out = static_cast<size_t>(w.shape[0]);
        quantized.fp4[p].in  = static_cast<size_t>(w.shape[1]) * 2;
        quantized.fp4[p].packed_offset = packed_total;
        packed_total += w.nbytes;
        scale_total += s.nbytes;
    }
    size_t scale_accum = 0;
    for (int p = 0; p < 3; ++p) {
        quantized.fp4[p].scale_offset = packed_total + scale_accum;
        scale_accum += scales[p].nbytes;
    }
    quantized.fp4_total_nbytes = packed_total + scale_total;

    // Stage 1 residency: keep a private RAM copy of the packed bytes (LRU,
    // bounded by cfg_.host_pack_cache_bytes) so repeated prefetch gathers hit
    // host DRAM instead of re-faulting cold mmap pages against the full
    // checkpoint.  The fill is the SAME on hit and miss: if the pack evicted
    // this key in between, it re-copies from the (cheap) resolved views.
    auto fill_pack = [&](uint8_t* dst, size_t n) {
        const TensorView* sources[6] = {
            &weights[0], &weights[1], &weights[2],
            &scales[0], &scales[1], &scales[2]};
        size_t off = 0;
        for (int r = 0; r < 6; ++r) {
            if (off + sources[r]->nbytes > n) {
                std::memset(dst, 0, n);  // corrupt: fill zero (fail closed)
                return;
            }
            std::memcpy(dst + off, sources[r]->data, sources[r]->nbytes);
            off += sources[r]->nbytes;
        }
#ifndef _WIN32
        // v10: the pack bytes now live in the host RAM LRU; drop the source
        // mmap pages from the page cache so they do not double-book RAM
        // (v9 OOM: 27.7 GiB host_pack + ~28 GiB of checkpoint pages read
        // through mmap exceeded the ~32 GiB box).  A later eviction + re-fill
        // re-faults these pages, which is exactly the pre-host_pack cost.
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            for (int r = 0; r < 6; ++r) {
                const uintptr_t start =
                    reinterpret_cast<uintptr_t>(sources[r]->data);
                const size_t len = sources[r]->nbytes;
                if (len == 0) continue;
                const uintptr_t page_mask =
                    static_cast<uintptr_t>(page_size) - 1;
                const uintptr_t aligned_start = start & ~page_mask;
                const size_t span = ((start + len + page_mask) & ~page_mask) -
                                    aligned_start;
                ::madvise(reinterpret_cast<void*>(aligned_start), span,
                          MADV_DONTNEED);
            }
        }
#endif
    };
    const uint8_t* pack_buf = pack_cache_.get(
        key, quantized.fp4_total_nbytes, fill_pack);
    if (!pack_buf) {
        std::fprintf(stderr,
            "[engine] host pack cache miss-allocation for expert (%d,%d); falling back to mmap regions\n",
            source_layer, expert);
    }

    // Metadata + region pointers for the target struct (existing entry on the
    // hit path, the fresh local copy on the miss path).
    auto existing = staging_int8_.find(key);
    QuantizedExpert* target =
        (existing != staging_int8_.end()) ? &existing->second : &quantized;
    for (int p = 0; p < 3; ++p) target->fp4[p] = quantized.fp4[p];
    target->fp4_total_nbytes = quantized.fp4_total_nbytes;
    if (pack_buf) {
        size_t off = 0;
        for (int p = 0; p < 3; ++p) {
            target->fp4_regions[p] = {pack_buf + off, weights[p].nbytes};
            target->fp4_region_nbytes[p] = weights[p].nbytes;
            off += weights[p].nbytes;
        }
        for (int p = 0; p < 3; ++p) {
            target->fp4_regions[3 + p] = {pack_buf + off, scales[p].nbytes};
            target->fp4_region_nbytes[3 + p] = scales[p].nbytes;
            off += scales[p].nbytes;
        }
    } else {
        // Fallback: verbatim mmap regions (previous behavior).
        target->fp4_regions[0] = {weights[0].data, weights[0].nbytes};
        target->fp4_regions[1] = {weights[1].data, weights[1].nbytes};
        target->fp4_regions[2] = {weights[2].data, weights[2].nbytes};
        target->fp4_regions[3] = {scales[0].data, scales[0].nbytes};
        target->fp4_regions[4] = {scales[1].data, scales[1].nbytes};
        target->fp4_regions[5] = {scales[2].data, scales[2].nbytes};
        for (int p = 0; p < 6; ++p) {
            target->fp4_region_nbytes[p] = target->fp4_regions[p].nbytes;
        }
    }
    if (profiler_.enabled()) {
        profiler_.add_cpu_ms(CpuStage::TensorResolution,
            std::chrono::duration<double, std::milli>(
                StageProfiler::now() - profile_begin).count());
        // The actual mmap->host gather is accounted for in the prefetcher's
        // cuda_submit (single copy); nothing is copied on the host here.
    }
    if (existing != staging_int8_.end()) {
        return &existing->second;
    }
    auto inserted = staging_int8_.emplace(key, std::move(quantized));
    return &inserted.first->second;
}

bool Engine::prepack_quantized_sources() {
    if (!cfg_.prepack_quantized_source || !cfg_.use_cuda ||
        cfg_.transfer_dtype == WeightTransferDType::Bf16) return true;

    std::unordered_set<int> source_layers;
    for (int layer = 0; layer < cfg_.num_layers; ++layer) {
        source_layers.insert(avail_layer(layer));
    }
    const size_t bytes_per_expert =
        cfg_.transfer_dtype == WeightTransferDType::Int4 ? (blob_elems_ + 1) / 2
        : cfg_.transfer_dtype == WeightTransferDType::Fp4E2m1
            ? blob_elems_ / 2 + (blob_elems_ + 31) / 32  // packed I8 + e8m0 scales
            : blob_elems_ * sizeof(int8_t);
    const size_t physical_experts = source_layers.size() *
                                    static_cast<size_t>(oracle_.num_experts());
    if (physical_experts > std::numeric_limits<size_t>::max() / bytes_per_expert) {
        std::fprintf(stderr, "[engine] quantized source prepack size overflow\n");
        return false;
    }
    const size_t total_bytes = physical_experts * bytes_per_expert;
    if (total_bytes > kPinnedStagingLimit) {
        if (cfg_.verbose) {
            std::fprintf(stderr,
                "[engine] quantized source prepack skipped: %zu bytes exceeds bounded %zu-byte cache\n",
                total_bytes, kPinnedStagingLimit);
        }
        return true;
    }

    const auto begin = std::chrono::steady_clock::now();
    for (int source_layer : source_layers) {
        for (int expert = 0; expert < oracle_.num_experts(); ++expert) {
            const QuantizedExpert* packed =
                cfg_.transfer_dtype == WeightTransferDType::Int4
                    ? get_staging_int4(source_layer, expert)
                    : cfg_.transfer_dtype == WeightTransferDType::Fp4E2m1
                        ? get_staging_fp4(source_layer, expert)
                        : get_staging_int8(source_layer, expert);
            if (!packed) {
                std::fprintf(stderr,
                    "[engine] quantized source prepack failed for layer %d expert %d\n",
                    source_layer, expert);
                return false;
            }
        }
    }
    stats_.quantized_prepack_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    stats_.quantized_prepack_experts = physical_experts;
    stats_.quantized_prepack_bytes = total_bytes;
    stats_.quantized_prepack_complete = true;
    return true;
}

bool Engine::stage_expert(int logical_layer, int source_layer, int expert, int priority) {
#ifdef DEE_CUDA
    // Central cache-staging boundary: any execution mode may follow the public
    // pointer-batched API. Release its completed cache pins before a different
    // path attempts eviction, including preload and legacy host entrypoints.
    if (cfg_.use_cuda && !drain_pointer_batch_pending()) {
        return false;
    }
#endif
    if (cfg_.use_cuda) {
        if (cfg_.transfer_dtype == WeightTransferDType::Int4) {
            const QuantizedExpert* quantized = get_staging_int4(source_layer, expert);
            if (!quantized) return false;
            const auto* source = quantized->pinned
                ? static_cast<const uint8_t*>(quantized->pinned)
                : reinterpret_cast<const uint8_t*>(quantized->host.data());
            return prefetcher_.prefetch_int4_to_f16(
                       source_layer, expert, source, blob_elems_,
                       static_cast<size_t>(inter_) * hidden_, quantized->scales,
                       priority, current_token_, logical_layer,
                       quantized->pinned != nullptr) >= 0;
        }
        if (cfg_.transfer_dtype == WeightTransferDType::Int8) {
            const QuantizedExpert* quantized = get_staging_int8(source_layer, expert);
            if (!quantized) return false;
            const auto* source = quantized->pinned
                ? static_cast<const int8_t*>(quantized->pinned) : quantized->host.data();
            return prefetcher_.prefetch_int8_to_f16(
                       source_layer, expert, source, blob_elems_,
                       static_cast<size_t>(inter_) * hidden_, quantized->scales,
                       priority, current_token_, logical_layer,
                       quantized->pinned != nullptr) >= 0;
        }
        if (cfg_.transfer_dtype == WeightTransferDType::Fp4E2m1) {
            const QuantizedExpert* quantized = get_staging_fp4(source_layer, expert);
            if (!quantized) return false;
            const void* region_src[6];
            size_t region_nbytes[6];
            size_t packed_offsets[3];
            size_t scale_offsets[3];
            size_t out[3];
            size_t in[3];
            for (int r = 0; r < 6; ++r) {
                region_src[r] = quantized->fp4_regions[r].data;
                region_nbytes[r] = quantized->fp4_regions[r].nbytes;
            }
            for (int p = 0; p < 3; ++p) {
                packed_offsets[p] = quantized->fp4[p].packed_offset;
                scale_offsets[p]  = quantized->fp4[p].scale_offset;
                out[p] = quantized->fp4[p].out;
                in[p]  = quantized->fp4[p].in;
            }
            return prefetcher_.prefetch_fp4_regions_to_f16(
                       source_layer, expert, region_src, region_nbytes,
                       quantized->fp4_total_nbytes,
                       packed_offsets, scale_offsets, out, in,
                       priority, current_token_, logical_layer) >= 0;
        }
        const uint16_t* blob = get_staging_bf16(source_layer, expert);
        if (!blob) {
            std::fprintf(stderr, "[engine] missing BF16 source weights for expert (%d,%d)\n",
                         source_layer, expert);
            return false;
        }
        const bool source_pinned = pinned_staging_bf16_.find(
            staging_key(source_layer, expert)) != pinned_staging_bf16_.end();
        const long transfer = cfg_.cache_dtype == DeviceCacheDType::Fp16
            ? prefetcher_.prefetch_bf16_to_f16(
                  source_layer, expert, blob, blob_elems_, priority,
                  current_token_, logical_layer, source_pinned)
            : prefetcher_.prefetch_bf16_to_f32(
                  source_layer, expert, blob, blob_elems_, priority,
                  current_token_, logical_layer, source_pinned);
        return transfer >= 0;
    }
    const float* blob = get_staging(source_layer, expert);
    if (!blob) {
        std::fprintf(stderr, "[engine] missing source weights for expert (%d,%d)\n", source_layer, expert);
        return false;
    }
    return prefetcher_.prefetch(source_layer, expert, blob, blob_bytes_, priority,
                                current_token_, logical_layer) >= 0;
}

void Engine::release_transient_bf16_sources() {
#ifdef DEE_CUDA
    size_t released_pinned = 0;
    for (const auto& entry : pinned_staging_bf16_) {
        if (entry.second) {
            DEE_CUDA_CHECK_NAMED(DEE_TA_FREE_HOST(entry.second, "entry"),
                                 "cudaFreeHost(transient BF16 expert source)");
            released_pinned += blob_elems_ * sizeof(uint16_t);
        }
    }
    pinned_staging_bf16_.clear();
    pinned_staging_bytes_ = released_pinned > pinned_staging_bytes_
        ? 0 : pinned_staging_bytes_ - released_pinned;
#endif
    // GPU cache blocks retain the converted expert. On an eviction the source
    // is reconstructed directly from mmap, bounding host memory per call.
    staging_bf16_.clear();
}

void Engine::release_transient_f32_sources() {
    // Milestone 2.5 fix (defects #3/#5): keep persistent pinned BF16 staging
    // across decode calls so first-touch pinning happens once per unique
    // expert instead of every decode step. Only the F32 staging_bf16_ fallback
    // (used by the CPU path) is dropped; host memory stays bounded by
    // kPinnedStagingLimit in get_staging_bf16. The forensic matrix measured
    // 1225 ms of synchronous cudaHostAlloc + 267 ms of mmap->pinned memcpy per
    // warm decode, dominated by the prior per-call free/re-pin churn.
    staging_bf16_.clear();
}

bool Engine::preload_all_experts() {
    std::unordered_set<int> source_layers;
    for (int layer = 0; layer < cfg_.num_layers; ++layer) source_layers.insert(avail_layer(layer));

    size_t loaded = 0;
    for (int source_layer : source_layers) {
        for (int expert = 0; expert < oracle_.num_experts(); ++expert) {
            prefetcher_.begin_batch();
            if (!stage_expert(source_layer, source_layer, expert, oracle_.num_experts() - expert) ||
                !prefetcher_.wait(source_layer, expert)) {
                std::fprintf(stderr, "[engine] failed to preload expert (%d,%d) for scenario %s\n",
                             source_layer, expert, benchmark_scenario_name(cfg_.scenario));
                return false;
            }
            if (++loaded % 64 == 0) prefetcher_.reset();
        }
    }
    prefetcher_.reset();
    return true;
}

bool Engine::prepare_profile_scenario() {
    const bool preload = cfg_.scenario == BenchmarkScenario::FullResident ||
                         cfg_.scenario == BenchmarkScenario::ResidentBypass ||
                         cfg_.scenario == BenchmarkScenario::ComputeOnly;
    if (preload) {
        // Preload is scenario setup, not measured work. Detach the profiler so
        // its bounded CUDA event pool cannot be consumed by setup transfers.
        cache_.set_profiler(nullptr);
        prefetcher_.set_profiler(nullptr);
        if (!preload_all_experts()) return false;
    }

    cache_.reset_stats();
    prefetcher_.reset_stats();
    scenario_requests_ = 0;
    scenario_resident_hits_ = 0;
    scenario_cold_loads_ = 0;
    profiler_.configure(cfg_.profile_stages, cfg_.trace_requests,
                        cache_blob_bytes_, oracle_.num_experts(), cfg_.profile_timeline);
    cache_.set_profiler(cfg_.profile_stages ? &profiler_ : nullptr);
    prefetcher_.set_profiler(cfg_.profile_stages ? &profiler_ : nullptr);
    oracle_.set_profiler(cfg_.profile_stages ? &profiler_ : nullptr);
    return true;
}

void Engine::forward_layer(int layer, const float* h_in, float* h_out) {
    std::vector<int> experts;
    const auto oracle_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    oracle_.predict(layer, h_in, cfg_.topk, experts);
    if (profiler_.enabled()) profiler_.add_cpu(CpuStage::Oracle, oracle_begin);

    // CPU/mock mode processes one expert at a time. This preserves the fixed
    // budget contract even when top-K is larger than the cache capacity.
    std::vector<float> acc(hidden_, 0.0f);
    const int source_layer = avail_layer(layer);
    profiler_.note_prediction(current_token_, layer, source_layer, experts);
    for (size_t k = 0; k < experts.size(); ++k) {
        int e = experts[k];
        prefetcher_.begin_batch();
        if (!stage_expert(layer, source_layer, e, static_cast<int>(cfg_.topk - k)) || !prefetcher_.wait(source_layer, e)) {
            stats_.fallbacks++;
            continue;
        }
        const void* p = cache_.data(source_layer, e);
        if (!p) {  // should not happen (wait guarantees resident)
            stats_.fallbacks++;
            continue;
        }
        swiglu((const float*)p, h_in, inter_, hidden_, acc.data());
        prefetcher_.mark_consumed(source_layer, e);
        cache_.touch(source_layer, e);
    }
    // 3) combine (mean over top-K) then stabilize (frozen layer-norm stand-in).
    //    The mock has no residual/LN, so without this the recurrent loop
    //    diverges to Inf over 40 layers. RMS-normalizing to 1.0 per layer keeps
    //    the signal bounded and finite while preserving the DEE pipeline logic.
    float inv = (experts.empty() ? 1.0f : 1.0f / (float)experts.size());
    double ss = 0.0;
    for (int i = 0; i < hidden_; ++i) { float v = acc[i] * inv; h_out[i] = v; ss += (double)v * v; }
    double rms = std::sqrt(ss / hidden_);
    if (rms > 1e-8) {
        float s = 1.0f / (float)rms;   // rescale RMS -> 1.0
        for (int i = 0; i < hidden_; ++i) h_out[i] *= s;
    }
}

bool Engine::init(const EngineConfig& cfg) {
    cfg_ = cfg;
    if (cfg.num_tokens <= 0 || cfg.topk <= 0 || cfg.num_layers <= 0 ||
        cfg.hidden <= 0 || cfg.base_layer < 0 || cfg.device_id < 0) {
        std::fprintf(stderr, "[engine] tokens, topk, layers, and hidden must be positive\n");
        return false;
    }
    if (cfg_.inter <= 0) cfg_.inter = 256;
    if (cfg_.num_experts <= 0) cfg_.num_experts = 256;
    hidden_ = cfg_.hidden;
    if (cfg.cache_dtype == DeviceCacheDType::Fp16 && !cfg.use_cuda) {
        std::fprintf(stderr, "[engine] FP16 device cache requires --cuda\n");
        return false;
    }
    if (cfg.transfer_dtype != WeightTransferDType::Bf16 &&
        (!cfg.use_cuda || cfg.cache_dtype != DeviceCacheDType::Fp16)) {
        std::fprintf(stderr, "[engine] INT8 transfer requires CUDA with an FP16 device cache\n");
        return false;
    }
    if (!prefetcher_.set_ring_size(cfg.prefetch_depth)) {
        std::fprintf(stderr, "[engine] invalid or late prefetch depth: %zu\n", cfg.prefetch_depth);
        return false;
    }
    // Expert dims are taken from the SHARD (so a mock inter=64 and the real
    // inter=256 are both handled). The Oracle's own MLP width (H=256) is a
    // separate, fixed quantity passed to oracle.load().
    inter_  = 256;            // provisional; overwritten below from the shard
    blob_elems_ = 3ULL * (size_t)inter_ * hidden_;
    blob_bytes_ = blob_elems_ * sizeof(float);
    cache_blob_bytes_ = cfg.cache_dtype == DeviceCacheDType::Fp16
        ? blob_elems_ * sizeof(uint16_t) : blob_bytes_;

    std::vector<std::string> shard_paths;
    if (!cfg.shard_path.empty()) shard_paths.push_back(cfg.shard_path);
    for (const std::string& path : cfg.shard_paths) {
        if (std::find(shard_paths.begin(), shard_paths.end(), path) == shard_paths.end())
            shard_paths.push_back(path);
    }
    if (shard_paths.empty()) {
        std::fprintf(stderr, "[engine] at least one shard path is required\n");
        return false;
    }
    for (const std::string& path : shard_paths) {
        if (!std::filesystem::is_regular_file(path)) {
            std::fprintf(stderr, "[engine] shard does not exist or is not a file: %s\n",
                         path.c_str());
            return false;
        }
    }
    cfg_.shard_path = shard_paths.front();
    cfg_.shard_paths = shard_paths;
    if (!mmap_.open(shard_paths.front())) {
        fprintf(stderr, "[engine] cannot open shard %s\n", shard_paths.front().c_str());
        return false;
    }
    resolver_.register_shard(&mmap_);
    for (size_t i = 1; i < shard_paths.size(); ++i) {
        auto mapping = std::make_unique<WeightMmap>();
        if (!mapping->open(shard_paths[i])) {
            std::fprintf(stderr, "[engine] cannot open shard %s\n", shard_paths[i].c_str());
            return false;
        }
        resolver_.register_shard(mapping.get());
        extra_mmaps_.push_back(std::move(mapping));
    }

    // DeepSeek-V4-Flash-0731 stores routed experts as pre-packed FP4 (I8)
    // weights + per-block F8_E8M0 scales under a different tensor naming.
    // Fp4E2m1 is that checkpoint's only transfer dtype, so key the resolver
    // dialect off it (Ornith keeps the default ORNITH naming).
    const bool deepseek_v4 = cfg.transfer_dtype == WeightTransferDType::Fp4E2m1;
    if (deepseek_v4) resolver_.set_model(TensorResolver::Model::DEEPSEEK_V4);

    // verify expert dims against the shard; derive inter_/hidden_ from it
    TensorView gv = resolver_.resolve_expert(cfg_.base_layer, 0, TensorResolver::GATE_PROJ);
    if (!gv.ok()) {
        fprintf(stderr, "[engine] cannot resolve expert 0 gate_proj in shard\n");
        return false;
    }
    if (gv.shape.size() != 2) {
        fprintf(stderr, "[engine] gate_proj shape rank != 2\n");
        return false;
    }
    // Packed FP4 gate is I8 [inter, hidden//2]: two e2m1fn values per byte, so
    // the logical input width is twice the stored column count.
    inter_  = (int)gv.shape[0];
    hidden_ = (int)(deepseek_v4 ? gv.shape[1] * 2 : gv.shape[1]);
    blob_elems_ = 3ULL * (size_t)inter_ * hidden_;
    blob_bytes_ = blob_elems_ * sizeof(float);
    cache_blob_bytes_ = cfg.cache_dtype == DeviceCacheDType::Fp16
        ? blob_elems_ * sizeof(uint16_t) : blob_bytes_;
    if (hidden_ != cfg.hidden) {
        fprintf(stderr, "[engine] shard hidden %d != configured %d\n", hidden_, cfg.hidden);
        return false;
    }
    if (!cfg.oracle_path.empty()) {
        std::ifstream oracle_file(cfg.oracle_path, std::ios::binary);
        if (!oracle_file) {
            std::fprintf(stderr, "[engine] oracle does not exist or is unreadable: %s\n", cfg.oracle_path.c_str());
            return false;
        }
        char oracle_prefix[64]{};
        oracle_file.read(oracle_prefix, sizeof(oracle_prefix) - 1);
        if (std::string(oracle_prefix).find("version https://git-lfs.github.com/spec/v1") != std::string::npos) {
            std::fprintf(stderr, "[engine] oracle is an unresolved Git LFS pointer: %s\n"
                                 "[engine] run: git lfs pull\n", cfg.oracle_path.c_str());
            return false;
        }
        if (!oracle_.load(cfg.oracle_path, hidden_, 256, 256)) {
            fprintf(stderr, "[engine] oracle load failed: %s\n", oracle_.error().c_str());
            return false;
        }
    } else {
        // Real-model integration mode: caller owns routing. Provide the
        // OracleScheduler with a no-op layer table so engine init bookkeeping
        // (num_layers, num_experts) still has correct values. forward_layer()
        // is NOT callable in this mode; routes come from the Python adapter.
        oracle_.set_no_op_layers(cfg_.num_layers, cfg_.hidden, 256, cfg_.num_experts);
        if (cfg.verbose) {
            std::fprintf(stderr,
                "[engine] real-model integration mode: oracle_path empty; routes come from Python (router=HF, %d experts/layer, %d layers)\n",
                cfg_.num_experts, cfg_.num_layers);
        }
    }
#ifdef DEE_CUDA
    if (cfg.use_cuda) {
        if (!DEE_CUDA_CHECK_NAMED(cudaSetDevice(cfg_.device_id), "cudaSetDevice(engine init)")) return false;
        if (oracle_.upload_to_gpu()) {
            const size_t scratch_elements = static_cast<size_t>(256 + 256);
            if (DEE_CUDA_CHECK_NAMED(
                    DEE_TA_MALLOC(&d_oracle_scratch_, scratch_elements * sizeof(float), "d_oracle_scratch_"),
                    "cudaMalloc(oracle scratch)")) {
                gpu_oracle_ready_ = true;
                if (cfg.verbose) {
                    std::fprintf(stderr, "[engine] GPU Oracle initialized (%zu layers)\n",
                                 static_cast<size_t>(oracle_.num_layers()));
                }
            }
        }
    }
#endif
    int nl = std::min(cfg.num_layers, oracle_.num_layers());
    cfg_.num_layers = nl;
    // VRAM budget. default: 4 experts. The arena backend is cudaMalloc when the
    // CUDA path is active, else a malloc'd host arena (mock backend).
    size_t budget = cfg.budget_bytes ? cfg.budget_bytes : (4 * blob_bytes_);
    const bool needs_full_residency = cfg_.scenario == BenchmarkScenario::FullResident ||
                                      cfg_.scenario == BenchmarkScenario::ResidentBypass ||
                                      cfg_.scenario == BenchmarkScenario::ComputeOnly;
    if (needs_full_residency) {
        std::unordered_set<int> source_layers;
        for (int layer = 0; layer < cfg_.num_layers; ++layer) source_layers.insert(avail_layer(layer));
        const size_t physical_experts = source_layers.size() * static_cast<size_t>(oracle_.num_experts());
        if (physical_experts > std::numeric_limits<size_t>::max() / cache_blob_bytes_) {
            std::fprintf(stderr, "[engine] full-resident scenario cache size overflow\n");
            return false;
        }
        budget = physical_experts * cache_blob_bytes_;
    }
    cfg_.budget_bytes = budget;
    if (cfg_.host_pack_cache_bytes == 0) {
        cfg_.host_pack_cache_bytes = 8ULL << 30;  // 8 GiB default host pack LRU
    }
    pack_cache_.set_budget(cfg_.host_pack_cache_bytes);
    Arena::Backend be;
    if (cfg.use_cuda) {
#ifdef DEE_CUDA
        be.kind = "cuda";
        be.alloc = [](size_t n) -> void* { void* p = nullptr; return DEE_CUDA_CHECK_NAMED(DEE_TA_MALLOC(&p, n, "p"), "cudaMalloc(expert cache)") ? p : nullptr; };
        be.free  = [](void* p) { if (p) DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(p, "p"), "cudaFree(expert cache)"); };
#else
        fprintf(stderr, "[engine] --cuda requested but this build has DEE_CUDA=OFF\n");
        return false;
#endif
    } else {
        be.kind = "host";
        be.alloc = [](size_t n) -> void* { return std::malloc(n); };
        be.free  = [](void* p) { std::free(p); };
    }
    if (!cache_.init(budget, be)) {
        fprintf(stderr, "[engine] cache init failed (budget %zu bytes)\n", budget);
        return false;
    }
    cache_.set_debug_validation(cfg_.debug_validate_cache);
    if (cfg_.debug_validate_cache && !cache_.validate_invariants()) {
        std::fprintf(stderr, "[engine] cache invariant validation failed after init\n");
        return false;
    }

#ifdef DEE_CUDA
    if (cfg.use_cuda) {
        if (!DEE_CUDA_CHECK_NAMED(cudaSetDevice(cfg_.device_id), "cudaSetDevice(engine buffers)")) return false;
        int device = -1;
        if (!DEE_CUDA_CHECK_NAMED(cudaGetDevice(&device), "cudaGetDevice")) return false;
        cudaDeviceProp prop{};
        if (!DEE_CUDA_CHECK_NAMED(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) return false;
        if (!DEE_CUDA_CHECK_NAMED(DEE_TA_STREAM_CREATE_FLAGS(&compute_stream_, cudaStreamNonBlocking, "compute_stream_"),
                                  "cudaStreamCreateWithFlags(compute)")) return false;
        if (!DEE_CUBLAS_CHECK_NAMED(DEE_TA_CUBLAS_CREATE(&cublas_handle_, "cublas_handle_"), "cublasCreate") ||
            !DEE_CUBLAS_CHECK_NAMED(cublasSetStream(cublas_handle_, compute_stream_), "cublasSetStream(compute)")) return false;
        auto dev_alloc = [](size_t n) -> float* { float* p = nullptr; return DEE_CUDA_CHECK_NAMED(DEE_TA_MALLOC(reinterpret_cast<void**>(&p), n, "p"), "cudaMalloc(engine work buffer)") ? p : nullptr; };
        d_h_in_  = dev_alloc((size_t)hidden_ * sizeof(float));
        d_h_out_ = dev_alloc((size_t)hidden_ * sizeof(float));
        d_hbuf_  = dev_alloc((size_t)inter_  * sizeof(float));
        d_ubuf_  = dev_alloc((size_t)inter_  * sizeof(float));
        d_ybuf_  = dev_alloc((size_t)cfg.topk * hidden_ * sizeof(float));
        if (cfg_.cache_dtype == DeviceCacheDType::Fp16) {
            d_h_in_half_ = dev_alloc((size_t)hidden_ * sizeof(uint16_t));
            d_activation_half_ = dev_alloc((size_t)inter_ * sizeof(uint16_t));
        }
        if (!d_h_in_ || !d_h_out_ || !d_hbuf_ || !d_ubuf_ || !d_ybuf_) {
            fprintf(stderr, "[engine] device work-buffer allocation failed\n");
            return false;
        }
        if (cfg_.cache_dtype == DeviceCacheDType::Fp16 &&
            (!d_h_in_half_ || !d_activation_half_)) {
            fprintf(stderr, "[engine] FP16 device work-buffer allocation failed\n");
            return false;
        }
        size_t freeB = 0, totalB = 0;
        if (DEE_CUDA_CHECK_NAMED(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo")) {
            cuda_total_ = totalB; cuda_free_ = freeB;
            stats_.cuda_total = totalB; stats_.cuda_free = freeB;
        }
        int runtime_version = 0;
        if (DEE_CUDA_CHECK_NAMED(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion")) {
            stats_.cuda_runtime_version = runtime_version;
        }
        stats_.cuda_device_name = prop.name;
        stats_.cuda_compute_major = prop.major;
        stats_.cuda_compute_minor = prop.minor;
    }
#endif

    if (!prefetcher_.init(cfg.use_cuda)) {
        fprintf(stderr, "[engine] prefetcher init failed\n");
        return false;
    }

    // This bounded persistent host representation is startup work. Keep it
    // outside measured decode and reset the stage profiler afterward so cold
    // construction is reported independently from steady-state throughput.
    if (!prepack_quantized_sources()) return false;

    profiler_.configure(cfg.profile_stages, cfg.trace_requests, cache_blob_bytes_,
                        oracle_.num_experts(), cfg.profile_timeline);
    cache_.set_profiler(cfg.profile_stages ? &profiler_ : nullptr);
    prefetcher_.set_profiler(cfg.profile_stages ? &profiler_ : nullptr);
    oracle_.set_profiler(cfg.profile_stages ? &profiler_ : nullptr);

    hidden_buf_[0].assign(hidden_, 0.0f);
    hidden_buf_[1].assign(hidden_, 0.0f);
    return true;
}

bool Engine::generate() {
    if (!prepare_profile_scenario()) return false;
#ifdef DEE_CUDA
    if (cfg_.use_cuda && cfg_.profile_timeline &&
        !profiler_.begin_cuda_timeline(static_cast<void*>(compute_stream_),
                                       prefetcher_.cuda_stream())) return false;
#endif
    auto t0 = std::chrono::steady_clock::now();

    // deterministic "prompt" embedding for token 0, layer 0 input
    std::vector<float>& h0 = hidden_buf_[0];
    for (int i = 0; i < hidden_; ++i) h0[i] = std::sin(0.01f * i) * 0.5f;

    int cur = 0, nxt = 1;
    for (int t = 0; t < cfg_.num_tokens; ++t) {
        current_token_ = t;
        const auto token_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
        float* h_in  = hidden_buf_[cur].data();
        float* h_out = hidden_buf_[nxt].data();
        for (int L = 0; L < cfg_.num_layers; ++L) {
            const auto layer_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
#ifdef DEE_CUDA
            if (cfg_.use_cuda) {
                if (!forward_layer_cuda(L, h_in, h_out)) return false;
            }
            else              forward_layer(L, h_in, h_out);
#else
            forward_layer(L, h_in, h_out);
#endif
            if (profiler_.enabled()) profiler_.add_layer_latency(layer_begin);
        }
        // autoregressive (recurrent mock): next token input = this token output
        std::swap(cur, nxt);
        // track peak VRAM (mock arena high-water / CUDA arena used)
        size_t used = cache_.used_bytes();
        if (used > stats_.peak_vram) stats_.peak_vram = used;
        if (profiler_.enabled()) profiler_.add_token_latency(token_begin);
    }

    prefetcher_.reset();   // bound inflight_/event churn between tokens

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    // gather stats
    const VramCacheManager::Stats& cs = cache_.stats();
    const AsyncPrefetcher::Stats& ps = prefetcher_.stats();
    const bool metadata_only = cfg_.scenario == BenchmarkScenario::CacheMetadataOnly;
    const uint64_t requests = metadata_only ? scenario_requests_ : ps.requests;
    const uint64_t resident_hits = metadata_only ? scenario_resident_hits_ : ps.resident_hits;
    const uint64_t inflight_hits = metadata_only ? 0 : ps.inflight_hits;
    const uint64_t cold_loads = metadata_only ? scenario_cold_loads_ : ps.cold_loads;
    if (requests != resident_hits + inflight_hits + cold_loads) {
        std::fprintf(stderr,
                     "[engine] request accounting invariant failed: requests=%llu resident=%llu inflight=%llu cold=%llu\n",
                     static_cast<unsigned long long>(requests),
                     static_cast<unsigned long long>(resident_hits),
                     static_cast<unsigned long long>(inflight_hits),
                     static_cast<unsigned long long>(cold_loads));
#ifdef DEE_CUDA_VALIDATE
        assert(requests == resident_hits + inflight_hits + cold_loads);
#endif
        return false;
    }
    if (cs.hits != resident_hits || cs.loads != cold_loads) {
        std::fprintf(stderr,
                     "[engine] cache/request accounting mismatch: cache_hits=%llu resident=%llu cache_loads=%llu cold=%llu\n",
                     static_cast<unsigned long long>(cs.hits),
                     static_cast<unsigned long long>(resident_hits),
                     static_cast<unsigned long long>(cs.loads),
                     static_cast<unsigned long long>(cold_loads));
#ifdef DEE_CUDA_VALIDATE
        assert(cs.hits == resident_hits && cs.loads == cold_loads);
#endif
        return false;
    }
    stats_.tokens = cfg_.num_tokens;
    stats_.elapsed_sec = sec;
    stats_.tok_per_sec = sec > 0 ? cfg_.num_tokens / sec : 0.0;
    stats_.cache_hits  = cs.hits;
    stats_.cache_loads = cs.loads;
    stats_.evictions   = cs.evictions;
    stats_.fallbacks   = cs.fallbacks;
    stats_.prefetch_issued = requests;
    stats_.prefetch_fallbacks = ps.fallbacks;
    stats_.resident_hits = resident_hits;
    stats_.inflight_hits = inflight_hits;
    stats_.cold_loads = cold_loads;
    stats_.duplicate_requests = ps.duplicate_requests;
#ifdef DEE_CUDA
    {
        const auto bs = oracle_.boundary_stats();
        stats_.oracle_boundary.gpu_calls = bs.gpu_calls;
        stats_.oracle_boundary.cpu_fallback_calls = bs.cpu_fallback_calls;
    }
#endif
    stats_.profile = profiler_.finish(sec * 1000.0, resident_hits, inflight_hits,
                                      cold_loads, ps.duplicate_requests, cs.evictions,
                                      cs.pinned_blocks_skipped);

    // validate output hidden is finite
    const float* out = hidden_buf_[cur].data();
    stats_.hidden_finite = true;
    stats_.final_hidden.assign(out, out + hidden_);
    for (int i = 0; i < hidden_; ++i) {
        if (!std::isfinite(out[i])) { stats_.hidden_finite = false; break; }
    }
    return true;
}

#ifdef DEE_CUDA
void Engine::cuda_cleanup() {
    if (!pointer_batch_pending_pins_.empty()) {
        bool compute_stream_synchronized = false;
        if (compute_stream_) {
            compute_stream_synchronized = DEE_CUDA_CHECK_NAMED(
                cudaStreamSynchronize(compute_stream_),
                "cudaStreamSynchronize(pointer batch teardown)");
        }
        if (compute_stream_synchronized) {
            drain_pointer_batch_pending(true);
        }
    }
    if (d_h_in_)  { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_h_in_, "d_h_in_"), "cudaFree(d_h_in)");  d_h_in_  = nullptr; }
    if (d_h_out_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_h_out_, "d_h_out_"), "cudaFree(d_h_out)"); d_h_out_ = nullptr; }
    if (d_hbuf_)  { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_hbuf_, "d_hbuf_"), "cudaFree(d_hbuf)");  d_hbuf_  = nullptr; }
    if (d_ubuf_)  { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_ubuf_, "d_ubuf_"), "cudaFree(d_ubuf)");  d_ubuf_  = nullptr; }
    if (d_ybuf_)  { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_ybuf_, "d_ybuf_"), "cudaFree(d_ybuf)");  d_ybuf_  = nullptr; }
    if (d_h_in_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_h_in_half_, "d_h_in_half_"), "cudaFree(d_h_in_half)"); d_h_in_half_ = nullptr; }
    if (d_activation_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_activation_half_, "d_activation_half_"), "cudaFree(d_activation_half)"); d_activation_half_ = nullptr; }
    if (d_router_weight_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_router_weight_half_, "d_router_weight_half_"), "cudaFree(router weight)"); d_router_weight_half_ = nullptr; }
    if (d_router_input_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_router_input_, "d_router_input_"), "cudaFree(router input)"); d_router_input_ = nullptr; }
    if (d_router_input_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_router_input_half_, "d_router_input_half_"), "cudaFree(router FP16 input)"); d_router_input_half_ = nullptr; }
    if (d_router_logits_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_router_logits_half_, "d_router_logits_half_"), "cudaFree(router FP16 logits)"); d_router_logits_half_ = nullptr; }
    router_capacity_tokens_ = 0;
    router_weight_layer_ = -1;
    if (d_moe_batch_input_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_input_, "d_moe_batch_input_"), "cudaFree(MoE batch input)"); d_moe_batch_input_ = nullptr; }
    if (d_moe_batch_input_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_input_half_, "d_moe_batch_input_half_"), "cudaFree(MoE batch FP16 input)"); d_moe_batch_input_half_ = nullptr; }
    if (d_moe_batch_gate_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_gate_half_, "d_moe_batch_gate_half_"), "cudaFree(MoE batch gate)"); d_moe_batch_gate_half_ = nullptr; }
    if (d_moe_batch_up_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_up_half_, "d_moe_batch_up_half_"), "cudaFree(MoE batch up)"); d_moe_batch_up_half_ = nullptr; }
    if (d_moe_batch_activation_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_activation_half_, "d_moe_batch_activation_half_"), "cudaFree(MoE batch activation)"); d_moe_batch_activation_half_ = nullptr; }
    if (d_moe_batch_output_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_batch_output_, "d_moe_batch_output_"), "cudaFree(MoE batch output)"); d_moe_batch_output_ = nullptr; }
    moe_batch_capacity_tokens_ = 0;
    if (d_moe_raw_f32_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_raw_f32_, "d_moe_raw_f32_"), "cudaFree(combined raw workspace)"); d_moe_raw_f32_ = nullptr; }
    moe_raw_capacity_selections_ = 0;
    if (d_moe_pointer_batch_gate_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_pointer_batch_gate_half_, "d_moe_pointer_batch_gate_half_"), "cudaFree(pointer batch gate)"); d_moe_pointer_batch_gate_half_ = nullptr; }
    if (d_moe_pointer_batch_up_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_pointer_batch_up_half_, "d_moe_pointer_batch_up_half_"), "cudaFree(pointer batch up)"); d_moe_pointer_batch_up_half_ = nullptr; }
    if (d_moe_pointer_batch_activation_half_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_pointer_batch_activation_half_, "d_moe_pointer_batch_activation_half_"), "cudaFree(pointer batch activation)"); d_moe_pointer_batch_activation_half_ = nullptr; }
    if (d_moe_pointer_table_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_moe_pointer_table_, "d_moe_pointer_table_"), "cudaFree(pointer batch table)"); d_moe_pointer_table_ = nullptr; }
    if (h_moe_pointer_table_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE_HOST(h_moe_pointer_table_, "h_moe_pointer_table_"), "cudaFreeHost(pointer batch table)"); h_moe_pointer_table_ = nullptr; }
    moe_pointer_batch_capacity_selections_ = 0;
    if (pointer_batch_complete_event_) { DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_DESTROY(pointer_batch_complete_event_, "pointer_batch_complete_event_"), "cudaEventDestroy(pointer batch complete)"); pointer_batch_complete_event_ = nullptr; }
    pointer_batch_completion_event_valid_ = false;
    pointer_batch_retirement_poisoned_ = false;
    if (h_moe_expert_ids_i64_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE_HOST(h_moe_expert_ids_i64_, "h_moe_expert_ids_i64_"), "cudaFreeHost(combined expert IDs)"); h_moe_expert_ids_i64_ = nullptr; }
    h_moe_expert_ids_capacity_ = 0;
    if (combined_output_ready_event_) { DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_DESTROY(combined_output_ready_event_, "combined_output_ready_event_"), "cudaEventDestroy(combined output ready)"); combined_output_ready_event_ = nullptr; }
    if (d_oracle_scratch_) { DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(d_oracle_scratch_, "d_oracle_scratch_"), "cudaFree(oracle_scratch)"); d_oracle_scratch_ = nullptr; }
    oracle_.free_gpu();
    if (cublas_handle_) { DEE_CUBLAS_CHECK_NAMED(DEE_TA_CUBLAS_DESTROY(cublas_handle_, "cublas_handle_"), "cublasDestroy"); cublas_handle_ = nullptr; }
    if (compute_stream_) { DEE_CUDA_CHECK_NAMED(DEE_TA_STREAM_DESTROY(compute_stream_, "compute_stream_"), "cudaStreamDestroy(compute)"); compute_stream_ = nullptr; }
}

// GPU forward: Oracle predicts experts -> H2D weight copies (secondary stream)
// -> SwiGLU kernels on the compute stream -> mean combine -> D2H. The host
// gates each kernel launch on prefetcher.wait() (cudaEventSynchronize of that
// expert's copy), so the compute stream only blocks when a weight isn't ready.
bool Engine::forward_layer_cuda(int layer, const float* h_in, float* h_out) {
    if (!DEE_CUDA_CHECK_NAMED(
            cudaSetDevice(cfg_.device_id),
            "cudaSetDevice(forward layer)") ||
        !drain_pointer_batch_pending()) {
        return false;
    }
    const auto host_scheduling_begin = profiler_.enabled()
        ? StageProfiler::now() : StageProfiler::TimePoint{};
    const auto finish_host_scheduling = [&]() {
        if (profiler_.enabled()) {
            profiler_.note_cpu_timeline(CpuTimelineKind::HostScheduling,
                                        host_scheduling_begin, current_token_, layer);
        }
    };
    std::vector<int> experts;
    if (cfg_.scenario == BenchmarkScenario::ComputeOnly) {
        experts.reserve(static_cast<size_t>(cfg_.topk));
        for (int k = 0; k < cfg_.topk; ++k) {
            experts.push_back((layer * cfg_.topk + k) % oracle_.num_experts());
        }
    } else if (gpu_oracle_ready_) {
        // Hidden must be on device before GPU Oracle runs
        const auto oracle_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
        if (cfg_.oracle_strict_margin > 0.0f) {
            // Boundary-aware GPU Oracle with strict CPU fallback for tight margins.
            // predict_gpu_boundary() does its own H2D of `h_in` and falls back
            // to the exact CPU predict() when margin < epsilon_margin.
            oracle_.predict_gpu_boundary(layer, h_in, d_h_in_, d_oracle_scratch_,
                                         cublas_handle_, static_cast<void*>(compute_stream_),
                                         cfg_.topk, experts, cfg_.oracle_strict_margin);
        } else {
            // Legacy raw GPU predict (no CPU fallback; may yield ULP-level
            // divergent routing for ~0.3% of calls).
            if (!DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(d_h_in_, h_in, (size_t)hidden_ * sizeof(float),
                                                      cudaMemcpyHostToDevice, compute_stream_),
                                      "cudaMemcpyAsync(hidden for GPU Oracle)")) return false;
            oracle_.predict_gpu(layer, d_h_in_, d_oracle_scratch_, cublas_handle_,
                                static_cast<void*>(compute_stream_), cfg_.topk, experts);
        }
        if (profiler_.enabled()) {
            const auto oracle_end = StageProfiler::now();
            profiler_.add_cpu_ms(CpuStage::Oracle,
                std::chrono::duration<double, std::milli>(oracle_end - oracle_begin).count());
        }
    } else {
        const auto oracle_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
        oracle_.predict(layer, h_in, cfg_.topk, experts);
        if (profiler_.enabled()) {
            const auto oracle_end = StageProfiler::now();
            profiler_.add_cpu_ms(CpuStage::Oracle,
                std::chrono::duration<double, std::milli>(oracle_end - oracle_begin).count());
            profiler_.note_cpu_timeline_interval(CpuTimelineKind::OracleOutput,
                oracle_begin, oracle_end, current_token_, layer);
        }
    }
    int K = (int)experts.size();
    if (K == 0) {
        for (int i = 0; i < hidden_; ++i) h_out[i] = 0.0f;
        finish_host_scheduling();
        return true;
    }

    const int source_layer = avail_layer(layer);
    profiler_.note_prediction(current_token_, layer, source_layer, experts);

    if (cfg_.scenario == BenchmarkScenario::OracleOnly) {
        std::memcpy(h_out, h_in, static_cast<size_t>(hidden_) * sizeof(float));
        finish_host_scheduling();
        return true;
    }

    if (cfg_.scenario == BenchmarkScenario::CacheMetadataOnly) {
        const int batch_size = std::max(1, static_cast<int>(cache_.budget_bytes() / cache_blob_bytes_));
        for (int first = 0; first < K; first += batch_size) {
            const int last = std::min(K, first + batch_size);
            std::vector<int> pinned;
            pinned.reserve(static_cast<size_t>(last - first));
            for (int k = first; k < last; ++k) {
                const int expert = experts[k];
                const bool resident = cache_.is_resident(source_layer, expert);
                if (!cache_.ensure(source_layer, expert, cache_blob_bytes_, cfg_.topk - k) ||
                    !cache_.pin(source_layer, expert)) return false;
                pinned.push_back(expert);
                ++scenario_requests_;
                if (resident) ++scenario_resident_hits_;
                else ++scenario_cold_loads_;
                const VramCacheManager::EnsureInfo info = cache_.last_ensure_info();
                profiler_.note_request(current_token_, layer, source_layer, expert,
                                       resident ? RequestKind::ResidentHit : RequestKind::ColdLoad,
                                       info.evicted ? info.evicted_key.layer : -1,
                                       info.evicted ? info.evicted_key.expert : -1,
                                       cfg_.topk - k,
                                       info.cache_bytes_before,
                                       info.cache_entries_before,
                                       info.cache_bytes_after,
                                       info.cache_entries_after,
                                       0, cache_blob_bytes_, 0, false,
                                       info.generation,
                                       cache_.pin_count(source_layer, expert),
                                       false, info.evicted_generation);
            }
            for (int expert : pinned) cache_.unpin(source_layer, expert);
        }
        std::memcpy(h_out, h_in, static_cast<size_t>(hidden_) * sizeof(float));
        finish_host_scheduling();
        return true;
    }

    // 1) issue all expert weight H2D copies (async, secondary stream)
    if (cfg_.scenario != BenchmarkScenario::TransferOnly) {
        if (profiler_.enabled()) {
            profiler_.set_cuda_context(current_token_, layer, -1,
                                       static_cast<size_t>(hidden_) * sizeof(float));
        }
        const size_t input_h2d_ticket = profiler_.enabled()
            ? profiler_.cuda_begin(GpuStage::H2D, static_cast<void*>(compute_stream_)) : static_cast<size_t>(-1);
        if (!DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(d_h_in_, h_in, (size_t)hidden_ * sizeof(float),
                                                  cudaMemcpyHostToDevice, compute_stream_),
                                  "cudaMemcpyAsync(hidden host to device)")) return false;
        if (profiler_.enabled()) {
            if (!profiler_.cuda_end(input_h2d_ticket, static_cast<void*>(compute_stream_))) return false;
            profiler_.note_h2d_copy(static_cast<size_t>(hidden_) * sizeof(float));
        }
        if (cfg_.cache_dtype == DeviceCacheDType::Fp16 &&
            !f32_to_f16_cuda(d_h_in_, d_h_in_half_, static_cast<size_t>(hidden_),
                             compute_stream_, nullptr)) return false;
    }

    const auto batch_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    const int batch_size = std::max(1, static_cast<int>(cache_.budget_bytes() / cache_blob_bytes_));
    if (profiler_.enabled()) profiler_.add_cpu(CpuStage::BatchConstruction, batch_begin);
    const bool bypass_cache = cfg_.scenario == BenchmarkScenario::ResidentBypass ||
                              cfg_.scenario == BenchmarkScenario::ComputeOnly;
    for (int first = 0; first < K; first += batch_size) {
        const int last = std::min(K, first + batch_size);
        if (!bypass_cache) {
            prefetcher_.begin_batch();
            for (int k = first; k < last; ++k) {
                if (!stage_expert(layer, source_layer, experts[k], cfg_.topk - k)) return false;
            }
        }
        for (int k = first; k < last; ++k) {
            const int e = experts[k];
            // Milestone 2.5 fix (defect #2): device-side wait, host-blocking
            // only as fallback (e.g. ResidentBypass/TransferOnly diagnostics).
            if (!bypass_cache &&
                !prefetcher_.wait_on_stream(source_layer, e, compute_stream_) &&
                !prefetcher_.wait(source_layer, e)) return false;
            if (cfg_.scenario == BenchmarkScenario::TransferOnly) continue;
            if (!bypass_cache && !cache_.pin(source_layer, e)) return false;
            const void* d_blob = cache_.data(source_layer, e);
            if (profiler_.enabled()) {
                profiler_.set_cuda_context(current_token_, layer, e);
            }
            const bool swiglu_ok = d_blob && (cfg_.cache_dtype == DeviceCacheDType::Fp16
                ? swiglu_expert_fp16_cuda(cublas_handle_, d_blob, d_h_in_half_, d_hbuf_, d_ubuf_,
                                           d_activation_half_, d_ybuf_ + (size_t)k * hidden_,
                                           inter_, hidden_, compute_stream_,
                                           profiler_.enabled() ? &profiler_ : nullptr)
                : swiglu_expert_cuda(cublas_handle_, static_cast<const float*>(d_blob), d_h_in_,
                                     d_hbuf_, d_ubuf_, d_ybuf_ + (size_t)k * hidden_,
                                     inter_, hidden_, compute_stream_,
                                     profiler_.enabled() ? &profiler_ : nullptr));
            if (!swiglu_ok) {
                if (!bypass_cache) cache_.unpin(source_layer, e);
                return false;
            }
            if (!bypass_cache) prefetcher_.mark_consumed(source_layer, e);
        }
        if (cfg_.scenario == BenchmarkScenario::TransferOnly) {
            // wait() has already completed every transfer in this batch, so
            // reclaim timing events without introducing another CUDA sync.
            if (profiler_.enabled() && !profiler_.cuda_collect_ready()) return false;
            continue;
        }
        // Keep cache blocks pinned until the compute stream has consumed them.
        const auto sync_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
        if (!DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_), "cudaStreamSynchronize(expert batch)")) return false;
        if (profiler_.enabled()) {
            const auto sync_end = StageProfiler::now();
            const double sync_ms = std::chrono::duration<double, std::milli>(
                sync_end - sync_begin).count();
            profiler_.add_cpu_ms(CpuStage::Synchronization, sync_ms);
            profiler_.note_host_wait(HostWaitReason::ComputeBatch, sync_begin, sync_end,
                                     current_token_, layer);
            profiler_.note_host_synchronization();
            if (!profiler_.cuda_collect_ready()) return false;
        }
        if (!bypass_cache) {
            for (int k = first; k < last; ++k) cache_.unpin(source_layer, experts[k]);
        }
    }
    if (cfg_.scenario == BenchmarkScenario::TransferOnly) {
        std::memcpy(h_out, h_in, static_cast<size_t>(hidden_) * sizeof(float));
        finish_host_scheduling();
        return true;
    }
    if (profiler_.enabled()) {
        profiler_.set_cuda_context(current_token_, layer, -1);
    }
    if (!combine_cuda(d_ybuf_, d_h_out_, K, hidden_, compute_stream_,
                      profiler_.enabled() ? &profiler_ : nullptr) ||
        !DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(h_out, d_h_out_, (size_t)hidden_ * sizeof(float),
                                              cudaMemcpyDeviceToHost, compute_stream_),
                              "cudaMemcpyAsync(hidden device to host)")) return false;
    const auto output_sync_begin = profiler_.enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    if (!DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_), "cudaStreamSynchronize(layer output)")) return false;
    if (profiler_.enabled()) {
        const auto output_sync_end = StageProfiler::now();
        const double sync_ms = std::chrono::duration<double, std::milli>(
            output_sync_end - output_sync_begin).count();
        profiler_.add_cpu_ms(CpuStage::Synchronization, sync_ms);
        profiler_.note_host_wait(HostWaitReason::LayerOutput, output_sync_begin,
                                 output_sync_end,
                                 current_token_, layer);
        profiler_.note_host_synchronization();
        if (!profiler_.cuda_collect_ready()) return false;
    }

    // 3) stabilize (frozen layer-norm stand-in) — same as CPU path
    double ss = 0.0;
    for (int i = 0; i < hidden_; ++i) ss += (double)h_out[i] * h_out[i];
    double rms = std::sqrt(ss / hidden_);
    if (rms > 1e-8) {
        float s = 1.0f / (float)rms;
        for (int i = 0; i < hidden_; ++i) h_out[i] *= s;
    }
    finish_host_scheduling();
    return true;
}

Engine::~Engine() {
    if (cfg_.use_cuda) {
        prefetcher_.synchronize_all();
        if (compute_stream_) {
            DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(compute_stream_),
                                 "cudaStreamSynchronize(engine teardown)");
        }
        if (profiler_.enabled()) profiler_.cuda_collect_ready();
        // profiler_ is declared after these borrowers, so C++ destroys it
        // before prefetcher_. AsyncPrefetcher::~AsyncPrefetcher() calls
        // reset()->synchronize_all(); leaving this pointer attached makes that
        // teardown write into StageProfiler vectors after their destruction.
        prefetcher_.set_profiler(nullptr);
        cache_.set_profiler(nullptr);
        oracle_.set_profiler(nullptr);
        for (const auto& entry : pinned_staging_bf16_) {
            DEE_CUDA_CHECK_NAMED(DEE_TA_FREE_HOST(entry.second, "entry"),
                                 "cudaFreeHost(persistent BF16 expert source)");
        }
        pinned_staging_bf16_.clear();
        for (const auto& entry : staging_int8_) {
            if (entry.second.pinned) {
                DEE_CUDA_CHECK_NAMED(DEE_TA_FREE_HOST(entry.second.pinned, "entry"),
                                     "cudaFreeHost(persistent INT8 expert source)");
            }
        }
        staging_int8_.clear();
        pinned_staging_bytes_ = 0;
        cuda_cleanup();
    }
}
#else
Engine::~Engine() = default;
#endif

size_t engine_abi_size() {
    return sizeof(Engine);
}

} // namespace dee
