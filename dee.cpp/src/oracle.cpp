// dee/oracle.cpp
#include "dee/oracle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define DEE_ORACLE_X86_TARGETS 1
#endif

namespace dee {

namespace {

bool oracle_profiling(StageProfiler* profiler) {
    return profiler && profiler->enabled();
}

template <typename T>
void timed_resize(std::vector<T>& values, size_t size, StageProfiler* profiler) {
    const size_t old_capacity = values.capacity();
    const bool allocates = size > old_capacity;
    const auto begin = allocates ? StageProfiler::now() : StageProfiler::TimePoint{};
    values.resize(size);
    if (allocates && oracle_profiling(profiler)) {
        profiler->add_oracle(OracleStage::Allocation, begin);
        profiler->note_oracle_allocation((values.capacity() - old_capacity) * sizeof(T));
    }
}

using DotProduct = float (*)(const float*, const float*, int);

float dot_product_scalar(const float* weights, const float* input, int count) {
    float sum = 0.0f;
    for (int index = 0; index < count; ++index) sum += weights[index] * input[index];
    return sum;
}

#ifdef DEE_ORACLE_X86_TARGETS
__attribute__((target("avx2,fma")))
float dot_product_avx2_fma(const float* weights, const float* input, int count);

__attribute__((target("avx512f,fma")))
float dot_product_avx512_f32(const float* weights, const float* input, int count) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    int index = 0;
    for (; index + 64 <= count; index += 64) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(weights + index),
                               _mm512_loadu_ps(input + index), sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(weights + index + 16),
                               _mm512_loadu_ps(input + index + 16), sum1);
        sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(weights + index + 32),
                               _mm512_loadu_ps(input + index + 32), sum2);
        sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(weights + index + 48),
                               _mm512_loadu_ps(input + index + 48), sum3);
    }
    __m512 total = _mm512_add_ps(_mm512_add_ps(sum0, sum1),
                                 _mm512_add_ps(sum2, sum3));
    float result = _mm512_reduce_add_ps(total);
    // AVX2 tail, then scalar tail (handled by the AVX2 routine).
    if (index < count) {
        result += dot_product_avx2_fma(weights + index, input + index, count - index);
    }
    return result;
}

__attribute__((target("avx2,fma")))
float dot_product_avx2_fma(const float* weights, const float* input, int count) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int index = 0;
    for (; index + 16 <= count; index += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(weights + index),
                               _mm256_loadu_ps(input + index), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(weights + index + 8),
                               _mm256_loadu_ps(input + index + 8), sum1);
    }
    __m256 sum = _mm256_add_ps(sum0, sum1);
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 combined = _mm_add_ps(low, high);
    combined = _mm_hadd_ps(combined, combined);
    combined = _mm_hadd_ps(combined, combined);
    float result = _mm_cvtss_f32(combined);
    for (; index < count; ++index) result += weights[index] * input[index];
    return result;
}
#endif

DotProduct select_dot_product() {
#ifdef DEE_ORACLE_X86_TARGETS
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
        return &dot_product_avx512_f32;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return &dot_product_avx2_fma;
    }
#endif
    return &dot_product_scalar;
}

// Linear: y = W * x + b,  W is (out, in) row-major.
static void linear(const std::vector<float>& W, const std::vector<float>& b,
                   const float* x, int in, int out, std::vector<float>& y,
                   OracleStage stage, StageProfiler* profiler) {
    timed_resize(y, static_cast<size_t>(out), profiler);
    const auto begin = oracle_profiling(profiler) ? StageProfiler::now() : StageProfiler::TimePoint{};
    static const DotProduct dot_product = select_dot_product();
    for (int o = 0; o < out; ++o) {
        const float* wrow = W.data() + (size_t)o * in;
        float acc = b.empty() ? 0.f : b[o];
        y[o] = dot_product(wrow, x, in) + acc;
    }
    if (oracle_profiling(profiler)) profiler->add_oracle(stage, begin);
}

static inline float relu(float v) { return v > 0.f ? v : 0.f; }

}  // namespace

bool OracleScheduler::load(const std::string& oracle_pt_path, int D, int H, int E) {
    D_ = D; H_ = H; E_ = E;
    PtLoader loader;
    if (!loader.open(oracle_pt_path)) { err_ = "PtLoader: " + loader.error(); return false; }

    // Count layers by probing "layers.L.net.0.weight"
    int L = 0;
    while (loader.tensors().count("layers." + std::to_string(L) + ".net.0.weight")) ++L;
    if (L == 0) { err_ = "no layers found in oracle.pt"; return false; }

    layers_.resize(L);
    for (int l = 0; l < L; ++l) {
        std::string base = "layers." + std::to_string(l) + ".";
        auto& w = layers_[l];
        if (!loader.read_tensor(base + "net.0.weight", w.w0) ||
            !loader.read_tensor(base + "net.0.bias",   w.b0) ||
            !loader.read_tensor(base + "net.2.weight", w.w2) ||
            !loader.read_tensor(base + "net.2.bias",   w.b2) ||
            !loader.read_tensor(base + "net.4.weight", w.w4) ||
            !loader.read_tensor(base + "net.4.bias",   w.b4)) {
            err_ = "missing tensor for layer " + std::to_string(l);
            return false;
        }
    }
    return true;
}

void OracleScheduler::forward(int layer, const float* hidden, std::vector<float>& logits) const {
    const auto lookup_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    const OracleLayerWeights* weights = &layers_.at(layer);
    if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::ModelLookup, lookup_begin);
    const auto& w = *weights;
    std::vector<float> h1, h2;
    linear(w.w0, w.b0, hidden, D_, H_, h1, OracleStage::Linear0, profiler_);
    const auto relu0_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    for (auto& v : h1) v = relu(v);
    if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::Relu0, relu0_begin);
    linear(w.w2, w.b2, h1.data(), H_, H_, h2, OracleStage::Linear1, profiler_);
    const auto relu1_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    for (auto& v : h2) v = relu(v);
    if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::Relu1, relu1_begin);
    linear(w.w4, w.b4, h2.data(), H_, E_, logits, OracleStage::Linear2, profiler_);
}

void OracleScheduler::predict(int layer, const float* hidden, int topk, std::vector<int>& out) const {
    if (oracle_profiling(profiler_)) profiler_->note_oracle_call();
    std::vector<float> logits;
    forward(layer, hidden, logits);
    // argsort by descending logit
    std::vector<int> idx;
    timed_resize(idx, static_cast<size_t>(E_), profiler_);
    const auto index_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    for (int i = 0; i < E_; ++i) idx[i] = i;
    if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::TopKOutput, index_begin);
    const auto sort_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::TopKSort, sort_begin);
    out.clear();
    int k = std::min(topk, E_);
    double output_allocation_ms = 0.0;
    const auto output_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    for (int i = 0; i < k; ++i) {
        if (out.size() == out.capacity() && oracle_profiling(profiler_)) {
            const size_t old_capacity = out.capacity();
            const auto allocation_begin = StageProfiler::now();
            out.push_back(idx[i]);
            const double allocation_ms = std::chrono::duration<double, std::milli>(
                StageProfiler::now() - allocation_begin).count();
            output_allocation_ms += allocation_ms;
            profiler_->add_oracle_ms(OracleStage::Allocation, allocation_ms);
            profiler_->note_oracle_allocation((out.capacity() - old_capacity) * sizeof(int));
        } else {
            out.push_back(idx[i]);
        }
    }
    if (oracle_profiling(profiler_)) {
        const double output_ms = std::chrono::duration<double, std::milli>(
            StageProfiler::now() - output_begin).count();
        profiler_->add_oracle_ms(OracleStage::TopKOutput,
                                 std::max(0.0, output_ms - output_allocation_ms));
    }
}

void OracleScheduler::predict_and_score(int layer, const float* hidden, int topk,
                                        std::vector<int>& out,
                                        std::vector<float>& logits_out) const {
    // Identical to predict() but exposes the intermediate logits. Callers use
    // this when they need the raw scores for routing-exactness comparison.
    if (oracle_profiling(profiler_)) profiler_->note_oracle_call();
    forward(layer, hidden, logits_out);
    std::vector<int> idx;
    timed_resize(idx, static_cast<size_t>(E_), profiler_);
    const auto index_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    for (int i = 0; i < E_; ++i) idx[i] = i;
    if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::TopKOutput, index_begin);
    const auto sort_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits_out[a] > logits_out[b]; });
    if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::TopKSort, sort_begin);
    out.clear();
    int k = std::min(topk, E_);
    double output_allocation_ms = 0.0;
    const auto output_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    for (int i = 0; i < k; ++i) {
        if (out.size() == out.capacity() && oracle_profiling(profiler_)) {
            const size_t old_capacity = out.capacity();
            const auto allocation_begin = StageProfiler::now();
            out.push_back(idx[i]);
            const double allocation_ms = std::chrono::duration<double, std::milli>(
                StageProfiler::now() - allocation_begin).count();
            output_allocation_ms += allocation_ms;
            profiler_->add_oracle_ms(OracleStage::Allocation, allocation_ms);
            profiler_->note_oracle_allocation((out.capacity() - old_capacity) * sizeof(int));
        } else {
            out.push_back(idx[i]);
        }
    }
    if (oracle_profiling(profiler_)) {
        const double output_ms = std::chrono::duration<double, std::milli>(
            StageProfiler::now() - output_begin).count();
        profiler_->add_oracle_ms(OracleStage::TopKOutput,
                                 std::max(0.0, output_ms - output_allocation_ms));
    }
}

} // namespace dee
