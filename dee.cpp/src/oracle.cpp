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

// INT8 dot product for fast approximate Linear0 scoring.
// Uses AVX2 to compute dot(int8_weights, float_input) with scale compensation.
__attribute__((target("avx2,fma")))
float dot_product_int8_f32_avx2(const int8_t* i8w, const float* x, int count, float wscale) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int index = 0;
    for (; index + 16 <= count; index += 16) {
        // Load 16 int8 weights, sign-extend to 32-bit, convert to float
        __m128i i8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(i8w + index));
        __m256i i32_lo = _mm256_cvtepi8_epi32(i8);
        __m256i i32_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(i8, 8));
        // Convert to float and multiply by weight scale
        __m256 wlo = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_lo), _mm256_set1_ps(wscale));
        __m256 whi = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_hi), _mm256_set1_ps(wscale));
        // Multiply with input
        sum0 = _mm256_fmadd_ps(wlo, _mm256_loadu_ps(x + index), sum0);
        sum1 = _mm256_fmadd_ps(whi, _mm256_loadu_ps(x + index + 8), sum1);
    }
    __m256 sum = _mm256_add_ps(sum0, sum1);
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 combined = _mm_add_ps(low, high);
    combined = _mm_hadd_ps(combined, combined);
    combined = _mm_hadd_ps(combined, combined);
    float result = _mm_cvtss_f32(combined);
    for (; index < count; ++index) result += static_cast<float>(i8w[index]) * wscale * x[index];
    return result;
}

float dot_product_int8_f32_scalar(const int8_t* i8w, const float* x, int count, float wscale) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += static_cast<float>(i8w[i]) * wscale * x[i];
    return sum;
}

float dot_product_int8_f32(const int8_t* i8w, const float* x, int count, float wscale) {
#ifdef DEE_ORACLE_X86_TARGETS
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return dot_product_int8_f32_avx2(i8w, x, count, wscale);
    }
#endif
    return dot_product_int8_f32_scalar(i8w, x, count, wscale);
}

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
        // Quantize W0 to INT8 for fast stage-1 scoring
        float max_abs = 0.0f;
        for (float v : w.w0) max_abs = std::max(max_abs, std::fabs(v));
        w.w0_int8_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        w.w0_int8.resize(w.w0.size());
        const float inv_scale = 1.0f / w.w0_int8_scale;
        for (size_t i = 0; i < w.w0.size(); ++i) {
            const long val = std::lrint(w.w0[i] * inv_scale);
            w.w0_int8[i] = static_cast<int8_t>(std::max(-127L, std::min(127L, val)));
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
    for (int i = 0; i < k; ++i) out.push_back(idx[i]);
}

void OracleScheduler::predict_twostage(int layer, const float* hidden, int topk,
                                       int stage2_margin, std::vector<int>& out) const {
    if (oracle_profiling(profiler_)) profiler_->note_oracle_call();
    const auto& w = layers_[layer];

    // Stage 1: Fast INT8 Linear0 scores for all E_ experts
    std::vector<float> fast_scores(E_);
    for (int o = 0; o < E_; ++o) {
        const int8_t* wrow = w.w0_int8.data() + static_cast<size_t>(o) * D_;
        float acc = w.b0.empty() ? 0.f : w.b0[o];
        fast_scores[o] = dot_product_int8_f32(wrow, hidden, D_, w.w0_int8_scale) + acc;
    }

    // Select top-(K+M) candidates from fast scores
    const int num_candidates = std::min(topk + stage2_margin, E_);
    std::vector<std::pair<float, int>> ranked(E_);
    for (int i = 0; i < E_; ++i) ranked[i] = {fast_scores[i], i};
    std::partial_sort(ranked.begin(), ranked.begin() + num_candidates, ranked.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Two-stage approach: INT8 Linear0 for fast approximate h1, then
    // FP32 refinement for top candidates to minimize routing error.

    // Step 1: INT8 Linear0 for all experts
    std::vector<float> h1_hybrid(H_);
    for (int o = 0; o < H_; ++o) {
        const int8_t* wrow = w.w0_int8.data() + static_cast<size_t>(o) * D_;
        float acc = w.b0.empty() ? 0.f : w.b0[o];
        h1_hybrid[o] = dot_product_int8_f32(wrow, hidden, D_, w.w0_int8_scale) + acc;
        h1_hybrid[o] = h1_hybrid[o] > 0.f ? h1_hybrid[o] : 0.f;  // ReLU
    }

    // Step 2: Select candidates with highest absolute h1 values
    std::vector<std::pair<float, int>> h1_ranked(H_);
    for (int i = 0; i < H_; ++i) h1_ranked[i] = {std::fabs(h1_hybrid[i]), i};
    const int h1_candidates = std::min(num_candidates * 2, H_);  // need enough for Linear1 coverage
    std::partial_sort(h1_ranked.begin(), h1_ranked.begin() + h1_candidates, h1_ranked.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Step 3: Recompute FP32 Linear0 for top candidates
    std::vector<bool> is_candidate(H_, false);
    for (int c = 0; c < h1_candidates; ++c) {
        const int o = h1_ranked[c].second;
        const float* wrow = w.w0.data() + static_cast<size_t>(o) * D_;
        float acc = w.b0.empty() ? 0.f : w.b0[o];
        float val = dot_product_avx2_fma(wrow, hidden, D_) + acc;
        h1_hybrid[o] = val > 0.f ? val : 0.f;
        is_candidate[o] = true;
    }

    // Step 4: Full FP32 Linear1+Linear2
    std::vector<float> h2(H_);
    for (int o = 0; o < H_; ++o) {
        const float* wrow = w.w2.data() + static_cast<size_t>(o) * H_;
        float acc = w.b2.empty() ? 0.f : w.b2[o];
        h2[o] = dot_product_avx2_fma(wrow, h1_hybrid.data(), H_) + acc;
        h2[o] = h2[o] > 0.f ? h2[o] : 0.f;
    }

    std::vector<float> logits(E_);
    for (int o = 0; o < E_; ++o) {
        const float* wrow = w.w4.data() + static_cast<size_t>(o) * H_;
        float acc = w.b4.empty() ? 0.f : w.b4[o];
        logits[o] = dot_product_avx2_fma(wrow, h2.data(), H_) + acc;
        logits[o] = 1.0f / (1.0f + std::exp(-logits[o]));
    }

    // Step 5: Top-K from exact FP32 logits
    std::vector<int> idx(E_);
    for (int i = 0; i < E_; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    out.clear();
    int k = std::min(topk, E_);
    for (int i = 0; i < k; ++i) out.push_back(idx[i]);
}

} // namespace dee
