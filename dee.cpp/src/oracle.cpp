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
__attribute__((target("avx512vnni,avx512f")))
int32_t dot_product_vnni_int8(const uint8_t* input, const int8_t* weights, int count) {
    __m512i acc = _mm512_setzero_si512();
    int index = 0;
    for (; index + 64 <= count; index += 64) {
        __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input + index));
        __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(weights + index));
        acc = _mm512_dpbusd_epi32(acc, a, b);
    }
    int32_t sum = _mm512_reduce_add_epi32(acc);
    for (; index < count; ++index) sum += static_cast<int>(input[index]) * static_cast<int>(weights[index]);
    return sum;
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

// static helpers for VNNI INT8 path
void OracleScheduler::quantize_weights(const std::vector<float>& W, int in, int out,
                                       std::vector<int8_t>& qout,
                                       std::vector<float>& scales,
                                       std::vector<int32_t>& row_sums) {
    qout.resize(static_cast<size_t>(out) * in);
    scales.resize(out);
    row_sums.resize(out);
    for (int o = 0; o < out; ++o) {
        float max_abs = 0.0f;
        for (int i = 0; i < in; ++i) {
            max_abs = std::max(max_abs, std::fabs(W[o * in + i]));
        }
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        scales[o] = scale;
        int32_t sum = 0;
        for (int i = 0; i < in; ++i) {
            int v = static_cast<int>(std::round(W[o * in + i] / scale));
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            qout[o * in + i] = static_cast<int8_t>(v);
            sum += v;
        }
        row_sums[o] = sum;
    }
}

void OracleScheduler::quantize_activation(const float* x, int n,
                                          std::vector<uint8_t>& qout,
                                          float& scale) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) max_abs = std::max(max_abs, std::fabs(x[i]));
    scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    qout.resize(n);
    for (int i = 0; i < n; ++i) {
        int v = static_cast<int>(std::round(x[i] / scale));
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        qout[i] = static_cast<uint8_t>(v + 128);
    }
}

void OracleScheduler::forward_vnni(int layer, const float* hidden,
                                   std::vector<float>& logits) const {
#ifdef DEE_ORACLE_X86_TARGETS
    const OracleLayerWeights& w = layers_[layer];
    float sx;

    // Linear0: hidden [D] -> h1 [H]
    {
        const auto linear0_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
        quantize_activation(hidden, D_, qx0_, sx);
        h1_vnni_.resize(H_);
        for (int o = 0; o < H_; ++o) {
            int32_t raw = dot_product_vnni_int8(qx0_.data(), w.w0_q.data() + o * D_, D_);
            int32_t signed_dot = raw - 128 * w.w0_r[o];
            h1_vnni_[o] = static_cast<float>(signed_dot) * sx * w.w0_s[o] + w.b0[o];
        }
        if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::Linear0, linear0_begin);
        const auto relu0_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
        for (auto& v : h1_vnni_) v = relu(v);
        if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::Relu0, relu0_begin);
    }

    // Linear1: h1 [H] -> h2 [H]
    {
        const auto linear1_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
        quantize_activation(h1_vnni_.data(), H_, qx1_, sx);
        h2_vnni_.resize(H_);
        for (int o = 0; o < H_; ++o) {
            int32_t raw = dot_product_vnni_int8(qx1_.data(), w.w2_q.data() + o * H_, H_);
            int32_t signed_dot = raw - 128 * w.w2_q_r[o];
            h2_vnni_[o] = static_cast<float>(signed_dot) * sx * w.w2_s[o] + w.b2[o];
        }
        if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::Linear1, linear1_begin);
        const auto relu1_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
        for (auto& v : h2_vnni_) v = relu(v);
        if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::Relu1, relu1_begin);
    }

    // Linear2: h2 [H] -> logits [E]
    {
        const auto linear2_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
        quantize_activation(h2_vnni_.data(), H_, qx2_, sx);
        timed_resize(logits, static_cast<size_t>(E_), profiler_);
        for (int o = 0; o < E_; ++o) {
            int32_t raw = dot_product_vnni_int8(qx2_.data(), w.w4_q.data() + o * H_, H_);
            int32_t signed_dot = raw - 128 * w.w4_q_r[o];
            logits[o] = static_cast<float>(signed_dot) * sx * w.w4_s[o] + w.b4[o];
        }
        if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::Linear2, linear2_begin);
    }
#else
    (void)layer;
    (void)hidden;
    (void)logits;
    // forward_vnni() is only invoked when use_vnni_ is true, which never
    // happens on non-x86 builds.  Keep the function defined so the class
    // layout is identical across platforms.
#endif
}

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

#ifdef DEE_ORACLE_X86_TARGETS
    use_vnni_ = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vnni");
#else
    use_vnni_ = false;
#endif

    if (use_vnni_) {
        // Pre-size reusable buffers once.
        qx0_.resize(D_);
        qx1_.resize(H_);
        qx2_.resize(H_);
        h1_vnni_.resize(H_);
        h2_vnni_.resize(H_);
        logits_buf_.reserve(static_cast<size_t>(E_));
        for (auto& w : layers_) {
            quantize_weights(w.w0, D_, H_, w.w0_q, w.w0_s, w.w0_r);
            quantize_weights(w.w2, H_, H_, w.w2_q, w.w2_s, w.w2_q_r);
            quantize_weights(w.w4, H_, E_, w.w4_q, w.w4_s, w.w4_q_r);
        }
    }

    return true;
}

void OracleScheduler::forward(int layer, const float* hidden, std::vector<float>& logits) const {
    const auto lookup_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    const OracleLayerWeights* weights = &layers_.at(layer);
    if (oracle_profiling(profiler_)) profiler_->add_oracle(OracleStage::ModelLookup, lookup_begin);
    if (use_vnni_) {
        forward_vnni(layer, hidden, logits);
        return;
    }
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
    logits_buf_.clear();
    forward(layer, hidden, logits_buf_);
    std::vector<float>& logits = logits_buf_;
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
