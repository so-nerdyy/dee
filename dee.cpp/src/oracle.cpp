// dee/oracle.cpp
//
// Implementation of OracleScheduler.
//
// PRESERVED from HEAD 31cb9e5:
//   - load(): PtLoader-based parser for the 3-layer-MLP Oracle pickle
//   - forward(): per-layer linear+relu+linear+relu+linear; AVX2/FMA dot
//   - predict(): argsort by descending logit, top-K
//   - upload_to_gpu(): cudaMalloc + cudaMemcpy(W*,b*) per layer
//   - free_gpu(): cudaFree per layer
//   - predict_gpu(): cuBLAS Sgemv/Saxpy + ReLU + D2H + CPU sigmoid + top-K
//   - predict_gpu_boundary(): exact-margin check + CPU fallback for tight cases
//   - StageProfiler integration (oracle_allocations / oracle_calls / etc.)
//
// ADDED (real-model integration mode):
//   - bool is_no_op_ flag (private)
//   - void OracleScheduler::set_no_op_layers(num_layers, D, H, E) - sizes the
//     stub layers_ table and flips is_no_op_ = true. ERR_ is cleared.
//   - inline void require_real_oracle(const char* op) - throws std::logic_error
//     with a clear remediation message.
//   - Guards in load(), forward(), predict(), upload_to_gpu(),
//     predict_gpu(), predict_gpu_boundary() that bail out in no-op mode.
//
// IMPORTANT: The is_no_op_ flag is OFF by default. Caller (Engine::init) only
// flips it via set_no_op_layers() when cfg.oracle_path is empty. The synthetic
// 31.647 tok/s baseline (which uses --oracle oracle.pt) is unaffected.

#include "dee/oracle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef DEE_CUDA
#include "dee/cuda_check.h"
#include "dee/cuda_convert.h"
#include <cuda_runtime.h>
#endif

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

}  // namespace

// Real-model integration stub: caller owns routing. Sizes the stub layers_
// table so num_layers() reports the right value; gating happens via is_no_op_
// at the top of every public method.
void OracleScheduler::set_no_op_layers(int num_layers, int D, int H, int E) {
    layers_.clear();
    layers_.resize(num_layers);  // placeholder-sized; contents are unused
    D_ = D;
    H_ = H;
    E_ = E;
    err_.clear();
    is_no_op_ = true;
#ifdef DEE_CUDA
    gpu_layers_.clear();
    bstats_ = {};
#endif
}

bool OracleScheduler::load(const std::string& oracle_pt_path, int D, int H, int E) {
    is_no_op_ = false;  // any successful load promotes us back to real-oracle mode
    D_ = D; H_ = H; E_ = E;
    PtLoader loader;
    if (!loader.open(oracle_pt_path)) { err_ = "PtLoader: " + loader.error(); return false; }

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
    require_real_oracle("forward");
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
    require_real_oracle("predict");
    if (oracle_profiling(profiler_)) profiler_->note_oracle_call();
    std::vector<float> logits;
    forward(layer, hidden, logits);
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

#ifdef DEE_CUDA

bool OracleScheduler::upload_to_gpu() {
    if (is_no_op_) {
        err_ = "upload_to_gpu called in no-op oracle mode";
        return false;
    }
    gpu_layers_.resize(layers_.size());

    for (size_t l = 0; l < layers_.size(); ++l) {
        const auto& cpu = layers_[l];
        auto& gpu = gpu_layers_[l];

        auto alloc_and_copy = [](float*& d_ptr, const std::vector<float>& src, const char* name) -> bool {
            const size_t bytes = src.size() * sizeof(float);
            if (!DEE_CUDA_CHECK_NAMED(cudaMalloc(&d_ptr, bytes), name)) return false;
            if (!DEE_CUDA_CHECK_NAMED(cudaMemcpy(d_ptr, src.data(), bytes, cudaMemcpyHostToDevice),
                                      name)) return false;
            return true;
        };

        if (!alloc_and_copy(gpu.d_w0, cpu.w0, "cudaMalloc(W0)") ||
            !alloc_and_copy(gpu.d_b0, cpu.b0, "cudaMalloc(b0)") ||
            !alloc_and_copy(gpu.d_w2, cpu.w2, "cudaMalloc(W2)") ||
            !alloc_and_copy(gpu.d_b2, cpu.b2, "cudaMalloc(b2)") ||
            !alloc_and_copy(gpu.d_w4, cpu.w4, "cudaMalloc(W4)") ||
            !alloc_and_copy(gpu.d_b4, cpu.b4, "cudaMalloc(b6)")) {
            free_gpu();
            return false;
        }
    }
    return true;
}

void OracleScheduler::free_gpu() {
    if (is_no_op_) return;
    for (auto& gpu : gpu_layers_) {
        auto safe_free = [](float*& p) {
            if (p) { cudaFree(p); p = nullptr; }
        };
        safe_free(gpu.d_w0); safe_free(gpu.d_b0);
        safe_free(gpu.d_w2); safe_free(gpu.d_b2);
        safe_free(gpu.d_w4); safe_free(gpu.d_b4);
    }
    gpu_layers_.clear();
}

void OracleScheduler::predict_gpu(int layer, const float* d_hidden, float* d_scratch,
                                  cublasHandle_t handle, void* stream,
                                  int topk, std::vector<int>& out) const {
    require_real_oracle("predict_gpu");
    if (oracle_profiling(profiler_)) profiler_->note_oracle_call();
    const auto& gpu = gpu_layers_[layer];
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    DEE_CUBLAS_CHECK_NAMED(cublasSetStream(handle, cuda_stream),
                           "cublasSetStream(oracle)");

    const float alpha = 1.0f, beta = 0.0f;
    const int D = D_, H = H_, E = E_;
    float* d_act = d_scratch;
    float* d_act2 = d_scratch + H;

    DEE_CUBLAS_CHECK_NAMED(cublasSgemv(handle, CUBLAS_OP_T, D, H,
                                       &alpha, gpu.d_w0, D, d_hidden, 1,
                                       &beta, d_act, 1), "cublasSgemv(W0)");
    DEE_CUBLAS_CHECK_NAMED(cublasSaxpy(handle, H, &alpha, gpu.d_b0, 1, d_act, 1),
                           "cublasSaxpy(b0)");
    oracle_relu_cuda(d_act, H, cuda_stream);

    DEE_CUBLAS_CHECK_NAMED(cublasSgemv(handle, CUBLAS_OP_T, H, H,
                                       &alpha, gpu.d_w2, H, d_act, 1,
                                       &beta, d_act2, 1), "cublasSgemv(W2)");
    DEE_CUBLAS_CHECK_NAMED(cublasSaxpy(handle, H, &alpha, gpu.d_b2, 1, d_act2, 1),
                           "cublasSaxpy(b2)");
    oracle_relu_cuda(d_act2, H, cuda_stream);

    DEE_CUBLAS_CHECK_NAMED(cublasSgemv(handle, CUBLAS_OP_T, H, H,
                                       &alpha, gpu.d_w4, H, d_act2, 1,
                                       &beta, d_act, 1), "cublasSgemv(W4)");
    DEE_CUBLAS_CHECK_NAMED(cublasSaxpy(handle, E, &alpha, gpu.d_b4, 1, d_act, 1),
                           "cublasSaxpy(b4)");

    std::vector<float> logits(E);
    const auto d2h_begin = oracle_profiling(profiler_) ? StageProfiler::now() : StageProfiler::TimePoint{};
    if (!DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(logits.data(), d_act,
                                              static_cast<size_t>(E) * sizeof(float),
                                              cudaMemcpyDeviceToHost, cuda_stream),
                              "cudaMemcpyAsync(oracle logits D2H)")) return;
    DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize(oracle)");
    if (oracle_profiling(profiler_)) {
        profiler_->add_oracle(OracleStage::Linear0, d2h_begin);
    }

    for (int i = 0; i < E; ++i) logits[i] = 1.0f / (1.0f + std::exp(-logits[i]));

    std::vector<int> idx(E);
    for (int i = 0; i < E; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });

    out.clear();
    int k = std::min(topk, E);
    for (int i = 0; i < k; ++i) out.push_back(idx[i]);
}

void OracleScheduler::predict_gpu_boundary(int layer, const float* h_in_cpu, float* d_hidden,
                                           float* d_scratch, cublasHandle_t handle, void* stream,
                                           int topk, std::vector<int>& out,
                                           float epsilon_margin) const {
    require_real_oracle("predict_gpu_boundary");
    if (oracle_profiling(profiler_)) profiler_->note_oracle_call();
    bstats_.gpu_calls++;

    if (!(epsilon_margin > 0.0f)) {
        predict_gpu(layer, d_hidden, d_scratch, handle, stream, topk, out);
        return;
    }

    const auto& gpu = gpu_layers_[layer];
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    DEE_CUBLAS_CHECK_NAMED(cublasSetStream(handle, cuda_stream),
                           "cublasSetStream(oracle boundary)");

    const float alpha = 1.0f, beta = 0.0f;
    const int D = D_, H = H_, E = E_;
    float* d_act = d_scratch;
    float* d_act2 = d_scratch + H;

    if (!DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(d_hidden, h_in_cpu,
                                              static_cast<size_t>(D) * sizeof(float),
                                              cudaMemcpyHostToDevice, cuda_stream),
                              "cudaMemcpyAsync(oracle hidden H2D)")) return;

    DEE_CUBLAS_CHECK_NAMED(cublasSgemv(handle, CUBLAS_OP_T, D, H,
                                       &alpha, gpu.d_w0, D, d_hidden, 1,
                                       &beta, d_act, 1), "cublasSgemv(W0)");
    DEE_CUBLAS_CHECK_NAMED(cublasSaxpy(handle, H, &alpha, gpu.d_b0, 1, d_act, 1),
                           "cublasSaxpy(b0)");
    oracle_relu_cuda(d_act, H, cuda_stream);

    DEE_CUBLAS_CHECK_NAMED(cublasSgemv(handle, CUBLAS_OP_T, H, H,
                                       &alpha, gpu.d_w2, H, d_act, 1,
                                       &beta, d_act2, 1), "cublasSgemv(W2)");
    DEE_CUBLAS_CHECK_NAMED(cublasSaxpy(handle, H, &alpha, gpu.d_b2, 1, d_act2, 1),
                           "cublasSaxpy(b2)");
    oracle_relu_cuda(d_act2, H, cuda_stream);

    DEE_CUBLAS_CHECK_NAMED(cublasSgemv(handle, CUBLAS_OP_T, H, H,
                                       &alpha, gpu.d_w4, H, d_act2, 1,
                                       &beta, d_act, 1), "cublasSgemv(W4)");
    DEE_CUBLAS_CHECK_NAMED(cublasSaxpy(handle, E, &alpha, gpu.d_b4, 1, d_act, 1),
                           "cublasSaxpy(b4)");

    std::vector<float> logits(E);
    if (!DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(logits.data(), d_act,
                                              static_cast<size_t>(E) * sizeof(float),
                                              cudaMemcpyDeviceToHost, cuda_stream),
                              "cudaMemcpyAsync(oracle logits D2H)")) return;
    DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize(oracle boundary)");

    for (int i = 0; i < E; ++i) logits[i] = 1.0f / (1.0f + std::exp(-logits[i]));

    std::vector<int> idx(E);
    for (int i = 0; i < E; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });

    bool ambiguous = false;
    const int K = std::min(topk, E);
    for (int i = 0; i + 1 < K; ++i) {
        const float margin = logits[idx[i]] - logits[idx[i + 1]];
        if (!(margin >= epsilon_margin)) { ambiguous = true; break; }
    }
    if (!ambiguous && K < E) {
        const float margin = logits[idx[K - 1]] - logits[idx[K]];
        if (!(margin >= epsilon_margin)) ambiguous = true;
    }

    if (ambiguous) {
        bstats_.cpu_fallback_calls++;
        predict(layer, h_in_cpu, topk, out);
        return;
    }

    out.clear();
    for (int i = 0; i < K; ++i) out.push_back(idx[i]);
}

#endif

} // namespace dee
