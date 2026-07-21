// dee/oracle.cpp
#include "dee/oracle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

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

// Linear: y = W * x + b,  W is (out, in) row-major.
static void linear(const std::vector<float>& W, const std::vector<float>& b,
                   const float* x, int in, int out, std::vector<float>& y,
                   OracleStage stage, StageProfiler* profiler) {
    timed_resize(y, static_cast<size_t>(out), profiler);
    const auto begin = oracle_profiling(profiler) ? StageProfiler::now() : StageProfiler::TimePoint{};
    for (int o = 0; o < out; ++o) {
        const float* wrow = W.data() + (size_t)o * in;
        float acc = b.empty() ? 0.f : b[o];
        float s = 0.f;
        for (int j = 0; j < in; ++j) s += wrow[j] * x[j];
        y[o] = s + acc;
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

} // namespace dee
