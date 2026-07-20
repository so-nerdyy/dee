// dee/oracle.cpp
#include "dee/oracle.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dee {

// Linear: y = W * x + b,  W is (out, in) row-major.
static void linear(const std::vector<float>& W, const std::vector<float>& b,
                   const float* x, int in, int out, std::vector<float>& y) {
    y.resize(out);
    for (int o = 0; o < out; ++o) {
        const float* wrow = W.data() + (size_t)o * in;
        float acc = b.empty() ? 0.f : b[o];
        float s = 0.f;
        for (int j = 0; j < in; ++j) s += wrow[j] * x[j];
        y[o] = s + acc;
    }
}

static inline float relu(float v) { return v > 0.f ? v : 0.f; }

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
    const auto& w = layers_.at(layer);
    std::vector<float> h1, h2;
    linear(w.w0, w.b0, hidden, D_, H_, h1);
    for (auto& v : h1) v = relu(v);
    linear(w.w2, w.b2, h1.data(), H_, H_, h2);
    for (auto& v : h2) v = relu(v);
    linear(w.w4, w.b4, h2.data(), H_, E_, logits);
}

void OracleScheduler::predict(int layer, const float* hidden, int topk, std::vector<int>& out) const {
    std::vector<float> logits;
    forward(layer, hidden, logits);
    // argsort by descending logit
    std::vector<int> idx(E_);
    for (int i = 0; i < E_; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    out.clear();
    int k = std::min(topk, E_);
    for (int i = 0; i < k; ++i) out.push_back(idx[i]);
}

} // namespace dee
