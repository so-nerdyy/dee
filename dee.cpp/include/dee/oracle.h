// dee/oracle.h
//
// Step 7 — OracleScheduler: runs the 3-layer MLP Oracle to predict which
// experts a layer will activate, so the prefetcher can pre-stage them.
//
// The Oracle is one small MLP per decoder layer (architecture fixed):
//   x: [hidden=2048] -> Linear(2048,256) -> ReLU -> Linear(256,256) -> ReLU
//                       -> Linear(256,256) -> logits[256]  (BCE => sigmoid)
// predict(layer, hidden) returns the top-K expert indices for that layer.

#pragma once

#include "dee/pt_loader.h"
#include "dee/profiling.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dee {

struct OracleLayerWeights {
    std::vector<float> w0, b0;   // (256, 2048), (256,)
    std::vector<float> w2, b2;   // (256, 256), (256,)
    std::vector<float> w4, b4;   // (256, 256), (256,)

    // VNNI INT8 quantized variants.  q_* are signed INT8 weights stored as
    // signed bytes (out, in) row-major.  row_scales are per-output-channel
    // FP32 scales.  row_sums are pre-computed sums of the signed weights.
    std::vector<int8_t> w0_q, w2_q, w4_q;
    std::vector<float> w0_s, w2_s, w4_s;
    std::vector<int32_t> w0_r, w2_q_r, w4_q_r;
};

class OracleScheduler {
public:
    bool load(const std::string& oracle_pt_path, int D = 2048, int H = 256, int E = 256);

    int num_layers() const { return (int)layers_.size(); }
    int num_experts() const { return E_; }

    // Run the MLP for `layer` on `hidden` (length D_) -> fill `logits` (length E_).
    void forward(int layer, const float* hidden, std::vector<float>& logits) const;

    // Predict the top-K expert indices for `layer` from `hidden`.
    // Returns them sorted by descending logit. `out` is cleared and filled.
    void predict(int layer, const float* hidden, int topk, std::vector<int>& out) const;

    // Predict the top-K expert indices for `layer` from `hidden`, also writing
    // the raw Oracle logits (length E_) into `logits_out`. `out` is cleared
    // filled with the top-K expert indices in descending logits order;
    // `logits_out` is cleared and resized to E_ exactly. The logits are the
    // raw scores the Oracle would sort to derive the top-K, so candidates may
    // compare them bit-for-bit against the validated baseline for per-request
    // exact-match routing validation. Behaviour is identical to predict();
    // this overload only exposes the intermediate scores.
    void predict_and_score(int layer, const float* hidden, int topk,
                            std::vector<int>& out,
                            std::vector<float>& logits_out) const;

    // Convenience: predict for the NEXT layer (prefetch target) given the
    // hidden state at the current layer's input.
    void predict_next(int current_layer, const float* hidden, int topk, std::vector<int>& out) const {
        int L = current_layer + 1;
        if (L >= num_layers()) L = current_layer; // clamp if last layer
        predict(L, hidden, topk, out);
    }

    const std::string& error() const { return err_; }
    void set_profiler(StageProfiler* profiler) { profiler_ = profiler; }

private:
    int D_ = 2048, H_ = 256, E_ = 256;
    std::vector<OracleLayerWeights> layers_;
    std::string err_;
    StageProfiler* profiler_ = nullptr;
    bool use_vnni_ = false;

    // Reusable VNNI INT8 buffers.  Mutable so the const forward path can reuse
    // them without allocating on every Oracle call.
    mutable std::vector<uint8_t> qx0_, qx1_, qx2_;
    mutable std::vector<float> h1_vnni_, h2_vnni_;
    mutable std::vector<float> logits_buf_;  // reused by predict()

    // VNNI INT8 forward used when AVX-512 VNNI is available.
    void forward_vnni(int layer, const float* hidden, std::vector<float>& logits) const;

    // Quantize one weight matrix (out, in) to signed INT8 per output channel.
    static void quantize_weights(const std::vector<float>& W, int in, int out,
                                  std::vector<int8_t>& qout,
                                  std::vector<float>& scales,
                                  std::vector<int32_t>& row_sums);

    // Quantize an activation vector to unsigned INT8 (offset by +128).
    static void quantize_activation(const float* x, int n,
                                    std::vector<uint8_t>& qout,
                                    float& scale);
};

} // namespace dee
