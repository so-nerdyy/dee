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

#include <string>
#include <vector>

#ifdef DEE_CUDA
#include <cublas_v2.h>
#endif

namespace dee {

struct OracleLayerWeights {
    std::vector<float> w0, b0;   // (256, 2048), (256,)
    std::vector<float> w2, b2;   // (256, 256), (256,)
    std::vector<float> w4, b4;   // (256, 256), (256,)
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

    // Convenience: predict for the NEXT layer (prefetch target) given the
    // hidden state at the current layer's input.
    void predict_next(int current_layer, const float* hidden, int topk, std::vector<int>& out) const {
        int L = current_layer + 1;
        if (L >= num_layers()) L = current_layer; // clamp if last layer
        predict(L, hidden, topk, out);
    }

#ifdef DEE_CUDA
    // Upload all layer weights to GPU and keep them resident. Must be called
    // after load() and before any predict_gpu() calls.
    bool upload_to_gpu();

    // Free GPU resources.
    void free_gpu();

    // GPU-accelerated predict: runs all three linears on device via cuBLAS,
    // D2H the 256 logits, then does sigmoid + top-K on CPU for exact routing.
    // `d_hidden` is the input on device (length D_). `d_scratch` is a
    // device buffer of at least H_ + E_ floats for intermediate activations.
    // Uses `stream` for all GPU work; syncs internally before returning.
    void predict_gpu(int layer, const float* d_hidden, float* d_scratch,
                     cublasHandle_t handle, void* stream,
                     int topk, std::vector<int>& out) const;

    // Boundary-aware GPU predict: like predict_gpu() but, after the 256
    // approximate logits are returned to the host, inspects the Kth vs
    // (K+1)th gap AND the minimum internal gap between adjacent sorted
    // top-K logits. When ANY gap is below `epsilon_margin` (in absolute
    // logit units), the call falls back to the exact CPU predict() on the
    // same host-side `h_in_cpu` to guarantee ordered, exact top-K routing.
    // Set `epsilon_margin` to 0 to disable (raw GPU predict; may mismatch).
    struct BoundaryStats {
        size_t gpu_calls = 0;          // total calls evaluated on GPU
        size_t cpu_fallback_calls = 0; // fallback to exact CPU predict
    };
    void predict_gpu_boundary(int layer, const float* h_in_cpu, float* d_hidden,
                              float* d_scratch, cublasHandle_t handle, void* stream,
                              int topk, std::vector<int>& out,
                              float epsilon_margin) const;
    BoundaryStats boundary_stats() const { return bstats_; }
    void reset_boundary_stats() { bstats_ = BoundaryStats{}; }
#endif

    const std::string& error() const { return err_; }
    void set_profiler(StageProfiler* profiler) { profiler_ = profiler; }

private:
    int D_ = 2048, H_ = 256, E_ = 256;
    std::vector<OracleLayerWeights> layers_;
    std::string err_;
    StageProfiler* profiler_ = nullptr;

#ifdef DEE_CUDA
    struct GpuLayerWeights {
        float* d_w0 = nullptr;
        float* d_b0 = nullptr;
        float* d_w2 = nullptr;
        float* d_b2 = nullptr;
        float* d_w4 = nullptr;
        float* d_b4 = nullptr;
    };
    std::vector<GpuLayerWeights> gpu_layers_;
    mutable BoundaryStats bstats_;
#endif
};

} // namespace dee
