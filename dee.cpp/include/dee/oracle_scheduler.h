#pragma once
// dee.cpp - Step 6: OracleScheduler.
//
// Wires ExpertOracle predictions to the AsyncPrefetcher. After each token is
// decoded, it predicts the experts the layer will need next and enqueues
// prefetches for those not already resident / in flight, so weights are in VRAM
// before the forward pass asks for them.

#include "dee/async_prefetcher.h"
#include "dee/oracle.h"
#include "dee/vram_cache_manager.h"

#include <vector>

namespace dee {

class OracleScheduler {
public:
    OracleScheduler(ExpertOracle& oracle, AsyncPrefetcher& prefetcher, VramCacheManager& cache,
                    int topk);

    // Called after a token is decoded for `layer`: predict the next experts and
    // enqueue prefetches for those not already cached / in flight.
    void on_token_decoded(int layer, const float* hidden_state, int hidden_dim);

    // Block until a specific predicted expert is resident + copied (per-expert
    // sync) before the forward pass reads it.
    bool ensure_ready(int layer, int expert_id);

    void synchronize_all();
    int topk() const { return topk_; }

private:
    ExpertOracle& oracle_;
    AsyncPrefetcher& pf_;
    VramCacheManager& cache_;
    int topk_;
};

}  // namespace dee
