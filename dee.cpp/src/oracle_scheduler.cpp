// dee.cpp - Step 6: OracleScheduler implementation.

#include "dee/oracle_scheduler.h"

namespace dee {

OracleScheduler::OracleScheduler(ExpertOracle& oracle, AsyncPrefetcher& prefetcher,
                                 VramCacheManager& cache, int topk)
    : oracle_(oracle), pf_(prefetcher), cache_(cache), topk_(topk) {}

void OracleScheduler::on_token_decoded(int layer, const float* hidden_state, int hidden_dim) {
    std::vector<int> pred(static_cast<size_t>(topk_));
    oracle_.predict(layer, hidden_state, hidden_dim, topk_, pred.data());
    for (int e : pred) {
        // Already resident, or a prefetch is already in flight -> don't re-issue.
        if (!cache_.contains(ExpertKey{layer, e}) && !pf_.is_inflight(layer, e)) {
            pf_.prefetch_expert(layer, e);
        }
    }
}

bool OracleScheduler::ensure_ready(int layer, int expert_id) {
    return pf_.synchronize_expert(layer, expert_id);
}

void OracleScheduler::synchronize_all() { pf_.synchronize_all(); }

}  // namespace dee
