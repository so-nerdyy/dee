// dee.cpp - Step 7: autoregressive generation loop implementation.

#include "dee/generation.h"

#include <algorithm>
#include <cstddef>

namespace dee {

Generator::Generator(CudaDevice& dev, VramCacheManager& cache, AsyncPrefetcher& pf,
                     WeightSource& weights, ExpertOracle& oracle, ExpertRouter& router,
                     TokenEmbedder& embedder, TokenSampler& sampler, const GeneratorConfig& cfg)
    : dev_(dev),
      cache_(cache),
      pf_(pf),
      weights_(weights),
      router_(router),
      embedder_(embedder),
      sampler_(sampler),
      scheduler_(oracle, pf, cache, cfg.topk),
      cfg_(cfg) {
    moe_out_.assign(cfg.hidden_dim, 0.f);
    expert_out_.assign(cfg.hidden_dim, 0.f);
    gate_scratch_.assign(cfg.inter_dim, 0.f);
    up_scratch_.assign(cfg.inter_dim, 0.f);
    act_scratch_.assign(cfg.inter_dim, 0.f);
    route_ids_.assign(cfg.topk, 0);
    route_scores_.assign(cfg.topk, 0.f);
    views_.assign(kNumExpertTensors, ExpertTensorView{});
}

void Generator::run_layers(float* x) {
    const int L = cfg_.num_layers, K = cfg_.topk, H = cfg_.hidden_dim, I = cfg_.inter_dim;

    for (int layer = 0; layer < L; ++layer) {
        // Map the 0-based position to the actual layer in the weight files so
        // real shards (e.g. model.layers.10) resolve. Identity for synthetic
        // sources, so existing tests are unaffected.
        const int ml = weights_.layer_index(layer);

        // (1) Oracle: predict the experts this layer will need next, and kick off
        //     async prefetches for any that are not already resident / in flight.
        scheduler_.on_token_decoded(ml, x, H);
        ++prediction_calls_;

        // (2) Router: which experts does THIS token actually use?
        router_.route(ml, x, H, K, route_ids_.data(), route_scores_.data());

        std::fill(moe_out_.begin(), moe_out_.end(), 0.f);
        for (int k = 0; k < K; ++k) {
            const int e = route_ids_[k];
            const ExpertKey key{ml, e};

            // (3) Cache-miss handling: synchronize THIS expert's per-expert event
            //     only when it is not already resident + copied. If the Oracle did
            //     not predict it (misprediction), issue a reactive prefetch first.
            if (pf_.is_ready(ml, e)) {
                ++cache_hits_;  // already in VRAM and copied -> no event wait
            } else {
                if (!pf_.is_inflight(ml, e)) {
                    pf_.prefetch_expert(ml, e);  // cold miss -> fetch now
                    ++reactive_prefetches_;
                }
                pf_.synchronize_expert(ml, e);   // per-expert event sync on miss
                ++cache_misses_;
            }
            cache_.touch(key);

            // (4) Read the now-resident weights out of the VRAM slot and run the
            //     per-expert SwiGLU MLP (the MoE compute node; ggml plugs in here).
            weights_.expert_tensors(ml, e, views_.data());
            const float* base = static_cast<const float*>(cache_.ptr_of(key));
            const std::byte* base_b = reinterpret_cast<const std::byte*>(base);
            const float* gu = reinterpret_cast<const float*>(
                base_b + views_[static_cast<int>(ExpertTensor::GateUp)].slot_offset);
            const float* dn = reinterpret_cast<const float*>(
                base_b + views_[static_cast<int>(ExpertTensor::Down)].slot_offset);
            expert_mlp(x, H, I, gu, dn, expert_out_.data(), gate_scratch_.data(),
                       up_scratch_.data(), act_scratch_.data());

            // (5) Accumulate the router-weighted expert output.
            const float s = route_scores_[k];
            for (int i = 0; i < H; ++i) moe_out_[i] += s * expert_out_[i];
        }

        // Residual connection: hidden = hidden + MoE(hidden).
        for (int i = 0; i < H; ++i) x[i] += moe_out_[i];
    }
}

std::vector<int> Generator::generate(const std::vector<int>& prompt, int max_new_tokens) {
    std::vector<float> x(cfg_.hidden_dim);
    if (prompt.empty() || max_new_tokens <= 0) return {};

    // Prefill: process the prompt tokens to build the context hidden.
    embedder_.embed(prompt.front(), x.data(), cfg_.hidden_dim);
    run_layers(x.data());
    for (size_t i = 1; i < prompt.size(); ++i) {
        embedder_.embed(prompt[i], x.data(), cfg_.hidden_dim);
        run_layers(x.data());
    }

    // Autoregressive decode: sample -> feed back -> repeat.
    std::vector<int> out;
    out.reserve(max_new_tokens);
    int steps = 0;
    while (steps < max_new_tokens) {
        const int next = sampler_.sample(x.data(), cfg_.hidden_dim);
        out.push_back(next);
        ++steps;
        if (next == cfg_.eos_token) break;
        embedder_.embed(next, x.data(), cfg_.hidden_dim);
        run_layers(x.data());
    }
    return out;
}

}  // namespace dee
