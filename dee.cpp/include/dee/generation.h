#pragma once
// dee.cpp - Step 7: autoregressive generation loop.
//
// Drives the token-by-token decode cycle that ties the whole system together:
// for each token the hidden state is fed to the OracleScheduler (which predicts
// the experts this layer will need next and kicks off async prefetches), the
// router selects the experts THIS token actually uses, and the MoE forward pass
// runs them. Per-expert events are synchronized ONLY on a cache miss (the expert
// is not already resident + copied); cache hits stream straight from VRAM with
// no event wait.
//
// The per-expert SwiGLU MLP (expert_mlp) is the MoE compute node. When ggml is
// vendored it becomes a GGML_OP_MUL_MAT subgraph; the structure here (router ->
// ensure-ready -> weighted expert_mlp -> residual) is the graph.

#include <cstddef>
#include <vector>

#include "dee/async_prefetcher.h"
#include "dee/moe_forward.h"
#include "dee/oracle.h"
#include "dee/oracle_scheduler.h"
#include "dee/vram_cache_manager.h"
#include "dee/weight_source.h"

namespace dee {

// Selects the actual experts (and their router weights) for one layer given the
// current hidden state. Production = the router linear head; tests inject a mock.
class ExpertRouter {
public:
    virtual ~ExpertRouter() = default;
    virtual void route(int layer, const float* hidden, int hidden_dim, int topk,
                       int* out_ids, float* out_scores) const = 0;
};

// Maps a token id to its embedding vector (hidden_dim floats).
class TokenEmbedder {
public:
    virtual ~TokenEmbedder() = default;
    virtual void embed(int token_id, float* hidden_out, int hidden_dim) const = 0;
};

// Maps the final hidden state to the next token id (LM head / sampler).
class TokenSampler {
public:
    virtual ~TokenSampler() = default;
    virtual int sample(const float* hidden, int hidden_dim) const = 0;
};

struct GeneratorConfig {
    int num_layers = 0;
    int topk = 0;             // experts selected (router) AND predicted (oracle) per layer
    int hidden_dim = 0;
    int inter_dim = 0;        // MoE intermediate dim (per expert)
    int eos_token = -1;       // stop generation when this token is sampled
};

// Autoregressive generator. Owns the OracleScheduler and orchestrates the
// prefetch / forward / synchronize cycle for every decoded token.
class Generator {
public:
    Generator(CudaDevice& dev, VramCacheManager& cache, AsyncPrefetcher& pf,
              WeightSource& weights, ExpertOracle& oracle, ExpertRouter& router,
              TokenEmbedder& embedder, TokenSampler& sampler, const GeneratorConfig& cfg);

    // Run autoregressive generation. `prompt` seeds the context; up to
    // `max_new_tokens` tokens are produced (fewer if EOS is sampled). Each
    // generated token is fed back as the next input (the autoregressive cycle).
    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens);

    // --- Observability (telemetry / tests) ---
    // Expert evaluations served from VRAM with no per-expert event wait.
    long cache_hits() const { return cache_hits_; }
    // Expert evaluations that required a per-expert event sync (cold miss or a
    // predicted-but-not-yet-copied expert).
    long cache_misses() const { return cache_misses_; }
    // Prefetches the loop had to issue itself because the Oracle did not predict
    // the expert the router actually selected (Oracle misprediction).
    long reactive_prefetches() const { return reactive_prefetches_; }
    // Number of times the Oracle was consulted (once per layer per token).
    long prediction_calls() const { return prediction_calls_; }

private:
    // Run all MoE layers over x (hidden_dim floats), in place.
    void run_layers(float* x);

    CudaDevice& dev_;
    VramCacheManager& cache_;
    AsyncPrefetcher& pf_;
    WeightSource& weights_;
    const ExpertRouter& router_;
    const TokenEmbedder& embedder_;
    const TokenSampler& sampler_;
    OracleScheduler scheduler_;  // wires oracle -> prefetcher
    GeneratorConfig cfg_;

    // Counters.
    long cache_hits_ = 0, cache_misses_ = 0, reactive_prefetches_ = 0, prediction_calls_ = 0;

    // Reusable scratch (single decode thread, so no realloc per token).
    std::vector<float> moe_out_, expert_out_, gate_scratch_, up_scratch_, act_scratch_;
    std::vector<int> route_ids_;
    std::vector<float> route_scores_;
    std::vector<ExpertTensorView> views_;
};

}  // namespace dee
