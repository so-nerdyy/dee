// dee/next_use.h
//
// Next-use estimator shared by the VRAM expert cache and the host pack cache.
//
// Why: decode sweeps the layers in a fixed cycle, so the expert used least
// recently is the one whose layer comes up next.  When one decode step asks
// for more experts than a tier holds, LRU therefore evicts every expert just
// before it is needed again (Phase 6 run p6m-l4-r5-b8-h16: 0 host hits out of
// 49,463 lookups).  Ranking victims by when they will be needed NEXT, instead
// of when they were used LAST, removes that failure.
//
// Model.  An expert of layer L can next be used when the sweep reaches L:
// d = (L - current_layer) mod cycle layer calls from now.  It is used at that
// visit with probability p, otherwise one cycle later, and so on:
//
//     E[next use] = d + cycle * (1/p - 1)
//
// p is an exponentially weighted reuse estimate kept per expert: every visit
// of its layer multiplies it by (1 - alpha), every use adds alpha.  An expert
// never seen before starts at p0.  The estimator keeps its statistics across
// evictions, so an expert that returns is ranked by its whole history.
//
// The victim is the unpinned resident with the LARGEST victim_rank(): the
// expected next use, except that an expert of the layer running now ranks
// below everything else.  The engine may still reach it later in this same
// call (its bytes are prepared first and staged afterwards), so it leaves
// only when nothing else can.
//
// This is a placement policy only.  It decides which resident expert leaves a
// tier; it never selects, skips, or substitutes an expert, so tokens and
// route journals are unchanged by construction.
//
// theory/theory/policy_sim.py (class NextUse) is the line-for-line Python
// mirror used to score this policy on committed route journals.
//
// Credit: next-use ("probabilistic Belady") eviction follows SeqMoE
// (arXiv 2609.12978); Belady's MIN (1966) is the offline optimum it
// approximates.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace dee {

class NextUseEstimator {
public:
    // Keys are (layer << 32) | expert, the engine's staging key layout.
    static uint64_t make_key(int layer, int expert) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
               static_cast<uint32_t>(expert);
    }
    static int layer_of(uint64_t key) { return static_cast<int>(key >> 32); }

    // cycle_layers <= 0 disables the estimator (every query returns 0).
    void configure(int cycle_layers, double alpha = 0.2, double p0 = 0.1,
                   double p_floor = 1e-3) {
        cycle_ = std::max(0, cycle_layers);
        alpha_ = std::min(1.0, std::max(1e-6, alpha));
        p0_ = std::min(1.0, std::max(p_floor, p0));
        p_floor_ = std::max(1e-9, p_floor);
        visits_.assign(static_cast<size_t>(cycle_), 0);
        stats_.clear();
        current_ = 0;
        started_ = false;
    }
    bool enabled() const { return cycle_ > 0; }
    int cycle() const { return cycle_; }
    int current_layer() const { return current_; }

    // The sweep reached `layer`: it is now the layer being executed.
    void begin_layer(int layer) {
        if (cycle_ <= 0) return;
        current_ = wrap(layer);
        ++visits_[static_cast<size_t>(current_)];
        started_ = true;
    }

    // `key` was requested during the current visit of its layer.  Idempotent
    // within one visit, so a hit followed by a touch counts once.
    void on_use(uint64_t key) {
        if (cycle_ <= 0) return;
        const uint32_t visit = visits_[static_cast<size_t>(wrap(layer_of(key)))];
        auto found = stats_.find(key);
        if (found == stats_.end()) {
            stats_.emplace(key, Stat{p0_ * (1.0 - alpha_) + alpha_, visit});
            return;
        }
        Stat& stat = found->second;
        if (stat.last_visit == visit) return;
        const double gap = static_cast<double>(visit - stat.last_visit);
        stat.ewma = stat.ewma * std::pow(1.0 - alpha_, gap) + alpha_;
        stat.last_visit = visit;
    }

    // Estimated probability that `key` is used at the next visit of its layer.
    double probability(uint64_t key) const {
        if (cycle_ <= 0) return 1.0;
        auto found = stats_.find(key);
        if (found == stats_.end()) return p0_;
        const int layer = wrap(layer_of(key));
        // Visits since the last use that are already known misses.  The
        // visit in progress is not decided yet for the current layer.
        int64_t elapsed =
            static_cast<int64_t>(visits_[static_cast<size_t>(layer)]) -
            static_cast<int64_t>(found->second.last_visit);
        if (started_ && layer == current_) --elapsed;
        double p = found->second.ewma;
        if (elapsed > 0) p *= std::pow(1.0 - alpha_, static_cast<double>(elapsed));
        return std::max(p_floor_, std::min(1.0, p));
    }

    // Layer calls until the sweep next reaches `key`'s layer (0 = running now).
    int distance(uint64_t key) const {
        if (cycle_ <= 0) return 0;
        return ((wrap(layer_of(key)) - current_) % cycle_ + cycle_) % cycle_;
    }

    // Expected layer calls until `key` is next used.  Larger = evict sooner.
    double expected_next_use(uint64_t key) const {
        if (cycle_ <= 0) return 0.0;
        return static_cast<double>(distance(key)) +
               static_cast<double>(cycle_) * (1.0 / probability(key) - 1.0);
    }

    // Eviction order: larger leaves first.  Experts of the running layer
    // rank below every other expert (see the header comment).
    double victim_rank(uint64_t key) const {
        if (cycle_ <= 0) return 0.0;
        if (started_ && wrap(layer_of(key)) == current_) return -1.0;
        return expected_next_use(key);
    }

    // Forget all history (between unrelated sequences / cache resets).
    void clear() {
        std::fill(visits_.begin(), visits_.end(), 0u);
        stats_.clear();
        current_ = 0;
        started_ = false;
    }
    size_t tracked() const { return stats_.size(); }

private:
    struct Stat {
        double ewma = 0.0;
        uint32_t last_visit = 0;
    };
    int wrap(int layer) const {
        if (cycle_ <= 0) return 0;
        return ((layer % cycle_) + cycle_) % cycle_;
    }

    int cycle_ = 0;
    double alpha_ = 0.2;
    double p0_ = 0.1;
    double p_floor_ = 1e-3;
    int current_ = 0;
    bool started_ = false;
    std::vector<uint32_t> visits_;
    std::unordered_map<uint64_t, Stat> stats_;
};

}  // namespace dee
