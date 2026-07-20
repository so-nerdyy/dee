"""
================================================================================
Step 3: VRAM Cache Manager (CPU test)  (Dynamic Expert Eviction MoE)
================================================================================
Pure-Python ExpertCache that SIMULATES a limited VRAM expert pool and uses the
Oracle's look-ahead predictions to drive eviction. No GPU memory is touched.

Capacity = 20 experts (matches our Oracle top_k_pred=20). At any time the cache
holds at most `capacity` experts in "VRAM". When a token needs its activated
experts, request_expert(id) checks presence and, on a miss, asynchronously
"prefetches" it from CPU RAM into VRAM (simulated; we simply record the op and
mark it present). If at capacity, evict() removes the LEAST-look-ahead-useful
expert rather than blind-LRU.

Look-ahead eviction policy:
  The Oracle predicts the top-20 experts likely needed for the NEXT token.
  Before serving token t, we pre-load those 20 (prefetch). When the cache is
  full and we must evict to make room for a prefetched expert, we evict the
  cached expert with the LOWEST priority = the one least present in the
  upcoming predicted set / least recently used among the non-predicted. This
  matches the brief: "evict the experts least likely to be needed in upcoming
  tokens", not blind LRU.

Simulation (run_simulation):
  Replays the REAL routing data collected in Step 1, using the REAL Oracle
  trained in Step 2 to predict ahead. For each token position:
    1. Oracle(hidden[t]) -> top-20 predicted experts for token t+1  (look-ahead)
    2. Pre-load those 20 into VRAM (prefetch, async-simulated)
    3. Request the TRUE 8 experts of token t+1 -> count hits / misses
  Outputs final Cache Hit Rate / Miss Rate to prove the logic before Step 4.

Run (CPU-only, reads dee-oracle-data volume):
    modal run modal_step3_cache_manager.py
================================================================================
"""

import os
import json
import time
import argparse

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy==1.26.4")
)

DATA_VOL = modal.Volume.from_name("dee-oracle-data", create_if_missing=True)
DATA_VOL_PATH = "/vol/data"

app = modal.App("dynamic-expert-eviction-step3")


# =========================================================================== #
# The Cache Manager (pure Python, no torch/GPU at class level)
# =========================================================================== #
class ExpertCache:
    """
    Simulated VRAM expert pool with look-ahead eviction.

    State:
        capacity        : max experts held in VRAM
        vram            : set of expert ids currently in VRAM
        last_used       : expert_id -> token index when last requested
        load_count      : expert_id -> how many times it's been loaded
        prefetch_count  : expert_id -> how many times it was preloaded by Oracle
        async_ops       : list of ("prefetch"|"evict", expert_id, token) log

    The async prefetch is simulated: request_expert / prefetch mark the expert
    present immediately and log the op, modeling "a background stream copies it
    in while compute proceeds". No real CUDA stream yet (that's Step 4).
    """

    def __init__(self, capacity: int = 20, top_k_pred: int = 20):
        self.capacity = capacity
        self._top_k_pred = top_k_pred
        self.vram = set()
        self.last_used = {}
        self.load_count = {}
        self.prefetch_count = {}
        self.async_ops = []
        self._lookahead_rank = {}     # expert -> rank in predicted list (0=top)
        self.clock = 0

    # ---- core ops -------------------------------------------------------- #
    def contains(self, expert_id: int) -> bool:
        return expert_id in self.vram

    def set_lookahead_ranked(self, predicted_ids):
        """
        Stash the Oracle's upcoming predictions AS A RANKED LIST so eviction can
        prefer high-rank (more-confidently-predicted) experts.
        self._lookahead_rank[expert] = position in predicted list (0 = top).
        """
        self._lookahead_rank = {e: i for i, e in enumerate(predicted_ids)}

    def prefetch(self, expert_id: int, token: int):
        """Simulate an asynchronous background copy from CPU RAM -> VRAM
        (full prefetch; loads even if already resident — kept for tests)."""
        self.clock += 1
        if expert_id not in self.vram:
            if len(self.vram) >= self.capacity:
                self.evict(token=token)
            self.vram.add(expert_id)
            self.load_count[expert_id] = self.load_count.get(expert_id, 0) + 1
        self.prefetch_count[expert_id] = self.prefetch_count.get(expert_id, 0) + 1
        self.last_used[expert_id] = token
        self.async_ops.append(("prefetch", expert_id, token))

    def prefetch_delta(self, predicted_ids, token: int):
        """
        SMARTER OVERLAP: only prefetch the DELTA — predicted experts not already
        resident in VRAM. Already-resident experts are skipped (no reload, no
        eviction). This minimizes churn: if 15 of 20 predicted are already in
        VRAM, we only load the 5 missing (evicting at most 5).
        """
        missing = [e for e in predicted_ids if e not in self.vram]
        for e in missing:
            self.clock += 1
            if len(self.vram) >= self.capacity:
                self.evict(token=token)
            self.vram.add(e)
            self.load_count[e] = self.load_count.get(e, 0) + 1
            self.prefetch_count[e] = self.prefetch_count.get(e, 0) + 1
            self.last_used[e] = token
            self.async_ops.append(("prefetch", e, token))
        return missing

    def request_expert(self, expert_id: int, token: int) -> bool:
        """
        Request an expert needed NOW. Returns True on hit, False on miss.
        On a miss, synchronously fetch it (simulated) + evict if over capacity.
        """
        self.clock += 1
        if expert_id in self.vram:
            self.last_used[expert_id] = token
            return True
        # MISS: bring it in now (models the on-demand fallback fetch)
        if len(self.vram) >= self.capacity:
            self.evict(token=token)
        self.vram.add(expert_id)
        self.load_count[expert_id] = self.load_count.get(expert_id, 0) + 1
        self.last_used[expert_id] = token
        self.async_ops.append(("fetch_on_miss", expert_id, token))
        return False

    def evict(self, token: int):
        """
        Remove ONE expert from VRAM using RANK-WEIGHTED look-ahead priority.
        _lookahead_rank: expert -> position in predicted list (0 = top priority).
        Keep score (HIGHER = keep):
            predicted  -> top_k_pred - rank   (top-ranked kept most)
            not predicted -> -1               (evicted first)
        Tie-break among equal keep-score: oldest last_used (LRU within class).
        This keeps high-rank predicted experts and only evicts low-rank /
        unpredicted ones — not blind LRU.
        """
        candidates = [e for e in self.vram]
        if not candidates:
            return None
        k = self._top_k_pred

        def keep_score(e):
            r = self._lookahead_rank.get(e, None)
            if r is None:
                base = -1                      # not predicted -> evict first
            else:
                base = k - r                   # higher rank (small r) -> higher
            return (base, self.last_used.get(e, 0))

        victim = min(candidates, key=keep_score)
        self.vram.discard(victim)
        self.async_ops.append(("evict", victim, token))
        return victim

    def stats(self):
        return {
            "vram_size": len(self.vram),
            "total_loads": sum(self.load_count.values()),
            "total_prefetches": sum(self.prefetch_count.values()),
            "n_async_ops": len(self.async_ops),
        }


# =========================================================================== #
# Simulation over real data + real Oracle
# =========================================================================== #
@app.function(image=image, cpu=4, volumes={DATA_VOL_PATH: DATA_VOL}, timeout=60 * 60)
def run_simulation(
    collected_dir: str,
    oracle_dir: str,
    capacity: int,
    top_k_pred: int,
    max_tokens: int,        # cap replay length per layer for speed (0 = all)
    verbose: bool,
) -> str:
    import torch

    log = lambda *a: print("[STEP3]", *a, flush=True)

    with open(os.path.join(oracle_dir, "oracle_meta.json")) as f:
        ometa = json.load(f)

    layer_files = sorted(
        f for f in os.listdir(collected_dir)
        if f.startswith("layer_") and f.endswith(".pt")
    )
    log(f"found {len(layer_files)} collected layer files; capacity={capacity} "
        f"top_k_pred={top_k_pred}")

    agg_hits = 0
    agg_miss = 0
    per_layer = []

    for lf in layer_files:
        layer_i = int(lf.split("_")[1].split(".")[0])
        data = torch.load(os.path.join(collected_dir, lf), map_location="cpu")
        H = data["hidden_states"].float()      # (P, H)
        E1 = data["next_experts_n1"].long()     # (P, k) experts N+1
        E2 = data["next_experts_n2"].long()     # (P, k) experts N+2
        P = H.shape[0]

        ow = torch.load(os.path.join(oracle_dir, f"oracle_layer_{layer_i:03d}.pt"),
                        map_location="cpu")
        num_experts = ow["num_experts"]
        mlp = torch.nn.Sequential(
            torch.nn.Linear(ometa["hidden_dim"], ometa["hidden_mlp"]),
            torch.nn.ReLU(),
            torch.nn.Linear(ometa["hidden_mlp"], ometa["hidden_mlp"]),
            torch.nn.ReLU(),
            torch.nn.Linear(ometa["hidden_mlp"], num_experts),
        )
        mlp.load_state_dict(ow["state_dict"])
        mlp.eval()

        cache = ExpertCache(capacity=capacity, top_k_pred=top_k_pred)
        hits = 0
        miss = 0
        T = min(P, max_tokens) if max_tokens and max_tokens > 0 else P

        # Replay with 2-ahead: at token t the Oracle predicts the UNION of
        # experts for {t+1, t+2}. We prefetch that union (delta), then serve
        # BOTH t+1 and t+2, checking hits against their TRUE expert sets.
        for t in range(T - 2):
            with torch.no_grad():
                pred_scores = mlp(H[t].unsqueeze(0))[0]            # (num_experts,)
                pred_topk = torch.topk(pred_scores, top_k_pred).indices.tolist()
            cache.set_lookahead_ranked(pred_topk)
            # Smarter overlap: only load predicted experts NOT already in VRAM
            cache.prefetch_delta(pred_topk, token=t)
            # Serve the two upcoming tokens; measure hits/misses vs true sets
            for offset, E in ((1, E1), (2, E2)):
                true_experts = E[t + offset].tolist()
                for eid in true_experts:
                    ok = cache.request_expert(eid, token=t + offset)
                    if ok:
                        hits += 1
                    else:
                        miss += 1
            if verbose and t % 20 == 0:
                log(f"layer {layer_i} token {t}: vram={sorted(cache.vram)} "
                    f"hits={hits} miss={miss}")

        layer_hit_rate = hits / (hits + miss) if (hits + miss) else 0.0
        agg_hits += hits
        agg_miss += miss
        per_layer.append({"layer": layer_i, "hits": hits, "miss": miss,
                          "hit_rate": layer_hit_rate})
        log(f"layer {layer_i}: hits={hits} miss={miss} hit_rate={layer_hit_rate:.4f} "
            f"vram_util={cache.stats()['vram_size']}/{capacity}")

    total = agg_hits + agg_miss
    overall_hit = agg_hits / total if total else 0.0
    overall_miss = agg_miss / total if total else 0.0
    min_lr = min(p["hit_rate"] for p in per_layer) if per_layer else 0
    max_lr = max(p["hit_rate"] for p in per_layer) if per_layer else 0
    mean_lr = sum(p["hit_rate"] for p in per_layer) / len(per_layer) if per_layer else 0

    summary = {
        "capacity": capacity,
        "top_k_pred": top_k_pred,
        "n_layers": len(per_layer),
        "agg_hits": agg_hits,
        "agg_miss": agg_miss,
        "overall_hit_rate": overall_hit,
        "overall_miss_rate": overall_miss,
        "per_layer_hit_rate_min": min_lr,
        "per_layer_hit_rate_max": max_lr,
        "per_layer_hit_rate_mean": mean_lr,
    }
    log("=" * 60)
    log(f"OVERALL CACHE HIT RATE : {overall_hit:.4f}  ({agg_hits}/{total})")
    log(f"OVERALL CACHE MISS RATE: {overall_miss:.4f}  ({agg_miss}/{total})")
    log(f"per-layer hit_rate min/mean/max: {min_lr:.4f}/{mean_lr:.4f}/{max_lr:.4f}")
    log("=" * 60)
    return json.dumps(summary)


# =========================================================================== #
# Capacity / predict-width SWEEP (pure-Python, CPU)
# =========================================================================== #
@app.function(image=image, cpu=4, volumes={DATA_VOL_PATH: DATA_VOL}, timeout=60 * 60)
def run_sweep(
    collected_dir: str,
    oracle_dir: str,
    capacities: list,
    top_k_preds: list,
    max_tokens: int,
) -> str:
    import torch

    log = lambda *a: print("[SWEEP]", *a, flush=True)

    with open(os.path.join(oracle_dir, "oracle_meta.json")) as f:
        ometa = json.load(f)
    layer_files = sorted(
        f for f in os.listdir(collected_dir)
        if f.startswith("layer_") and f.endswith(".pt")
    )
    # Preload all layer data + oracles once
    layers = []
    for lf in layer_files:
        layer_i = int(lf.split("_")[1].split(".")[0])
        data = torch.load(os.path.join(collected_dir, lf), map_location="cpu")
        ow = torch.load(os.path.join(oracle_dir, f"oracle_layer_{layer_i:03d}.pt"),
                        map_location="cpu")
        num_experts = ow["num_experts"]
        mlp = torch.nn.Sequential(
            torch.nn.Linear(ometa["hidden_dim"], ometa["hidden_mlp"]),
            torch.nn.ReLU(),
            torch.nn.Linear(ometa["hidden_mlp"], ometa["hidden_mlp"]),
            torch.nn.ReLU(),
            torch.nn.Linear(ometa["hidden_mlp"], num_experts),
        )
        mlp.load_state_dict(ow["state_dict"])
        mlp.eval()
        layers.append((layer_i, data["hidden_states"].float(),
                       data["next_experts_n1"].long(),
                       data["next_experts_n2"].long(), mlp))

    results = []
    best = None  # lowest capacity with hit_rate > 0.95
    for cap in capacities:
        for tk in top_k_preds:
            if tk > cap:
                continue   # can't predict more than we can hold
            agg_hits, agg_miss = 0, 0
            for (layer_i, H, E1, E2, mlp) in layers:
                P = H.shape[0]
                T = min(P, max_tokens) if max_tokens and max_tokens > 0 else P
                cache = ExpertCache(capacity=cap, top_k_pred=tk)
                for t in range(T - 2):
                    with torch.no_grad():
                        pred = mlp(H[t].unsqueeze(0))[0]
                        pred_topk = torch.topk(pred, tk).indices.tolist()
                    cache.set_lookahead_ranked(pred_topk)
                    cache.prefetch_delta(pred_topk, token=t)
                    for offset, E in ((1, E1), (2, E2)):
                        for eid in E[t + offset].tolist():
                            ok = cache.request_expert(eid, token=t + offset)
                            if ok:
                                agg_hits += 1
                            else:
                                agg_miss += 1
            total = agg_hits + agg_miss
            hr = agg_hits / total if total else 0.0
            rec = {"capacity": cap, "top_k_pred": tk, "hit_rate": hr,
                   "miss_rate": 1 - hr, "hits": agg_hits, "miss": agg_miss}
            results.append(rec)
            log(f"cap={cap:2d} topk={tk:2d} -> hit_rate={hr:.4f}")
            if hr > 0.95:
                if best is None or cap < best["capacity"]:
                    best = rec
    log("=" * 60)
    log("SWEEP DONE. configs tested: %d" % len(results))
    if best:
        log(f"BEST (lowest capacity, hit>0.95): capacity={best['capacity']} "
            f"top_k_pred={best['top_k_pred']} hit_rate={best['hit_rate']:.4f}")
    else:
        log("NO config reached >0.95 hit rate. Report full table.")
    log("=" * 60)
    return json.dumps({"results": results, "best": best})


@app.local_entrypoint()
def main(
    collected_dir: str = DATA_VOL_PATH + "/collected2",
    oracle_dir: str = DATA_VOL_PATH + "/oracle2",
    capacity: int = 24,
    top_k_pred: int = 24,
    max_tokens: int = 0,
    verbose: bool = False,
    sweep: bool = False,
    capacities: str = "24,32,40,48",
    top_k_preds: str = "16,20,24",
):
    if sweep:
        caps = [int(x) for x in capacities.split(",")]
        tks = [int(x) for x in top_k_preds.split(",")]
        print(f"[LOCAL] Sweep: capacities={caps} top_k_preds={tks}", flush=True)
        t0 = time.time()
        out = run_sweep.remote(
            collected_dir=collected_dir,
            oracle_dir=oracle_dir,
            capacities=caps,
            top_k_preds=tks,
            max_tokens=max_tokens,
        )
        print(f"[LOCAL] sweep finished in {time.time()-t0:.1f}s")
        print("[LOCAL] sweep result:", out)
    else:
        print(f"[LOCAL] Step3 simulation: capacity={capacity} top_k_pred={top_k_pred}",
              flush=True)
        t0 = time.time()
        result = run_simulation.remote(
            collected_dir=collected_dir,
            oracle_dir=oracle_dir,
            capacity=capacity,
            top_k_pred=top_k_pred,
            max_tokens=max_tokens,
            verbose=verbose,
        )
        print(f"[LOCAL] finished in {time.time()-t0:.1f}s")
        print("[LOCAL] summary:", result)
