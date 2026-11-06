#!/usr/bin/env python3
"""CACHE1.1 — expert access-pattern analysis from sealed DS10 v12 evidence.

Pure local analysis (zero Kaggle time).  Reads the sealed evidence
token_trace (per-token, per-layer selected experts + cache counters) and
computes:

  - total expert accesses, unique experts, per-layer/per-token statistics
  - reuse across consecutive tokens (same layer, same expert at t+1)
  - reuse across tokens within a layer (does layer L revisit experts?)
  - expert popularity distribution (power-law vs uniform) per layer and global
  - theoretical max hit rate under oracle eviction at several budgets
  - what the measured ffn_cache_counters say happened during DS10
  - shared-expert presence (expert_id == -1 or negative)

Run:  python cache1_analysis.py <evidence.json> [--json out.json]
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict

TOP_K = 6  # DeepSeek-V4-Flash-0731 routing depth (verify from trace)


def flatten_expert_groups(sel: list) -> list[int]:
    """Flatten whatever nesting the selected_experts field uses.

    Can be [[e1..e6]] (single position) or [[...],[...]] (multi-position
    prefill) or flat [e1..e6].  Returns the flat int list.
    """
    out: list[int] = []
    def rec(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                rec(y)
        else:
            out.append(int(x))
    rec(sel)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    evidence_path = sys.argv[1]
    d = json.load(open(evidence_path, encoding="utf-8"))
    g = d.get("gates", d)
    tt = g.get("token_trace", {})
    tokens = sorted(tt.keys(), key=lambda k: int(k.split("_")[1]))
    if not tokens:
        print("NO token_trace in evidence")
        return 1

    # access[tok_idx][layer] = [expert_ids...]
    access: dict[int, dict[int, list[int]]] = {}
    counters: dict[int, dict[int, dict]] = {}
    n_layers_seen = 0
    for tname in tokens:
        tok = tt[tname]
        idx = int(tname.split("_")[1])
        access[idx] = {}
        counters[idx] = {}
        for row in tok.get("layers", []):
            lid = int(row["layer"])
            n_layers_seen = max(n_layers_seen, lid + 1)
            access[idx][lid] = flatten_expert_groups(row.get("selected_experts", []))
            ctr = row.get("ffn_cache_counters")
            counters[idx][lid] = ctr if isinstance(ctr, dict) else {}
    n_tokens = len(tokens)
    print(f"tokens in trace: {n_tokens}, layers per token: {n_layers_seen}")

    # ---- 1. aggregate accesses ----
    total_accesses = 0
    unique_global = set()
    per_layer_uniq: dict[int, set] = defaultdict(set)
    per_layer_accesses: dict[int, int] = defaultdict(int)
    for idx in sorted(access):
        for lid, exps in access[idx].items():
            total_accesses += len(exps)
            for e in exps:
                unique_global.add((lid, e))
                per_layer_uniq[lid].add(e)
            per_layer_accesses[lid] += len(exps)

    print(f"\n== 1. VOLUME ==\n"
          f"total expert accesses (all tokens, all layers): {total_accesses}\n"
          f"unique (layer,expert) pairs accessed: {len(unique_global)}")

    # ---- 2. reuse across consecutive tokens (t -> t+1, same layer) ----
    consec_hits = 0
    consec_pairs = 0
    for t in range(n_tokens - 1):
        for lid in range(n_layers_seen):
            a = set(access.get(t, {}).get(lid, []))
            b = set(access.get(t + 1, {}).get(lid, []))
            if not a and not b:
                continue
            consec_pairs += 1
            consec_hits += len(a & b)
    consec_reuse = consec_hits / max(1, consec_pairs * TOP_K)
    print(f"\n== 2. CONSECUTIVE-TOKEN REUSE (same layer, t->t+1) ==\n"
          f"overlapping expert-slots across adjacent tokens: {consec_hits} "
          f"/ {consec_pairs * TOP_K} slots ({consec_reuse*100:.1f}%)")

    # ---- 3. within-layer revisit frequency ----
    revisit_slots = 0
    for lid in range(n_layers_seen):
        for t in range(n_tokens):
            exps = set(access.get(t, {}).get(lid, []))
            for e in exps:
                # count occurrences of e in this layer across ALL tokens
                occ = sum(1 for t2 in range(n_tokens)
                          if e in set(access.get(t2, {}).get(lid, [])))
                if occ >= 2:
                    revisit_slots += 1
    print(f"\n== 3. WITHIN-LAYER REVISIT ==\n"
          f"(layer,expert) slots used by >=2 tokens: {revisit_slots} "
          f"/ {len(unique_global)} ({revisit_slots/max(1,len(unique_global))*100:.1f}%)")

    # ---- 4. popularity distribution ----
    global_freq: Counter[int] = Counter()
    layer_freqs: dict[int, Counter] = defaultdict(Counter)
    for idx in sorted(access):
        for lid, exps in access[idx].items():
            for e in exps:
                global_freq[(lid, e)] += 1
                layer_freqs[lid][e] += 1
    freqs = sorted(global_freq.values(), reverse=True)
    n = len(freqs)
    top10_share = sum(freqs[:10]) / max(1, total_accesses)
    # power-law fit proxy: fraction of accesses covered by top 5% of experts
    top5n = max(1, int(n * 0.05))
    top5_share = sum(freqs[:top5n]) / max(1, total_accesses)
    mean_freq = total_accesses / max(1, n)
    print(f"\n== 4. POPULARITY ==\n"
          f"unique (layer,expert): {n}, mean frequency: {mean_freq:.3f}\n"
          f"top-10 (layer,expert) share of accesses: {top10_share*100:.1f}%\n"
          f"top-5% experts share of accesses: {top5_share*100:.1f}%  "
          f"(~uniform => {0.05*100:.1f}%; heavy tail => much higher)")

    # ---- 5. theoretical max hit rate under oracle eviction ----
    # Model: streaming accesses, cache of C experts, evict the entry whose
    # NEXT use is furthest in the future (Belady).  We simulate per-GPU
    # budget assuming per-expert FP16 payload ~50 MiB (verify from headers);
    # report as function of expert capacity AND as budgeted GiB.
    # Access stream ordered by (token, layer): first GPU holds layers < split.
    # We approximate with a single unified cache for the hit-rate shape, then
    # split by the DS10 partition for the byte budgets.
    stream: list[tuple[int, tuple[int, int]]] = []
    for t in range(n_tokens):
        for lid in range(n_layers_seen):
            for e in access.get(t, {}).get(lid, []):
                stream.append((t, (lid, e)))

    def belady_hitrate(capacity: int) -> float:
        nxt = defaultdict(list)
        for i, (_, key) in enumerate(stream):
            nxt[key].append(i)
        for k in nxt:
            nxt[k].append(10**9)
        ptr = {k: 0 for k in nxt}
        cache: dict[tuple[int, int], int] = {}
        hits = 0
        for i, (_, key) in enumerate(stream):
            if key in cache:
                hits += 1
            else:
                if len(cache) >= capacity:
                    # evict entry with farthest next use
                    victim = max(cache, key=lambda k: nxt[k][ptr[k]])
                    del cache[victim]
                cache[key] = i
            ptr[key] += 1
        return hits / max(1, len(stream))

    print(f"\n== 5. THEORETICAL MAX HIT RATE (Belady/oracle, all 43 layers "
          f"one cache) ==\nstream length: {len(stream)}")
    for cap in (32, 64, 128, 256, 512, 1024):
        hr = belady_hitrate(cap)
        print(f"  capacity {cap:5d} experts: oracle hit {hr*100:6.1f}%")

    # Byte budgets: FP16 expert payload estimate
    EXPERT_MB = 50.0  # placeholder; corrected below from model module if found
    print(f"\n  (per-expert FP16 estimate {EXPERT_MB:.0f} MiB — refine from "
          f"DS10 headers)")
    for budget_gib in (1, 2, 4, 8):
        cap = int(budget_gib * 1024 / EXPERT_MB)
        hr = belady_hitrate(cap)
        print(f"  budget {budget_gib:3d} GiB (~{cap} experts): oracle hit "
              f"{hr*100:6.1f}%")

    # ---- 6. measured cache counters from evidence ----
    print(f"\n== 6. MEASURED ffn_cache_counters (sum over tokens/layers) ==")
    agg: Counter = Counter()
    for idx in counters:
        for lid, ctr in counters[idx].items():
            for k, v in ctr.items():
                agg[k] += int(v) if isinstance(v, (int, float)) else 0
    for k, v in agg.most_common(20):
        print(f"  {k}: {v}")

    # ---- 7. shared experts ----
    shared_slots = sum(1 for idx in access for lid in access[idx]
                       for e in access[idx][lid] if e < 0)
    print(f"\n== 7. SHARED EXPERTS ==\n"
          f"negative (shared) expert ids in access: {shared_slots}")

    if len(sys.argv) > 2 and sys.argv[2] == "--json":
        out = {
            "n_tokens": n_tokens,
            "n_layers": n_layers_seen,
            "total_accesses": total_accesses,
            "unique_pairs": len(unique_global),
            "consecutive_token_reuse_pct": round(consec_reuse * 100, 2),
            "revisit_slots_pct": round(
                revisit_slots / max(1, len(unique_global)) * 100, 2),
            "top10_share_pct": round(top10_share * 100, 2),
            "top5pct_share_pct": round(top5_share * 100, 2),
            "belady": {str(c): round(belady_hitrate(c) * 100, 2)
                       for c in (32, 64, 128, 256, 512, 1024)},
            "cache_counters": dict(agg),
            "shared_slots": shared_slots,
        }
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
