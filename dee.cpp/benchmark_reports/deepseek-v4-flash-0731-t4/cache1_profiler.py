#!/usr/bin/env python3
"""Phase A — Cache Characterization profiler for the DeepSeek V4 Flash DS10 campaign.

Measurement-only tooling (NO runtime changes, NO GPU). Consumes sealed DS10
evidence JSON (and any cache1 run evidence dirs) and computes, from the
per-token/per-layer expert trace:

  - per-layer, per-token cache hit rate (ffn_cache_counters)
  - expert popularity distribution (are selections power-law or uniform?)
  - consecutive-token reuse rate (does expert e reappear in top-K at t+1?)
  - temporal reuse distance (steps between consecutive accesses to an expert)
  - eviction pressure (cache0 ensures/loads/evictions, pinned blocks)
  - bytes transferred (provider fetch_stats + H2D)
  - decode latency profile (per-token ms, p50/p95/max)
  - theoretical Belady hit-rate ceiling at several cache budgets (oracle eviction)

Outputs:
  - JSON report to --output
  - Markdown bottleneck-ranking summary to stdout / --summary

Usage:
  python cache1_profiler.py <evidence.json> [more evidence.json...] \
      [--output report.json] [--summary REPORT.md] [--cache-budgets 1,2,4,8]

Stdlib only. Evidence JSONs: ds10-evidence.json files (v12 seal or cache1 runs).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_evidence(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


EXPERT_BYTES = 50331648  # 48 MiB FP16 payload (matches cache1_policy_sim.py)


def _tokens(trace: dict) -> list[dict]:
    if not isinstance(trace, dict):
        return []
    return [trace[k] for k in sorted(trace.keys()) if k.startswith("token_")]


def flatten_expert_groups(sel) -> list[int]:
    """Flatten nested [[...],[...]] expert groups to a flat int list."""
    out: list[int] = []

    def rec(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                rec(y)
        else:
            out.append(int(x))

    rec(sel)
    return out


def build_stream(token_trace: dict) -> list[tuple[int, tuple[int, int]]]:
    """Emit one access per DISTINCT expert per (token, layer) call plus the
    shared expert, exactly like cache1_policy_sim.build_stream (the real FFN
    groups duplicate expert selections within a call before hitting cache)."""
    stream: list[tuple[int, tuple[int, int]]] = []
    n_layers = 0
    for tname, tok in sorted(token_trace.items(),
                             key=lambda kv: int(kv[0].split("_")[1])):
        idx = int(tname.split("_")[1])
        for row in tok.get("layers", []):
            lid = int(row["layer"])
            n_layers = max(n_layers, lid + 1)
            for e in dict.fromkeys(
                    flatten_expert_groups(row.get("selected_experts", []))):
                stream.append((idx, (lid, e)))
            stream.append((idx, (lid, -1)))  # shared expert per layer call
    return stream, n_layers


def expert_accesses(token_trace: dict) -> list[tuple[int, int, int, int]]:
    """Return (token_idx, layer, pos, expert) for every routed expert access."""
    out: list[tuple[int, int, int, int]] = []
    for ti, tok in enumerate(_tokens(token_trace)):
        for layer in tok.get("layers", []):
            lid = layer.get("layer")
            exps = layer.get("selected_experts") or []
            for pos, row in enumerate(exps):
                if isinstance(row, list):
                    for exp in row:
                        out.append((ti, lid, pos, int(exp)))
                else:
                    out.append((ti, lid, pos, int(row)))
    return out


def per_layer_hit_rates(token_trace: dict) -> dict:
    """Aggregate ffn_cache_counters per layer across all tokens."""
    agg: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tok in _tokens(token_trace):
        for layer in tok.get("layers", []):
            cc = layer.get("ffn_cache_counters") or {}
            a = agg[layer.get("layer")]
            for k in ("requests", "hits", "misses", "staging_waits"):
                a[k] += int(cc.get(k, 0))
    out: dict[str, object] = {"layers": {}}
    tot = {"requests": 0, "hits": 0, "misses": 0, "staging_waits": 0}
    for lid in sorted(agg):
        a = agg[lid]
        for k in tot:
            tot[k] += a[k]
        rate = (a["hits"] / a["requests"]) if a["requests"] else 0.0
        out["layers"][str(lid)] = {
            "requests": a["requests"], "hits": a["hits"], "misses": a["misses"],
            "staging_waits": a["staging_waits"],
            "hit_rate": round(rate, 5),
        }
    out["total"] = tot
    out["total_hit_rate"] = round(
        tot["hits"] / tot["requests"], 5) if tot["requests"] else 0.0
    return out


def popularity(accesses: list[tuple[int, int, int, int]]) -> dict:
    """Expert popularity: frequency across the whole trace, power-law check."""
    by_expert: Counter[int] = Counter(a[3] for a in accesses)
    n = len(accesses)
    ranked = sorted(by_expert.items(), key=lambda kv: -kv[1])
    # Pareto: share of requests served by the top 20% of distinct experts
    n_experts = len(ranked)
    top20 = ranked[: max(1, int(math.ceil(0.2 * n_experts)))]
    top20_share = sum(v for _, v in top20) / n if n else 0.0
    top1_share = (ranked[0][1] / n) if ranked and n else 0.0
    return {
        "n_accesses": n,
        "n_distinct_experts": n_experts,
        "top_experts": [{"expert": e, "count": c} for e, c in ranked[:10]],
        "top1_share": round(top1_share, 5),
        "top10_share": round(
            sum(v for _, v in ranked[: max(1, int(0.1 * n_experts))]) / n, 5)
        if n else 0.0,
        "top20_share": round(top20_share, 5),
        "gini": _gini([v for _, v in ranked]),
    }


def _gini(values: list[int]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    cum = 0.0
    for i, v in enumerate(values, start=1):
        cum += (2 * i - n - 1) * v
    denom = n * sum(values)
    return round(cum / denom, 5) if denom else 0.0


def reuse_metrics(stream: list[tuple[int, tuple[int, int]]]) -> dict:
    """Consecutive-token reuse and temporal reuse distance, computed on the
    deduped+shared access stream (same as cache1_policy_sim.build_stream), so
    the numbers match the sealed CACHE1.1 analysis (37.5% consecutive reuse)."""
    by_key: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ti, key in stream:
        by_key[key].append(ti)
    next_tok_reuse = 0
    total_pairs = 0
    reuse_distances: list[int] = []
    for key, toks in by_key.items():
        toks_sorted = sorted(set(toks))
        for a, b in zip(toks_sorted, toks_sorted[1:]):
            if b == a + 1:
                next_tok_reuse += 1
            total_pairs += 1
            reuse_distances.append(b - a)
    # Overall consecutive-token reuse rate: fraction of (layer,expert) keys
    # seen at token t that also appear at token t+1.
    seen: dict[tuple[int, int], int] = {}
    consec = 0
    total = 0
    for ti, key in stream:
        if key in seen and seen[key] == ti - 1:
            consec += 1
        seen[key] = ti
        total += 1
    return {
        "consecutive_token_reuse_rate": round(
            consec / total, 5) if total else 0.0,
        "same_layer_expert_pairs": total_pairs,
        "next_tok_reuse_fraction": round(
            next_tok_reuse / total_pairs, 5) if total_pairs else 0.0,
        "reuse_distance_median": float(statistics.median(reuse_distances))
        if reuse_distances else None,
        "reuse_distance_p90": _percentile(reuse_distances, 90),
    }


def _percentile(values: list[int], p: float) -> float | None:
    if not values:
        return None
    vs = sorted(values)
    k = (len(vs) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vs[int(k)])
    return float(vs[f] * (c - k) + vs[c] * (k - f))


def belady_per_gpu(stream: list, split: int, capacity: int) -> tuple[float, float]:
    """Oracle (Belady MIN) hit-rate ceiling per GPU with `capacity` experts,
    matching cache1_policy_sim.belady_per_gpu."""
    results = []
    for lo, hi in ((0, split), (split, 10 ** 9)):
        sub = [(tok, key) for tok, key in stream if lo <= key[0] < hi]
        nxt = defaultdict(list)
        for i, (_, key) in enumerate(sub):
            nxt[key].append(i)
        for k in nxt:
            nxt[k].append(10 ** 9)
        ptr = {k: 0 for k in nxt}
        cache: dict = {}
        hits = 0
        for i, (_, key) in enumerate(sub):
            if key in cache:
                hits += 1
            else:
                if len(cache) >= capacity:
                    victim = max(cache, key=lambda k: nxt[k][ptr[k]])
                    del cache[victim]
                cache[key] = i
            ptr[key] += 1
        results.append(100.0 * hits / max(1, len(sub)))
    return tuple(results)  # type: ignore[return-value]


def belady_ceiling(stream: list, n_layers: int,
                   budgets_gib: list[int]) -> dict:
    """Oracle hit-rate ceiling at GiB budgets per GPU (48 MiB FP16 experts)."""
    split = (n_layers + 1) // 2
    out = {}
    for budget_gib in budgets_gib:
        cap = int(budget_gib << 30) // EXPERT_BYTES
        h0, h1 = belady_per_gpu(stream, split, cap)
        out[str(budget_gib)] = {
            "gib_per_gpu": budget_gib,
            "experts_per_gpu": cap,
            "gpu0_hit_pct": round(h0, 2),
            "gpu1_hit_pct": round(h1, 2),
            "mean_hit_pct": round((h0 + h1) / 2.0, 2),
        }
    return out


def provider_and_cache(runtime: dict | None) -> dict:
    """Extract cache0/cache1/provider counters from a runtime snapshot."""
    if not runtime:
        return {}
    out: dict[str, object] = {}
    for key in ("cache0", "cache1", "shared_expert_host", "provider"):
        v = runtime.get(key)
        if isinstance(v, dict):
            out[key] = v
    return out


def decode_profile(gates: dict) -> dict:
    ms = gates.get("decode_timings_ms") or []
    msf = [float(x) for x in ms if isinstance(x, (int, float))]
    if not msf:
        return {"n_tokens": 0}
    return {
        "n_tokens": len(msf),
        "per_token_ms_median": round(statistics.median(msf), 2),
        "p50_ms": round(statistics.median(msf), 2),
        "p95_ms": round(_percentile_sorted(sorted(msf), 95) or 0.0, 2),
        "max_ms": round(max(msf), 2),
        "sum_ms": round(sum(msf), 2),
        "tok_per_s": round(len(msf) / (sum(msf) / 1000.0), 3)
        if sum(msf) else 0.0,
        "prefill_ms": round(float(gates.get("prefill_ms", 0.0)), 2),
        "decode_wall_s": round(float(gates.get("decode_wall_s", 0.0)), 3),
        "decode_tok_per_s_gate": round(
            float(gates.get("decode_tok_per_s", 0.0)), 3),
    }


def _percentile_sorted(vs: list[float], p: float) -> float | None:
    if not vs:
        return None
    k = (len(vs) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vs[int(k)])
    return float(vs[f] * (c - k) + vs[c] * (k - f))


def analyze(evidence: dict, cache_budgets: list[int]) -> dict:
    gates = evidence.get("gates", {})
    trace = gates.get("token_trace") or {}
    accesses = expert_accesses(trace)          # raw routed accesses
    stream, n_layers = build_stream(trace)     # deduped + shared, like the sim
    return {
        "run_id": evidence.get("run_id"),
        "stage": evidence.get("stage"),
        "model_revision": evidence.get("model_revision"),
        "verdict": evidence.get("verdict"),
        "gpu_count": evidence.get("gpu_count"),
        "tokens": _tokens(trace),
        "n_layers": n_layers,
        "n_expert_accesses": len(accesses),
        "n_stream_accesses": len(stream),
        "per_layer_hit_rates": per_layer_hit_rates(trace),
        "popularity": popularity(accesses),
        "reuse": reuse_metrics(stream),
        "belady_ceiling_gib": belady_ceiling(stream, n_layers, cache_budgets),
        "decode": decode_profile(gates),
        "runtime_after_warm": provider_and_cache(
            gates.get("runtime_after_warm")),
        "runtime_alternate_budget": provider_and_cache(
            gates.get("runtime_alternate_budget")),
        "fetch_stats": gates.get("memory", {}).get("fetch_stats"),
        "gates": {
            "tokens_match_sealed_ds10": gates.get("tokens_match_sealed_ds10"),
            "cold_warm_equal": gates.get("cold_warm_equal"),
            "deterministic_rerun": gates.get("deterministic_rerun"),
            "cache_capacity_variation_equal": gates.get(
                "cache_capacity_variation_equal"),
        },
    }


def bottleneck_ranking(reports: list[dict]) -> list[dict]:
    """Rank bottlenecks by contribution to total decode wall time."""
    ranked: list[dict] = []
    for r in reports:
        dec = r.get("decode", {})
        wall_ms = dec.get("sum_ms") or 0.0
        if wall_ms <= 0:
            continue
        # HTTP fetch share: estimated from fetch bytes vs decode time is not
        # directly measurable; use hit rate as the proxy for avoidable traffic.
        plh = r.get("per_layer_hit_rates", {})
        hit_rate = plh.get("total_hit_rate", 0.0) or 0.0
        fetch = r.get("fetch_stats") or {}
        fetch_bytes = fetch.get("bytes", 0) or 0
        ranked.append({
            "run_id": r.get("run_id"),
            "decode_wall_s": round(wall_ms / 1000.0, 2),
            "decode_tok_per_s": dec.get("tok_per_s"),
            "cache_hit_rate": hit_rate,
            "http_bytes": fetch_bytes,
            "http_requests": fetch.get("requests", 0),
            "n_expert_accesses": r.get("n_expert_accesses"),
            "estimate": (
                "decode wall time dominated by HTTP expert fetches" if
                hit_rate < 0.5 and fetch_bytes > 0 else
                "decode wall time not clearly fetch-bound"),
        })
    ranked.sort(key=lambda x: -(x["decode_wall_s"] or 0))
    return ranked


def write_markdown(reports: list[dict], out_path: str) -> None:
    lines = ["# Phase A — Cache Characterization Report\n"]
    for r in reports:
        lines.append(f"## Run {r.get('run_id')} (stage={r.get('stage')})")
        lines.append(f"- verdict: {r.get('verdict')}")
        lines.append(f"- expert accesses: {r.get('n_expert_accesses')}")
        plh = r.get("per_layer_hit_rates", {})
        lines.append(f"- overall cache hit rate: {plh.get('total_hit_rate')}")
        pop = r.get("popularity", {})
        lines.append("- popularity: top1={top1_share} top10={top10_share} "
                     "top20={top20_share} gini={gini} "
                     "(distinct={n_distinct_experts})".format(**pop))
        reuse = r.get("reuse", {})
        lines.append(
            "- consecutive-token reuse: {consecutive_token_reuse_rate} "
            "(p90 reuse distance {reuse_distance_p90})".format(**reuse))
        bel = r.get("belady_ceiling_gib", {})
        bel_str = ", ".join(
            f"{k}GiB→{v['mean_hit_pct']}%" for k, v in sorted(bel.items()))
        lines.append(f"- Belady (oracle) ceiling: {bel_str}")
        dec = r.get("decode", {})
        lines.append(
            f"- decode: {dec.get('n_tokens')} tokens, "
            f"median {dec.get('p50_ms')} ms/tok, p95 {dec.get('p95_ms')} ms, "
            f"max {dec.get('max_ms')} ms, {dec.get('tok_per_s')} tok/s")
        fs = r.get("fetch_stats") or {}
        lines.append(f"- HTTP: {fs.get('requests')} requests, "
                     f"{round((fs.get('bytes') or 0) / 1e9, 2)} GB")
        lines.append("")
    ranking = bottleneck_ranking(reports)
    lines.append("## Bottleneck ranking (by decode wall time)\n")
    lines.append("| run | decode_s | tok/s | hit_rate | http_GB | estimate |")
    lines.append("|---|---|---|---|---|---|")
    for b in ranking:
        lines.append(
            f"| {b['run_id']} | {b['decode_wall_s']} | "
            f"{b['decode_tok_per_s']} | {b['cache_hit_rate']} | "
            f"{round((b['http_bytes'] or 0) / 1e9, 2)} | {b['estimate']} |")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence", nargs="+", help="ds10-evidence.json paths")
    ap.add_argument("--output", default=None, help="JSON report path")
    ap.add_argument("--summary", default=None, help="Markdown summary path")
    ap.add_argument("--cache-budgets", default="1,2,4,8",
                    help="Belady ceiling budgets (units)")
    args = ap.parse_args()
    budgets = [int(x) for x in args.cache_budgets.split(",")]
    reports = []
    for p in args.evidence:
        try:
            reports.append(analyze(_load_evidence(p), budgets))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: {p}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if args.output:
        Path(args.output).write_text(
            json.dumps(reports, indent=1), encoding="utf-8")
        print(f"wrote {args.output}")
    if args.summary:
        write_markdown(reports, args.summary)
        print(f"wrote {args.summary}")
    else:
        write_markdown(reports, "/tmp/phase_a_report.md")
        print(Path("/tmp/phase_a_report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
