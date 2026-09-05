#!/usr/bin/env python3
"""Offline matched-experiment (A/B) noise & effect analysis for dee.cpp exact runs.

Part of research/exact-staging (goal: statistics for ~1-4% effects).

Consumes ONLY machine-readable run JSON:
  - results/seal_host_reuse/derived_per_step_decode.json (hash-traceable to
    seal 1486403; committed derived extracts, decode walls + per-step walls)
  - optional extra profiles via --profile TAG=PATH (e.g. v63/v64 sealed
    campaign profiles); each is sha256-recorded as provenance.

Label discipline (enforced in the output schema):
  MEASURED              read from run files (with provenance)
  DERIVED               arithmetic on measured values
  EXPLORATORY_STATISTIC small-sample statistic; NOT confirmatory
  SIMULATED             model output; never a measurement
  UNKNOWN               missing measurement, never inferred

Small-sample honesty: n=3 baselines and n=2 pairs. Nothing here assumes
normality is proven; raw values, median, and MAD are reported alongside
means; bootstrap intervals are labeled exploratory and are degenerate for
n<3 distinct resamples. Per-token walls are serially dependent (one token's
route mix sets up the next) and are NEVER treated as independent replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent / "research" / "exact-staging"
DERIVED = RESEARCH / "results" / "seal_host_reuse" / "derived_per_step_decode.json"

SEAL = "14864034a7354e0e29e11c1c09f18b0863afe6a0"
RECORD_BYTES = 13_369_344

# Config groups (label; provenance in RECALIBRATION.md / EVIDENCE_INVENTORY
# of research/exact-critical-path @ 2db0fde for the lane eras).
GROUP_BASELINE = "exact-on-demand-6lane-noreuse"
GROUP_REUSE = "exact-on-demand-6lane-reuse"

MATCHED_PAIRS = [
    {"pair_id": "reuse-original",
     "notebook_series": "dee.cpp/tmp/host-reuse-ab-20260904 (v1/v2 fetches)",
     "baseline_run": "ab_baseline", "candidate_run": "ab_candidate",
     "config": GROUP_BASELINE + " -> " + GROUP_REUSE,
     "only_delta": "host-buffer reuse flag"},
    {"pair_id": "reuse-replication",
     "notebook_series": "dee.cpp/tmp/host-reuse-rep-20260904 (v1/v2 fetches)",
     "baseline_run": "rep_baseline", "candidate_run": "rep_candidate",
     "config": GROUP_BASELINE + " -> " + GROUP_REUSE,
     "only_delta": "host-buffer reuse flag"},
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_runs(extra_profiles: list[tuple[str, Path]] | None = None) -> dict:
    """name -> run dict. Extra profiles get wall + per-token from profile.json."""
    raw = json.loads(DERIVED.read_text())["runs"]
    runs = {}
    for name, r in raw.items():
        runs[name] = {
            "decode_wall_s": r["decode_wall_s"],
            "step_walls_ms": r["step_walls_ms"],
            "group": GROUP_REUSE if name.endswith("candidate") else GROUP_BASELINE,
            "profiling_perturbation": name == "profiled_candidate",
            "provenance": f"seal {SEAL} (derived_per_step_decode.json)",
        }
    for tag, p in (extra_profiles or []):
        d = json.loads(p.read_text())
        walls = d.get("decode_timings_ms") or []
        runs[tag] = {
            "decode_wall_s": d["decode_wall_s"],
            "step_walls_ms": walls,
            "group": "exact-3lane-era (per EVIDENCE_INVENTORY @2db0fde)",
            "profiling_perturbation": bool(d.get("profile_stages_enabled")),
            "provenance": f"{p} sha256={sha256(p)}",
        }
    return runs


def baseline_stats(walls: list[float]) -> dict:
    """Spread of same-config baseline walls. MEASURED inputs, DERIVED stats."""
    n = len(walls)
    med = statistics.median(walls)
    mean = sum(walls) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in walls) / (n - 1)) if n > 1 else None
    mad = statistics.median([abs(x - med) for x in walls])
    return {
        "n_runs": n,
        "walls_s": [round(w, 3) for w in walls],
        "mean_s": round(mean, 3),
        "median_s": round(med, 3),
        "sd_s": round(sd, 3) if sd is not None else None,
        "mad_s": round(mad, 3),
        "spread_s": round(max(walls) - min(walls), 3),
        "labels": ["MEASURED (walls)", "DERIVED (summary stats)"],
    }


def paired_effect(pairs: list[dict], runs: dict) -> dict:
    """Paired deltas for matched baseline/candidate pairs."""
    deltas, pcts, rows = [], [], []
    for pr in pairs:
        b = runs[pr["baseline_run"]]["decode_wall_s"]
        c = runs[pr["candidate_run"]]["decode_wall_s"]
        d = c - b
        deltas.append(d)
        pcts.append(100 * (c / b - 1))
        rows.append({
            "pair_id": pr["pair_id"],
            "baseline_run": pr["baseline_run"], "baseline_s": b,
            "candidate_run": pr["candidate_run"], "candidate_s": c,
            "delta_s": round(d, 3), "delta_pct": round(pcts[-1], 2),
            "correctness": "both arms ACCEPT_CORRECTNESS (seal)",
            "labels": ["MEASURED (walls)", "DERIVED (delta)"],
        })
    mean = sum(deltas) / len(deltas)
    med = statistics.median(deltas)
    out = {
        "n_pairs": len(deltas),
        "pairs": rows,
        "mean_delta_s": round(mean, 3),
        "median_delta_s": round(med, 3),
        "mean_delta_pct": round(100 * mean / statistics.mean(
            [runs[p["baseline_run"]]["decode_wall_s"] for p in pairs]), 2),
        "direction_consistent": all(d < 0 for d in deltas),
        "sign_consistency": f"{sum(d < 0 for d in deltas)}/{len(deltas)} negative",
        "labels": ["DERIVED (deltas)", "EXPLORATORY_STATISTIC (summary: n=2)"],
    }
    return out


def bootstrap_ci(deltas: list[float], iters: int = 10_000,
                 seed: int = 20260904) -> dict:
    """Percentile bootstrap of the mean delta. EXPLORATORY; degenerate for
    n<3 (resampling 2 values yields only 3 distinct means)."""
    n = len(deltas)
    rng = random.Random(seed)
    if n < 2:
        return {"ci95_mean_delta_s": None,
                "note": "insufficient pairs", "label": "UNKNOWN"}
    means = []
    for _ in range(iters):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[min(iters - 1, int(0.975 * iters))]
    return {
        "ci95_mean_delta_s": [round(lo, 3), round(hi, 3)],
        "iters": iters, "seed": seed,
        "label": "EXPLORATORY_STATISTIC",
        "caveat": (f"n={n} pairs; bootstrap resamples {n} values, so the "
                   "interval is degenerate/overconfident — direction only"),
    }


def lag1_autocorr(xs: list[float]) -> float:
    n = len(xs)
    m = sum(xs) / n
    num = sum((xs[i] - m) * (xs[i + 1] - m) for i in range(n - 1))
    den = sum((x - m) ** 2 for x in xs)
    return num / den if den else 0.0


def per_token_paired_analysis(pairs: list[dict], runs: dict) -> dict:
    """Per-token paired timing. 15 token walls are a time series on ONE run,
    NOT 15 independent replicates — reported with dependence caveats only."""
    out = {}
    for pr in pairs:
        a = runs[pr["baseline_run"]]["step_walls_ms"]
        b = runs[pr["candidate_run"]]["step_walls_ms"]
        if not a or not b or len(a) != len(b):
            out[pr["pair_id"]] = {"per_token": "UNKNOWN (missing/unequal steps)"}
            continue
        deltas = [bb - aa for aa, bb in zip(a, b)]
        mean_d = sum(deltas) / len(deltas)
        neg = sum(d < 0 for d in deltas)
        out[pr["pair_id"]] = {
            "n_tokens": len(deltas),
            "mean_delta_ms": round(mean_d, 1),
            "median_delta_ms": round(statistics.median(deltas), 1),
            "tokens_improved": f"{neg}/{len(deltas)}",
            "lag1_autocorr_baseline": round(lag1_autocorr(a), 3),
            "lag1_autocorr_candidate": round(lag1_autocorr(b), 3),
            "dependence_caveat": (
                "per-token walls are serially dependent (autocorrelated route "
                "mix); the 15 token walls of one run are ONE replicate each, "
                "so per-token sign counts are descriptive, not a test"),
            "label": "EXPLORATORY_STATISTIC",
        }
    return out


def signal_to_noise(mean_delta_s: float, sigma_s: float) -> dict:
    snr = abs(mean_delta_s) / sigma_s if sigma_s else None
    if snr is None:
        verdict = "UNKNOWN"
    elif snr < 1:
        verdict = "inside baseline noise (not detectable without pairing)"
    elif snr < 2:
        verdict = "suggestive; requires matched pairs + replication"
    else:
        verdict = "detectable with a small matched-pair budget"
    return {"snr_mean_delta_over_baseline_sd": round(snr, 2) if snr else None,
            "verdict": verdict, "label": "DERIVED"}


def profiling_perturbation(runs: dict) -> dict:
    """Profiled reuse candidate vs unprofiled reuse replication candidate."""
    a = runs.get("rep_candidate", {}).get("decode_wall_s")
    b = runs.get("profiled_candidate", {}).get("decode_wall_s")
    if a is None or b is None:
        return {"delta_s": "UNKNOWN"}
    return {
        "rep_candidate_unprofiled_s": a,
        "profiled_candidate_s": b,
        "delta_s": round(b - a, 3),
        "note": "stage profiling perturbation at decode level appears small "
                "but the two runs differ in notebook session too; treat as "
                "EXPLORATORY bound only",
        "label": "EXPLORATORY_STATISTIC",
    }


def build(extra_profiles: list[tuple[str, Path]]) -> dict:
    runs = load_runs(extra_profiles)
    base_names = [n for n, r in runs.items()
                  if r["group"] == GROUP_BASELINE and not r["profiling_perturbation"]]
    base = baseline_stats([runs[n]["decode_wall_s"] for n in sorted(base_names)])
    eff = paired_effect(MATCHED_PAIRS, runs)
    boot = bootstrap_ci([p["delta_s"] for p in eff["pairs"]])
    tok = per_token_paired_analysis(MATCHED_PAIRS, runs)
    sd = base["sd_s"] or 0.0
    out = {
        "schema": "exact-staging/ab-noise-v1",
        "source_seal": SEAL,
        "runs": {n: {"decode_wall_s": r["decode_wall_s"],
                     "group": r["group"],
                     "profiling_perturbation": r["profiling_perturbation"],
                     "provenance": r["provenance"]}
                 for n, r in sorted(runs.items())},
        "baseline_variance": {
            "group": GROUP_BASELINE,
            **base,
            "between_notebook_sd_s": base["sd_s"],
            "within_notebook_sd_s": "UNKNOWN (no same-config repeats within "
                                    "one notebook; BLOCKED_BY_MISSING_MEASUREMENT)",
        },
        "paired_ab_effect": eff,
        "bootstrap_ci": boot,
        "per_token_paired": tok,
        "profiling_perturbation": profiling_perturbation(runs),
        "signal_to_noise_reuse_effect": signal_to_noise(eff["mean_delta_s"], sd),
        "lane_era_context": {
            "note": "v63/v64 (3-lane era) vs v65/host-reuse (6-lane) are "
                    "DIFFERENT configs; included for context only, excluded "
                    "from baseline variance",
            "walls_s": {n: runs[n]["decode_wall_s"] for n in sorted(runs)
                        if "lane-era" in runs[n]["group"]} or "not provided",
        },
        "key_numbers_for_planning": {
            "baseline_sd_s": base["sd_s"],
            "baseline_mad_s": base["mad_s"],
            "baseline_spread_s": base["spread_s"],
            "observed_reuse_mean_delta_s": eff["mean_delta_s"],
            "labels": ["DERIVED"],
        },
        "claim_rules": "no TPS claims; no acceptance claims; measured vs "
                       "exploratory labels enforced per field",
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", action="append", default=[],
                    help="extra run profile TAG=PATH (sha256 recorded)")
    ap.add_argument("--out", type=Path,
                    default=RESEARCH / "results" / "ab_noise.json")
    args = ap.parse_args()
    extras = []
    for spec in args.profile:
        tag, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"profile missing: {spec}")
        extras.append((tag, p))
    out = build(extras)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({
        "baseline_variance": out["baseline_variance"],
        "paired_ab_effect": {k: out["paired_ab_effect"][k] for k in
                             ("n_pairs", "mean_delta_s", "median_delta_s",
                              "direction_consistent")},
        "bootstrap_ci": out["bootstrap_ci"]["ci95_mean_delta_s"],
        "snr": out["signal_to_noise_reuse_effect"],
    }, indent=1))


if __name__ == "__main__":
    main()
