#!/usr/bin/env python3
"""A/B run-count / power planner for small exact-run effects (advisory only).

Part of research/exact-staging. Answers, from MEASURED noise:
  how many matched pairs are needed to distinguish a 1/2/3/5% decode effect?

Method (small-sample honest):
  - Paired-difference SD sigma_d estimated from the two host-reuse pairs
    (n=2 -> EXPLORATORY_STATISTIC) and, conservatively, the between-notebook
    baseline SD (n=3, DERIVED). No normality claim: results are decision
    rules on direction consistency and a z-approximation labeled as such.
  - Direction rule: P(single pair shows improvement) = Phi(E/sigma_d);
    P(all n pairs improve) = Phi(E/sigma_d)^n. Require >= 0.90.
  - Mean rule (approximate z, exploratory): n = (z * sigma_d / E)^2, z=2.
  - Longer-decode strategy: mechanical 1/sqrt scaling with an autocorrelation
    effective-sample correction, labeled SIMULATED/EXPLORATORY, plus the
    workload caveat (longer generation changes the route/recurrence mix and
    can change the effect itself; direct comparison then invalid).

This tool does NOT change formal acceptance rules. It prices evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent / "research" / "exact-staging"
NOISE = RESEARCH / "results" / "ab_noise.json"

BASELINE_MEAN_S = 72.281      # mean of 3 same-config baselines (ab_noise.json)
Z = 2.0                       # ~95% one-sided for the mean rule (approximation)
DIRECTION_TARGET = 0.90       # P(all pairs improve)


def _baseline_mean(noise: dict) -> float:
    """Live value from ab_noise.json; falls back to the documented constant."""
    v = noise.get("baseline_variance", {}).get("mean_s")
    return float(v) if v else BASELINE_MEAN_S


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pairs_direction_rule(effect_s: float, sigma_d: float,
                         max_pairs: int = 8) -> int | None:
    """Smallest n with P(all n paired deltas negative) >= DIRECTION_TARGET.
    p^n DECREASES in n, so if p^1 < target no n works -> None."""
    if effect_s <= 0:
        return None
    p = phi(effect_s / sigma_d)
    if p <= 0.5:
        return None
    for n in range(1, max_pairs + 1):
        if p ** n >= DIRECTION_TARGET:
            return n
    return None


def pairs_mean_rule(effect_s: float, sigma_d: float) -> int | None:
    if effect_s <= 0:
        return None
    return max(1, math.ceil((Z * sigma_d / effect_s) ** 2))


def z_for_p(p: float) -> float:
    """Inverse normal CDF via bisection (no scipy dependency)."""
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def effective_sample_size(n_tokens: int, rho1: float) -> int:
    """First-order autocorrelation correction: n_eff ~ n(1-rho)/(1+rho)."""
    if n_tokens <= 0:
        return 0
    return max(1, round(n_tokens * (1.0 - rho1) / (1.0 + rho1)))


def build(noise: dict) -> dict:
    base_mean = _baseline_mean(noise)
    sd_between = noise["baseline_variance"]["sd_s"]
    deltas = [p["delta_s"] for p in noise["paired_ab_effect"]["pairs"]]
    sd_paired = statistics.stdev(deltas) if len(deltas) > 1 else sd_between
    # lag-1 autocorrelation of per-token walls (mean over pairs' baselines):
    rho = statistics.mean(
        noise["per_token_paired"][p]["lag1_autocorr_baseline"]
        for p in noise["per_token_paired"]
        if isinstance(noise["per_token_paired"].get(p), dict)
        and "lag1_autocorr_baseline" in noise["per_token_paired"][p])

    effects = {f"{pct}_pct": pct / 100 * base_mean for pct in (1, 2, 3, 5)}
    sigma_scenarios = {
        "paired_delta_sd_exploratory_n2": round(sd_paired, 3),
        "between_notebook_sd_conservative": round(sd_between, 3),
    }
    table = []
    for sname, sig in sigma_scenarios.items():
        for ename, e_s in effects.items():
            nd = pairs_direction_rule(e_s, sig)
            table.append({
                "sigma_d_s": sig, "sigma_source": sname,
                "effect": ename, "effect_s": round(e_s, 3),
                "pairs_direction_rule_p90": (
                    nd if nd is not None else "UNATTAINABLE (<=8 pairs; "
                    "single-pair direction probability < 0.90)"),
                "pairs_mean_rule_z2": pairs_mean_rule(e_s, sig),
                "snr_single_pair": round(e_s / sig, 2),
                "labels": ["DERIVED (rules)", "EXPLORATORY_STATISTIC (sigma)"],
            })

    # Strategy assessment -------------------------------------------------
    n_tok, n_eff = 15, effective_sample_size(15, rho)
    longer = {
        "mechanical_model": "sigma_wall/T ~ sigma_rel / sqrt(n_eff); "
                            "n_eff = n(1-rho1)/(1+rho1) (first-order correction)",
        "per_token_walls_n": n_tok,
        "lag1_autocorr_mean": round(rho, 3),
        "effective_independent_tokens_15": n_eff,
        "longer_decode_example": {
            "tokens": 64,
            "n_eff_if_same_rho": effective_sample_size(64, rho),
            "expected_rel_sd_reduction_vs_15tok": round(
                math.sqrt(n_eff / effective_sample_size(64, rho)), 3),
            "label": "SIMULATED (mechanical model; UNVALIDATED)",
            "caveat": "longer generation changes the recurrence/miss mix and "
                      "therefore the effect itself; direct comparison to the "
                      "15-token effect is then INVALID. Also invalidates "
                      "replay-validated counter comparability.",
        },
    }
    strategies = [
        {"strategy": "more matched pairs",
         "cost": "2 Kaggle runs per pair",
         "when": "always; it is the only lever that scales evidence without "
                 "changing the workload",
         "quantified": "see pairs table"},
        {"strategy": "interleaved baseline/candidate in ONE notebook session "
                     "(A/B/A/B or A/B)",
         "cost": "same total runs; removes between-notebook offset from the "
                 "paired delta",
         "when": "highest-value structural change: between-notebook SD "
                 f"(~{sd_between:.2f} s) is the dominant noise term and "
                 "pairing within a session should collapse it",
         "blocker": "within-session SD is UNKNOWN until measured once "
                    "(BLOCKED_BY_MISSING_MEASUREMENT)",
         "recommendation": "measure once in the next A/B by logging both arms "
                           "from one session if the harness permits"},
        {"strategy": "longer decode",
         "quantified": longer,
         "when": "only as a SEPARATE experiment; do not mix with 15-token "
                 "comparisons"},
        {"strategy": "paired notebooks (both arms in notebook 1, repeat in "
                     "notebook 2)",
         "cost": "4 runs",
         "when": "this is exactly the host-reuse design; it works and should "
                 "be the default replication shape"},
        {"strategy": "repeated same-notebook arms",
         "cost": ">=4 runs in one session",
         "when": "cheapest way to estimate within-session SD; add one repeat "
                 "arm if a session is cheap to extend"},
    ]

    # Pack-cap specific answers (effect = SIMULATED 51 misses x service time)
    model = json.loads((RESEARCH / "results" / "recalibrated_model.json").read_text())
    miss_service = model["model"]["miss_service_ms"]
    rec = model["recommended_candidate"]
    pack_effect_s = rec["miss_reduction_vs_8gib5"] * miss_service / 1000
    pack_plan = {
        "predicted_effect_s": round(pack_effect_s, 3),
        "label": "SIMULATED (recalibrated model; the A/B is the arbiter)",
        "is_one_pair_informative": {
            "answer": pack_effect_s / sd_paired >= 2,
            "snr_vs_paired_sigma": round(pack_effect_s / sd_paired, 2),
            "snr_vs_between_sigma": round(pack_effect_s / sd_between, 2),
            "note": "IF the simulated ~2.5 s holds, one pair is informative "
                    "(SNR ~2.7-2.8) and two pairs (pre-budgeted) give "
                    "direction consistency >=97% and a mean-rule pass. The "
                    "risk is not power but EFFECT REALITY: the marginal "
                    "misses removed at a larger budget are the easiest ones, "
                    "so realized service-per-avoided-miss may differ from the "
                    "global 48.7 ms marginal",
        },
        "pairs_budgeted_upfront": 2,
        "is_15_tokens_enough": {
            "answer": True,
            "reason": "predicted effect ~2.5 s exceeds even the conservative "
                      "between-notebook SD (~0.93 s) by ~2.7x; per-token "
                      "resolution is not needed",
            "caveat": "15 tokens is also what the replay counters validate; "
                      "changing length would desync the comparison",
        },
        "would_longer_decode_help": {
            "answer": "not for THIS hypothesis",
            "reason": "pack-miss savings scale with the miss stream, not with "
                      "token count alone; a longer run adds prefill and "
                      "changes recurrence mix, invalidating counter matching",
        },
        "stopping_rule_preregistered": {
            "budget": "2 matched pairs, both run regardless of pair-1 "
                      "outcome unless FUTILITY fires",
            "futility_after_pair_1": "STOP if candidate wall > baseline wall "
                                     "+ 1 baseline SD (~0.93 s) — the "
                                     "hypothesis predicted an improvement; a "
                                     "slower candidate is an obvious null",
            "no_early_accept": "never stop after pair 1 on a favorable "
                               "result; pair 2 is mandatory replication",
            "after_pair_2": "informative-positive iff both deltas negative "
                            "AND mean delta > baseline MAD (~0.905 s); "
                            "otherwise UNRESOLVED (not evidence of absence)",
            "no_post_hoc_stopping": "this rule is fixed before the first run; "
                                    "no result-driven re-interpretation",
        },
        "effect_warranting_formal_acceptance": {
            "threshold": "both pairs improve AND mean delta >= 2x baseline SD "
                         f"(~{round(2 * sd_between, 2)} s) AND correctness "
                         "gates pass on all arms",
            "then": "hand to the campaign's formal acceptance flow (this "
                    "package does not grant acceptance)",
        },
    }

    return {
        "schema": "exact-staging/ab-power-plan-v1",
        "inputs": {
            "baseline_mean_s": base_mean,
            "baseline_sd_s": sd_between,
            "paired_delta_sd_s_exploratory": round(sd_paired, 3),
            "labels": ["DERIVED", "EXPLORATORY_STATISTIC (paired sigma, n=2)"],
        },
        "pairs_required_table": table,
        "smallest_detectable_effect": {
            "definition": "effect with SNR>=1 vs conservative sigma (single "
                          "pair, direction-only)",
            "s_single_pair_direction_only_s": round(sd_between, 3),
            "s_single_pair_direction_only_pct": round(
                100 * sd_between / base_mean, 2),
            "s_two_pairs_direction_p90_s": round(
                sd_between * z_for_p(math.sqrt(DIRECTION_TARGET)), 3),
            "s_two_pairs_direction_p90_pct": round(
                100 * sd_between * z_for_p(math.sqrt(DIRECTION_TARGET))
                / base_mean, 2),
            "note": "with the pre-registered 2-pair design, an effect of "
                    "~1.6x sigma (~2.1%) is the practical floor for a clean "
                    "directional read; 1% effects are UNATTAINABLE under the "
                    "direction rule at any pair count <=8 and need the "
                    "interleaved-session strategy first (within-session SD "
                    "is currently UNKNOWN)",
        },
        "strategies": strategies,
        "pack_cap_experiment_plan": pack_plan,
        "advisory_only": "does not modify formal acceptance rules; no TPS "
                         "claims; sigma estimates are n<=3 statistics",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", type=Path, default=NOISE)
    ap.add_argument("--out", type=Path,
                    default=RESEARCH / "results" / "ab_power_plan.json")
    args = ap.parse_args()
    noise = json.loads(args.noise.read_text())
    out = build(noise)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({
        "smallest_detectable_effect": out["smallest_detectable_effect"],
        "pairs_table_1_2_pct": [t for t in out["pairs_required_table"]
                                if t["effect"] in ("1_pct", "2_pct") and
                                t["sigma_source"].startswith("between")],
        "pack_plan": {k: out["pack_cap_experiment_plan"][k]
                      for k in ("predicted_effect_s", "is_one_pair_informative",
                                "pairs_budgeted_upfront",
                                "is_15_tokens_enough")},
    }, indent=1))


if __name__ == "__main__":
    main()
