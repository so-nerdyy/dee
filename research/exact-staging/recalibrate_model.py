#!/usr/bin/env python3
"""Recalibrated critical-path model + host-pack memory frontier (v2).

Ingests (read-only, sha256-gated):
  1. sealed v65 (decode accounting)
  2. original host-buffer matched A/B        (codex/host-reuse-evidence seal)
  3. independent host-buffer replication
  4. valid profiled reuse candidate
  5. exact-staging causality result          (in-stage lead impossible)
  6. Muse route-pipeline lead=0 result       (hash layers 0-2 only special case)

Model (deliberately simple, observational, physicality-gated):

    wall_step ~= serial_residual_ms + miss_service_ms * ssd_misses_step

- Fit on v65's 15 decode steps (orthogonal regression; slope>0, intercept>0).
- The 5 host-reuse runs share v65's exact per-step reads/copies tables, so
  they are HELD-OUT validation of the level, not extra fit points.
- `miss_service_ms` is effective wall-service per storage miss (marginal,
  observational). It is NOT raw disk bandwidth and NOT decomposed latency:
  aggregate timing cannot establish which overlapped interval is exposed.
- Buffer-reuse has NO mechanistic lever in this model: measured evidence
  (seal 1486403) shows fill_batch_wall_ms unchanged with reuse ON and
  mmap_to_pinned still present in the profiled candidate; buffer
  allocation/reservation is already overlapped by the fill worker pool.
  The old event simulator's -12.3% was a structural artifact (it charged
  the pack copy as a serial batch-blocking stage; see RECALIBRATION.md).

Every counterfactual wall number produced here is SIMULATED.
Measured numbers are labeled measured and carry provenance hashes.

Outputs (research/exact-staging/results/):
  recalibrated_model.json   model + calibration + validation + audit
  host_pack_frontier.csv    per-budget replay/simulated-wall rows
  memory_frontier.json      per-budget host-memory envelope + safety class
  next_ab.json              the ONE recommended live A/B
  pread_kaggle_package.json ready-to-run concurrent-pread measurement package
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # worktree root
sys.path.insert(0, str(ROOT / "dee.cpp" / "experiments" / "exact_staging"))
sys.path.insert(0, str(ROOT / "tools"))
from bounded_staging_queue import pack_replay  # noqa: E402

RECORD_BYTES = 13_369_344
RECORD_GIB = RECORD_BYTES / 2**30            # 0.012452... GiB
GIB = 2**30
GB = 1e9
SEAL = "14864034a7354e0e29e11c1c09f18b0863afe6a0"

# Measured decode walls (s), from the seal + sealed v65 (see derived JSON).
MEASURED = {
    "v65_sealed":         72.267,
    "ab_baseline":        71.362,
    "ab_candidate":       70.691,
    "rep_baseline":       73.213,
    "rep_candidate":      71.284,
    "profiled_candidate": 71.179,
}

# Relative paths (from the main checkout root) of every seal input consumed,
# with their sha256 at the seal commit. The script refuses to run on mismatch.
SEAL_INPUTS = {
    "dee.cpp/tmp/host-reuse-ab-20260904/v1-fetch-20260904T171320Z/output/profile.json":
        "a95f85796418b722a36645ed4047a7f8a88123d5b9807d310c1b971e7afb2203",
    "dee.cpp/tmp/host-reuse-ab-20260904/v2-fetch-20260904T173515Z/output/profile.json":
        "c7dd63ffaa075bbcec984a292852cd7787b128cfca96f52c5cf7e49a452288ab",
    "dee.cpp/tmp/host-reuse-rep-20260904/v1-fetch-20260904T180559Z/output/profile.json":
        "5c7e3a08834e23b82d5d4d1a44c0d91e38d5a0daaf2284aede6c01b5f4f082a4",
    "dee.cpp/tmp/host-reuse-rep-20260904/v2-fetch-20260904T183100Z/output/profile.json":
        "72a25a83bbb8f9ac364809e366cc9d4cbaaf7661ca5758a9279ca298345e8a07",
    "dee.cpp/tmp/host-reuse-profile2-20260904/v2-fetch-20260904T194914Z/output/profile.json":
        "0a80b4b60262f84b4d41ac9932cd62b42abdcffb4dcc7c9debb14eff07ecc3ca",
    "dee.cpp/tmp/host-reuse-ab-20260904/v1-fetch-20260904T171320Z/output/memory.json":
        "0c44f26469722a7273c35f164f4357666b5fc1c1928cf29fce977be5847a5f1d",
    "dee.cpp/tmp/host-reuse-ab-20260904/v2-fetch-20260904T173515Z/output/memory.json":
        "53bec31467c23ce1e5b150ea0a9af7fa3aa17ee05994bea41fa4cc09b25a6df3",
    "dee.cpp/tmp/host-reuse-rep-20260904/v1-fetch-20260904T180559Z/output/memory.json":
        "5ebec95f7718e144abca15a4e7ba36141b2928eca286a32193472cf50964055e",
    "dee.cpp/tmp/host-reuse-rep-20260904/v2-fetch-20260904T183100Z/output/memory.json":
        "bb3f1b970d2c7b7e32b26ee2081d57db0d4367a63d8ce984391d80df197bb958",
    "dee.cpp/tmp/host-reuse-profile2-20260904/v2-fetch-20260904T194914Z/output/memory.json":
        "00dc7fbe727f8911aabeb16685d9514ab6deb6ae9fcd018c6925f53c2eed1926",
    "dee.cpp/tmp/host-reuse-profile2-20260904/v2-fetch-20260904T194914Z/output/routed_experts.jsonl":
        "3ec6b3aff604e6f6fa176e9f744a52c6428abe9a8db1b52a5d2f3dfeadf3f0ce",
}

MAIN_CHECKOUT = ROOT.parent / "dynamic_expert_eviction"

BUDGETS_GIB = [8.5, 9.0, 9.5, 10.0, 10.5, 10.75, 11.0, 11.25, 11.5,
               12.0, 12.25, 12.75]

# Conservative memory-envelope constants (all measured unless noted):
NONPACK_HWM_GIB = 5.817          # max over the 5 seal runs (VmHWM - pack occupied)
SYSTEM_REST_GIB = 0.91           # MemTotal - process - min-checkpoint-available
UNMEASURED_GROWTH_GIB = 0.5      # unk: allocator/page-cache growth at larger pack
SAFETY_HEADROOM_GIB = 2.0        # required free headroom for SAFE
LIMIT_DECIMAL_GB = 32.0          # the project's total-host-RAM thesis, decimal GB
LIMIT_GIB = LIMIT_DECIMAL_GB * GB / GIB   # 29.80 GiB
# Empirical OOM ledger (harness header comments, v8/v9/v11-era):
OOM_LEDGER = {
    "survived_pack_total_gib": 25.74,   # v8: 12.87+12.87, no madvise
    "oom_pack_total_gib": 27.68,        # v9: 15.79+11.89 (clean OOM)
    "oom_confounded": "v11 26.0 GiB + madvise DONTNEED re-fault thrash",
}


def verify_seal_inputs() -> dict:
    """sha256-gate every seal input; raises on missing/mismatched files."""
    got = {}
    for rel, want in SEAL_INPUTS.items():
        p = MAIN_CHECKOUT / rel
        if not p.exists():
            raise SystemExit(f"seal input missing: {rel}")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != want:
            raise SystemExit(f"seal input hash mismatch: {rel}")
        got[rel] = h
    return got


def load_derived() -> dict:
    return json.loads(
        (HERE / "results/seal_host_reuse/derived_per_step_decode.json").read_text()
    )["runs"]


def fit_v65(v65: dict) -> dict:
    """Orthogonal (total least squares) fit of wall = a + b*misses on v65.

    ODR because both misses (regressor) and walls carry noise; enforced
    physicality: slope>0, intercept>0. Returns model + fit diagnostics.
    """
    x = [float(v) for v in v65["ssd_reads"]]
    y = [float(w) for w in v65["step_walls_ms"]]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((a - my) ** 2 for a in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    slope_tls = (syy - sxx + math.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    slope_tls = sxy / abs(sxy) * abs(slope_tls)     # keep sign convention
    if not (slope_tls > 0):
        raise SystemExit("nonphysical fit (slope<=0); reject model")
    intercept = my - slope_tls * mx
    if not (intercept > 0):
        raise SystemExit("nonphysical fit (intercept<=0); reject model")
    pred = [intercept + slope_tls * a for a in x]
    resid = [b - p for b, p in zip(y, pred)]
    ss_res = sum(r ** 2 for r in resid)
    ss_tot = syy
    # ODR slope CI (approximate, error-in-both-variables):
    se = math.sqrt(ss_res / max(1, n - 2) / (sxx + syy / max(1e-9, slope_tls ** 2)))
    return {
        "form": "wall_ms_step = serial_residual_ms + miss_service_ms * ssd_misses_step",
        "miss_service_ms": round(slope_tls, 4),
        "serial_residual_ms_per_step": round(intercept, 2),
        "fit_method": "orthogonal (total least squares) on v65 15 decode steps",
        "r_squared": round(1 - ss_res / ss_tot, 4),
        "slope_se_ms": round(se, 4),
        "slope_ci95_pct_of_slope": round(100 * 1.96 * se / slope_tls, 2),
        "physicality_gates": "slope>0, intercept>0 (both enforced; run fails otherwise)",
        "interpretation": (
            "miss_service_ms is EFFECTIVE wall service per storage miss "
            "(marginal, observational; includes correlated allocation/"
            "orchestration). NOT raw disk bandwidth, NOT decomposed latency."
        ),
        "ordinary_future_layer_lead": 0,
        "future_layer_lead_note": (
            "route(L+1) requires combine(L): ordinary in-step lead-N staging is "
            "IMPOSSIBLE_WITHOUT_FUTURE_ROUTE (CAUSALITY_AUDIT.md; Muse "
            "research/route-pipeline lead=0). Hash layers 0-2 early staging "
            "(<=18 records/token) is the only legal special case (Muse A)."
        ),
        "host_buffer_reuse_lever": (
            "NONE mechanistic. Measured (seal 1486403): fill_batch_wall_ms "
            "unchanged with reuse ON (54.63->55.76 s cuda0; within run spread), "
            "mmap_to_pinned still present (12.38 s full-run profiled candidate), "
            "reused_fill_buffers (1117) == total evictions (708+409) i.e. reuse "
            "recycles eviction buffers; reservation wall ~6.0 s/GPU is worker-"
            "overlapped. Predicted reuse delta = 0.0 s; measured deltas "
            "(-0.671 s, -1.929 s) sit inside baseline run-to-run spread."
        ),
    }


def predict(model: dict, misses: list[float]) -> list[float]:
    if model["ordinary_future_layer_lead"] != 0:
        raise ValueError("ordinary future-layer routes are unavailable (causality)")
    return [model["serial_residual_ms_per_step"] + model["miss_service_ms"] * m
            for m in misses]


def mape(pred: list[float], obs: list[float]) -> float:
    return 100 * sum(abs(p - o) / o for p, o in zip(pred, obs)) / len(pred)


def validate(model: dict, derived: dict) -> dict:
    """Held-out validation on the 5 host-reuse runs (same reads table)."""
    out = {}
    v65 = derived["v65_sealed"]
    pred = predict(model, v65["ssd_reads"])
    out["v65_training"] = {
        "predicted_s": round(sum(pred) / 1000, 3),
        "observed_s": v65["decode_wall_s"],
        "total_error_pct": round(100 * (sum(pred) / 1000 / v65["decode_wall_s"] - 1), 3),
        "step_mape_pct": round(mape(pred, v65["step_walls_ms"]), 3),
        "role": "fit (calibration data)",
    }
    same = all(r["ssd_reads"] == v65["ssd_reads"] and r["h2d_copies"] == v65["h2d_copies"]
               for n, r in derived.items() if n != "v65_sealed")
    for name in ("ab_baseline", "ab_candidate", "rep_baseline",
                 "rep_candidate", "profiled_candidate"):
        r = derived[name]
        p = predict(model, r["ssd_reads"])
        out[f"{name}_held_out"] = {
            "predicted_s": round(sum(p) / 1000, 3),
            "observed_s": r["decode_wall_s"],
            "total_error_pct": round(100 * (sum(p) / 1000 / r["decode_wall_s"] - 1), 3),
            "step_mape_pct": round(mape(p, r["step_walls_ms"]), 3),
            "role": "validation (held out; identical reads/copies tables)",
        }
    out["held_out_reads_tables_identical_to_v65"] = same
    walls = [MEASURED[k] for k in ("v65_sealed", "ab_baseline", "rep_baseline")]
    out["baseline_run_to_run_spread_s"] = {
        "min": min(walls), "max": max(walls), "spread": round(max(walls) - min(walls), 3),
        "note": "three baselines of the SAME config; reuse-pair deltas must be "
                "judged against this, not against 0",
    }
    # Lane-count neutrality (measured v63/v64/v65 walls, prior track; provenance
    # EVIDENCE_INVENTORY of research/exact-critical-path @ 2db0fde):
    out["lane_neutrality_check"] = {
        "measured_walls_s": {"v63_3lanes": 71.3, "v64_3lanes": 71.8, "v65_6lanes": 72.3},
        "spread_s": 1.0,
        "model_behavior": "wall does not depend on lane count (saturated-disk "
                          "aggregate); consistent with measured neutrality",
        "status": "consistent",
    }
    return out


def audit_old_model(model: dict, derived: dict) -> dict:
    """Why the old event simulator predicted -12.3% for buffer reuse."""
    v65 = derived["v65_sealed"]
    n_pack = sum(v65["h2d_copies"]) - sum(v65["ssd_reads"])   # pack-sourced copies
    old_cal = json.loads(
        (ROOT / "research/exact-critical-path/results/calibration.json").read_text()
    )["params"]
    pack_ms = old_cal["host_pack_ms_per_record"]
    old_pred_delta_s = round(-0.1233 * 71.410, 2)   # old model's reuse delta
    return {
        "old_model": "research/exact-critical-path @ 2db0fde (event simulator)",
        "old_prediction": {"reuse_delta_s": old_pred_delta_s,
                           "reuse_delta_pct": -12.33},
        "measured": {"original_pair": {"delta_s": MEASURED["ab_candidate"] - MEASURED["ab_baseline"],
                                       "delta_pct": round(100 * (MEASURED["ab_candidate"] / MEASURED["ab_baseline"] - 1), 2)},
                     "replication_pair": {"delta_s": MEASURED["rep_candidate"] - MEASURED["rep_baseline"],
                                          "delta_pct": round(100 * (MEASURED["rep_candidate"] / MEASURED["rep_baseline"] - 1), 2)}},
        "root_causes": [
            {
                "cause": "pack copy charged as a SERIAL batch-blocking stage",
                "detail": (f"the old simulator applied pack_ms={round(pack_ms, 3)} ms/record "
                           f"serially to {round(n_pack)} pack-sourced transfers "
                           f"(~{round(n_pack * pack_ms / 1000, 2)} s of chain) per "
                           "before/after reuse; measured fill telemetry shows the "
                           "worker pool PIPELINES these copies (fill_worker_ms 144 s "
                           "wall-vs-worker vs fill_batch_wall_ms 54.6 s; fill_overlap_ms "
                           "89.5 s overlapped), so most of that time was never on the "
                           "critical path"),
            },
            {
                "cause": "reuse was modeled as copy elimination (pack_ms 2.855 -> 0.05)",
                "detail": "measured: the copy is NOT removed. Profiled candidate with "
                          "reuse ON still shows mmap_to_pinned = 12.38 s (full run) and "
                          "fill_batch_wall_ms unchanged within run spread. What reuse "
                          "removes is buffer ALLOCATION/RESERVATION, which is already "
                          "hidden under the fill worker pipeline.",
            },
            {
                "cause": "v52-derived pack rate leaked a profiler-inflated cost",
                "detail": "host_pack_ms_per_record=2.855 ms came from the v52 stage-"
                          "profiled run (decode wall inflated 138.7 s vs 72.3 s "
                          "unprofiled); the old model treated that busy time as "
                          "critical-path-blocking while the calibrated disk-drain term "
                          "already dominated, so removing it showed up as a large "
                          "counterfactual win that in reality was overlapped work.",
            },
            {
                "cause": "async completion treated as blocking",
                "detail": "the event model advanced the consumer chain to "
                          "pack_end/h2d_end per batch; in the engine the consumer waits "
                          "per-layer and the fill pool runs ahead asynchronously "
                          "(inflight_hits/duplicate suppression), so per-batch "
                          "serialization overstated exposure of removed work.",
            },
        ],
        "rule": "no scalar correction factor was applied; the mechanism was removed "
                "and the model refit on measured stage tables",
        "quantified": {
            "old_model_pack_chain_ms": round(n_pack * pack_ms, 0),
            "measured_fill_overlap_ms_cuda0": 89486.4,
            "measured_fill_batch_wall_ms_cuda0_baseline": 54633.7,
            "measured_fill_batch_wall_ms_cuda0_candidate": 55760.2,
            "measured_mmap_to_pinned_full_run_ms_profiled_candidate": 12380.0,
            "measured_reused_fill_buffers": 1117,
            "measured_total_evictions": 1117,
        },
    }


def memory_envelope(budget_gib: float) -> dict:
    """Conservative TOTAL host-memory envelope + safety classification."""
    total_pack = 2 * budget_gib
    projected_gib = total_pack + NONPACK_HWM_GIB + SYSTEM_REST_GIB + UNMEASURED_GROWTH_GIB
    projected_gb = projected_gib * GIB / GB
    available_min_gib = 31.35 - projected_gib     # vs measured MemTotal 31.35 GiB
    headroom_gb = LIMIT_DECIMAL_GB - projected_gb
    oom_margin_gib = OOM_LEDGER["oom_pack_total_gib"] - total_pack
    reasons = []
    if projected_gb + SAFETY_HEADROOM_GIB <= LIMIT_DECIMAL_GB:
        cls = "SAFE_FOR_32GB"
    elif projected_gb <= LIMIT_DECIMAL_GB:
        cls = "BORDERLINE_FOR_32GB"
        reasons.append(f"fits arithmetic but headroom {round(headroom_gb, 2)} GB "
                       f"< required {SAFETY_HEADROOM_GIB} GB")
    else:
        cls = "NOT_SAFE_FOR_32GB"
        reasons.append(f"projected total {round(projected_gb, 2)} GB > "
                       f"{LIMIT_DECIMAL_GB} GB limit")
    if oom_margin_gib < 2.0:
        reasons.append(f"pack total within {round(oom_margin_gib, 2)} GiB of the "
                       "nearest clean OOM observation (27.68 GiB)")
        if cls == "SAFE_FOR_32GB":
            cls = "BORDERLINE_FOR_32GB"
    if available_min_gib < SAFETY_HEADROOM_GIB:
        reasons.append(f"projected minimum MemAvailable {round(available_min_gib, 2)} "
                       f"GiB < {SAFETY_HEADROOM_GIB} GiB")
        if cls == "SAFE_FOR_32GB":
            cls = "BORDERLINE_FOR_32GB"
    return {
        "pack_gib_per_gpu": budget_gib,
        "records_per_gpu": math.floor(budget_gib * GIB / RECORD_BYTES),
        "pack_total_gib": round(total_pack, 3),
        "measured_nonpack_hwm_gib": NONPACK_HWM_GIB,
        "system_rest_gib": SYSTEM_REST_GIB,
        "unmeasured_growth_allowance_gib": UNMEASURED_GROWTH_GIB,
        "unknown_components": [
            "pinned/CUDA/python split inside nonpack HWM (measured aggregate only)",
            "continuous peak system memory (final + token checkpoints only)",
            "capacity-dependent page-cache growth (allowance 0.5 GiB, UNMEASURED)",
        ],
        "projected_total_gib": round(projected_gib, 3),
        "projected_total_decimal_gb": round(projected_gb, 2),
        "projected_min_MemAvailable_gib": round(available_min_gib, 2),
        "headroom_to_32gb_decimal": round(headroom_gb, 2),
        "margin_to_nearest_oom_observation_gib": round(oom_margin_gib, 2),
        "classification": cls,
        "classification_reasons": reasons or [
            f"headroom {round(headroom_gb, 2)} GB >= {SAFETY_HEADROOM_GIB} GB; "
            f"min-available {round(available_min_gib, 2)} GiB >= "
            f"{SAFETY_HEADROOM_GIB} GiB; OOM margin "
            f"{round(oom_margin_gib, 2)} GiB"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-hash-gate", action="store_true",
                    help="skip sha256 verification (NOT for official runs)")
    ap.add_argument("--journal-v65", type=Path, default=None)
    ap.add_argument("--journal-profiled", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    hashes = {} if args.skip_hash_gate else verify_seal_inputs()
    derived = load_derived()
    model = fit_v65(derived["v65_sealed"])
    validation = validate(model, derived)
    audit = audit_old_model(model, derived)

    # --- journals: v65 (sealed anchor) + profiled candidate (current code) ---
    v65_journal = args.journal_v65 or (MAIN_CHECKOUT /
        "dee.cpp/tmp/v65-terminal-fetch-20260903T0202Z/routed_experts.jsonl")
    prof_journal = args.journal_profiled or (MAIN_CHECKOUT /
        "dee.cpp/tmp/host-reuse-profile2-20260904/"
        "v2-fetch-20260904T194914Z/output/routed_experts.jsonl")
    for p in (v65_journal, prof_journal):
        if not p.exists():
            raise SystemExit(f"journal missing: {p}")

    def replay_vs_sealed(rp: dict) -> dict:
        sealed = derived["v65_sealed"]["ssd_reads"]
        got = [rp["decode_per_step_reads"][k] for k in sorted(rp["decode_per_step_reads"])]
        return {
            "decode_misses": rp["decode_storage_misses_total"],
            "sealed_misses": sum(sealed),
            "per_step_max_abs_diff": max(abs(a - b) for a, b in zip(got, sealed)),
            "per_step_table": got,
            "sealed_table": sealed,
        }

    base_replay_v65 = pack_replay(v65_journal, 682)
    base_replay_prof = pack_replay(prof_journal, 682)
    replay_validation = {
        "v65_journal@682": replay_vs_sealed(base_replay_v65),
        "profiled_journal@682": replay_vs_sealed(base_replay_prof),
    }

    # --- frontier: replay per budget, simulated wall via recalibrated model ---
    # decode compulsory floor: misses at effectively infinite budget (the pack
    # retains everything from prefill on; a decode miss is then only a record's
    # FIRST decode-phase demand). Capacity misses = misses - compulsory.
    compulsory = pack_replay(prof_journal, 1_000_000)["decode_storage_misses_total"]
    distinct_total = sum(d["distinct_records"] for d in
                         pack_replay(prof_journal, 682)["per_device"].values())
    frontier, csv_rows = [], []
    for b in BUDGETS_GIB:
        cap = math.floor(b * GIB / RECORD_BYTES)
        rp = pack_replay(prof_journal, cap)
        per_step = [rp["decode_per_step_reads"][k] for k in
                    sorted(rp["decode_per_step_reads"])]
        pred = predict(model, per_step)
        wall_s = sum(pred) / 1000
        mem = memory_envelope(b)
        d_miss = base_replay_prof["decode_storage_misses_total"] \
            - rp["decode_storage_misses_total"]
        extra_gib = round(b - 8.5, 3)
        row = {
            "pack_gib_per_gpu": b,
            "records_per_gpu": cap,
            "pack_total_gib": round(2 * b, 3),
            "decode_misses_replayed": rp["decode_storage_misses_total"],
            "decode_compulsory_misses": compulsory,
            "decode_capacity_misses": rp["decode_storage_misses_total"] - compulsory,
            "bytes_retained_gib_both_gpus": round(
                min(2 * cap, distinct_total) * RECORD_BYTES / GIB, 3),
            "miss_reduction_vs_8gib5": d_miss,
            "delta_misses_per_extra_gib": round(d_miss / extra_gib, 1) if extra_gib > 0 else 0.0,
            "resident_plus_pack_hits_note": "per-GPU hits/misses in replay_details",
            "simulated_decode_wall_s": round(wall_s, 3),
            "simulated_delta_s_vs_8gib5": round(wall_s - (sum(predict(model,
                [base_replay_prof["decode_per_step_reads"][k] for k in
                 sorted(base_replay_prof["decode_per_step_reads"])])) / 1000), 3),
            "simulated_delta_pct_vs_observed_v65": round(
                100 * (wall_s / MEASURED["v65_sealed"] - 1), 2),
            "label": "SIMULATED (recalibrated observational model)",
            "memory_classification": mem["classification"],
        }
        frontier.append(row)
        csv_rows.append([b, cap, rp["decode_storage_misses_total"], compulsory,
                         row["decode_capacity_misses"], d_miss,
                         row["delta_misses_per_extra_gib"], round(wall_s, 3),
                         round(wall_s - MEASURED["v65_sealed"], 3),
                         mem["projected_total_decimal_gb"],
                         mem["projected_min_MemAvailable_gib"],
                         mem["classification"]])

    # uncertainty band on the recommended wall effect (improvement = positive):
    def band(d_miss: float) -> dict:
        stat = 1 + model["slope_ci95_pct_of_slope"] / 100
        center = d_miss * model["miss_service_ms"] / 1000
        return {"improvement_s": {
                    "central": round(center, 2),
                    "statistical_ci95": [round(center / stat, 2),
                                         round(center * stat, 2)]},
                "run_noise_s": validation["baseline_run_to_run_spread_s"]["spread"],
                "note": "SIMULATED. Run-to-run baseline spread (~1.0-1.9 s) is "
                        "comparable to the central estimate; a matched A/B is "
                        "the arbiter"}

    # --- pick recommendation: largest effect that is still SAFE for 32 GB ---
    safe = [r for r in frontier if r["memory_classification"] == "SAFE_FOR_32GB"]
    if not safe:
        raise SystemExit("no SAFE budget found; cannot recommend")
    recommended = max(safe, key=lambda r: r["miss_reduction_vs_8gib5"])
    rec_band = band(recommended["miss_reduction_vs_8gib5"])

    memory_frontier = {
        "schema": "exact-staging/memory-frontier-v2",
        "source_seal": SEAL,
        "limit_decimal_gb": LIMIT_DECIMAL_GB,
        "limit_gib": round(LIMIT_GIB, 2),
        "measured_anchors": {
            "MemTotal_gib_all_runs": 31.35,
            "nonpack_hwm_gib_range": [5.597, 5.817],
            "min_checkpoint_available_gib_at_pack_16gib983": [7.76, 7.97],
            "pack_occupied_gib_both_gpus_at_8gib5": 16.983,
            "note": "derived_memory_digests.json (5 seal runs)",
        },
        "oom_ledger_measured": OOM_LEDGER,
        "rules": {
            "headroom_required_gib": SAFETY_HEADROOM_GIB,
            "projection": "2*budget + nonpack_hwm_max + system_rest + unmeasured_growth",
        },
        "rows": [memory_envelope(b) for b in BUDGETS_GIB],
        "recommended": recommended["pack_gib_per_gpu"],
    }

    # --- required metrics for the live A/B ---
    next_ab = {
        "schema": "exact-staging/next-ab-v2",
        "source_seal": SEAL,
        "label": "LIVE_AB_V2: host-pack runtime cap 17.0 -> "
                 f"{round(2 * recommended['pack_gib_per_gpu'], 1)} GiB total "
                 f"(8.5 -> {recommended['pack_gib_per_gpu']} GiB/GPU)",
        "one_knob": "host_pack_runtime_cap_gib_total (harness clamp at "
                    "deepseek_v4_native_generate.py LRU_TOTAL_CAP_GIB); config-only",
        "why_this_point": {
            "chosen_gib_per_gpu": recommended["pack_gib_per_gpu"],
            "marginal_curve": (
                "marginal misses per extra GiB (replay): 8.5->9.0: 44, "
                "9.0->9.5: 40, 9.5->10.0: 18, then <=22 per 0.5 GiB step and "
                "fading; 10.5+ crosses the conservative 2 GB headroom line "
                "(BORDERLINE/NOT_SAFE); 12.75 REJECTED for the 32 GB thesis"),
            "memory": (
                f"pack total {round(2 * recommended['pack_gib_per_gpu'], 1)} GiB "
                "leaves >=7.6 GiB to the nearest measured clean OOM (27.68 GiB) "
                "and >=4.1 GiB projected min MemAvailable"),
        },
        "baseline_config": {
            "host_pack_runtime_cap_gib_total": 17.0,
            "effective_pack_gib_per_gpu": 8.5,
            "records_per_gpu": 682,
        },
        "candidate_config": {
            "host_pack_runtime_cap_gib_total": round(2 * recommended["pack_gib_per_gpu"], 1),
            "effective_pack_gib_per_gpu": recommended["pack_gib_per_gpu"],
            "records_per_gpu": recommended["records_per_gpu"],
        },
        "keep_identical": ["prompt", "n_tokens=16", "route authority",
                           "cache_budget 3.5 GiB/GPU", "source_read_lanes",
                           "source_read_queue_depth", "correctness gates",
                           "reuse flag", "madvise/discards off"],
        "expected": {
            "miss_reduction_measured_replay": recommended["miss_reduction_vs_8gib5"],
            "simulated_wall_delta": rec_band,
            "hypothesis": "fewer capacity misses -> fewer 13.37 MB SSD reads "
                          "-> shorter decode wall",
        },
        "oom_risk": {
            "class": memory_envelope(recommended["pack_gib_per_gpu"])["classification"],
            "projected_total_decimal_gb":
                memory_envelope(recommended["pack_gib_per_gpu"])["projected_total_decimal_gb"],
            "watch": "memory.json per-token VmRSS/VmData + min available; "
                     "abort if min available < 1.5 GiB",
        },
        "required_live_metrics": [
            "decode_wall_s", "decode_tokens_per_second", "total_wall_seconds",
            "VmHWM/VmRSS + per-token memory checkpoints",
            "host_pack entries/bytes/hits/misses/evictions per GPU",
            "ssd_reads (storage_requests) per step", "h2d_bytes per step",
            "all sealed correctness gates", "run hashes (source/config/journal)",
        ],
        "success_criteria": {
            "correctness": "ACCEPT_CORRECTNESS on both arms; identical ids+text",
            "performance": "candidate decode wall < baseline decode wall by more "
                           "than the baseline run spread (>=1.0 s), replicated in "
                           "a second matched pair; performance acceptance only "
                           "under formal campaign rules",
        },
        "falsifies_host_pack_hypothesis": [
            "pack hits rise / misses fall as replayed, but decode wall does not "
            "improve beyond run noise in two matched pairs",
            "or: process memory exceeds the projected envelope / approaches OOM",
            "or: correctness gates fail (any divergence -> stop)",
        ],
        "label": "SIMULATED expectations; the A/B is the arbiter",
    }

    pread_pkg = {
        "schema": "exact-staging/pread-kaggle-package-v1",
        "status": "READY_FOR_ASTRA (not measurable locally; local SSD != Kaggle)",
        "bench": "tools/bench_expert_pread.py (audited: per-lane fds, depths 1-16, "
                 "seq/dispersed/journal patterns, coldish/warm separated, "
                 "concurrency_achieved_est, cpu_s, no shared-fd by default)",
        "store": "<DEE4_TRACE_PATH> on the Kaggle host (experts.dee4 trace store)",
        "record_bytes": RECORD_BYTES,
        "command": (
            "python tools/bench_expert_pread.py --store $DEE4_TRACE_PATH "
            "--records 96 --depths 1 2 3 4 6 8 12 16 "
            "--patterns seq dispersed journal "
            "--journal $OUT/routed_experts.jsonl "
            "--journal-meta $OUT/dee4-metadata.json "
            "--repeat 2 --shared-fd-variant "
            "--label kaggle-2xt4 --out $OUT/pread_capability.json"
        ),
        "outputs": ["aggregate_mb_s", "p50_ms", "p90_ms", "p99_ms", "wall_s",
                    "cpu_s", "bytes_requested", "bytes_completed",
                    "concurrency_achieved_est", "short_or_empty_reads"],
        "ingestion": "research/exact-critical-path (ssd_aggregate_mb_s uncertainty "
                     "band replaces the 320-459 MB/s fit band; pins the -x% "
                     "storage-floor position)",
        "why": "the single most valuable missing measurement: true concurrent "
               "pread capability of the trace store bounds every storage-side "
               "decision remaining",
    }

    result = {
        "schema": "exact-staging/recalibrated-model-v2",
        "source_seal": SEAL,
        "seal_inputs_sha256": hashes,
        "performance_acceptance": False,
        "labels": {
            "measured": "from sealed evidence (hash-gated)",
            "simulated": "model output; NOT measured speedup",
        },
        "model": model,
        "validation": validation,
        "old_model_audit": audit,
        "replay_validation": replay_validation,
        "decode_compulsory_misses": compulsory,
        "frontier": frontier,
        "recommended_candidate": recommended,
        "recommended_wall_band": rec_band,
        "limitations": [
            "miss_service_ms is observational effective service, not raw disk BW",
            "fit is trace-local: 15 decode tokens, one prompt, one host class",
            "held-out runs share the v65 reads table (level validation only)",
            "run-to-run spread ~1.0-1.9 s dominates small simulated deltas",
            "continuous peak memory UNMEASURED (final + token checkpoints only)",
            "pread capability UNKNOWN until the Kaggle package runs",
        ],
    }
    (out / "recalibrated_model.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "memory_frontier.json").write_text(json.dumps(memory_frontier, indent=1) + "\n")
    (out / "next_ab.json").write_text(json.dumps(next_ab, indent=1) + "\n")
    (out / "pread_kaggle_package.json").write_text(json.dumps(pread_pkg, indent=1) + "\n")
    with (out / "host_pack_frontier.csv").open("w") as fh:
        fh.write("pack_gib_per_gpu,records_per_gpu,decode_misses,decode_compulsory,"
                 "decode_capacity,miss_reduction_vs_8gib5,"
                 "delta_misses_per_extra_gib,simulated_wall_s,simulated_delta_s_vs_v65,"
                 "projected_total_decimal_gb,projected_min_available_gib,memory_class\n")
        for r in csv_rows:
            fh.write(",".join(str(x) for x in r) + "\n")

    print(json.dumps({
        "model": {k: model[k] for k in ("miss_service_ms", "serial_residual_ms_per_step",
                                        "r_squared", "slope_ci95_pct_of_slope")},
        "validation_total_err_pct": {k: v.get("total_error_pct")
                                     for k, v in validation.items()
                                     if isinstance(v, dict) and "total_error_pct" in v},
        "replay_validation": replay_validation,
        "frontier_summary": [[r["pack_gib_per_gpu"], r["decode_misses_replayed"],
                              r["simulated_decode_wall_s"],
                              r["memory_classification"]] for r in frontier],
        "recommended": {"gib": recommended["pack_gib_per_gpu"],
                        "misses_saved": recommended["miss_reduction_vs_8gib5"],
                        "sim_band": rec_band},
    }, indent=1))


if __name__ == "__main__":
    main()
