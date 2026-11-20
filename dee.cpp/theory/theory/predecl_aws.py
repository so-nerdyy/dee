"""Phase-6 AWS addendum to the frozen pre-registered matrix (PREDICTIONS_PHASE6.md 6+).

The frozen matrix (2026-09-30) was registered against Modal cells whose B_SSD
values were spec ASSUMPTIONs; its rule makes any run whose measured B_SSD falls
outside 0.9-1.1x spec UNTESTABLE for decode tok/s and $/1k tok.  The campaign
moved to AWS EC2 (us-east-2), whose instance-store NVMe bandwidth is unpublished.

This addendum registers NEW rows for the AWS cells WITHOUT editing the frozen
ones.  It reuses the frozen closed form unchanged (roofline.roofline_point +
predecl._gpu_derivations) with three explicit substitutions:

  1. B_SSD is a PARAMETER bound to the run's own measured value: bench_runner's
     fio randread probe (13,369,344 B blocks, O_DIRECT, libaio, iodepth 6,
     numjobs 3 -- the anchor's 3-lane QD6 definition), reported as
     summary.json `measured_b_ssd_gib_s`.  The prediction FUNCTION is frozen at
     this file's registration commit; evaluating it at the measured B_SSD is
     the registered prediction.  The 0.9-1.1x B_SSD corner is kept as the
     instrument-noise band around the measured value.
  2. VRAM expert budget = what the AWS harness actually provisions
     (aws/phase6/config.py GPU_SPECS budget_mib: 3584 MiB on L4/A10 = 281
     records, the sealed-anchor budget; 28672 MiB on L40S = 2,248 records).
     The frozen matrix's 22 GiB/GPU on L4/A10 is not provisionable: the
     DSv4-Flash dense backbone the runner keeps resident is 8.238 GiB
     (1,564 tensors, 8,845,959,388 B) and an L4 has ~22.0 GiB usable.
  3. Price = the instance's all-in on-demand us-east-2 rate (Price List API,
     2026-10-08, ASSUMPTION); rate_s = usd_per_hour / 3600, no separate
     CPU/RAM/volume line items (AWS bills the instance whole; S3 storage is
     outside the per-token window).

Scope: DSv4-Flash only (no MiMo dense checkpoint or complete store exists on
AWS at registration).  b in {1, 8}.  Host budgets map to the smallest instance
of the GPU's family with RAM - 12 GiB >= host (the harness refuses otherwise).
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace

from .constants import ALL_CELLS, C, COMPUTE_EFF, T4_TOUCH_OVERHEAD_US
from .predecl import DATA_DIR, METRICS, _gpu_derivations
from .roofline import GIB, build_pop, dense_ms_for, load_geoms, roofline_point

REGISTRATION_UTC = C(
    "predecl_aws_registration_utc", "2026-10-08", "DERIVED",
    "AWS addendum registered before any AWS GPU run existed (G/VT quota 0 at registration)",
    "the git commit introducing theory/predecl_aws.py is the registration of record")

# eta_storage exactly as roofline.run() uses it (anchor-calibrated).
ETA_STORAGE = 0.29170032225974407

MiB = 1 << 20
MI_TO_REC = 13369344

AWS_CELLS = C(
    "predecl_aws_cells", {
        # cell label -> (frozen GPU cell the constants come from, instance, usd/h,
        #                VRAM expert budget MiB, host budgets GiB served by it)
        "aws-L4-g6.2xlarge": ("1xL4", "g6.2xlarge", 0.9776, 3584, (16,)),
        "aws-L4-g6.8xlarge": ("1xL4", "g6.8xlarge", 2.0144, 3584, (64,)),
        "aws-A10-g5.2xlarge": ("1xA10", "g5.2xlarge", 1.212, 3584, (16,)),
        "aws-A10-g5.8xlarge": ("1xA10", "g5.8xlarge", 2.448, 3584, (64,)),
        "aws-L40S-g6e.2xlarge": ("1xL40S", "g6e.2xlarge", 2.24208, 28672, (16,)),
        "aws-L40S-g6e.4xlarge": ("1xL40S", "g6e.4xlarge", 3.00424, 28672, (64,)),
        "aws-L40S-g6e.16xlarge": ("1xL40S", "g6e.16xlarge", 7.57719, 28672, (256,)),
    }, "ASSUMPTION",
    "dee.cpp/aws/phase6/config.py on research/phase6-aws @ 754fe85 (INSTANCE_TYPES "
    "prices from AWS Price List API 2026-10-08; GPU_SPECS budget_mib; smallest-fit sizing "
    "with HOST_RAM_RESERVE_GIB=12)",
    "GPU B_h2d / F_peak inherited from the frozen Modal cell of the same GPU (same "
    "ASSUMPTION tags); B_SSD replaced by the measured value at scoring time")

BSSD_GRID_GBPS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
BATCHES = (1, 8)
MODEL = "dsv4_flash"

_COLS = ("cell", "frozen_gpu_cell", "instance", "model", "b", "host_gib", "B_ssd_gbps",
         "metric", "nominal", "lo", "hi", "kill_lo", "kill_hi", "limiter",
         "depends_on_B_ssd", "source_tag")


def _frozen_cell(name: str):
    for c in ALL_CELLS:
        if c.name == name:
            return c
    raise KeyError(name)


def aws_cell(label: str, B_ssd_Bps: float):
    gpu_cell, instance, usd_h, budget_mib, _ = AWS_CELLS.value[label]
    base = _frozen_cell(gpu_cell)
    return replace(base, name=label, B_ssd=float(B_ssd_Bps),
                   vram_bytes=float(budget_mib * MiB),
                   price_gpu_s=usd_h / 3600.0, price_cpu_core_s=0.0,
                   price_ram_gib_s=0.0,
                   provenance=f"AWS {instance} on-demand us-east-2 ${usd_h}/h (2026-10-08); "
                              f"GPU constants from frozen {gpu_cell}",
                   B_ssd_tag="MEASURED-AT-SCORING", price_tag="ASSUMPTION")


_GEOM = None
_POP = None


def _geom_pop():
    global _GEOM, _POP
    if _GEOM is None:
        _GEOM = load_geoms()[MODEL]
        _POP = build_pop(_GEOM)
    return _GEOM, _POP


def _roof_rows(cell, b: int, host: int) -> dict:
    """Reproduce roofline.run()'s two horizon rows for one (cell, b, host)."""
    g, pop = _geom_pop()
    t0 = float(T4_TOUCH_OVERHEAD_US.value)
    flops_eff = COMPUTE_EFF.value * cell.F_peak
    dense = dense_ms_for(cell, g)
    h_slots = host * GIB / g.record_bytes / max(1, cell.n_gpu)
    out = {}
    for horizon, finite, hname in ((512, False, "steady_state"), (16, True, "finite_16tok")):
        p = roofline_point(g, pop, cell, b, h_slots, dense, t0, flops_eff,
                           horizon_tokens=horizon, finite=finite,
                           batches=([7] + [1] * 15) if finite else None,
                           eta_storage=ETA_STORAGE)
        out[(cell.name, MODEL, b, hname, host)] = {
            "t_dense": p.t_dense, "t_compute": p.t_compute, "t_h2d": p.t_h2d,
            "t_storage": p.t_storage, "cold_records_per_tok": p.cold_records_per_tok,
            "tps_pred": p.tps_pred, "limiter": p.limiter,
        }
    return out


UNITS_CORRECTION = C(
    "predecl_aws_b8_units_correction", {
        "defect": "roofline_point at b>1 returns t_storage / t_h2d / cold_records_per_tok "
                  "PER BATCH STEP (D(b) = sum_i 1-(1-pi_i)^b distinct records per step) but "
                  "t_dense / t_compute PER EMITTED TOKEN (divided by b); t_pred sums them, so "
                  "the frozen b=8 decode_tps is neither aggregate nor per-stream throughput. "
                  "Evidence: 1xL4-class host16 b=8 steady cold_records_per_tok = 482 > "
                  "L*k = 258 record touches per token -- only possible per step.",
        "correction": "step_time = b*(t_dense + t_compute) + (1-eta_s)*t_storage_step + "
                      "(1-eta_h)*t_h2d_step;  decode_tps (AGGREGATE over the b streams) = "
                      "b / step_time;  cold_records_per_tok = cold_step / b",
        "scope": "applied to AWS b=8 rows only; b=1 rows are bit-identical to the frozen form "
                 "(step == token); the frozen PREDICTIONS_PHASE6.md b=8 rows are NOT edited -- "
                 "the defect is recorded in its addendum",
    }, "DERIVED", "theory/roofline.py roofline_point (D(b) accounting) + this file",
    "registered before any AWS run existed")


def _b8_corrected(label: str, b: int, host: int, roof: dict, cell) -> dict:
    """Unit-consistent b>1 derivation: same corners/kill rules as _gpu_derivations."""
    from itertools import product

    from .predecl import (B_HI_MULT, B_LO_MULT, D4_LAYERS, D4_TOPK, EFF_NOM, ETA_H_HI,
                          ETA_H_LO, ETA_H_NOM, ETA_S_HI, ETA_S_LO, ETA_S_NOM, P_REC, T0_HI,
                          T0_LO, T0_NOM_S, W_TOUCH)
    rs = roof[(label, MODEL, b, "steady_state", host)]
    rf = roof[(label, MODEL, b, "finite_16tok", host)]
    flops_term = D4_LAYERS.value * D4_TOPK.value * W_TOUCH / (EFF_NOM * cell.F_peak)
    alpha = max(0.0, min(1.0, (rs["t_compute"] - flops_term) / rs["t_compute"]))
    cold_step = float(rs["cold_records_per_tok"])

    def step(t0, eff, bm, es, eh) -> float:
        comp = rs["t_compute"] * (alpha * (t0 / T0_NOM_S) + (1.0 - alpha) * (EFF_NOM / eff))
        stor = cold_step * P_REC / (cell.B_ssd * bm)
        return b * (rs["t_dense"] + comp) + (1.0 - es) * stor + (1.0 - eh) * rs["t_h2d"]

    corners = list(product((T0_LO, T0_HI), (0.3, 0.8), (B_LO_MULT, B_HI_MULT),
                           (ETA_S_LO, ETA_S_HI), (ETA_H_LO, ETA_H_HI)))
    s_nom = step(T0_NOM_S, EFF_NOM, 1.0, ETA_S_NOM, ETA_H_NOM)
    s_best, s_worst = min(step(*a) for a in corners), max(step(*a) for a in corners)
    tps = (b / s_nom, b / s_worst, b / s_best)
    usd = tuple(1000.0 * cell.price_gpu_s / t for t in tps)        # nom, hi-cost, lo-cost
    cold_tok_steady = cold_step / b
    cold_finite = float(rf["cold_records_per_tok"])
    cold_lo = min(cold_tok_steady, cold_finite)
    cold_hi = 2.0 * max(cold_tok_steady, cold_finite)
    return {
        "decode_tps": (tps[0], tps[1], tps[2], (0.70 * tps[1], 1.43 * tps[2])),
        "usd_per_1k_tok": (usd[0], usd[2], usd[1], (0.60 * usd[2], 1.67 * usd[1])),
        "cold_records_per_tok": (cold_tok_steady, cold_lo, cold_hi,
                                 (0.75 * cold_lo, 1.25 * cold_hi)),
    }


def predict(label: str, b: int, host: int, B_ssd_Bps: float) -> dict:
    """THE REGISTERED PREDICTION for one AWS cell at the run's measured B_SSD (B/s).

    Returns {metric: (nominal, lo, hi, (kill_lo, kill_hi))} plus 'limiter'.
    b=1: the frozen closed form, unchanged.  b>1: the UNITS_CORRECTION step model
    (decode_tps is AGGREGATE over the b streams; cold_records_per_tok per emitted token).
    """
    if host not in AWS_CELLS.value[label][4]:
        raise ValueError(f"{label} does not serve host={host} GiB")
    cell = aws_cell(label, B_ssd_Bps)
    roof = _roof_rows(cell, b, host)
    steady = roof[(label, MODEL, b, "steady_state", host)]
    frontier = {(label, host, b): {"rate_s": cell.price_gpu_s, "limiter": steady["limiter"]}}
    d = _gpu_derivations(label, MODEL, b, host, roof, frontier, cell=cell)
    out = {m: d[m][:4] for m in METRICS}
    if b > 1:
        out.update(_b8_corrected(label, b, host, roof, cell))
    out["limiter"] = steady["limiter"]
    return out


def self_check() -> dict:
    """The pipeline must reproduce frozen rows when fed the frozen Modal cell."""
    frozen = {}
    with open(DATA_DIR / "predecl_matrix.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            frozen[(r["cell"], r["model"], int(r["b"]), int(r["host_gib"]), r["metric"])] = r
    worst = 0.0
    checked = 0
    for name in ("1xL4", "1xA10", "1xL40S"):
        cell = _frozen_cell(name)
        for b in BATCHES:
            for host in (16, 64, 256):
                roof = _roof_rows(cell, b, host)
                fr_rate = None
                with open(DATA_DIR / "serve_cost_frontier.csv", newline="",
                          encoding="utf-8") as fh:
                    for r in csv.DictReader(fh):
                        if (r["cell"], int(r["host_gib"]), int(r["batch"])) == (name, host, b):
                            fr_rate = (float(r["hardware_rate_s"]), r["limiter"])
                frontier = {(name, host, b): {"rate_s": fr_rate[0], "limiter": fr_rate[1]}}
                d = _gpu_derivations(name, MODEL, b, host, roof, frontier, cell=cell)
                for m in METRICS:
                    ref = frozen[(name, MODEL, b, host, m)]
                    for i, col in enumerate(("nominal", "lo", "hi")):
                        a, e = float(d[m][i]), float(ref[col])
                        worst = max(worst, abs(a - e) / max(abs(e), 1e-12))
                    checked += 1
    return {"rows_checked": checked, "max_rel_err": worst, "pass": worst < 1e-9}


def run() -> dict:
    chk = self_check()
    if not chk["pass"]:
        raise RuntimeError(f"AWS addendum pipeline does not reproduce the frozen matrix: {chk}")
    rows = []
    for label, (gpu_cell, instance, _, _, hosts) in AWS_CELLS.value.items():
        for b in BATCHES:
            for host in hosts:
                for gbps in BSSD_GRID_GBPS:
                    p = predict(label, b, host, gbps * 1e9)
                    for m in METRICS:
                        nom, lo, hi, (klo, khi) = p[m]
                        rows.append({
                            "cell": label, "frozen_gpu_cell": gpu_cell, "instance": instance,
                            "model": MODEL, "b": b, "host_gib": host, "B_ssd_gbps": gbps,
                            "metric": m, "nominal": nom, "lo": lo, "hi": hi,
                            "kill_lo": klo, "kill_hi": khi, "limiter": p["limiter"],
                            "depends_on_B_ssd": m in ("decode_tps", "usd_per_1k_tok"),
                            "source_tag": "CLOSED-FORM-ONLY; AWS addendum; "
                                          "B_SSD=measured-at-scoring",
                        })
    path = DATA_DIR / "predecl_aws_matrix.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    meta = {
        "registration_utc": REGISTRATION_UTC.value,
        "self_check": chk,
        "cells": {k: {"frozen_gpu_cell": v[0], "instance": v[1], "usd_per_hour": v[2],
                      "vram_budget_mib": v[3], "vram_records": int(v[3] * MiB // MI_TO_REC),
                      "host_gib": list(v[4])} for k, v in AWS_CELLS.value.items()},
        "eta_storage": ETA_STORAGE,
        "bssd_grid_gbps": list(BSSD_GRID_GBPS),
        "scoring_rule": "evaluate theory.predecl_aws.predict(cell, b, host, B_ssd) at "
                        "B_ssd = summary.json measured_b_ssd_gib_s * 2^30 (fio randread, "
                        "13,369,344 B, iodepth 6, numjobs 3, O_DIRECT); the CSV grid is a "
                        "human-readable sample of that frozen function",
        "rows": len(rows),
    }
    with open(DATA_DIR / "predecl_aws_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
