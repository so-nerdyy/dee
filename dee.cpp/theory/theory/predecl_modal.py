"""Phase-6 MODAL addendum to the frozen pre-registered matrix (PREDICTIONS_PHASE6.md Addendum 3).

The campaign returned to Modal on 2026-11-04, before any Phase-6 hardware run existed on
any platform. The frozen section-1 rows stay registered and are scored as written. Two of
their inputs are known to be unscorable or unrealizable on real Modal containers, so this
addendum registers NEW rows beside them without editing a frozen value:

  1. B_SSD. The frozen rows assume Modal NVMe at the Cell spec (L4 2.5 / 2xL4 5.0 / A10 3.0 /
     L40S 6.0 / RTX 8.0 GB/s, ASSUMPTION A5). Section 0 prerequisite 5 makes decode tok/s and
     $/1k UNTESTABLE outside 0.9-1.1x spec, and nobody has measured Modal's disk. Here B_SSD is a
     PARAMETER bound to the run's own fio randread probe (`summary.json measured_b_ssd_gib_s`,
     13,369,344 B blocks, O_DIRECT, libaio, iodepth 6, numjobs 3 -- the definition Addendum 2
     registered). The prediction FUNCTION is frozen at this file's commit; evaluating it at the
     measured B_SSD is the registered prediction.
  2. VRAM. The frozen rows assume 22 GiB/GPU (L4, A10), 48 GiB (L40S), 96 GiB (RTX PRO 6000) of
     expert cache. The 8.238 GiB DSv4 dense backbone stays resident, so 22 GiB is not
     realizable on a ~22 GiB L4/A10 (Addendum 2 item 2). This addendum registers the budget
     `modal/phase6/modal_p6.py` actually provisions: 3,584 MiB (L4, 2xL4, A10), 28,672 MiB
     (L40S), 61,440 MiB (RTX PRO 6000) per GPU.
  3. Price. rate_s = what the Modal container is billed per second at the frozen A7 schedule:
     n_gpu x GPU + 8 CPU cores (the bench function's `cpu=8`) + the container's RAM size
     (Modal bills the memory reservation). Container RAM is the bench function's `memory` at
     host 16 / 64 GiB and 288 GiB at host 256 (host + reserve; the frozen 96 / 128 / 192 GiB
     containers cannot hold a 256 GiB host tier).

Everything else is the frozen closed form, unchanged: b=1 rows use `predecl._gpu_derivations`
verbatim (MiMo-Flash = DSv4 x MIMO_TOUCH_RATIO, as frozen); b=8 rows use Addendum 2's
UNITS_CORRECTION step model (decode tok/s AGGREGATE, cold rec/tok per emitted token), with the
same MiMo scaling applied to its outputs. Scoring protocol = Addendum 2's (b=1 finite 16-token
window; b=8 last 32 of >= 64 decode steps, scorable only if that window's cold rate is inside
the row's cold kill band).
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace

from .constants import ALL_CELLS, C
from .predecl import DATA_DIR, METRICS, MIMO_TOUCH_RATIO, _gpu_derivations
from .predecl_aws import _b8_corrected, _roof_rows
from .roofline import GIB

REGISTRATION_UTC = C(
    "predecl_modal_registration_utc", "2026-11-04", "DERIVED",
    "Modal addendum registered before any Phase-6 hardware run existed (Modal or AWS)",
    "the git commit introducing theory/predecl_modal.py is the registration of record")

MiB = 1 << 20
CPU_CORES = 8                   # modal_p6._bench_dec: cpu=8 on every bench function
HOST_256_CONTAINER_MIB = 294912  # 288 GiB: 256 GiB host tier + 32 GiB for torch/CUDA/OS
MODELS = ("dsv4_flash", "mimo_v2_flash")
BATCHES = (1, 8)
HOSTS = (16, 64, 256)
BSSD_GRID_GBPS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)

MODAL_CELLS = C(
    "predecl_modal_cells", {
        # modal bench fn -> (frozen GPU cell, VRAM expert budget MiB/GPU, container MiB @ 16/64)
        "L4": ("1xL4", 3584, 98304),
        "2xL4": ("2xL4", 3584, 98304),
        "A10": ("1xA10", 3584, 98304),
        "L40S": ("1xL40S", 28672, 131072),
        "RTXPRO6000": ("1xRTXPRO6000", 61440, 196608),
    }, "ASSUMPTION",
    "dee.cpp/modal/phase6/modal_p6.py GPU_SPECS (budget MiB, container memory MiB), "
    "research/phase6-modal 2026-11-04",
    "GPU B_h2d / F_peak / prices inherited from the frozen Cell of the same GPU (same tags); "
    "B_SSD replaced by the measured value at scoring time")


def _frozen_cell(name: str):
    for c in ALL_CELLS:
        if c.name == name:
            return c
    raise KeyError(name)


def container_mib(gpu: str, host: int) -> int:
    return HOST_256_CONTAINER_MIB if host == 256 else MODAL_CELLS.value[gpu][2]


def rate_s(gpu: str, host: int) -> float:
    """Registered all-in Modal $/s for the bench container (frozen A7 price schedule)."""
    base = _frozen_cell(MODAL_CELLS.value[gpu][0])
    return (base.price_gpu_s * base.n_gpu + base.price_cpu_core_s * CPU_CORES
            + base.price_ram_gib_s * container_mib(gpu, host) / 1024.0)


def label(gpu: str) -> str:
    return f"modal-{gpu}"


def modal_cell(gpu: str, B_ssd_Bps: float):
    gpu_cell, budget_mib, _ = MODAL_CELLS.value[gpu]
    base = _frozen_cell(gpu_cell)
    return replace(base, name=label(gpu), B_ssd=float(B_ssd_Bps),
                   vram_bytes=float(budget_mib * MiB),
                   provenance=f"Modal bench_{gpu}; GPU constants from frozen {gpu_cell}",
                   B_ssd_tag="MEASURED-AT-SCORING")


def _predict_cell(cell, model: str, b: int, host: int, rate: float) -> dict:
    roof = _roof_rows(cell, b, host)
    steady = roof[(cell.name, "dsv4_flash", b, "steady_state", host)]
    frontier = {(cell.name, host, b): {"rate_s": rate, "limiter": steady["limiter"]}}
    d = _gpu_derivations(cell.name, model, b, host, roof, frontier, cell=cell)
    out = {m: d[m][:4] for m in METRICS}
    if b > 1:
        # _b8_corrected prices with cell.price_gpu_s; feed it the all-in rate.
        b8 = _b8_corrected(cell.name, b, host, roof, replace(cell, price_gpu_s=rate))
        if model != "dsv4_flash":   # frozen MiMo rule: DSv4 x MIMO_TOUCH_RATIO
            f = MIMO_TOUCH_RATIO.value
            tps, usd, cold = b8["decode_tps"], b8["usd_per_1k_tok"], b8["cold_records_per_tok"]
            b8 = {
                "decode_tps": (tps[0] / f, tps[1] / f, tps[2] / f, (tps[3][0] / f, tps[3][1] / f)),
                "usd_per_1k_tok": (usd[0] * f, usd[1] * f, usd[2] * f,
                                   (usd[3][0] * f, usd[3][1] * f)),
                "cold_records_per_tok": (cold[0] * f, cold[1] * f, cold[2] * f,
                                         (cold[3][0] * f, cold[3][1] * f)),
            }
        out.update(b8)
    out["limiter"] = steady["limiter"]
    return out


def predict(gpu: str, model: str, b: int, host: int, B_ssd_Bps: float) -> dict:
    """THE REGISTERED PREDICTION for one Modal run at its measured B_SSD (B/s).

    Returns {metric: (nominal, lo, hi, (kill_lo, kill_hi))} plus 'limiter'.
    """
    if gpu not in MODAL_CELLS.value:
        raise ValueError(f"unknown Modal bench function {gpu!r}")
    if model not in MODELS or b not in BATCHES or host not in HOSTS:
        raise ValueError(f"unregistered (model, b, host) = ({model}, {b}, {host})")
    return _predict_cell(modal_cell(gpu, B_ssd_Bps), model, b, host, rate_s(gpu, host))


def self_check() -> dict:
    """With the frozen B_SSD / VRAM / rate fed back in, b=1 rows must equal the frozen matrix."""
    frozen = {}
    with open(DATA_DIR / "predecl_matrix.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            frozen[(r["cell"], r["model"], int(r["b"]), int(r["host_gib"]), r["metric"])] = r
    rates = {}
    with open(DATA_DIR / "serve_cost_frontier.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rates[(r["cell"], int(r["host_gib"]), int(r["batch"]))] = float(r["hardware_rate_s"])
    worst, checked = 0.0, 0
    for gpu, (gpu_cell, _, _) in MODAL_CELLS.value.items():
        base = _frozen_cell(gpu_cell)
        for model in MODELS:
            for host in HOSTS:
                pred = _predict_cell(base, model, 1, host, rates[(gpu_cell, host, 1)])
                for m in METRICS:
                    ref = frozen[(gpu_cell, model, 1, host, m)]
                    for i, col in enumerate(("nominal", "lo", "hi")):
                        a, e = float(pred[m][i]), float(ref[col])
                        worst = max(worst, abs(a - e) / max(abs(e), 1e-12))
                    checked += 1
    return {"rows_checked": checked, "max_rel_err": worst, "pass": worst < 1e-9}


def run() -> dict:
    chk = self_check()
    if not chk["pass"]:
        raise RuntimeError(f"Modal addendum does not reproduce the frozen b=1 rows: {chk}")
    cols = ("cell", "frozen_gpu_cell", "model", "b", "host_gib", "B_ssd_gbps", "metric",
            "nominal", "lo", "hi", "kill_lo", "kill_hi", "limiter", "rate_s", "source_tag")
    rows = []
    for gpu, (gpu_cell, _, _) in MODAL_CELLS.value.items():
        for model in MODELS:
            for b in BATCHES:
                for host in HOSTS:
                    for gbps in BSSD_GRID_GBPS:
                        p = predict(gpu, model, b, host, gbps * 1e9)
                        for m in METRICS:
                            nom, lo, hi, (klo, khi) = p[m]
                            rows.append({
                                "cell": label(gpu), "frozen_gpu_cell": gpu_cell, "model": model,
                                "b": b, "host_gib": host, "B_ssd_gbps": gbps, "metric": m,
                                "nominal": float(nom), "lo": float(lo), "hi": float(hi),
                                "kill_lo": float(klo), "kill_hi": float(khi),
                                "limiter": p["limiter"], "rate_s": rate_s(gpu, host),
                                "source_tag": "CLOSED-FORM-ONLY; Modal addendum; "
                                              "B_SSD=measured-at-scoring"})
    with open(DATA_DIR / "predecl_modal_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    meta = {
        "registration_utc": REGISTRATION_UTC.value,
        "self_check": chk,
        "cells": {label(g): {"frozen_gpu_cell": v[0], "vram_budget_mib": v[1],
                             "container_mib": {h: container_mib(g, h) for h in HOSTS},
                             "rate_s": {h: rate_s(g, h) for h in HOSTS}}
                  for g, v in MODAL_CELLS.value.items()},
        "bssd_grid_gbps": list(BSSD_GRID_GBPS),
        "scoring_rule": "evaluate theory.predecl_modal.predict(gpu, model, b, host, B_ssd) at "
                        "B_ssd = summary.json measured_b_ssd_gib_s * 2^30; the CSV grid is a "
                        "human-readable sample of that frozen function",
        "rows": len(rows),
    }
    with open(DATA_DIR / "predecl_modal_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
