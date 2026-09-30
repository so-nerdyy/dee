"""theory.predecl -- Phase-7 component D: the PRE-REGISTERED PHASE-6 PREDICTION MATRIX.

Frozen deliverables produced by this module:

* ``data/predecl_matrix.csv``  -- one row per (cell, model, batch, host_gib, metric)
  with nominal / range / kill band / source tag / model citation / prediction IDs.
* ``data/predecl_meta.json``   -- registration timestamp, assumptions ledger,
  scoring rules, sim/solve status at freeze.
* ``data/predecl_tables.md``   -- the same matrix rendered as the markdown tables
  pasted verbatim into ``PREDICTIONS_PHASE6.md`` sections 1-2.

``run(results_SIM=None, results_SOLVE=None)`` is deterministic (no RNG). With
``results_SIM`` / ``results_SOLVE`` left None (the state at freeze time) every row
carries source_tag ``CLOSED-FORM-ONLY (sim pending at freeze)``. If Component A's
simulator output is passed in later as a mapping
``{(cell, model, batch, host): {"decode_tps": ..., "host_hit_rate": ...,
"cold_records_per_tok": ..., "usd_per_1k_tok": ...}}`` (plain dict values also
accepted), matching rows are tagged SIM-PREDICTED and the sim-vs-closed-form
disagreement is recorded in ``predecl_meta.json`` under ``sim_reconciliation``
-- WITHOUT altering any frozen range.

Closed-form substrate (all in-repo, deterministic):

* ``data/roofline_curves.csv`` rows keyed (cell, model, batch, horizon, host) --
  the THEORY.md 5.1 closed form instantiated on the empirical/ transferred
  popularity models. Nominal decode TPS is rebuilt from its own components with
  T_pred = t_dense + t_compute + (1-eta_s) t_storage + (1-eta_h) t_h2d
  (THEORY.md 5.1) and cross-checked against the row's tps_pred column.
* ``data/serve_cost_frontier.csv`` -- hardware_rate_s per (cell, host, batch);
  $/1k tok = 1000 * rate_s / TPS (the corrected formula; THEORY.md 6's printed
  formula was dimensionally wrong, Phase-7 correction 2026-09-30).
* ``data/cache_store_knees.csv`` -- host-tier hit-rate knees (H50/H80/H95).
* ``data/provenance.csv`` -- constant provenance (via theory.constants).

MiMo-V2.6-Flash (``mimo_v2_flash``) has no in-repo popularity model of its own
(``S_TRANSFER``): its per-token terms are the DSv4-Flash closed form scaled by
MIMO_TOUCH_RATIO = (48 layers x topk) / (43 x 6) record touches per decode token.
The ``zipf_mandelbrot_transferred`` rows in roofline_curves.csv are NOT used for
MiMo nominal values: they imply D(b=1) ~= 6 distinct records/token against the
structural requirement L x topk = 288 (see PREDICTIONS_PHASE6.md section 7).
"""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

from .constants import (
    ALL_CELLS,
    C,
    COMPUTE_EFF,
    CPU_EXEC_MS_PER_EXPERT,
    D4_LAYERS,
    D4_RECORD,
    D4_TOPK,
    D4_UNIVERSE_BYTES,
    D4_UNIVERSE_RECORDS,
    DENSE_BASELINE,
    MODAL_PRICES,
    SERVE_ASSUMPTIONS,
    T4_FLOP_PER_TOUCH,
    T4_TOUCH_OVERHEAD_US,
    VOLUME_PRICE_GIB_MO,
    WS_KNEE,
)

GIB = 1073741824.0

# ==========================================================================
# Registered pre-declaration constants (provenance picked up automatically)
# ==========================================================================

REGISTRATION_UTC = C(
    "predecl_registration_utc", "2026-09-30T03:45:16Z", "DERIVED",
    "Phase-7 campaign clock at freeze (component D); no Phase-6 hardware run existed at this instant",
    "the git commit timestamp of PREDICTIONS_PHASE6.md is the pre-registration; "
    "frozen ranges must not be edited later -- later information is addenda only")

MIMO_LAYERS = C(
    "predecl_mimo_layers", 48, "DERIVED",
    "data/cache_store_knees.csv mimo_v2_flash universe_records 12288 = 48x256; "
    "AGENT_BRIEF.md L22-23 and THEORY.md L40-41 (48x256 records, 13,369,344 B)",
    "AGENT_BRIEF.md quotes 12,032 records for MiMo-V2.6-Flash -- inconsistent with "
    "48x256 = 12,288 (PREDICTIONS_PHASE6.md section 7)")

MIMO_TOPK = C(
    "predecl_mimo_topk", 6, "ASSUMPTION",
    "top-6 routed assumed for MiMo-V2.6-Flash matching DSv4-Flash; the authoritative "
    "value lives in tools/phase3/specs/mimo_v2_flash.json (outside component D read scope)",
    "DATA NEEDED: confirm routed top-k for MiMo-V2.6-Flash")

MIMO_TOUCH_RATIO = C(
    "predecl_mimo_touch_ratio",
    (MIMO_LAYERS.value * MIMO_TOPK.value) / (D4_LAYERS.value * D4_TOPK.value),
    "DERIVED",
    "record touches per decode token: (48 x topk 6) / (43 x 6) = 288/258 = 1.1163; "
    "MiMo-V2.6-Flash records are byte-identical to DEE4 (13,369,344 B) so W_touch is unchanged",
    "MiMo per-token compute / spill / cold terms = DSv4 closed form scaled by this ratio")

HOST_HIT_CURVE = C(
    "predecl_host_hit_curve", {
        "form": "piecewise log-linear through (0,0),(N50,0.50),(N80,0.80),(N95,0.95),"
                "(2.5*N95,0.99); 1.0 once per-GPU host slots >= universe records",
        "dsv4_flash": {"n50": 900, "n80": 1523, "n95": 2459,
                       "universe": D4_UNIVERSE_RECORDS.value},
        "mimo_v2_flash": {"n50": 939, "n80": 1590, "n95": 2566, "universe": 12288},
        "calibration_check": "curve reads 47.9% at 682 records/GPU vs anchor measured "
                             "host share 51.3% pooled-dedup (residual -3.4pp); P2 band 0.48-0.54",
        "known_model_error": "steady-state Che reads +25pp HIGH on short windows; "
                             "finite-window Che reads up to -7pp LOW "
                             "(FALSIFICATION.md known-failure 1)",
    }, "DERIVED",
    "knee points from data/cache_store_knees.csv (H50/H80/H95 record counts); "
    "host-share prior 51.3% pooled-dedup (46.8%/56.1% per-GPU) from data/anchor_gate.json "
    "host_share.definitions (Phase-7 correction 2026-09-30)",
    "host-tier hit rate vs per-GPU host slots; host budgets are TOTAL, split evenly over GPUs")

PREDECL_ETA = C(
    "predecl_eta_ranges", {
        "eta_s": [0.15, 0.292, 0.45],   # fill-over-compute hiding (P5)
        "eta_h": [0.30, 0.50, 0.70],    # H2D hiding (THEORY.md 5.1 ASSUMPTION)
    }, "ASSUMPTION",
    "eta_s nominal 0.292 CALIBRATED on the 2xT4 anchor (THEORY.md 5.1); range = FALSIFICATION.md P5; "
    "eta_h nominal 0.5 ASSUMPTION (THEORY.md 5.1), range chosen here",
    "used as the T_pred sensitivity corners")

PREDECL_FINITE_HIT = C(
    "predecl_finite_hit_prior", {
        "form": "piecewise linear in pooled host GiB through (0,0),(16,0.51),(32,0.53),"
                "(64,0.54),(256,0.54); the SHORT-WINDOW (4-20 decode token) LRU prior",
        "points_gib": [0, 16, 32, 64, 256],
        "points_hit": [0.0, 0.51, 0.53, 0.54, 0.54],
        "calibration": "0.51 at 16 GiB pooled = midpoint of FALSIFICATION P2's registered "
                       "0.48-0.54 (ws-policy LRU replay, knee 16 GiB pooled; P2b: LRU -2.8pp "
                       "vs MIN there; LRU reaches MIN by 32 GiB)",
        "saturation": "0.536-0.54 = the finite-window repeat ceiling (5,099 dedup cache "
                      "requests - 2,364 unique records)/5,099 on the sealed window; P2c "
                      "850-1000 of 2,364-2,500 records ever repeat",
        "known_model_error": "steady-state Che over-predicts short-window hits by +25pp at "
                             "16 GiB (FALSIFICATION known-failure 1) and is carried as the "
                             "UPPER band edge (valid for >=512-token serving windows), NOT "
                             "as the short-run nominal",
    }, "DERIVED",
    "P2/P2c/P2b + ws_lru_knee_gib (provenance.csv) + sealed-window repeat structure "
    "(provenance.csv d4_sealed_accesses_dedup=5099, d4_sealed_unique_records=2364)",
    "host-tier hit rate nominal for short windows; independent of cell and GPU count "
    "(pooled host budget basis); MiMo-Flash uses the same prior (S_TRANSFER) with wider band")

PREDECL_T0_RANGE = C(
    "predecl_t0_range_us", [50.0, float(T4_TOUCH_OVERHEAD_US.value), 420.0], "ASSUMPTION",
    "per-touch dispatch cost: nominal 387.47 us = T4_TOUCH_OVERHEAD_US (CALIBRATED, "
    "carried to all cells); range = union of FALSIFICATION.md P8c (50-150 us batched kernel) "
    "and P8b (300-420 us anchor dispatch path)",
    "used as the T_pred sensitivity corner t_0 in [50, 420] us")

PREDECL_BSSD_MULT = C(
    "predecl_bssd_mult_range", [0.9, 1.1], "ASSUMPTION",
    "B_SSD uncertainty band around each Cell spec value (spec values are ASSUMPTION "
    "per Cell provenance); a live run must report its measured B_SSD or the row is UNTESTABLE",
    "used as the T_pred sensitivity corner")

PREDECL_KILL_RULES = C(
    "predecl_kill_rules", {
        "decode_tps": "kill if measured outside [0.70*lo, 1.43*hi]",
        "host_hit_rate": "kill if measured outside nominal -/+ 0.15 (DSv4) or -/+ 0.22 (MiMo, S_TRANSFER)",
        "cold_records_per_tok": "kill if measured outside [0.75*lo, 1.25*hi] of the "
                                "[steady-Che, 2x finite-Che] bracket (P4 construction)",
        "usd_per_1k_tok": "kill if measured outside [0.60*lo, 1.67*hi]",
        "grading": "PASS = every measured metric inside its [lo,hi] range; KILLED = any metric "
                   "outside its kill band (or outside a referenced FALSIFICATION id's own "
                   "registered kill band, whichever is tighter); UNTESTABLE = missing "
                   "prerequisite/artifact, or the value lands in the deliberate gap band "
                   "between range edge and kill edge (weakened, neither confirmed nor killed)",
        "registered_id_rule": "where a row references a FALSIFICATION.md id, that id's own "
                              "registered kill criterion additionally applies and is the "
                              "tighter one where they conflict",
    }, "ASSUMPTION", "component D scoring design (method, not a measurement)",
    "kill margins are deliberate: wide enough that instrumentation noise cannot kill, "
    "narrow enough that a wrong regime (storage-bound vs compute-bound) always kills")

DENSE_PRICE_READINGS = C(
    "predecl_dense_price_readings", {
        "note": "DENSE_BASELINE price_gpu_s=0.003223 $/GPU/s (= $11.60/GPU/h) conflicts with "
                "its own note '$3.223/GPU/h' (= 0.000895 $/GPU/s); 3.57x ambiguity "
                "(Phase-7 correction 2026-09-30). Both readings carried; every break-even "
                "kill criterion states which reading it assumes.",
        "expensive_reading_usd_per_1k": (
            DENSE_BASELINE.value["n_gpu"] * DENSE_BASELINE.value["price_gpu_s"]
            / DENSE_BASELINE.value["achieved_tok_s"] * 1000.0),
        "cheap_reading_price_gpu_s": 3.223 / 3600.0,
        "cheap_reading_usd_per_1k": (
            DENSE_BASELINE.value["n_gpu"] * (3.223 / 3600.0)
            / DENSE_BASELINE.value["achieved_tok_s"] * 1000.0),
        "sensitivity_ratio": DENSE_BASELINE.value["price_gpu_s"] / (3.223 / 3600.0),
    }, "ASSUMPTION", "dee.cpp/theory/theory/constants.py DENSE_BASELINE (both readings)",
    "dense-residency baseline $/1k tok: 0.12892 (expensive reading) vs 0.03581 (cheap "
    "reading); P15b 'dee beats dense at b=8' holds only under the expensive reading")

CPU_KERNEL_BOUND = C(
    "predecl_cpu_kernel_bound", {
        "portable_torch_ms_per_expert": CPU_EXEC_MS_PER_EXPERT.value,
        "tuned_kernel_bound_ms_per_expert": [25.0, 30.0],
        "max_speedup_vs_measured": 90.0,
    }, "DERIVED",
    "CPU_EXEC_MS_PER_EXPERT (MEASURED, AGENTS.md CPU 6/10) vs the 25-30 ms/expert "
    "arithmetic bound (~90x over, AGENTS.md research assimilation)",
    "CPU-only rows: nominal = portable-torch MEASURED; range upper edge = the unbuilt "
    "tuned-kernel bound (DATA NEEDED: tuned AVX2 expert kernel)")

# Grid
PHASE6_CELLS = ("1xL4", "2xL4", "1xA10", "1xL40S", "1xRTXPRO6000")
REF_CELLS = ("2xT4", "CPU-only")
CELL_ORDER = PHASE6_CELLS + REF_CELLS
MODELS = ("dsv4_flash", "mimo_v2_flash")
BATCHES = (1, 8)
HOSTS_GIB = (16, 64, 256)
METRICS = ("decode_tps", "host_hit_rate", "cold_records_per_tok", "usd_per_1k_tok")

DISPLAY = {
    "1xL4": "1xL4", "2xL4": "2xL4", "1xA10": "1xA10", "1xL40S": "1xL40S",
    "1xRTXPRO6000": "1xRTX PRO 6000", "2xT4": "2xT4", "CPU-only": "CPU-only",
    "dsv4_flash": "DSv4-Flash", "mimo_v2_flash": "MiMo-Flash",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGS_DIR = Path(__file__).resolve().parent.parent / "figs"
MATRIX_CSV = DATA_DIR / "predecl_matrix.csv"
META_JSON = DATA_DIR / "predecl_meta.json"
TABLES_MD = FIGS_DIR / "predecl_tables.md"

T0_NOM_S = T4_TOUCH_OVERHEAD_US.value * 1e-6  # per-touch dispatch cost in seconds (387.47 us)
EFF_NOM = COMPUTE_EFF.value                  # 0.6
ETA_S_NOM, ETA_H_NOM = 0.292, 0.5
W_TOUCH = T4_FLOP_PER_TOUCH.value            # FLOP per record touch
P_REC = D4_RECORD.value                      # bytes per packed expert record
ETA_S_LO, ETA_S_HI = PREDECL_ETA.value["eta_s"][0], PREDECL_ETA.value["eta_s"][2]
ETA_H_LO, ETA_H_HI = PREDECL_ETA.value["eta_h"][0], PREDECL_ETA.value["eta_h"][2]
T0_LO, T0_HI = PREDECL_T0_RANGE.value[0] * 1e-6, PREDECL_T0_RANGE.value[2] * 1e-6
B_LO_MULT, B_HI_MULT = PREDECL_BSSD_MULT.value

_CSV_COLS = ("cell", "model", "b", "host_gib", "metric", "nominal", "lo", "hi",
             "kill_lo", "kill_hi", "source_tag", "model_citation", "prediction_ids")


# ==========================================================================
# Substrate loaders
# ==========================================================================

def _load_roof() -> dict:
    out = {}
    with open(DATA_DIR / "roofline_curves.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["cell"], r["model"], int(r["batch"]), r["horizon"],
                   int(float(r["host_gib_total"])))
            out[key] = {k: (float(v) if k not in ("cell", "model", "horizon", "limiter",
                                                  "popularity_source") else v)
                        for k, v in r.items()}
            out[key]["batch"] = int(r["batch"])
            out[key]["host"] = int(float(r["host_gib_total"]))
    return out


def _load_frontier() -> dict:
    out = {}
    with open(DATA_DIR / "serve_cost_frontier.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["cell"], int(r["host_gib"]), int(r["batch"]))
            out[key] = {"rate_s": float(r["hardware_rate_s"]),
                        "usd_total": float(r["usd_per_1k_tok_total"]),
                        "tps_pred": float(r["tps_pred"]),
                        "limiter": r["limiter"]}
    return out


# ==========================================================================
# Closed-form pieces
# ==========================================================================

def _cell(name: str):
    for c in ALL_CELLS:
        if c.name == name:
            return c
    raise KeyError(name)


def _host_hit_frac(records: float, model: str) -> float:
    """HOST_HIT_CURVE: piecewise log-linear host-tier hit rate vs per-GPU slots."""
    spec = HOST_HIT_CURVE.value[model]
    n_universe = float(spec["universe"])
    if records >= n_universe:
        return 1.0
    pts = [(1.0, 0.0), (float(spec["n50"]), 0.50), (float(spec["n80"]), 0.80),
           (float(spec["n95"]), 0.95), (2.5 * float(spec["n95"]), 0.99)]
    x = max(records, 1.0)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            frac = (math_log(x / x0)) / (math_log(x1 / x0))
            return min(0.995, y0 + (y1 - y0) * frac)
    return 0.995


def math_log(x: float) -> float:
    from math import log
    return log(x)


def _tpred(t_dense, t_compute, t_storage, t_h2d, eta_s, eta_h) -> float:
    """THEORY.md 5.1 overlap-corrected prediction (the model that fits the anchor)."""
    return t_dense + t_compute + (1.0 - eta_s) * t_storage + (1.0 - eta_h) * t_h2d


def _finite_hit(host_gib: int, model: str) -> float:
    """PREDECL_FINITE_HIT: short-window LRU hit prior vs pooled host GiB (S_TRANSFER for MiMo)."""
    pts = list(zip(PREDECL_FINITE_HIT.value["points_gib"],
                   PREDECL_FINITE_HIT.value["points_hit"]))
    x = float(host_gib)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def _gpu_derivations(cell_name: str, model: str, b: int, host: int, roof: dict, frontier: dict):
    """Derive the 4 metric predictions for one GPU cell from the closed form."""
    prim = "finite_16tok" if b == 1 else "steady_state"   # P1 uses finite at b=1; P1b steady at b=8
    rp = roof[(cell_name, "dsv4_flash", b, prim, host)]
    rsteady = roof[(cell_name, "dsv4_flash", b, "steady_state", host)]
    rfinite = roof[(cell_name, "dsv4_flash", b, "finite_16tok", host)]
    c = _cell(cell_name)
    f = 1.0 if model == "dsv4_flash" else MIMO_TOUCH_RATIO.value

    # t_compute (THEORY.md 5.1: D t_0 / b + L k W_touch / (eff F)) is decomposed into its
    # dispatch share alpha and arithmetic share (1-alpha) by subtracting the W_tok/(eff F)
    # term; each share is scaled by its own sensitivity corner (t_0 range / COMPUTE_EFF range).
    flops_term = D4_LAYERS.value * D4_TOPK.value * W_TOUCH / (EFF_NOM * c.F_peak)
    alpha = max(0.0, min(1.0, (rp["t_compute"] - flops_term) / rp["t_compute"]))

    t_dense = rp["t_dense"] * f
    t_h2d = rp["t_h2d"] * f
    cold_prim = rp["cold_records_per_tok"] * f     # M_h at the primary horizon
    cold_steady = rsteady["cold_records_per_tok"] * f
    cold_finite = rfinite["cold_records_per_tok"] * f
    # bracket construction = FALSIFICATION.md P4 (corrected anchor bracket 67.84-323.92):
    #   [steady-Che, 2 x finite-Che]; nominal = finite-Che (anchor check: 162 predicted
    #   vs 155.06 measured storage_requests/tok, +4.5%). At b=8 the batch-mixture u_i(8)
    #   can make steady-Che the LARGER model (e.g. 2xT4 host16: steady 420 vs finite 179),
    #   so the bracket is order-clamped: [min(s,f), 2 x max(s,f)].
    cold_lo = min(cold_steady, cold_finite)
    cold_hi = 2.0 * max(cold_steady, cold_finite)
    cold_nom = cold_finite

    def t_compute(t0: float, eff: float) -> float:
        return f * rp["t_compute"] * (alpha * (t0 / T0_NOM_S) + (1.0 - alpha) * (EFF_NOM / eff))

    def t_storage(b_mult: float) -> float:
        return cold_prim * P_REC / (c.B_ssd * b_mult)

    def T(t0, eff, b_mult, es, eh) -> float:
        return _tpred(t_dense, t_compute(t0, eff), t_storage(b_mult), t_h2d, es, eh)

    # steady-horizon twin (used ONLY for $/1k tok): THEORY.md 6 economics use the
    # steady-state TPS basis at both b -- this reproduces serve_cost_frontier.csv exactly
    # and P15c's ratio identity 1.08 (1xL4) ... 1.79 (1xRTX PRO 6000) ... 2.29 (2xT4)
    # vs the dense baseline 0.12892 $/1k tok.
    alpha_s = max(0.0, min(1.0, (rsteady["t_compute"] - flops_term) / rsteady["t_compute"]))

    def t_compute_s(t0: float, eff: float) -> float:
        return f * rsteady["t_compute"] * (alpha_s * (t0 / T0_NOM_S) + (1.0 - alpha_s) * (EFF_NOM / eff))

    def T_s(t0, eff, b_mult, es, eh) -> float:
        # the steady-horizon twin uses the STEADY row's own t_dense/t_h2d (spill differs by
        # horizon) -- this is what reproduces serve_cost_frontier.csv nominal values exactly
        return _tpred(rsteady["t_dense"] * f, t_compute_s(t0, eff),
                      cold_steady * P_REC / (c.B_ssd * b_mult), rsteady["t_h2d"] * f, es, eh)

    corners = list(product((T0_LO, T0_HI), (0.3, 0.8), (B_LO_MULT, B_HI_MULT),
                           (ETA_S_LO, ETA_S_HI), (ETA_H_LO, ETA_H_HI)))
    T_worst = max(T(*args) for args in corners)
    T_best = min(T(*args) for args in corners)
    T_nom = T(T0_NOM_S, EFF_NOM, 1.0, ETA_S_NOM, ETA_H_NOM)
    T_s_worst = max(T_s(*args) for args in corners)
    T_s_best = min(T_s(*args) for args in corners)
    T_s_nom = T_s(T0_NOM_S, EFF_NOM, 1.0, ETA_S_NOM, ETA_H_NOM)

    # Nominals are ANCHORED to the substrate's own tps_pred column (the roofline stage's
    # T_pred instantiation), which reproduces the registered numbers exactly: P1 1.35
    # (1xL4 b=1 h64), P1b 14.6 (2xL4 b=8 h256), P14 0.095 (1xL40S b=8 h256), P15b ratios
    # 0.70/0.74/0.93, P15c ratios 1.08-2.29, THEORY.md 6 table (0.527 2xT4, 93.8 CPU).
    # The T_pred sensitivity rebuild above supplies the RELATIVE range envelopes only.
    tps_nom = rp["tps_pred"] / f
    tps_lo = tps_nom * (T_nom / T_worst)
    tps_hi = tps_nom * (T_nom / T_best)
    tps_kill = (0.70 * tps_lo, 1.43 * tps_hi)

    # host-tier hit rate as the FINITE/STEADY band (brief: "Che finite/steady band ... known
    # +25pp steady-state error on short windows"): nominal = finite-window LRU prior (the
    # short-run expectation, P2-consistent); upper edge = steady-Che (long >=512-token
    # serving windows; known to over-predict short windows by +25pp). Pooled-budget basis.
    pooled_slots = host * GIB / P_REC
    w_lo = 0.07 if model == "dsv4_flash" else 0.12
    w_hi = 0.05 if model == "dsv4_flash" else 0.08
    hit_nom = _finite_hit(host, model)
    hit_steady = _host_hit_frac(pooled_slots, model)
    hit_rng = (max(0.0, hit_nom - w_lo), min(1.0, hit_steady + w_hi))
    hit_kill = (max(0.0, hit_nom - 2.0 * w_lo), min(1.0, hit_steady + 2.0 * w_hi))

    cold_kill = (0.75 * cold_lo, 1.25 * cold_hi)

    rate_s = frontier[(cell_name, host, b)]["rate_s"]
    # $/1k tok = 1000 * rate_s / TPS, steady-horizon TPS at the row's own b (THEORY.md 6
    # economics convention; the substrate's rs["tps_pred"] reproduces serve_cost_frontier.csv
    # and the registered P14/P15b/P15c identities exactly). Range = relative T_s envelope.
    steady_tps = rsteady["tps_pred"] / f
    usd_nom = 1000.0 * rate_s / steady_tps
    usd_lo = usd_nom * (T_s_best / T_s_nom)
    usd_hi = usd_nom * (T_s_worst / T_s_nom)
    usd_kill = (0.60 * usd_lo, 1.67 * usd_hi)

    horizon = prim
    limiter = frontier[(cell_name, host, b)]["limiter"]
    substrate = (f"roofline_curves.csv ({cell_name},dsv4_flash,{b},{prim},{host})"
                 + (f" scaled x{MIMO_TOUCH_RATIO.value:.4f} (MIMO_TOUCH_RATIO)" if f != 1.0 else ""))
    extra = (f"; nominal anchored to the substrate tps_pred (reproduces FALSIFICATION P1/"
             f"P1b/P14/P15b/P15c registered centers); range = T_pred sensitivity envelope at "
             f"B_SSD 0.9-1.1x spec (Cell B_ssd {c.B_ssd:.2g} B/s), COMPUTE_EFF 0.3-0.8, "
             f"t_0 50-420us, eta_s 0.15-0.45, eta_h 0.3-0.7; horizon={horizon}; "
             f"limiter={limiter}")
    return {
        "decode_tps": (tps_nom, tps_lo, tps_hi, tps_kill,
                       "THEORY.md 5.1 T_pred = t_dense + t_compute + (1-eta_s)t_storage "
                       "+ (1-eta_h)t_h2d; substrate " + substrate + extra),        "host_hit_rate": (hit_nom, hit_rng[0], hit_rng[1], hit_kill,
                          "FINITE/STEADY Che band: nominal = finite-window LRU prior "
                          "(PREDECL_FINITE_HIT: 0.51 at 16 GiB pooled = P2 0.48-0.54 "
                          "midpoint; saturation 0.536-0.54 = sealed-window repeat ceiling); "
                          "upper edge = steady-Che (HOST_HIT_CURVE, cache_store_knees.csv) "
                          "which over-predicts short windows by +25pp (known-failure 1) and "
                          "governs >=512-token serving windows; pooled host budget basis, "
                          f"({pooled_slots:.0f} pooled slots); independent of cell and b; "
                          "at host 16 GiB rows tagged P2 the P2 kill band 0.45-0.57 governs "
                          "short-window measurements (registered-ID rule)"),
        "cold_records_per_tok": (cold_nom, cold_lo, cold_hi, cold_kill,
                                 "FALSIFICATION.md P4 bracket [steady-Che, 2x finite-Che] on "
                                 + substrate + " (order-clamped min/max at b=8); nominal = "
                                 "finite-Che (anchor: 162 vs 155.06 measured, +4.5%); "
                                 "corrected anchor bracket 67.84-323.92"),
        "usd_per_1k_tok": (usd_nom, usd_lo, usd_hi, usd_kill,
                           "THEORY.md 6 cost frontier: $/1k tok = 1000 * rate_s / TPS "
                           "with the steady-horizon TPS at the row's own b (economics "
                           "convention); nominal = substrate rs tps_pred -> reproduces "
                           "serve_cost_frontier.csv exactly and the registered P14 "
                           "(0.07-0.13) / P15b (0.70/0.74/0.93 ratios) / P15c (1.08-2.29 "
                           "ratios) identities; range = relative T_s envelope; "
                           f"rate_s from serve_cost_frontier.csv ({rate_s:.6g} $/s); "
                           "dense-baseline comparison carried under BOTH price readings "
                           "(0.12892 vs 0.03581 $/1k tok, 3.57x ambiguity)"),
        "_internals": {"alpha_dispatch": alpha, "T_nom": T_nom, "tps_csv_row": rp["tps_pred"],
                       "rate_s": rate_s, "limiter": limiter, "horizon": horizon,
                       "pooled_slots": pooled_slots, "f": f},
    }


def _cpu_derivations(model: str, b: int, host: int, roof: dict, frontier: dict):
    """CPU-only reference rows: portable-torch MEASURED execution dominates."""
    prim = "finite_16tok" if b == 1 else "steady_state"
    rp = roof[("CPU-only", "dsv4_flash", b, prim, host)]
    rsteady = roof[("CPU-only", "dsv4_flash", b, "steady_state", host)]
    rfinite = roof[("CPU-only", "dsv4_flash", b, "finite_16tok", host)]
    f = 1.0 if model == "dsv4_flash" else MIMO_TOUCH_RATIO.value
    tps_nom = rp["tps_pred"] / f
    tps_lo, tps_hi = 0.5 * tps_nom, 90.0 * tps_nom   # tuned-kernel bound ceiling (~90x)
    tps_kill = (0.25 * tps_nom, 200.0 * tps_nom)
    rec = host * GIB / P_REC
    w_lo = 0.07 if model == "dsv4_flash" else 0.12
    w_hi = 0.05 if model == "dsv4_flash" else 0.08
    hit_nom = _finite_hit(host, model)
    hit_steady = _host_hit_frac(rec, model)
    cold_steady = rsteady["cold_records_per_tok"] * f
    cold_finite = rfinite["cold_records_per_tok"] * f
    cold_nom = cold_finite
    cold_lo, cold_hi = min(cold_steady, cold_finite), 2.0 * max(cold_steady, cold_finite)
    cold_kill = (0.75 * cold_lo, 1.25 * cold_hi)
    rate_s = frontier[("CPU-only", host, b)]["rate_s"]
    tps_steady = rsteady["tps_pred"] / f
    usd_nom = 1000.0 * rate_s / tps_steady
    usd_lo, usd_hi = 1000.0 * rate_s / (90.0 * tps_steady), 1000.0 * rate_s / (0.5 * tps_steady)
    usd_kill = (0.60 * usd_lo, 1.67 * usd_hi)
    cpu_note = ("CPU_EXEC_MS_PER_EXPERT 2750 ms MEASURED (portable-torch, AGENTS.md CPU 6/10); "
                "range upper edge = 25-30 ms/expert tuned-kernel bound (unbuilt, ~90x); "
                f"horizon={prim}")
    return {
        "decode_tps": (tps_nom, tps_lo, tps_hi, tps_kill,
                       "roofline_curves.csv CPU-only row (compute-bound, CPU expert execution); "
                       + cpu_note),
        "host_hit_rate": (hit_nom, max(0.0, hit_nom - w_lo), min(1.0, hit_steady + w_hi),
                          (max(0.0, hit_nom - 2.0 * w_lo), min(1.0, hit_steady + 2.0 * w_hi)),
                          f"FINITE/STEADY Che band (same construction as GPU rows) at "
                          f"{rec:.0f} pooled slots (single pooled tier); nominal = finite-"
                          "window LRU prior, upper = steady-Che"),
        "cold_records_per_tok": (cold_nom, cold_lo, cold_hi, cold_kill,
                                 "roofline_curves.csv CPU-only row cold count, P4 bracket "
                                 "[min(steady,finite), 2x max(steady,finite)]; "
                                 + cpu_note),
        "usd_per_1k_tok": (usd_nom, usd_lo, usd_hi, usd_kill,
                           "1000 * rate_s / TPS with rate_s from serve_cost_frontier.csv "
                           f"({rate_s:.6g} $/s, CPU cores + RAM only); CPU cost dominates; "
                           "dense comparison meaningless here (360-5800x worse per THEORY.md 6)"),
        "_internals": {"rate_s": rate_s, "limiter": "compute", "horizon": prim,
                       "rec_per_gpu": rec, "f": f},
    }


# ==========================================================================
# Prediction-ID mapping (FALSIFICATION.md)
# ==========================================================================

def _pred_ids(cell: str, model: str, b: int, host: int, metric: str) -> str:
    ref = cell in REF_CELLS
    if metric == "decode_tps":
        ids = []
        if cell == "1xL4" and b == 1 and host == 64:
            ids.append("P1")
        if cell == "2xL4" and b == 8 and host == 256:
            ids.append("P1b")
        if not ref:
            if b == 8 and host == 256 and cell in ("2xL4", "1xL40S", "1xRTXPRO6000"):
                ids.append("P15b")
            if b == 1 and host == 256:
                ids.append("P15c")
        if cell == "2xT4":
            ids += ["P3c", "P5"]
        return ",".join(dict.fromkeys(ids)) or "-"
    if metric == "host_hit_rate":
        return "P2,P2b" if host == 16 else "P2"
    if metric == "cold_records_per_tok":
        base = ["P3", "P4"]
        if cell == "2xT4":
            base.append("P3b")
        return ",".join(base)
    # usd_per_1k_tok
    ids = []
    if cell == "1xL40S" and b == 8 and host == 256:
        ids.append("P14")
    if not ref and b == 8 and host == 256 and cell in ("2xL4", "1xL40S", "1xRTXPRO6000"):
        ids.append("P15b")
    if not ref and b == 1 and host == 256:
        ids.append("P15c")
    if b == 8 and host == 256:
        ids.append("P15")
    return ",".join(dict.fromkeys(ids)) or "-"


# ==========================================================================
# Matrix assembly
# ==========================================================================

def _sim_lookup(results_SIM, cell: str, model: str, b: int, host: int, metric: str):
    if results_SIM is None:
        return None
    try:
        v = None
        if isinstance(results_SIM, dict):
            for key in ((cell, model, b, host),
                        f"{cell}|{model}|{b}|{host}",
                        f"{cell}_{model}_b{b}_h{host}"):
                if key in results_SIM:
                    v = results_SIM[key]
                    break
        if isinstance(v, dict):
            v = v.get(metric)
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def build_rows(results_SIM=None, results_SOLVE=None):
    roof, frontier = _load_roof(), _load_frontier()
    rows = []
    sim_rec = []
    for model in MODELS:
        for b in BATCHES:
            for cell in CELL_ORDER:
                for host in HOSTS_GIB:
                    if cell == "CPU-only":
                        d = _cpu_derivations(model, b, host, roof, frontier)
                    else:
                        d = _gpu_derivations(cell, model, b, host, roof, frontier)
                    tag_base = ("REFERENCE-NON-PHASE6; " if cell in REF_CELLS else "")
                    for metric in METRICS:
                        nom, lo, hi, kill, cite = d[metric]
                        sim_v = _sim_lookup(results_SIM, cell, model, b, host, metric)
                        if sim_v is None:
                            tag = tag_base + "CLOSED-FORM-ONLY (sim pending at freeze)"
                        else:
                            tag = tag_base + "SIM-PREDICTED+CLOSED-FORM"
                            sim_rec.append({
                                "cell": cell, "model": model, "b": b, "host_gib": host,
                                "metric": metric, "closed_form_nominal": nom, "sim": sim_v,
                                "abs_disagreement": abs(sim_v - nom),
                                "rel_disagreement": (abs(sim_v - nom) / nom if nom else None),
                            })
                        rows.append({
                            "cell": cell, "model": model, "b": b, "host_gib": host,
                            "metric": metric, "nominal": nom, "lo": lo, "hi": hi,
                            "kill_lo": kill[0], "kill_hi": kill[1],
                            "source_tag": tag, "model_citation": cite,
                            "prediction_ids": _pred_ids(cell, model, b, host, metric),
                        })
    return rows, sim_rec


def _meta(results_SIM, results_SOLVE, sim_rec) -> dict:
    return {
        "registration": {
            "utc": REGISTRATION_UTC.value,
            "frozen_commit": "<orchestrator fills SHA>",
            "campaign": "Phase-7 component D -- pre-registered Phase-6 prediction matrix",
            "freeze_rule": "ranges and kill criteria frozen at commit; later information "
                           "may only be appended as dated addenda in PREDICTIONS_PHASE6.md",
            "hardware_runs_existing_at_freeze": 0,
        },
        "scope": {
            "scored_cells": 60,
            "scored_grid": {"cells": list(PHASE6_CELLS), "models": list(MODELS),
                            "b": list(BATCHES), "host_gib_total": list(HOSTS_GIB)},
            "reference_rows_non_phase6": {"cells": list(REF_CELLS), "count": 24,
                                          "note": "2xT4 anchor (calibration/correctness "
                                                  "instrument) and CPU-only (portable-torch "
                                                  "measured); not Phase-6 Modal configs"},
            "metrics": list(METRICS),
            "storage_provisioning": {
                "dsv4_flash_full_store": f"{D4_UNIVERSE_BYTES.value / GIB:.3f} GiB packed "
                                         f"({D4_UNIVERSE_RECORDS.value} records x {P_REC} B)",
                "mimo_v2_flash_full_store": "153.0 GiB packed (12,288 records x 13,369,344 B; "
                                            "AGENT_BRIEF.md says 12,032 records -- inconsistent, "
                                            "see PREDICTIONS_PHASE6.md section 7)",
                "per_cell": "every cell needs the full packed store on local NVMe "
                            "(>= 153 GiB scratch for MiMo-Flash) plus the host budget in RAM "
                            "and the per-GPU VRAM expert cache (L4/A10 22 GiB = 1,766 records; "
                            "L40S 48 GiB = 3,754; RTX PRO 6000 96 GiB = 7,508); at host 256 GiB "
                            "the DSv4-Flash store (146.6 GiB) fits whole in host RAM and the "
                            "MiMo-Flash store (153.0 GiB) fits tightly -- cold start amortizes "
                            "after the first fill",
            },
        },
        "assumptions": {
            "COMPUTE_EFF": {"value": COMPUTE_EFF.value, "range": [0.3, 0.8], "tag": "ASSUMPTION",
                            "cite": "FALSIFICATION.md data-needed (GPU GEMM microbenchmark "
                                    "missing); 0.6 of vendor dense-tensor peak"},
            "t_0_per_touch_us": {"value": T0_NOM_S * 1e6, "range": [50.0, 420.0],
                                 "tag": "CALIBRATED on anchor, carried cross-GPU",
                                 "cite": "T4_TOUCH_OVERHEAD_US (provenance.csv); ranges = "
                                         "FALSIFICATION P8b 300-420, P8c 50-150 (union 50-420)"},
            "eta_s_fill_hiding": {"value": ETA_S_NOM, "range": [ETA_S_LO, ETA_S_HI],
                                  "tag": "CALIBRATED on anchor",
                                  "cite": "THEORY.md 5.1; FALSIFICATION P5 (0.15-0.45)"},
            "eta_h_h2d_hiding": {"value": ETA_H_NOM, "range": [ETA_H_LO, ETA_H_HI],
                                 "tag": "ASSUMPTION",
                                 "cite": "THEORY.md 5.1 (eta_h = 0.5 ASSUMPTION)"},
            "B_SSD_mult": {"value": 1.0, "range": [B_LO_MULT, B_HI_MULT], "tag": "ASSUMPTION",
                           "cite": "Cell.B_ssd spec values are ASSUMPTION (Cell provenance); "
                                   "live runs must report measured B_SSD"},
            "popularity_transfer_S_TRANSFER": {
                "value": "DSv4-Flash closed form x MIMO_TOUCH_RATIO for MiMo-Flash",
                "range": "MiMo ranges widened (hit +/-12pp vs +/-7pp; touch ratio assumes top-6)",
                "tag": "ASSUMPTION",
                "cite": "FALSIFICATION.md data-needed (no MiMo router traces); "
                        "S_TRANSFER (provenance.csv); MIMO_TOPK is ASSUMPTION (data needed)"},
            "host_hit_curve": dict(HOST_HIT_CURVE.value),
            "finite_hit_prior": dict(PREDECL_FINITE_HIT.value),
            "modal_prices": {"value": MODAL_PRICES.value, "tag": "ASSUMPTION",
                             "cite": "AGENT_BRIEF price schedule; per-Cell price_tag=ASSUMPTION"},
            "serve_structure": dict(SERVE_ASSUMPTIONS.value),
            "volume_price": {"value": VOLUME_PRICE_GIB_MO.value, "tag": "ASSUMPTION",
                             "cite": "$0.09/GiB/mo, first 1 TiB free; 146.6 GiB store = "
                                     "$3.8/mo above free tier (THEORY.md 6) -- negligible"},
            "dense_baseline_price_ambiguity": dict(DENSE_PRICE_READINGS.value),
            "cpu_kernel_bound": dict(CPU_KERNEL_BOUND.value),
            "ws_lru_knee_gib": {"value": WS_KNEE.value, "tag": "MEASURED",
                                "cite": "provenance.csv ws_lru_knee_gib (sealed-trace pooled LRU)"},
        },
        "scoring_rules": dict(PREDECL_KILL_RULES.value) | {
            "metric_artifact_map": {
                "decode_tps": "result.json decode_tok_s (cross-check decode_tokens/decode_wall_s)",
                "host_hit_rate": "result.json host_pack hits/(hits+misses) "
                                 "(pooled_dedup definition per data/anchor_gate.json "
                                 "host_share.definitions; 51.3% anchor figure)",
                "cold_records_per_tok": "per_token_accounting storage_requests / decode tokens "
                                        "(byte_accounting.storage_requests_per_generated_token)",
                "usd_per_1k_tok": "Modal billing receipt: 1000 * (billed_s * rate_s) / "
                                  "emitted_tokens; cross-check 1000 * rate_s / decode_tok_s",
            },
            "prerequisites_per_cell": "billed Modal run on the named config at the named host "
                                     "budget, full packed store on local NVMe, >= 16 decode "
                                     "tokens, stage profile captured, measured B_SSD reported",
        },
        "horizon_convention": {
            "b1_decode_tps": "finite_16tok primary (matches FALSIFICATION P1's short-window "
                             "calibration)",
            "b8_decode_tps": "steady_state primary (matches P1b's steady-state regime)",
            "usd_per_1k_tok": "steady-state TPS basis at BOTH b values (THEORY.md 6 economics "
                              "convention; reproduces serve_cost_frontier.csv exactly and "
                              "P15c's ratio identity 1.08 (1xL4) ... 2.29 (2xT4) vs dense "
                              "0.12892 $/1k tok)",
            "secondary_horizon": "quoted as the cold bracket; cold bracket = P4 construction "
                                 "[min(steady,finite), 2x max(steady,finite)]",
        },
        "sim_status_at_freeze": {
            "results_SIM": "None -- Component A simulator outputs (data/sim_*.json) were not "
                           "present at freeze; sim column reads 'pending sim'",
            "results_SOLVE": "None -- Component B solver outputs (data/solve_*.csv) were not "
                             "present at freeze",
        },
        "sim_reconciliation": (sim_rec or None),
        "corrections_carried": [
            "P4 cold bracket corrected to [67.84, 323.92] "
            "(Phase-7 orchestrator correction 2026-09-30)",
            "host_pack hit share ground truth 51.3% pooled-dedup (46.8%/56.1% per-GPU); the "
            "Phase-7 brief's 23-25% band is the cuda0-only share 24.0% "
            "(data/anchor_gate.json host_share.definitions)",
            "dense H100 price ambiguity: 0.003223 $/GPU/s vs its note $3.223/GPU/h "
            "(3.57x); both readings carried",
            "$/1k tok = 1000 * rate_s / TPS (THEORY.md 6's printed formula was dimensionally "
            "wrong)",
        ],
        "constants_registered_here": [
            "predecl_registration_utc", "predecl_mimo_layers", "predecl_mimo_topk",
            "predecl_mimo_touch_ratio", "predecl_host_hit_curve", "predecl_finite_hit_prior",
            "predecl_eta_ranges",
            "predecl_t0_range_us", "predecl_bssd_mult_range", "predecl_kill_rules",
            "predecl_dense_price_readings", "predecl_cpu_kernel_bound",
        ],
    }


# ==========================================================================
# Markdown table rendering (pasted verbatim into PREDICTIONS_PHASE6.md)
# ==========================================================================

def _fmt(x: float) -> str:
    if 0.0 <= x < 0.01:
        return "0"
    if x >= 1000:
        return f"{x:.0f}"
    if x >= 100:
        return f"{x:.1f}"
    return f"{x:.3g}"


def _metric_cell(row: dict, metric: str) -> str:
    nom, lo, hi = row["nominal"], row["lo"], row["hi"]
    klo, khi = row["kill_lo"], row["kill_hi"]
    if metric == "host_hit_rate":
        nom, lo, hi, klo, khi = (100 * v for v in (nom, lo, hi, klo, khi))
    return (f"{_fmt(nom)} [{_fmt(lo)}-{_fmt(hi)}] "
            f"kill <{_fmt(klo)}|>{_fmt(khi)}")


def render_tables(rows: list) -> str:
    by_group = {}
    for r in rows:
        by_group.setdefault((r["model"], r["b"]), []).append(r)
    out = []
    for (model, b) in sorted(by_group, key=lambda g: (MODELS.index(g[0]), g[1])):
        grp = by_group[(model, b)]
        prim = "finite 16-token" if b == 1 else "steady-state"
        title = f"{DISPLAY[model]}, b={b} (primary horizon: {prim})"
        is_ref = any(r["cell"] in REF_CELLS for r in grp)
        out.append(f"#### {title}\n")
        out.append("| cell | host GiB | decode tok/s | host hit % | cold rec/tok | $/1k tok "
                   "| src | IDs |")
        out.append("|---|---|---|---|---|---|---|---|")
        cell_order = {c: i for i, c in enumerate(CELL_ORDER)}
        host_order = {h: i for i, h in enumerate(HOSTS_GIB)}
        key = {}
        for r in grp:
            key[(r["cell"], r["host_gib"], r["metric"])] = r
        for cell in sorted({r["cell"] for r in grp}, key=lambda c: cell_order[c]):
            for host in sorted({r["host_gib"] for r in grp if r["cell"] == cell},
                               key=lambda h: host_order[h]):
                r_tps = key[(cell, host, "decode_tps")]
                r_hit = key[(cell, host, "host_hit_rate")]
                r_cold = key[(cell, host, "cold_records_per_tok")]
                r_usd = key[(cell, host, "usd_per_1k_tok")]
                star = " *(ref)*" if cell in REF_CELLS else ""
                ids = ",".join(dict.fromkeys(
                    i for r in (r_tps, r_hit, r_cold, r_usd)
                    for i in r["prediction_ids"].split(",") if i != "-")) or "-"
                out.append(
                    f"| {DISPLAY[cell]}{star} | {host} "
                    f"| {_metric_cell(r_tps, 'decode_tps')} "
                    f"| {_metric_cell(r_hit, 'host_hit_rate')} "
                    f"| {_metric_cell(r_cold, 'cold_records_per_tok')} "
                    f"| {_metric_cell(r_usd, 'usd_per_1k_tok')} "
                    f"| CF | {ids} |")
        out.append("")
        if is_ref:
            out.append("*(ref) = reference row, non-Phase-6.*\n")
    return "\n".join(out)


def render_example_rows(rows: list) -> str:
    picks = [("1xL4", "dsv4_flash", 1, 64), ("2xL4", "dsv4_flash", 8, 256),
             ("1xL40S", "dsv4_flash", 8, 256), ("1xL4", "mimo_v2_flash", 1, 16),
             ("2xT4", "dsv4_flash", 1, 16)]
    idx = {(r["cell"], r["model"], r["b"], r["host_gib"], r["metric"]): r for r in rows}
    out = ["| cell | model | b | host | metric | nominal | lo | hi | kill_lo | kill_hi | "
           "source_tag | prediction_ids |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell, model, b, host in picks:
        for metric in METRICS:
            r = idx[(cell, model, b, host, metric)]
            out.append(
                f"| {cell} | {model} | {b} | {host} | {metric} "
                f"| {_fmt(r['nominal'])} | {_fmt(r['lo'])} | {_fmt(r['hi'])} "
                f"| {_fmt(r['kill_lo'])} | {_fmt(r['kill_hi'])} "
                f"| {r['source_tag']} | {r['prediction_ids']} |")
    return "\n".join(out)


# ==========================================================================
# Entry point
# ==========================================================================

def run(results_SIM=None, results_SOLVE=None):
    """Regenerate data/predecl_matrix.csv + data/predecl_meta.json deterministically.

    results_SIM / results_SOLVE: optional Component-A simulator / Component-B solver
    outputs (see module docstring for the accepted mapping shape). Both may be None.
    """
    rows, sim_rec = build_rows(results_SIM, results_SOLVE)
    with open(MATRIX_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in _CSV_COLS})
    meta = _meta(results_SIM, results_SOLVE, sim_rec)
    with open(META_JSON, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # markdown fragments consumed by PREDICTIONS_PHASE6.md (sections 1-2 + examples)
    n_scored = sum(1 for r in rows if r["cell"] in PHASE6_CELLS)
    n_ref = len(rows) - n_scored
    tables = [
        "<!-- generated by theory.predecl -- verbatim source of PREDICTIONS_PHASE6.md "
        "sections 1-2; do not hand-edit -->",
        f"<!-- scored metric-rows: {n_scored} ({n_scored // 4} config cells x 4 metrics); "
        f"reference metric-rows: {n_ref} ({n_ref // 4} ref cells x 4 metrics) -->",
        "",
        "Cell format: `nominal [lo-hi] kill <kill_lo|>kill_hi`. `CF` = CLOSED-FORM-ONLY "
        "(sim pending at freeze). Host budgets are TOTAL GiB (split evenly over GPUs; the "
        "engine host tier is per-GPU). Host hit % is the host-tier hit share "
        "(hits/(hits+misses), pooled_dedup definition). Host hit % is a FINITE/STEADY band: "
        "nominal = finite-window LRU prior (short 4-20 token runs should land near it), the "
        "upper edge = steady-Che (valid for >=512-token serving windows; known +25pp "
        "over-prediction on short windows). Decode tok/s horizon: finite "
        "16-token at b=1 (P1 convention), steady-state at b=8 (P1b convention). $/1k tok is "
        "on the steady-state TPS basis at both b (THEORY.md 6 economics convention). "
        "Cold-rec lower edge `0` is the steady-Che floor (the model legitimately reads ~0 "
        "cold once host+VRAM cover the working set; a cold start's compulsory reads live in "
        "the nominal/horizon terms) -- the falsifiable content of that metric is the UPPER "
        "edge and the P4 bracket.",
        "",
        render_tables(rows),
        "",
        "### Example rows verbatim (machine-readable column form)",
        "",
        render_example_rows(rows),
        "",
    ]
    with open(TABLES_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tables))
    return {"rows": rows, "meta": meta, "sim_reconciliation": sim_rec}
