"""Core model for the dee-serve provisioning solver (Phase-7 component C).

The capacity problem, as a constrained optimisation (full statement with
equation citations in ``theory/solve/SOLVER.md``):

    minimise    $/1k tok = 1000 * (r * replica_rate_s + volume_rate_s)
                        / (lambda * L)          [demand convention]
                $/1k tok = 1000 * replica_rate_s / X_k                [sat.]
    over        cell (discrete, constants.ALL_CELLS), host GiB
                (continuous, fine grid 4-1024), VRAM GiB (continuous
                within the cell's card memory), prefetch budget
                ({0, one legal hint budget}), decode batch policy
                b (streams/replica), replica count r (integer)
    subject to  Poisson arrivals at lambda, request shape L decode
                tokens (SERVE_ASSUMPTIONS), optional per-stream decode
                SLO, host/VRAM memory limits of the cell, storage
                bandwidth of the cell, stability + headroom
                (rho <= SOLVE_RHO_MAX), exactness contract (hints may
                only waste bandwidth, never change execution).

Named methods only: enumeration over the discrete grid; Che's common-TTL
residence approximation for cache hit rates (THEORY.md 3.1) with the
two-level miss-through form (THEORY.md 5.1); the overlap-corrected token
roofline T_pred (THEORY.md 5.1); Little's law; an M/M/1 response-time
inflation (Kleinrock inverse-utilisation law); the batch-overlap
correction u_i(b) = 1-(1-pi_i)^b (THEORY.md 5.1).

Deterministic: no RNG anywhere in this package.

All constants declared here are registered through ``constants.C`` at
import time so ``data/provenance.csv`` picks them up automatically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..cache import CheTable
from ..constants import (ALL_CELLS, C, COMPUTE_EFF, DENSE_BASELINE, GiB_F,
                         SERVE_ASSUMPTIONS, T4_TOUCH_OVERHEAD_US,
                         VOLUME_PRICE_GIB_MO, Cell)
from ..roofline import build_pop, dense_ms_for, load_geoms

GIB = float(1 << 30)
MONTH_S = 30.0 * 86400.0


# ==========================================================================
# Solve-owned constants (provenance registered at import).
# ==========================================================================
SOLVE_SLO_SET = C(
    "solve_slo_set", [5, 20, 50], "ASSUMPTION",
    "component-C task statement (SLO set chosen by the solver)",
    "per-stream decode tok/s SLOs for 512-token requests; equivalent mean "
    "response budgets L/SLO = 102.4 s / 25.6 s / 10.2 s", "tok/s",
    uncertainty=(1, 100))

SOLVE_LAMBDA_GRID = C(
    "solve_lambda_grid", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    "ASSUMPTION", "component-C task statement (arrival-rate grid 0.01-10 req/s)",
    "Poisson arrival rates per second", "req/s")

SOLVE_LEN_GRID = C(
    "solve_request_len_grid", [64, 128, 256, 512, 1024, 2048, 4096],
    "ASSUMPTION",
    "component-C task statement (break-even surface axis: request length); "
    "512 is the SERVE_ASSUMPTIONS request shape",
    "decode tokens per request", "tokens")

SOLVE_BATCH_GRID = C(
    "solve_batch_policy_grid", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
    "ASSUMPTION",
    "dee-serve continuous-batching window (no measured scheduler exists yet)",
    "decode batch policy b = concurrently decoded request streams per "
    "replica; ceiling 32 = SOLVE_BATCH_CAP", "streams", uncertainty=(4, 64))

SOLVE_BATCH_CAP = C(
    "solve_batch_cap", 32, "ASSUMPTION",
    "dee-serve continuous-batching window (no measured scheduler exists yet)",
    "max concurrently decoded request streams per replica", "streams",
    uncertainty=(8, 64))

SOLVE_RHO_MAX = C(
    "solve_rho_headroom", 0.7, "ASSUMPTION",
    "standard capacity-planning headroom for Poisson burstiness",
    "max replica utilisation rho = lambda_r * L / X admitted as feasible",
    "fraction", uncertainty=(0.5, 0.9))

SOLVE_CORES = C(
    "solve_cpu_cores_per_replica", 2, "ASSUMPTION",
    "Modal container floor for a GPU worker (orchestration/IO threads)",
    "billed CPU cores per dee replica; deliberately NOT the full cell "
    "n_cpu_cores constant (a scheduler does not pin 8-32 cores to one "
    "worker) - the delta vs serving.py's accounting is reported as a "
    "disagreement in SOLVER.md", "cores", uncertainty=(1, 4))

SOLVE_PF = C(
    "solve_prefetch_arms", {"off": 0,
                            "m1_announced_per_call": 6.119047619047619,
                            "m1_precision": 0.27456225680933855,
                            "m1_recall": 0.22716297786720321},
    "MEASURED",
    "dee.cpp/theory/data/prefetch_predictor.csv (m_per_source=1 row: "
    "announced_records_per_call=6.119, precision=0.2746, recall=0.2272)",
    "the two legal prefetch budgets compared: hints off (0) and one legal "
    "hint budget (1 candidate per source expert, the measured next-layer "
    "empirical-conditional predictor). Wrong hints can only waste bandwidth "
    "(exactness contract).")

SOLVE_PF_MODEL = C(
    "solve_prefetch_cost_model",
    "dt = (wasted - saved) * (1 - eta_s) * t_service; "
    "saved = A_tok * miss_frac * precision; wasted = A_tok * miss_frac - saved",
    "DERIVED",
    "THEORY.md 5.1 fill-service model + data/prefetch_predictor.csv",
    "a right hint replaces one demand-gated fill earlier in the same "
    "saturated device (saves one exposed service slice), a wrong hint adds "
    "one; both credited/debited at the same hiding fraction (symmetric "
    "treatment). At measured precision 0.275 < 0.5 the sign is negative "
    "under ANY symmetric hiding treatment (P12-consistent: k=1 hints worth "
    "~0 s on the sealed bank shape).")

SOLVE_DENSE_REF_ACTIVE = C(
    "solve_dense_ref_active_params", 13.0e9, "MEASURED",
    "AGENTS.md project overview (~13B active/token, ~284B backbone, DSv4-Flash)",
    "reference active parameters/token of the dense-baseline anchor model",
    "params")

SOLVE_DENSE_BACKBONE = C(
    "solve_dense_backbone_gib", 17.4, "DERIVED",
    "constants.DENSE_BASELINE.resident_bytes 568 GiB minus routed-universe "
    "bf16 (11776 records x 25,165,824 params x 2 B = 550.7 GiB)",
    "non-routed (attention + dense/shared) bf16 footprint of DSv4-Flash; "
    "carried unchanged to the other three stores (ASSUMPTION sub-range "
    "10-40 GiB - their attention shapes are not in-repo, data needed)", "GiB",
    uncertainty=(10.0, 40.0))

SOLVE_DENSE_PRICE_GPU_S = C(
    "solve_dense_price_gpu_s_computed", 0.003223, "ASSUMPTION",
    "constants.DENSE_BASELINE price_gpu_s AS COMPUTED (0.003223 $/GPU/s = "
    "$11.60/GPU/h); the constant's own note and serving.py claim "
    "'$3.223/GPU/h' = 0.000895 $/GPU/s - the constant is AMBIGUOUS "
    "(orchestrator review 2026-09). Both readings are carried through the "
    "break-even surface; this is the primary (computed) reading.",
    "dense-baseline H100 price per GPU per second, computed reading",
    "$/GPU/s", uncertainty=(0.000895, 0.003223))

SOLVE_DENSE_PRICE_GPU_S_CHEAP = C(
    "solve_dense_price_gpu_s_cheap", 0.000895, "ASSUMPTION",
    "the '$3.223/GPU/h' reading of constants.DENSE_BASELINE's note "
    "(3.223/3600) - 3.57x cheaper than the computed reading",
    "dense-baseline H100 price per GPU per second, cheap reading",
    "$/GPU/s", uncertainty=(0.000895, 0.003223))

SOLVE_DENSE_TPS = C(
    "solve_dense_tok_s", 200.0, "ASSUMPTION",
    "constants.DENSE_BASELINE (deliberately high achieved aggregate decode "
    "for a 13B-active MoE on 8xH100)",
    "dense baseline aggregate decode throughput at the reference model",
    "tok/s", uncertainty=(100.0, 400.0))

SOLVE_DENSE_SCALE = C(
    "solve_dense_scaling_law",
    "X_dense = 200 * (13.0e9 / active_params); "
    "n_H100 = ceil((universe_params*2 B)/GIB + 17.4) / 74.5 GiB",
    "ASSUMPTION",
    "dense decode is memory-bandwidth bound in active parameters (no "
    "in-repo dense serving measurement exists - data needed)",
    "cross-model dense baseline scaling; deliberately CONSERVATIVE TOWARD "
    "THE COMPETITOR: aggregate throughput does NOT fall with resident "
    "footprint (only with active parameters), price scales only with the "
    "H100 count needed to hold the bf16 weights (74.5 usable GiB per "
    "H100-80GB), and the dense side is granted rho<=1.0 with no SLO and no "
    "headroom while dee must hold rho<=0.7 plus the SLO")

SOLVE_H100_USABLE = C(
    "solve_h100_usable_gib", 74.5, "ASSUMPTION",
    "H100-80GB nominal 80 GiB minus runtime/allocator reserve",
    "usable bf16 weight capacity per dense-baseline H100", "GiB",
    uncertainty=(70.0, 80.0))

SOLVE_MONTH_S = C("solve_month_seconds", MONTH_S, "DERIVED", "30-day month",
                  "converts VOLUME_PRICE_GIB_MO to $/s", "s")

SOLVE_N_TAU_DEFAULT = 200

SOLVE_N_TAU = C(
    "solve_che_tau_grid", 200, "ASSUMPTION",
    "CheTable tau-grid resolution (theory/cache.py default 400)",
    "log-spaced tau grid used for the Che TTL inversion in the solver's "
    "20k-point enumeration; 400 is kept for the dsv4 disagreement "
    "cross-check and the interpolation residual is reported in SOLVER.md",
    "points", uncertainty=(100, 400))

SOLVE_KNEE_METHOD = C(
    "solve_knee_method", "maximum perpendicular distance from the chord "
    "between the frontier endpoints in (log10 TPS, log10 $/1k-tok) space",
    "ASSUMPTION", "named knee heuristic (kneedle-style)",
    "how the 'knee' allocation is selected on each model's Pareto frontier")

SOLVE_T4_PRICE_ARTIFACT = C(
    "solve_t4_zero_price_artifact", 0.0, "MEASURED",
    "constants.T4_CELL price_gpu_s=0.0 (anchor cell ran on a free Kaggle "
    "tier; no purchasable T4 price exists in the price table)",
    "2xT4 carries a ZERO GPU price: its cost-frontier position is a price-"
    "table artifact, not a purchasable quote. The solver reports it but "
    "flags is_purchasable=False and computes the T4 break-even price at "
    "which it would match the best purchasable cell.", "$/GPU/s")


# ==========================================================================
# Enumeration grids (--fast coarsens)
# ==========================================================================
HOST_GRID = [4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0,
             192.0, 256.0, 384.0, 512.0, 768.0, 1024.0]
HOST_GRID_FAST = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
VFRAC_GRID = [0.05, 0.25, 0.6, 1.0]
VFRAC_GRID_FAST = [0.25, 1.0]
LEN_GRID = [64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0]
LAMBDA_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
SLO_GRID = [5.0, 20.0, 50.0]
PF_ARMS = ["off", "m1"]
R_MAX = 4096


def grids(fast: bool = False):
    return ((HOST_GRID_FAST if fast else HOST_GRID),
            (VFRAC_GRID_FAST if fast else VFRAC_GRID),
            tuple(SOLVE_BATCH_GRID.value))


# ==========================================================================
# Che two-level hit model with cached inversion tables
# ==========================================================================
class HitCache:
    """Two-level Che TTL miss-through with cached inversion tables.

    Same equations as ``roofline.hit_rates`` (THEORY.md 3.1 + 5.1) but the
    host-level Che table is cached per (batch, VRAM capacity) so a 20k-point
    enumeration does not rebuild N x n_tau tables per query.  The cache is
    tiny (FIFO, cap 8): the enumeration loops host capacity innermost, so one
    host table serves a whole host sweep before it is displaced.
    """

    def __init__(self, pop, geom, n_tau: int = SOLVE_N_TAU_DEFAULT):
        self.pop = pop
        self.geom = geom
        self.n_tau = n_tau
        self._top: Dict[int, CheTable] = {}
        self._host: Dict[Tuple[int, int], Tuple[CheTable, np.ndarray]] = {}
        self._fifo: List[Tuple[int, int]] = []

    def _u_w(self, k: int):
        u = self.pop.u(k).reshape(-1)
        tot = float(u.sum())
        return u, (u / max(1e-30, tot))

    def hits(self, m_v: float, m_h: float, k: int) -> Dict[str, float]:
        u, w = self._u_w(k)
        tab = self._top.get(k)
        if tab is None:
            tab = CheTable(w, n_tau=self.n_tau)
            self._top[k] = tab
        tau_v = tab.tau_at_capacity(float(max(1, int(m_v)))) if m_v > 0 else 0.0
        h_v = 1.0 - np.exp(-w * tau_v)
        u_h = u * (1.0 - h_v)
        tot_h = float(u_h.sum())
        if tot_h <= 0 or m_h <= 0:
            h_h = np.zeros_like(h_v)
            w_h = w
        else:
            key = (k, int(m_v))
            ent = self._host.get(key)
            if ent is None:
                w_h = u_h / tot_h
                ent = (CheTable(w_h, n_tau=self.n_tau), w_h)
                self._host[key] = ent
                self._fifo.append(key)
                while len(self._fifo) > 8:
                    self._host.pop(self._fifo.pop(0), None)
            tab_h, w_h = ent
            tau_h = tab_h.tau_at_capacity(float(max(1, int(m_h))))
            h_h = 1.0 - np.exp(-w_h * tau_h)
        tot = float(u.sum())
        H_v = float((u * h_v).sum() / max(1e-30, tot))
        miss_v = float((u * (1.0 - h_v)).sum())
        miss_h = float((u * (1.0 - h_v) * (1.0 - h_h)).sum())
        H_h = 1.0 - miss_h / max(1e-30, miss_v) if miss_v > 0 else 0.0
        return {"H_v": H_v, "H_h": float(H_h),
                "miss_through_fraction": miss_h / max(1e-30, tot),
                "miss_v_fraction": miss_v / max(1e-30, tot),
                "D_per_token": tot}


# ==========================================================================
# Dense-residency baseline (both price readings; conservative to competitor)
# ==========================================================================
@dataclass(frozen=True)
class DenseSpec:
    model: str
    active_params: float
    dense_gib: float
    n_h100: int
    price_per_gpu_s: float
    tok_s: float

    @property
    def price_s(self) -> float:
        return self.n_h100 * self.price_per_gpu_s

    def cost_usd_per_1k_tok(self) -> float:
        return self.price_s / self.tok_s * 1000.0


DENSE_PRICE_READINGS = {
    "computed_0.003223": float(SOLVE_DENSE_PRICE_GPU_S.value),
    "cheap_0.000895": float(SOLVE_DENSE_PRICE_GPU_S_CHEAP.value),
}


def dense_spec(geom, price_gpu_s: Optional[float] = None) -> DenseSpec:
    if price_gpu_s is None:
        price_gpu_s = DENSE_PRICE_READINGS["computed_0.003223"]
    params_per_rec = geom.flops_per_touch / 2.0      # 2 FLOP per param per tok
    routed_active = geom.touches_per_token * params_per_rec
    backbone_active = (float(SOLVE_DENSE_REF_ACTIVE.value)
                       - 276.0 * 25165824.0)          # DSv4-calibrated
    active = routed_active + backbone_active
    universe_params = geom.universe_records * params_per_rec
    dense_gib = universe_params * 2.0 / GIB + float(SOLVE_DENSE_BACKBONE.value)
    n_h100 = max(1, int(math.ceil(dense_gib / float(SOLVE_H100_USABLE.value))))
    tok_s = float(SOLVE_DENSE_TPS.value) * (
        float(SOLVE_DENSE_REF_ACTIVE.value) / active)
    return DenseSpec(geom.name, active, dense_gib, n_h100,
                     float(price_gpu_s), tok_s)


# ==========================================================================
# Enumeration table
# ==========================================================================
@dataclass(frozen=True)
class AllocRow:
    """One enumerated allocation operating point at batch policy ``k``.

    Hit-derived quantities are scenario-independent; timing is recomputed
    cheaply per (COMPUTE_EFF, t_0, prefetch arm) scenario.
    """
    model: str
    cell: Cell
    host_gib: float
    vram_gib: float
    k: int
    D: float
    miss_through: float
    miss_v_fraction: float
    H_v: float
    H_h: float
    cold_recs: float
    spill_recs: float
    t_service: float
    t_h2d: float
    t_dense: float

    @property
    def alloc_id(self) -> Tuple[str, str, float, float]:
        return (self.model, self.cell.name, self.host_gib, self.vram_gib)


def point_timing(row: AllocRow, eff: float, t0_us: float,
                 pf: str = "off") -> Dict[str, float]:
    """Overlap-corrected token prediction T_pred (THEORY.md 5.1) + prefetch."""
    g_touch = _TOUCHES[row.model]
    g_flop = _FLOP[row.model]
    cell = row.cell
    t_demand = row.cold_recs * row.t_service
    if cell.n_gpu == 0:
        t_compute = g_touch * 2750.0e-3 / max(1, row.k)
    else:
        launch = row.D * (t0_us * 1e-6) / max(1, row.k)
        arith = g_touch * (g_flop / max(1.0, eff * cell.F_peak))
        t_compute = launch + arith
    t_pf = 0.0
    if pf == "m1":
        A_tok = float(SOLVE_PF.value["m1_announced_per_call"]) * g_touch / 6.0
        # announced per call is per layer-call; layers = touches / topk(6)
        prec = float(SOLVE_PF.value["m1_precision"])
        saved = A_tok * row.miss_through * prec
        wasted = A_tok * row.miss_through - saved
        t_pf = (wasted - saved) * (1.0 - _ETA_S[0]) * row.t_service
    t_pred = (row.t_dense + t_compute
              + (1.0 - _ETA_S[0]) * t_demand
              + (1.0 - _ETA_H[0]) * row.t_h2d
              + t_pf)
    X = 1.0 / t_pred if t_pred > 0 else float("inf")
    terms = {"storage": t_demand + t_pf, "h2d": row.t_h2d,
             "compute": t_compute, "dense": row.t_dense}
    return {"X": X, "t_pred": t_pred, "t_compute": t_compute,
            "t_demand": t_demand, "t_pf": t_pf,
            "limiter": max(terms, key=terms.get)}


# module-level scenario state used by point_timing (set by SolveModel)
_TOUCHES: Dict[str, int] = {}
_FLOP: Dict[str, float] = {}
_ETA_S = [0.2917]
_ETA_H = [0.5]


class SolveModel:
    """Enumerated allocation table + closed-form deployment solver."""

    def __init__(self, results_D=None, results_B=None, fast: bool = False):
        self.fast = fast
        self.geoms = load_geoms()
        self.pops = {n: build_pop(g) for n, g in self.geoms.items()}
        self.hits = {n: HitCache(self.pops[n], self.geoms[n])
                     for n in self.geoms}
        self.dense = {n: dense_spec(g) for n, g in self.geoms.items()}
        self.dense_by_reading = {
            rd: {n: dense_spec(g, p) for n, g in self.geoms.items()}
            for rd, p in DENSE_PRICE_READINGS.items()}
        self.t0_us = float(T4_TOUCH_OVERHEAD_US.value)
        self.compute_eff = float(COMPUTE_EFF.value)
        self.eta_s = self._eta_storage(results_D)
        self.eta_h = 0.5                                # ASSUMPTION (roofline)
        _ETA_S[0] = self.eta_s
        _ETA_H[0] = self.eta_h
        for n, g in self.geoms.items():
            _TOUCHES[n] = g.touches_per_token
            _FLOP[n] = g.flops_per_touch
        self.host_grid, self.vfrac_grid, self.k_grid = grids(fast)
        self.cells: List[Cell] = list(ALL_CELLS)
        self.cell_by_name = {c.name: c for c in self.cells}
        self.table: List[AllocRow] = []
        self.results_D = results_D
        self.results_B = results_B

    # ---- calibrated fill-hiding fraction (CALIBRATED, anchor residual) ----
    @staticmethod
    def _eta_storage(results_D) -> float:
        if results_D and isinstance(results_D, dict):
            a = results_D.get("anchor") or {}
            if "calibrated_eta_storage" in a:
                return float(a["calibrated_eta_storage"])
        # Standalone run: reuse the value the roofline stage calibrated into
        # its own artifact (that stage's files are not ours to regenerate).
        import json

        from .. import paths

        p = paths.DATA_DIR / "anchor_check.json"
        if p.exists():
            try:
                return float(json.loads(p.read_text(encoding="utf-8"))
                             ["calibrated_eta_storage"])
            except Exception:
                pass
        return 0.2917

    # ---- enumeration ------------------------------------------------------
    def build_table(self) -> List[AllocRow]:
        if self.table:
            return self.table
        rows: List[AllocRow] = []
        for mname, g in self.geoms.items():
            hc = self.hits[mname]
            for cell in self.cells:
                n_gpu = max(1, cell.n_gpu)
                vram_cap = cell.vram_bytes / GIB if cell.n_gpu else 0.0
                for host_gib in self.host_grid:
                    for vf in self.vfrac_grid:
                        vram_gib = vram_cap * vf
                        v_slots = (int(vram_gib * GIB / g.record_bytes)
                                   if cell.n_gpu else 0)
                        m_v = n_gpu * v_slots
                        m_h = host_gib * GIB / g.record_bytes
                        for k in self.k_grid:
                            hr = hc.hits(float(m_v), float(m_h), k)
                            D = hr["D_per_token"]
                            cold_recs = D * hr["miss_through_fraction"]
                            spill_recs = D * (1.0 - hr["H_v"])
                            t_service = g.record_bytes / max(1.0, cell.B_ssd)
                            t_h2d = (g.record_bytes * spill_recs
                                     / max(1.0, cell.B_h2d)) if cell.B_h2d > 0 \
                                else 0.0
                            dense_ms = dense_ms_for(cell, g)
                            t_dense = (dense_ms * 1e-3) * (0.35 + 0.65 * k) \
                                / max(1, k)
                            rows.append(AllocRow(
                                model=mname, cell=cell, host_gib=host_gib,
                                vram_gib=vram_gib, k=k, D=D,
                                miss_through=hr["miss_through_fraction"],
                                miss_v_fraction=hr["miss_v_fraction"],
                                H_v=hr["H_v"], H_h=hr["H_h"],
                                cold_recs=cold_recs, spill_recs=spill_recs,
                                t_service=t_service, t_h2d=t_h2d,
                                t_dense=t_dense))
        self.table = rows
        return rows

    def alloc_index(self) -> Dict[Tuple[str, str, float, float],
                                 Dict[int, AllocRow]]:
        idx: Dict[Tuple[str, str, float, float], Dict[int, AllocRow]] = {}
        for r in self.table:
            idx.setdefault(r.alloc_id, {})[r.k] = r
        return idx

    # ---- money -------------------------------------------------------------
    @staticmethod
    def replica_rate(cell: Cell, host_gib: float) -> Dict[str, float]:
        gpu = cell.n_gpu * cell.price_gpu_s
        cpu = float(SOLVE_CORES.value) * cell.price_cpu_core_s
        ram = cell.price_ram_gib_s * host_gib
        return {"gpu": gpu, "cpu": cpu, "ram": ram, "total": gpu + cpu + ram}

    @staticmethod
    def volume_rate_s(store_gib: float) -> float:
        return (max(0.0, store_gib - 1024.0)
                * float(VOLUME_PRICE_GIB_MO.value) / MONTH_S)

    # ---- deployment solve (closed form over the enumerated table) ----------
    @staticmethod
    def rho_fill(b: int) -> float:
        """Batch-fillability floor on rho for a full batch of b streams.

        Mean concurrency under the M/M/1 batch model is
        K = rho*b/(1-rho) (Little + Kleinrock); a batch of b fills when
        K >= b-1, i.e. rho >= (b-1)/(2b-1).  Below that floor the declared
        batch policy would run underfilled and X(b) overstates capacity, so
        such operating points are declared infeasible (smaller b covers them).
        """
        return (b - 1.0) / (2.0 * b - 1.0) if b > 1 else 0.0

    def lam_r_max(self, row: AllocRow, L: float,
                  slo: Optional[float], eff: float, t0_us: float,
                  pf: str) -> Tuple[float, Dict[str, float]]:
        """Max per-replica arrival rate this allocation/batch can serve.

        Closed form: rho* = min(SOLVE_RHO_MAX, 1 - slo*b/X) clipped to the
        batch-fillability window [rho_fill(b), 1); lam_r_max = rho* * X / L.
        """
        tm = point_timing(row, eff, t0_us, pf)
        X = tm["X"]
        b = row.k
        rho_hi = float(SOLVE_RHO_MAX.value)
        if slo is not None:
            rho_hi = min(rho_hi, 1.0 - float(slo) * b / max(1e-12, X))
        rho_lo = self.rho_fill(b)
        if rho_hi <= rho_lo or X <= 0:
            return 0.0, tm
        lam = rho_hi * X / L
        return lam, tm

    def solve_deployment(self, idx, model: str, lam: float, L: float,
                         slo: Optional[float], pf: str = "off",
                         eff: Optional[float] = None,
                         t0_us: Optional[float] = None,
                         cells: Optional[Sequence[str]] = None,
                         host_gib_fixed: Optional[float] = None) -> dict:
        """Minimum-cost feasible deployment for traffic (lam, L) [+ SLO].

        Enumerates (cell, host GiB, VRAM GiB) x batch policy; replica count
        r = ceil(lam / lam_r_max) (integer replication, cheapest at the
        smallest feasible r); cost = r*replica_rate + volume_rate.
        Returns the winning allocation dict or None if infeasible.
        """
        eff = self.compute_eff if eff is None else float(eff)
        t0_us = self.t0_us if t0_us is None else float(t0_us)
        best = None
        for (mname, cname, host_gib, vram_gib), by_k in idx.items():
            if mname != model:
                continue
            if cells is not None and cname not in cells:
                continue
            if host_gib_fixed is not None and abs(host_gib - host_gib_fixed) > 1e-9:
                continue
            cell = by_k[min(by_k)].cell
            rate = self.replica_rate(cell, host_gib)
            # closed-form capacity bound at the full policy batch (candidate
            # generator for r); _report_point below verifies the realised
            # fixed-point operating point and rejects optimistic bounds.
            row_max = by_k[max(self.k_grid)]
            lam_bound, _tm = self.lam_r_max(row_max, L, slo, eff, t0_us, pf)
            if lam_bound <= 0:
                continue
            r0 = max(1, int(math.ceil(lam / lam_bound - 1e-12)))
            rep = None
            r_used = r0
            for r_try in range(r0, min(r0 + 8, R_MAX + 1)):
                rep = self._report_point(by_k, lam / r_try, L, r_try,
                                         eff, t0_us, pf, slo)
                if rep is not None:
                    r_used = r_try
                    break
            if rep is None:
                continue
            r = r_used
            lam_r = lam / r
            total_rate = r * rate["total"] + self.volume_rate_s(
                self.geoms[model].universe_bytes / GIB)
            cost_1k = 1000.0 * total_rate / (lam * L)
            cand = dict(rep, r=r, lam_r=lam_r, rate=rate,
                        volume_rate_s=self.volume_rate_s(
                            self.geoms[model].universe_bytes / GIB),
                        total_rate_s=total_rate,
                        usd_per_1k_tok=cost_1k,
                        demand_tps=lam * L)
            key = (cand["usd_per_1k_tok"], cand["total_rate_s"], r,
                   cand["host_gib"], cand["vram_gib"])
            if best is None or key < best[0]:
                best = (key, cand)
        return best[1] if best else None

    def _report_point(self, by_k: Dict[int, AllocRow], lam_r: float, L: float,
                      r: int, eff: float, t0_us: float, pf: str,
                      slo: Optional[float]) -> Optional[dict]:
        """Realised operating point at per-replica rate lam_r (reporting).

        Snaps the realised batch occupancy to the policy grid via the
        Little fixed point b = min(b_policy, max(1, ceil(lam_r * W))) and
        recomputes the realised timing at the snapped batch.  If the fixed
        point settles below the batch-fillability floor (the declared policy
        would run underfilled and X(b) overstates capacity) the point is
        declared infeasible — smaller b covers it.
        """
        b_pol = max(by_k)
        b = b_pol
        X = point_timing(by_k[b], eff, t0_us, pf)["X"]
        for _ in range(40):
            rho = lam_r * L / max(1e-12, X)
            if rho >= 1.0:
                return None
            W = (L * b / max(1e-12, X)) / (1.0 - rho)
            K = lam_r * W
            b_next = _snap_k(int(min(b_pol, max(1, math.ceil(K - 1e-9)))),
                             self.k_grid, b_pol)
            if b_next == b:
                break
            b = b_next
            X = point_timing(by_k[b], eff, t0_us, pf)["X"]
        rho = lam_r * L / max(1e-12, X)
        if rho >= 1.0 or rho < self.rho_fill(b) - 1e-9:
            return None
        W = (L * b / max(1e-12, X)) / (1.0 - rho)
        tm = point_timing(by_k[b], eff, t0_us, pf)
        per_stream = L / W
        if slo is not None and per_stream < float(slo) - 1e-9:
            return None
        row = by_k[b]
        core = {"storage": tm["t_demand"] + tm["t_pf"], "h2d": row.t_h2d,
                "compute": tm["t_compute"], "dense": row.t_dense}
        return {
            "model": row.model, "cell": row.cell.name,
            "host_gib": row.host_gib, "vram_gib": row.vram_gib,
            "batch_policy": b_pol, "batch_realized": b,
            "X_replica": X, "tps_capacity": r * X,
            "rho": rho, "W_s": W, "per_stream_tps": per_stream,
            "K_little": lam_r * W, "limiter": max(core, key=core.get),
            "t_pred": tm["t_pred"], "t_compute": tm["t_compute"],
            "t_demand": tm["t_demand"], "t_h2d": row.t_h2d,
            "t_dense": row.t_dense, "t_pf": tm["t_pf"],
            "cold_recs_per_tok": row.cold_recs, "H_v": row.H_v, "H_h": row.H_h,
            "miss_through": row.miss_through, "D_per_tok": row.D,
            "prefetch": pf,
        }


def _snap_k(b: int, k_grid: Sequence[int], b_pol: int) -> int:
    for kk in k_grid:
        if kk >= b:
            return int(min(kk, b_pol))
    return int(b_pol)
