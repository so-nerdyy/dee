"""Deliverable 3 ("for X traffic, provision Y") + the disagreement report
vs the theory layer's hand-computed cost frontier.

build_provisioning: for each (model, arrival rate, per-stream SLO) find the
minimum-cost feasible deployment at the 512-token request shape
(SERVE_ASSUMPTIONS convention) -- full line items, honest infeasibility.

build_disagreement: every place the solver's numbers disagree with
THEORY.md 6 / data/serve_cost_frontier.csv (the prior hand-computed
frontier) and with serving.py's CPU accounting, quantified.
"""
from __future__ import annotations

import csv
import math
from typing import Dict, List, Optional, Tuple

from .model import LAMBDA_GRID, SLO_GRID, SOLVE_CORES, SolveModel

PROV_L = 512.0        # request shape for the provisioning table (tokens)


def build_provisioning(sm: SolveModel, idx):
    rows: List[dict] = []
    infeasible: List[dict] = []
    for model in sorted(sm.geoms):
        for lam in LAMBDA_GRID:
            for slo in SLO_GRID:
                sol = sm.solve_deployment(idx, model, float(lam), PROV_L,
                                          float(slo), pf="off",
                                          eff=sm.compute_eff,
                                          t0_us=sm.t0_us)
                if sol is None:
                    infeasible.append({
                        "model": model, "lambda_req_s": lam,
                        "slo_tok_s": slo, "L_req": PROV_L,
                        "reason": "no feasible (cell, host, VRAM, batch, r) "
                                  "at rho<=0.7 with batch-fillability and "
                                  "the per-stream SLO"})
                    continue
                d_hi = sm.dense_by_reading["computed_0.003223"][
                    model].cost_usd_per_1k_tok()
                d_lo = sm.dense_by_reading["cheap_0.000895"][
                    model].cost_usd_per_1k_tok()
                rate = sol["rate"]
                rows.append({
                    "model": model, "lambda_req_s": lam, "slo_tok_s": slo,
                    "L_req": PROV_L,
                    "cell": sol["cell"], "host_gib": sol["host_gib"],
                    "vram_gib": sol["vram_gib"],
                    "prefetch": sol["prefetch"],
                    "batch_policy": sol["batch_policy"],
                    "batch_realized": sol["batch_realized"],
                    "replicas": sol["r"], "lam_r_per_replica": sol["lam_r"],
                    "X_replica_tps": sol["X_replica"],
                    "tps_capacity": sol["tps_capacity"],
                    "demand_tps": sol["demand_tps"],
                    "rho": sol["rho"], "W_s": sol["W_s"],
                    "per_stream_tps": sol["per_stream_tps"],
                    "K_little": sol["K_little"], "limiter": sol["limiter"],
                    "gpu_rate_s": rate["gpu"], "cpu_rate_s": rate["cpu"],
                    "ram_rate_s": rate["ram"],
                    "volume_rate_s": sol["volume_rate_s"],
                    "total_rate_s": sol["total_rate_s"],
                    "usd_per_1k_tok": sol["usd_per_1k_tok"],
                    "dense_usd_per_1k_computed": d_hi,
                    "dense_usd_per_1k_cheap": d_lo,
                    "ratio_vs_dense_computed": sol["usd_per_1k_tok"] / d_hi,
                    "ratio_vs_dense_cheap": sol["usd_per_1k_tok"] / d_lo,
                    "cold_recs_per_tok": sol["cold_recs_per_tok"],
                    "H_v": sol["H_v"], "H_h": sol["H_h"]})
    return rows, infeasible


def build_disagreement(sm: SolveModel, idx):
    """Quantified disagreements vs data/serve_cost_frontier.csv + accounting."""
    from .. import paths

    rows: List[dict] = []
    frontier = paths.DATA_DIR / "serve_cost_frontier.csv"
    if not frontier.exists():
        rows.append({"quantity": "serve_cost_frontier.csv",
                     "theory_value": "missing", "solver_value": "",
                     "rel_delta": "", "note": "frontier artifact absent; "
                     "the disagreement cross-check could not run"})
        return rows

    with open(frontier, newline="", encoding="utf-8") as f:
        theory_rows = list(csv.DictReader(f))

    model = "dsv4_flash"
    for tr in theory_rows:
        cname = tr["cell"]
        host = float(tr["host_gib"])
        b = int(tr["batch"])
        # the solver's comparable point: same (cell, host), full VRAM
        key = _find_alloc(sm, idx, model, cname, host)
        if key is None:
            continue
        by_k = idx[key]
        k_near = min(by_k, key=lambda k: abs(k - b))
        row = by_k[k_near]
        from .model import point_timing

        tm = point_timing(row, sm.compute_eff, sm.t0_us, "off")
        X = tm["X"]
        rate = sm.replica_rate(row.cell, host)
        vol = sm.volume_rate_s(sm.geoms[model].universe_bytes / (1 << 30))
        cost_solver = 1000.0 * (rate["total"] + vol) / X if X > 0 \
            else float("inf")
        tps_theory = float(tr["tps_pred"])
        cost_theory = float(tr["usd_per_1k_tok_total"])
        tag = "%s host=%g b=%d" % (cname, host, b)
        if tps_theory > 0:
            rows.append({
                "quantity": "tps_pred " + tag,
                "theory_value": tps_theory, "solver_value": X,
                "rel_delta": X / tps_theory - 1.0,
                "note": "serving.py batch=%d vs solver k_grid nearest %d; "
                        "solver adds batch-fillability + M/M/1 deployment "
                        "conventions at saturation" % (b, k_near)})
        if cost_theory > 0 and math.isfinite(cost_solver):
            rows.append({
                "quantity": "usd_per_1k_tok " + tag,
                "theory_value": cost_theory, "solver_value": cost_solver,
                "rel_delta": cost_solver / cost_theory - 1.0,
                "note": "includes volume amortisation + SOLVE_CORES=2 CPU "
                        "accounting vs serving.py's full cell cores"})
    # CPU accounting disagreement (documented in SOLVE_CORES note)
    for cname in sorted({r.cell.name for r in sm.table}):
        cell = sm.cell_by_name[cname]
        rows.append({
            "quantity": "cpu cores billed " + cname,
            "theory_value": cell.n_cpu_cores,
            "solver_value": float(SOLVE_CORES.value),
            "rel_delta": float(SOLVE_CORES.value) / max(1, cell.n_cpu_cores)
            - 1.0,
            "note": "serving.py bills the full cell core count per replica; "
                    "the solver bills a 2-core container floor (SOLVE_CORES, "
                    "ASSUMPTION 1-4 cores) -- dee cost delta"})
    return rows


def _find_alloc(sm: SolveModel, idx, model: str, cname: str,
                host: float) -> Optional[Tuple[str, str, float, float]]:
    best = None
    for key in idx:
        if key[0] != model or key[1] != cname:
            continue
        if abs(key[2] - host) > 1e-6:
            continue
        if best is None or key[3] > best[3]:
            best = key
    return best
