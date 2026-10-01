"""Deliverable 1: cost-throughput Pareto frontier per model + knee allocation.

A frontier point is a per-replica allocation archetype
(cell, host GiB, VRAM GiB, prefetch budget, batch policy b) at SATURATION:

    throughput   X(b)          (tok/s, overlap-corrected roofline, THEORY.md 5.1)
    $/1k tok     1000 * (replica_rate_s + volume_rate_s) / X(b)

Replicas scale linearly (r copies cost r copies and deliver r·X); the
volume share therefore amortizes as 1/r and the frontier is stated per
replica at r=1 (the conservative volume charge).  Queueing headroom
(rho <= SOLVE_RHO_MAX) is a deployment constraint and is enforced in
deliverable 3 (``solve_provisioning.csv``), not in this capacity curve.

Dominance: point (X1, c1) is dominated when some point has X2 >= X1 and
c2 <= c1 with one strict.  Both prefetch arms are enumerated so the hint
lever stays visible; dominated arms are flagged, never dropped silently.
The "knee" is the maximum perpendicular distance from the chord between
the frontier endpoints in (log10 X, log10 $/1k tok) space (named
kneedle-style heuristic).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .model import AllocRow, PF_ARMS, SolveModel, point_timing


def _candidates(sm: SolveModel, idx) -> List[dict]:
    out = []
    for (mname, cname, host_gib, vram_gib), by_k in idx.items():
        cell = by_k[min(by_k)].cell
        rate = sm.replica_rate(cell, host_gib)
        vol = sm.volume_rate_s(sm.geoms[mname].universe_bytes / (1 << 30))
        for b in sorted(by_k):
            row = by_k[b]
            for pf in PF_ARMS:
                tm = point_timing(row, sm.compute_eff, sm.t0_us, pf)
                X = tm["X"]
                if X <= 0 or not math.isfinite(X):
                    continue
                cost = 1000.0 * (rate["total"] + vol) / X
                out.append({
                    "model": mname, "cell": cname,
                    "host_gib": host_gib, "vram_gib": vram_gib,
                    "prefetch": pf, "batch": b,
                    "tps_capacity": X,
                    "usd_per_1k_tok": cost,
                    "usd_per_1k_gpu": 1000.0 * rate["gpu"] / X,
                    "usd_per_1k_cpu": 1000.0 * rate["cpu"] / X,
                    "usd_per_1k_ram": 1000.0 * rate["ram"] / X,
                    "usd_per_1k_volume": 1000.0 * vol / X,
                    "replica_rate_s": rate["total"],
                    "volume_rate_s": vol,
                    "limiter": tm["limiter"],
                    "t_pred": tm["t_pred"], "t_compute": tm["t_compute"],
                    "t_demand": tm["t_demand"], "t_h2d": row.t_h2d,
                    "t_dense": row.t_dense, "t_pf": tm["t_pf"],
                    "H_v": row.H_v, "H_h": row.H_h,
                    "cold_recs_per_tok": row.cold_recs,
                    "cold_mib_per_tok": row.cold_recs
                    * sm.geoms[mname].record_bytes / (1 << 20),
                    "spill_mib_per_tok": row.spill_recs
                    * sm.geoms[mname].record_bytes / (1 << 20),
                    "D_per_tok": row.D, "miss_through": row.miss_through,
                    "is_purchasable": cname != "2xT4",
                    "pareto": False, "dominated_by": "", "is_knee": False,
                })
    return out


def _mark_pareto(rows: List[dict]) -> None:
    by_model: Dict[str, List[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for m, rs in by_model.items():
        for i, a in enumerate(rs):
            dominator = None
            for j, b in enumerate(rs):
                if i == j:
                    continue
                if (b["tps_capacity"] >= a["tps_capacity"]
                        and b["usd_per_1k_tok"] <= a["usd_per_1k_tok"]
                        and (b["tps_capacity"] > a["tps_capacity"]
                             or b["usd_per_1k_tok"] < a["usd_per_1k_tok"])):
                    if (dominator is None
                            or (b["usd_per_1k_tok"], -b["tps_capacity"])
                            < (dominator["usd_per_1k_tok"],
                               -dominator["tps_capacity"])):
                        dominator = b
            if dominator is None:
                a["pareto"] = True
            else:
                a["dominated_by"] = "%s|host=%.0f|vram=%.1f|pf=%s|b=%d" % (
                    dominator["cell"], dominator["host_gib"],
                    dominator["vram_gib"], dominator["prefetch"],
                    dominator["batch"])


def _knee_of(points: List[dict]) -> Optional[dict]:
    """Kneedle-style knee: max perpendicular distance from the chord
    between the frontier endpoints in (log TPS, log $/1k) space."""
    pts = sorted(points, key=lambda r: r["tps_capacity"])
    if len(pts) < 3:
        return pts[0] if pts else None
    x0 = math.log10(pts[0]["tps_capacity"])
    y0 = math.log10(pts[0]["usd_per_1k_tok"])
    x1 = math.log10(pts[-1]["tps_capacity"])
    y1 = math.log10(pts[-1]["usd_per_1k_tok"])
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1.0
    best, best_d = None, -1.0
    for p in pts:
        x = math.log10(p["tps_capacity"])
        y = math.log10(p["usd_per_1k_tok"])
        d = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / norm
        if d > best_d:
            best, best_d = p, d
    return best


def build_pareto(sm: SolveModel, idx) -> Tuple[List[dict], List[dict]]:
    rows = _candidates(sm, idx)
    _mark_pareto(rows)
    rows.sort(key=lambda r: (r["model"], r["tps_capacity"],
                             r["usd_per_1k_tok"], r["prefetch"]))

    knees = []
    models = sorted({r["model"] for r in rows})
    best_purch = None
    for m in models:
        front = [r for r in rows
                 if r["model"] == m and r["pareto"] and r["prefetch"] == "off"]
        knee = _knee_of(front)
        if knee is not None:
            knee["is_knee"] = True
        # prefetch-arm penalty at the knee allocation (lever visibility)
        pf_gap = None
        if knee is not None:
            twin = next((r for r in rows
                         if r["model"] == m and r["prefetch"] == "m1"
                         and r["cell"] == knee["cell"]
                         and r["host_gib"] == knee["host_gib"]
                         and abs(r["vram_gib"] - knee["vram_gib"]) < 1e-9
                         and r["batch"] == knee["batch"]), None)
            if twin is not None:
                pf_gap = {
                    "m1_tps_capacity": twin["tps_capacity"],
                    "m1_usd_per_1k_tok": twin["usd_per_1k_tok"],
                    "tps_ratio_m1_over_off": (twin["tps_capacity"]
                                              / knee["tps_capacity"]),
                    "cost_ratio_m1_over_off": (twin["usd_per_1k_tok"]
                                               / knee["usd_per_1k_tok"]),
                    "t_pf_s_per_tok": twin["t_pf"],
                }
        dense = sm.dense[m]
        knee_entry = dict(knee) if knee else {}
        knee_entry.update({
            "model": m,
            "dense_cost_usd_per_1k_tok": dense.cost_usd_per_1k_tok(),
            "dense_cost_cheap_reading": sm.dense_by_reading[
                "cheap_0.000895"][m].cost_usd_per_1k_tok(),
            "ratio_knee_over_dense": (
                knee["usd_per_1k_tok"] / dense.cost_usd_per_1k_tok()
                if knee else None),
            "prefetch_arm_penalty": pf_gap,
            "provisioning_line_item": {
                "gpu_rate_s": sm.replica_rate(
                    sm.cell_by_name[knee["cell"]], knee["host_gib"])["gpu"],
                "cpu_rate_s": sm.replica_rate(
                    sm.cell_by_name[knee["cell"]], knee["host_gib"])["cpu"],
                "ram_rate_s": sm.replica_rate(
                    sm.cell_by_name[knee["cell"]], knee["host_gib"])["ram"],
                "volume_rate_s": knee["volume_rate_s"],
                "replica_rate_s": knee["replica_rate_s"],
            } if knee else {},
        })
        knees.append(knee_entry)
        if knee is not None and knee["is_purchasable"]:
            if (best_purch is None
                    or knee["usd_per_1k_tok"] < best_purch["usd_per_1k_tok"]):
                best_purch = knee

    # 2xT4 price-table artifact: the per-GPU $/s at which the T4 knee would
    # match the best purchasable knee's $/1k tok (T4 price is 0 in the table).
    for k in knees:
        if k.get("cell") == "2xT4" and best_purch and k.get("tps_capacity"):
            cell = sm.cell_by_name["2xT4"]
            rate = sm.replica_rate(cell, k["host_gib"])
            target = best_purch["usd_per_1k_tok"]
            need_rate = target * k["tps_capacity"] / 1000.0
            head = need_rate - rate["cpu"] - rate["ram"] - k["volume_rate_s"]
            k["t4_breakeven_price_gpu_s"] = max(0.0, head / max(1, cell.n_gpu))
            k["t4_breakeven_note"] = (
                "per-GPU $/s at which 2xT4 matches the best purchasable "
                "knee ($%.4f/1k tok, %s); the price table's T4 rate is 0.0 "
                "(free-tier artifact)" % (target, best_purch["cell"]))
            k["best_purchasable_knee_cell"] = best_purch["cell"]
    return rows, knees
