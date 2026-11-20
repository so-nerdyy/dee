"""Deliverable 2 (break-even surface vs dense residency) + deliverable 4
(COMPUTE_EFF / t_0 sensitivity) of the provisioning solver.

The break-even surface states, over (arrival rate lambda x request length L)
per model and per dense-price reading, the cost ratio

    ratio(lambda, L) = $/1k tok (cheapest feasible dee deployment)
                       / $/1k tok (dense-residency baseline)

with the dee side solved by ``SolveModel.solve_deployment`` (constrained
optimisation over cell x host GiB x VRAM GiB x prefetch budget x replicas)
and the dense side from ``SolveModel.dense_by_reading`` (both H100 price
readings of the ambiguous DENSE_BASELINE constant -- see the Phase-7
price-ambiguity disclosure in THEORY.md 6).

Sensitivity re-solves the recommended deployment under each
(COMPUTE_EFF, t_0) scenario and reports how the recommendation and the
break-even ratio move.  Named methods only (enumeration + the closed forms
cited in ``model.py``).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .model import (DENSE_PRICE_READINGS, LEN_GRID, LAMBDA_GRID, SLO_GRID,
                    C, SolveModel)

# Surface grid: request-length subset (full LEN_GRID is 7 points; 5 keeps the
# 200-solve surface tractable while covering the 512-token economics
# convention point).  Registered for provenance.
SOLVE_BE_L_GRID = C(
    "solve_breakeven_L_grid", [64.0, 256.0, 512.0, 2048.0, 4096.0],
    "ASSUMPTION", "theory/solve/model.py LEN_GRID subset (surface runtime)",
    "request length (decode tokens) grid of the break-even surface; "
    "512 is the SERVE_ASSUMPTIONS economics convention", "tokens")

REF_LAM = 1.0        # reference traffic for sensitivity rows (req/s)
REF_L = 512.0        # reference request length (tokens)
REF_SLO = 20.0       # reference per-stream SLO (tok/s)


def _dense_cost(sm: SolveModel, model: str, reading: str) -> float:
    return sm.dense_by_reading[reading][model].cost_usd_per_1k_tok()


def build_breakeven(sm: SolveModel, idx):
    """(be_rows, contour_rows, be_summary) -- see module docstring."""
    be_rows: List[dict] = []
    contour_rows: List[dict] = []
    summary_rows: List[dict] = []
    lams = list(LAMBDA_GRID)
    for model in sorted(sm.geoms):
        # the dee deployment does not depend on the dense price reading:
        # solve once per (L, lam), then score both readings.
        solved: Dict[Tuple[float, float], Optional[dict]] = {}
        for L in SOLVE_BE_L_GRID.value:
            for lam in lams:
                solved[(L, lam)] = sm.solve_deployment(
                    idx, model, float(lam), float(L), None,
                    pf="off", eff=sm.compute_eff, t0_us=sm.t0_us)
        for reading in DENSE_PRICE_READINGS:
            dcost = _dense_cost(sm, model, reading)
            for L in SOLVE_BE_L_GRID.value:
                series: List[Tuple[float, float, float]] = []   # (lam, ratio, cost)
                for lam in lams:
                    sol = solved[(L, lam)]
                    if sol is None:
                        be_rows.append({
                            "model": model, "price_reading": reading,
                            "L_req": L, "lambda_req_s": lam,
                            "cell": "INFEASIBLE", "host_gib": "",
                            "vram_gib": "", "batch_policy": "",
                            "batch_realized": "", "replicas": "",
                            "X_replica_tps": "", "demand_tps": lam * L,
                            "dee_usd_per_1k_tok": "",
                            "dense_usd_per_1k_tok": dcost,
                            "cost_ratio_dee_over_dense": "",
                            "dee_wins": False, "limiter": "",
                            "rho": "", "per_stream_tps": ""})
                        continue
                    ratio = sol["usd_per_1k_tok"] / dcost
                    series.append((float(lam), ratio, sol["usd_per_1k_tok"]))
                    be_rows.append({
                        "model": model, "price_reading": reading,
                        "L_req": L, "lambda_req_s": lam,
                        "cell": sol["cell"], "host_gib": sol["host_gib"],
                        "vram_gib": sol["vram_gib"],
                        "batch_policy": sol["batch_policy"],
                        "batch_realized": sol["batch_realized"],
                        "replicas": sol["r"],
                        "X_replica_tps": sol["X_replica"],
                        "demand_tps": sol["demand_tps"],
                        "dee_usd_per_1k_tok": sol["usd_per_1k_tok"],
                        "dense_usd_per_1k_tok": dcost,
                        "cost_ratio_dee_over_dense": ratio,
                        "dee_wins": ratio < 1.0,
                        "limiter": sol["limiter"], "rho": sol["rho"],
                        "per_stream_tps": sol["per_stream_tps"]})
                # break-even contour: lam where ratio crosses 1.0
                cross = _contour_points(model, reading, L, series)
                contour_rows.extend(cross)
                wins = [s for s in series if s[1] < 1.0]
                summary_rows.append({
                    "model": model, "price_reading": reading, "L_req": L,
                    "max_lam_dee_wins": max((s[0] for s in wins), default=0.0),
                    "min_lam_dee_loses": min(
                        (s[0] for s in series if s[1] >= 1.0), default=""),
                    "ratio_at_ref_lam1": _ratio_at(series, REF_LAM),
                    "dee_wins_anywhere": bool(wins),
                    "verdict": ("dee wins for lambda <= %.2f req/s"
                                % max((s[0] for s in wins), default=0.0))
                    if wins else "dense wins at every tested lambda"})
    be_summary = {"rows": summary_rows,
                  "dense_price_readings": dict(DENSE_PRICE_READINGS),
                  "note": ("dee side = min-cost feasible deployment "
                           "(SolveModel.solve_deployment, rho<=0.7 + batch-"
                           "fillability); dense side = DENSE_BASELINE scaled "
                           "per SOLVE_DENSE_SCALE, both price readings")}
    return be_rows, contour_rows, be_summary


def _ratio_at(series, lam: float) -> Optional[float]:
    for l, r, _c in series:
        if abs(l - lam) < 1e-12:
            return r
    return None


def _contour_points(model: str, reading: str, L: float,
                    series: List[Tuple[float, float, float]]) -> List[dict]:
    out: List[dict] = []
    s = sorted(series)
    for (l0, r0, _c0), (l1, r1, _c1) in zip(s[:-1], s[1:]):
        if (r0 - 1.0) * (r1 - 1.0) < 0 or r0 == 1.0:
            # log-lam interpolation of the ratio=1 crossing
            if r1 != r0:
                t = (1.0 - r0) / (r1 - r0)
            else:
                t = 0.0
            lam_star = 10.0 ** (math.log10(l0) + t * (math.log10(l1)
                                                       - math.log10(l0)))
            out.append({"model": model, "price_reading": reading,
                        "L_req": L, "lam_lo": l0, "lam_hi": l1,
                        "ratio_lo": r0, "ratio_hi": r1,
                        "lambda_breakeven": lam_star,
                        "note": "ratio=1.0 crossing, log-lambda interpolation"})
    return out


# ==========================================================================
# Deliverable 4: COMPUTE_EFF / t_0 sensitivity (+ dense-price reading)
# ==========================================================================
def build_sensitivity(sm: SolveModel, idx, SCENARIOS):
    sens_rows: List[dict] = []
    flips: List[dict] = []
    for name, eff, t0_us in SCENARIOS:
        for model in sorted(sm.geoms):
            sol = sm.solve_deployment(
                idx, model, REF_LAM, REF_L, REF_SLO, pf="off",
                eff=float(eff), t0_us=float(t0_us))
            for reading in DENSE_PRICE_READINGS:
                dcost = _dense_cost(sm, model, reading)
                if sol is None:
                    sens_rows.append({
                        "scenario": name, "compute_eff": eff, "t0_us": t0_us,
                        "model": model, "price_reading": reading,
                        "cell": "INFEASIBLE", "host_gib": "", "vram_gib": "",
                        "batch_realized": "", "replicas": "",
                        "X_replica_tps": "", "usd_per_1k_tok": "",
                        "dense_usd_per_1k_tok": dcost,
                        "ratio_at_ref_traffic": "", "dee_wins": False})
                    flips.append({"scenario": name, "model": model,
                                  "price_reading": reading,
                                  "flip": "INFEASIBLE at "
                                          "lambda=1 req/s, L=512, SLO=20"})
                    continue
                ratio = sol["usd_per_1k_tok"] / dcost
                sens_rows.append({
                    "scenario": name, "compute_eff": eff, "t0_us": t0_us,
                    "model": model, "price_reading": reading,
                    "cell": sol["cell"], "host_gib": sol["host_gib"],
                    "vram_gib": sol["vram_gib"],
                    "batch_realized": sol["batch_realized"],
                    "replicas": sol["r"], "X_replica_tps": sol["X_replica"],
                    "usd_per_1k_tok": sol["usd_per_1k_tok"],
                    "dense_usd_per_1k_tok": dcost,
                    "ratio_at_ref_traffic": ratio,
                    "dee_wins": ratio < 1.0})
    # headline flips: win/lose transitions across scenarios per (model, reading)
    by_key: Dict[Tuple[str, str], List[dict]] = {}
    for r in sens_rows:
        by_key.setdefault((r["model"], r["price_reading"]), []).append(r)
    for (model, reading), rs in sorted(by_key.items()):
        states = {bool(r["dee_wins"]) for r in rs}
        if len(states) > 1:
            flips.append({
                "scenario": "ALL", "model": model, "price_reading": reading,
                "flip": "verdict FLIPS across scenarios: "
                        + ", ".join("%s->%s" % (r["scenario"],
                                                "win" if r["dee_wins"]
                                                else "lose") for r in rs)})
    sens_summary = {
        "reference_point": {"lambda_req_s": REF_LAM, "L_req": REF_L,
                            "slo_tok_s": REF_SLO},
        "flips": flips,
        "n_rows": len(sens_rows),
        "note": ("each scenario re-solves the min-cost deployment under "
                 "(COMPUTE_EFF, t_0) and scores it against the dense "
                 "baseline under BOTH price readings")}
    return sens_rows, sens_summary


# ==========================================================================
# Figures
# ==========================================================================
def plot_all(sm: SolveModel, pareto_rows, knee_rows, be_rows, contour_rows,
             sens_rows, be_summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..util import figure, savefig

    models = sorted(sm.geoms)

    # ---- fig 1: pareto frontier per model (computed price reading) --------
    ncol = 2
    nrow = int(math.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 7.2),
                             squeeze=False)
    for ax, m in zip(axes.flat, models):
        rs = [r for r in pareto_rows if r["model"] == m]
        dom = [r for r in rs if not r["pareto"]]
        par = [r for r in rs if r["pareto"]]
        if dom:
            ax.scatter([r["tps_capacity"] for r in dom],
                       [r["usd_per_1k_tok"] for r in dom], s=6, c="#bbbbbb",
                       alpha=0.5, label="dominated")
        if par:
            pts = sorted(par, key=lambda r: r["tps_capacity"])
            ax.plot([r["tps_capacity"] for r in pts],
                    [r["usd_per_1k_tok"] for r in pts], ".-", ms=4,
                    color="#1f77b4", label="Pareto (hints off)")
        for k in knee_rows:
            if k.get("model") == m and k.get("tps_capacity"):
                ax.plot([k["tps_capacity"]], [k["usd_per_1k_tok"]], "r*",
                        ms=14, label="knee")
        d_hi = _dense_cost(sm, m, "computed_0.003223")
        d_lo = _dense_cost(sm, m, "cheap_0.000895")
        ax.axhline(d_hi, color="r", ls="--", lw=1.0,
                   label="dense $%.3f/1k (computed price)" % d_hi)
        ax.axhline(d_lo, color="g", ls=":", lw=1.0,
                   label="dense $%.3f/1k (cheap price)" % d_lo)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(m)
        ax.set_xlabel("throughput tok/s (per replica)")
        ax.set_ylabel("$ / 1k tok")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, "solve_pareto")

    # ---- fig 2: break-even ratio surface (computed reading) ---------------
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 7.2),
                             squeeze=False)
    reading = "computed_0.003223"
    for ax, m in zip(axes.flat, models):
        rows = [r for r in be_rows if r["model"] == m
                and r["price_reading"] == reading
                and r["cell"] != "INFEASIBLE"]
        lams = sorted({r["lambda_req_s"] for r in rows})
        Ls = sorted({r["L_req"] for r in rows})
        grid = [[next((r["cost_ratio_dee_over_dense"] for r in rows
                       if r["L_req"] == L and r["lambda_req_s"] == lam),
                      float("nan"))
                 for lam in lams] for L in Ls]
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       extent=[math.log10(min(lams)), math.log10(max(lams)),
                               0, len(Ls) - 1],
                       cmap="RdYlGn_r", vmin=0.2, vmax=3.0)
        ax.set_yticks(range(len(Ls)))
        ax.set_yticklabels(["%g" % L for L in Ls], fontsize=7)
        ax.set_xticks([math.log10(l) for l in lams])
        ax.set_xticklabels(["%g" % l for l in lams], fontsize=6,
                           rotation=45)
        cs = ax.contour(grid, levels=[1.0], colors="k", linewidths=1.2)
        ax.clabel(cs, fmt="break-even", fontsize=6)
        ax.set_title("%s (computed price)" % m, fontsize=9)
        ax.set_xlabel("lambda req/s (log)")
        ax.set_ylabel("L tokens")
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.suptitle("dee/dense cost ratio (green = dee wins); contour = 1.0; "
                 "cheap-price reading is 3.57x worse for dee", fontsize=9)
    fig.tight_layout()
    savefig(fig, "solve_breakeven")

    # ---- fig 3: sensitivity ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    scen_names: List[str] = []
    for r in sens_rows:
        if r["scenario"] not in scen_names:
            scen_names.append(r["scenario"])
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        vals = []
        for s in scen_names:
            row = next((r for r in sens_rows if r["model"] == m
                        and r["scenario"] == s
                        and r["price_reading"] == "computed_0.003223"), None)
            vals.append(row["ratio_at_ref_traffic"] if row
                        and row["ratio_at_ref_traffic"] != "" else float("nan"))
        xs = [j + i * width for j in range(len(scen_names))]
        ax.bar(xs, vals, width=width, label=m)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xticks([j + 0.4 for j in range(len(scen_names))])
    ax.set_xticklabels(scen_names, fontsize=7, rotation=20)
    ax.set_ylabel("dee/dense $ per 1k tok (computed price)")
    ax.set_title("break-even ratio at lambda=1 req/s, L=512, SLO=20 vs "
                 "(COMPUTE_EFF, t_0) scenario")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    savefig(fig, "solve_sensitivity")
