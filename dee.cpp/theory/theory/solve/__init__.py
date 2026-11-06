"""Phase-7 component C: the dee-serve provisioning solver.

``run(results_D, results_B)`` regenerates every solve artifact:
``data/solve_pareto.csv``      Pareto frontier ($/1k tok vs throughput)
``data/solve_knees.json``      knee allocations + full provisioning line items
``data/solve_breakeven_surface.csv``  cost-ratio surface vs the dense baseline
``data/solve_breakeven_contour.csv``  break-even (ratio=1.0) contour points
``data/solve_provisioning.csv`` "for X traffic, provision Y"
``data/solve_sensitivity.csv``  COMPUTE_EFF / t_0 sensitivity rows
``data/solve_disagreement.csv``  vs THEORY.md 6 / data/serve_cost_frontier.csv
``figs/solve_pareto.png`` ``figs/solve_breakeven.png`` ``figs/solve_sensitivity.png``
``solve/SOLVER.md``            the report

Deterministic (no RNG).  ``results_D``/``results_B`` may be None (standalone
stage run); the module then reuses the roofline stage's own calibrated
artifact and recomputes everything else through ``theory.roofline`` /
``theory.cache`` internals.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ..util import dlog, write_csv, write_json
from .model import (DENSE_PRICE_READINGS, PF_ARMS, AllocRow, R_MAX, SolveModel,
                    point_timing)

from .pareto import build_pareto
from .surfaces import build_breakeven, build_sensitivity, plot_all
from .report import write_solver_md
from .provisioning import build_provisioning, build_disagreement

MODELS = ["dsv4_flash", "mimo_v2_flash", "mimo_v2_pro", "minimax_m3"]
SCENARIOS = [("eff06_t387", 0.6, 387.0), ("eff03_t50", 0.3, 50.0),
             ("eff08_t50", 0.8, 50.0), ("eff03_t400", 0.3, 400.0),
             ("eff08_t400", 0.8, 400.0)]


def run(results_D=None, results_B=None) -> Dict[str, object]:
    import time

    t0 = time.time()
    dlog("J. provisioning solver (constrained optimisation)")
    sm = SolveModel(results_D, results_B)
    table = sm.build_table()
    idx = sm.alloc_index()
    dlog("   enumerated", len(table), "allocation rows (",
         len(sm.cells), "cells x", len(sm.host_grid), "host GiB x",
         len(sm.vfrac_grid), "VRAM fracs x", len(sm.k_grid), "batch policies x",
         len(sm.geoms), "stores )")

    pareto_rows, knee_rows = build_pareto(sm, idx)
    write_csv("solve_pareto.csv", pareto_rows)
    write_json("solve_knees.json", {"knees": knee_rows,
                                    "knee_method": "max perpendicular "
                                    "distance from chord in (log TPS, "
                                    "log $/1k tok) space",
                                    "scenario": "eff06_t387",
                                    "prefetch": "off"})

    be_rows, contour_rows, be_summary = build_breakeven(sm, idx)
    write_csv("solve_breakeven_surface.csv", be_rows)
    write_csv("solve_breakeven_contour.csv", contour_rows)

    prov_rows, prov_infeasible = build_provisioning(sm, idx)
    write_csv("solve_provisioning.csv", prov_rows)
    write_csv("solve_provisioning_infeasible.csv", prov_infeasible)

    sens_rows, sens_summary = build_sensitivity(sm, idx, SCENARIOS)
    write_csv("solve_sensitivity.csv", sens_rows)

    disag_rows = build_disagreement(sm, idx)
    write_csv("solve_disagreement.csv", disag_rows)

    plot_all(sm, pareto_rows, knee_rows, be_rows, contour_rows, sens_rows,
             be_summary)

    out = {"pareto": pareto_rows, "knees": knee_rows,
           "breakeven": be_rows, "contour": contour_rows,
           "breakeven_summary": be_summary,
           "provisioning": prov_rows, "provisioning_infeasible": prov_infeasible,
           "sensitivity": sens_rows, "sensitivity_summary": sens_summary,
           "disagreement": disag_rows, "model": sm}
    write_solver_md(out, sm)
    dlog("   solver stage done in %.1fs; %d pareto rows, %d provisioning rows,"
         " %d disagreement rows" % (time.time() - t0, len(pareto_rows),
                                    len(prov_rows), len(disag_rows)))
    return out
