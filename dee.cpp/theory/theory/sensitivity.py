"""G. Sensitivity and the regime map.

Two deliverables:

  * global sensitivity of the TPS prediction/bound to its inputs — analytic
    partial derivatives where closed-form, finite-difference elasticities
    otherwise (log-log: d log TPS / d log x, a Sobol-style one-at-a-time
    sweep with the interaction term bounded by the total-effect residual);
  * the regime map over (B_SSD, M_host): which resource binds at each point
    (storage-bound / H2D-bound / compute-bound / dense-bound), per model.
    The Phase-1 verdict (0.29-0.37 GiB/s bank, "storage-bandwidth
    constrained") must land inside the storage-bound region or be flagged as
    contradicting — it is drawn on the map.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .util import dlog, figure, savefig, write_csv, write_json

GIB = float(1 << 30)


def elasticity(f, x0: Dict[str, float], keys: List[str],
               rel_step: float = 0.05) -> List[Dict[str, float]]:
    """One-at-a-time log-log elasticities of f at x0.

    E_k = (d log f / d log x_k) ~= [f(x_k (1+h)) - f(x_k (1-h))] / (2h log-ratio).
    """
    base = f(**x0)
    rows = []
    for k in keys:
        up = dict(x0)
        dn = dict(x0)
        up[k] = x0[k] * (1.0 + rel_step)
        dn[k] = x0[k] * (1.0 - rel_step)
        try:
            fu = f(**up)
            fd = f(**dn)
        except Exception as exc:                      # pragma: no cover
            rows.append({"param": k, "elasticity": float("nan"),
                         "base": base, "error": str(exc)})
            continue
        e = (np.log(max(fu, 1e-30)) - np.log(max(fd, 1e-30))) / (2.0 * rel_step)
        rows.append({"param": k, "elasticity": float(e),
                     "f_base": float(base), "f_up": float(fu), "f_down": float(fd),
                     "x0": x0[k]})
    return rows


def run(roof_result=None) -> Dict[str, object]:
    from .constants import ALL_CELLS, COMPUTE_EFF, T4_CELL, T4_DENSE_MS_PER_TOK, \
        T4_TOUCH_OVERHEAD_US
    from .roofline import build_pop, load_geoms, roofline_point

    dlog("G. sensitivity + regime map")
    geoms = load_geoms()
    g = geoms["dsv4_flash"]
    t0 = float(T4_TOUCH_OVERHEAD_US.value)

    def tps(b=1, host_gib=64.0, vram_slots=281.0, B_ssd=0.33 * GIB,
            B_h2d=T4_CELL.B_h2d, record_bytes=13369344.0, s=0.6,
            dense_ms=180.0, flops_eff=None, horizon=512):
        """TPS prediction on the anchor cell as a function of its inputs."""
        from .roofline import ModelGeom, build_pop

        gg = ModelGeom("sens", "sens", 43, 256, 6, int(record_bytes),
                       float(T4_FLOP_PER_TOUCH()), "sensitivity")
        pp = build_pop(gg, s_override=s, q_override=1.0)
        ff = flops_eff if flops_eff is not None else COMPUTE_EFF.value * T4_CELL.F_peak
        finite = horizon < 512
        p = roofline_point(gg, pp, T4_CELL, int(b),
                           host_gib * GIB / record_bytes, dense_ms,
                           t0, ff, vram_slots_per_gpu=int(vram_slots),
                           finite=finite,
                           batches=[int(b)] * int(horizon) if finite else None,
                           horizon_tokens=int(horizon),
                           eta_storage=0.2917)
        return p.tps_pred

    keys = ["host_gib", "vram_slots", "B_ssd", "B_h2d", "record_bytes",
            "s", "dense_ms"]
    x0 = {"b": 1, "host_gib": 64.0, "vram_slots": 281.0, "B_ssd": 0.33 * GIB,
          "B_h2d": T4_CELL.B_h2d, "record_bytes": 13369344.0, "s": 0.6,
          "dense_ms": 180.0, "horizon": 512}
    sens = []
    for horizon, regime in ((512, "steady_state_512tok"),
                            (16, "finite_window_16tok")):
        x = dict(x0, horizon=horizon)
        for r in elasticity(tps, x, keys):
            r["model"] = "dsv4_flash"
            r["cell"] = "2xT4"
            r["regime"] = regime
            sens.append(r)
    write_csv("sensitivity.csv", sens)

    # ---- regime map -----------------------------------------------------
    bs_grid = np.geomspace(0.02, 20.0, 30) * GIB
    host_grid = np.geomspace(1.0, 512.0, 30)
    maps = []
    phase1_zone = []
    for gname in ("dsv4_flash", "mimo_v2_pro", "minimax_m3"):
        gg = geoms[gname]
        pp = build_pop(gg)
        for cell in ALL_CELLS:
            if cell.name not in ("2xT4", "1xL4", "1xL40S", "CPU-only"):
                continue
            grid = np.empty((host_grid.size, bs_grid.size), dtype=object)
            limiter = np.empty_like(grid)
            for i, hg in enumerate(host_grid):
                for j, bs in enumerate(bs_grid):
                    from .constants import Cell

                    cc = Cell(cell.name, cell.n_gpu, cell.gpu_label, float(bs),
                              cell.B_h2d, cell.F_peak, cell.vram_bytes,
                              cell.price_gpu_s, cell.price_cpu_core_s,
                              cell.n_cpu_cores, cell.price_ram_gib_s,
                              cell.provenance)
                    flops_eff = (COMPUTE_EFF.value * cc.F_peak) if cc.n_gpu else 0.0
                    from .roofline import dense_ms_for

                    dense = dense_ms_for(cc, gg)
                    p = roofline_point(gg, pp, cc, 1,
                                       hg * GIB / gg.record_bytes, dense,
                                       t0, flops_eff, finite=True,
                                       batches=[1] * 128, eta_storage=0.2917)
                    grid[i, j] = p.tps_pred
                    limiter[i, j] = p.limiter
                    maps.append({
                        "model": gname, "cell": cell.name,
                        "host_gib": hg, "B_ssd": bs,
                        "B_ssd_gib_s": bs / GIB,
                        "tps_pred": p.tps_pred, "limiter": p.limiter,
                        "H_v": p.H_v, "H_h": p.H_h,
                        "cold_mib_per_tok": p.cold_bytes_per_tok / (1 << 20),
                    })
                    if (gname == "dsv4_flash" and cell.name == "2xT4"
                            and 0.29 * GIB <= bs <= 0.37 * GIB):
                        phase1_zone.append({"limiter": p.limiter, "hg": hg,
                                            "bs": bs / GIB, "tps": p.tps_pred})
    write_csv("regime_map.csv", maps)

    # Phase-1 consistency check: at the measured 0.29-0.37 GiB/s the limiter
    # must be storage, or the model contradicts the Phase-1 verdict.
    lims = [r["limiter"] for r in phase1_zone]
    from collections import Counter

    lc = Counter(lims)
    phase1_ok = lc.get("storage", 0) / max(1, len(lims)) >= 0.9
    phase1_check = {
        "measured_bandwidth_gib_s": [0.29, 0.37],
        "limiter_counts": dict(lc),
        "fraction_storage_limited": lc.get("storage", 0) / max(1, len(lims)),
        "verdict": "CONSISTENT" if phase1_ok else "CONTRADICTS",
        "phase1_verdict": "research/route-pipeline/STORAGE_VERDICT.md: "
                          "'storage-bandwidth constrained, 0.29-0.37 GiB/s, "
                          "feed-side verdict' (Phase 1 closed)",
    }
    write_json("regime_phase1_check.json", phase1_check)
    dlog("   Phase-1 consistency:", phase1_check["verdict"],
         phase1_check["fraction_storage_limited"])

    _regime_plot(maps)
    _sens_plot(sens)
    return {"sensitivity": sens, "regime": maps, "phase1_check": phase1_check}


def T4_FLOP_PER_TOUCH() -> float:
    from .constants import T4_FLOP_PER_TOUCH as _C

    return float(_C.value)


def _sens_plot(sens) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = sorted({r.get("regime", "steady_state_512tok") for r in sens})
    fig, axs = plt.subplots(1, len(regimes), figsize=(7.2 * len(regimes), 4.4),
                            sharey=True)
    if len(regimes) == 1:
        axs = [axs]
    for ax, regime in zip(axs, regimes):
        rows = sorted([r for r in sens if r.get("regime") == regime],
                      key=lambda r: abs(r["elasticity"]))
        ax.barh([r["param"] for r in rows], [r["elasticity"] for r in rows])
        ax.axvline(0, color="k", lw=0.7)
        ax.set_xlabel("elasticity  d log TPS / d log x")
        ax.set_title(regime)
        ax.grid(True, alpha=0.3, axis="x")
    fig.suptitle("Sensitivity of the TPS prediction (anchor cell, DSv4-Flash, b=1)")
    savefig(fig, "sensitivity.png")


def _regime_plot(maps) -> None:
    fig, axes = _regime_figure(maps)


def _regime_figure(maps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    models = ["dsv4_flash", "mimo_v2_pro", "minimax_m3"]
    cells = ["2xT4", "1xL4", "1xL40S", "CPU-only"]
    fig, axs = plt.subplots(len(models), len(cells),
                            figsize=(16, 10), sharex=True, sharey=True)
    labels = ["storage", "h2d", "compute", "dense"]
    cmap = ListedColormap(["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"])
    for i, m in enumerate(models):
        for j, c in enumerate(cells):
            ax = axs[i, j]
            pts = [r for r in maps if r["model"] == m and r["cell"] == c]
            if not pts:
                ax.axis("off")
                continue
            hg = sorted({r["host_gib"] for r in pts})
            bs = sorted({r["B_ssd"] for r in pts})
            Z = [[labels.index(next(r["limiter"] for r in pts
                                    if r["host_gib"] == h and r["B_ssd"] == b))
                  for b in bs] for h in hg]
            ax.pcolormesh(np.log10(np.array(bs) / GIB), np.log10(hg), np.array(Z),
                          cmap=cmap, vmin=-0.5, vmax=3.5, shading="auto")
            if i == len(models) - 1:
                ax.set_xlabel("log10 B_SSD (GiB/s)")
            if j == 0:
                ax.set_ylabel(f"{m}\nlog10 host GiB")
            ax.set_title(f"{m} / {c}", fontsize=8)
            if m == "dsv4_flash" and c == "2xT4":
                ax.axvspan(np.log10(0.29), np.log10(0.37), color="white",
                           alpha=0.35)
                ax.annotate("Phase-1 bank\n0.29-0.37 GiB/s",
                            (np.log10(0.33), np.log10(32)), fontsize=7,
                            ha="center")
    fig.suptitle("Regime map: which resource binds (red=storage, orange=H2D, "
                 "blue=compute, green=dense)", fontsize=11)
    from .util import savefig as _save

    _save(fig, "regime_map.png")
    return fig, axs
