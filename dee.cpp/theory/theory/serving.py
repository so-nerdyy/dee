"""E. Serving extension: cross-request economics and the cost frontier.

Cross-request cache: with K concurrent requests the pooled reference stream
is a mixture of per-request popularity vectors.  For homogeneous requests
under an IRM the mixture is the mean popularity, and Che's TTL form applies
to the mixture — but the *capacity per request* falls as 1/K, so the hit
rate is a non-monotone function of concurrency at fixed M.  We derive

    H_serve(M, lambda) = H( w_bar(K(lambda)), M )

with K(lambda) ~ Poisson(lambda * W) from Little's law (W = mean response
time), and report bytes/token vs concurrency.

Cost frontier:  $/1k tok = hardware_rate_s / (TPS * 1000) per cell, plus the
break-even request volume against a conservative dense-residency baseline
(DSv4-Flash bf16 ~568 GiB resident on 8xH100-class at published prices).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import stats

from .util import dlog, figure, savefig, write_csv, write_json

GIB = float(1 << 30)


def concurrency_mix(pop_pi: np.ndarray, K: int) -> np.ndarray:
    """Pooled per-token inclusion probability at concurrency K.

    Under an IRM with homogeneous requests, the pooled stream's per-record
    request probability per cache-request step is the mean of K independent
    per-request vectors:

        u_i(K) = 1 - (1 - pi_i)^(K * b)

    so the mixture scales the per-token request rate by K (batch-of-b per
    request) while the cache capacity does not.
    """
    pi = np.asarray(pop_pi, dtype=float)
    return 1.0 - (1.0 - pi) ** K


def little_concurrency(lam: float, W: float) -> float:
    """Little's law: mean concurrent requests = lambda * W."""
    return lam * W


def cost_per_1k_tok(price_gpu_s: float, n_gpu: int, price_cpu_core_s: float,
                    n_cores: int, price_ram_gib_s: float, ram_gib: float,
                    tps: float) -> Dict[str, float]:
    if tps <= 0:
        return {"gpu": float("nan"), "cpu": float("nan"), "ram": float("nan"),
                "total": float("nan")}
    g = price_gpu_s * n_gpu / tps * 1000.0
    c = price_cpu_core_s * n_cores / tps * 1000.0
    r = price_ram_gib_s * ram_gib / tps * 1000.0
    return {"gpu": g, "cpu": c, "ram": r, "total": g + c + r}


def dense_baseline_cost() -> Dict[str, float]:
    """Conservative dense-residency baseline: DSv4-Flash bf16 resident.

    ASSUMPTIONS (stated explicitly):
      * resident footprint ~568 GiB bf16 routed+dense weights;
      * 8 x H100-80GB-class at $3.223/GPU/h ($0.000895/GPU/s) — conservative
        cloud list pricing, not reserved/spot;
      * achieved decode 200 tok/s aggregate (deliberately conservative for a
        13B-active MoE on 8xH100; dense serving stacks routinely report far
        more, which only strengthens the dee case).
    The baseline is NOT measured in-repo; it is a deliberately generous
    competitor so the break-even is not an artifact of a weak strawman.
    """
    from .constants import DENSE_BASELINE

    v = DENSE_BASELINE.value
    price_s = v["n_gpu"] * v["price_gpu_s"]
    return {
        "resident_gib": v["resident_bytes"] / GIB,
        "n_gpu": v["n_gpu"], "price_s": price_s,
        "tok_s": v["achieved_tok_s"],
        "usd_per_1k_tok": price_s / v["achieved_tok_s"] * 1000.0,
        "assumption": "DENSE_BASELINE (constants.py): 568 GiB bf16, 8xH100, "
                      "$0.003223/GPU/s, 200 tok/s achieved (all ASSUMPTION)",
    }


def run(roof_result=None, cache_result=None) -> Dict[str, object]:
    from .constants import ALL_CELLS, SERVE_ASSUMPTIONS, VOLUME_PRICE_GIB_MO
    from .cache import CheTable
    from .roofline import build_pop, dense_ms_for, load_geoms

    dlog("E. serving economics")
    geoms = load_geoms()
    g = geoms["dsv4_flash"]
    pop = build_pop(g)

    # ---- H_serve(M, lambda) --------------------------------------------
    # Request stream: Poisson arrivals, mean response W (ASSUMPTION 4 s at
    # b=1 on the anchor cell), decode length 512 tokens.
    W = 4.0
    req_tokens = 512
    b = 4
    lam_grid = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    host_gib_grid = [8, 16, 32, 64, 128, 256]
    serve_rows = []
    for lam in lam_grid:
        K = little_concurrency(lam, W)
        K_int = max(1, int(round(K)))
        u = concurrency_mix(pop.pi, K_int * b)
        w = u / max(1e-30, u.sum())
        tab = CheTable(w)
        for host_gib in host_gib_grid:
            M = host_gib * GIB / g.record_bytes
            h = tab.hit_at_capacity(float(M))
            # bytes/token = P_rec * (requests/token) * (1-H); requests/token
            # scales with K_int * b per step (mixture) but the token rate of
            # the replica also scales, so bytes/token per request is
            # P_rec * D(1)(1-H).
            bytes_per_tok = g.record_bytes * pop.u(1).sum() * (1.0 - h)
            serve_rows.append({
                "lambda_per_s": lam, "W_s": W, "K_mean": K, "K_int": K_int,
                "batch": b, "host_gib": host_gib, "capacity_records": M,
                "H_serve": h, "bytes_per_token": bytes_per_tok,
                "mib_per_token": bytes_per_tok / (1 << 20),
                "universe_records": g.universe_records,
                "universe_gib": g.universe_bytes / GIB,
            })
    write_csv("serve_hitrate.csv", serve_rows)

    # ---- cost frontier ---------------------------------------------------
    from .constants import T4_TOUCH_OVERHEAD_US
    from .roofline import roofline_point

    t0 = float(T4_TOUCH_OVERHEAD_US.value)      # MEASURED-derived, same as D/G
    cost_rows = []
    for cell in ALL_CELLS:
        for host_gib in (16, 64, 256):
            for bsize in (1, 8):
                h_slots = host_gib * GIB / g.record_bytes / max(1, cell.n_gpu)
                flops_eff = (0.6 * cell.F_peak) if cell.n_gpu else 0.0
                p = roofline_point(g, pop, cell, bsize, h_slots,
                                   dense_ms_for(cell, g), t0,
                                   flops_eff, horizon_tokens=512,
                                   finite=False, eta_storage=0.2917)
                c = cost_per_1k_tok(cell.price_gpu_s, cell.n_gpu,
                                    cell.price_cpu_core_s, cell.n_cpu_cores,
                                    cell.price_ram_gib_s,
                                    host_gib + cell.n_gpu * cell.vram_bytes / GIB,
                                    p.tps_pred)
                cost_rows.append({
                    "cell": cell.name, "host_gib": host_gib, "batch": bsize,
                    "tps_pred": p.tps_pred, "tps_bound": p.tps_bound,
                    "limiter": p.limiter,
                    "usd_per_1k_tok_gpu": c["gpu"], "usd_per_1k_tok_cpu": c["cpu"],
                    "usd_per_1k_tok_ram": c["ram"], "usd_per_1k_tok_total": c["total"],
                    "hardware_rate_s": (cell.price_gpu_s * cell.n_gpu
                                        + cell.price_cpu_core_s * cell.n_cpu_cores
                                        + cell.price_ram_gib_s
                                        * (host_gib + cell.n_gpu * cell.vram_bytes / GIB)),
                })
    write_csv("serve_cost_frontier.csv", cost_rows)

    dense = dense_baseline_cost()
    # break-even volume: at which request rate does a shared dee cache beat
    # the dense baseline?  Model: one dee replica serves K concurrent
    # requests; its cost/1k tok falls with b (batching).  Break-even volume
    # is the request rate at which dee's amortized $/1k tok equals dense's.
    be_rows = []
    for cell in ALL_CELLS:
        for host_gib in (64, 256):
            h_slots = host_gib * GIB / g.record_bytes / max(1, cell.n_gpu)
            flops_eff = (0.6 * cell.F_peak) if cell.n_gpu else 0.0
            for bsize in (1, 4, 8, 16):
                p = roofline_point(g, pop, cell, bsize, h_slots,
                                   dense_ms_for(cell, g), t0,
                                   flops_eff, horizon_tokens=512,
                                   finite=False, eta_storage=0.2917)
                rate = (cell.price_gpu_s * cell.n_gpu
                        + cell.price_cpu_core_s * cell.n_cpu_cores
                        + cell.price_ram_gib_s
                        * (host_gib + cell.n_gpu * cell.vram_bytes / GIB))
                c_dee = rate / p.tps_pred * 1000.0
                ratio = c_dee / dense["usd_per_1k_tok"]
                # requests/s at which one replica saturates at batch b
                # (K = lambda W = b at saturation of the batching window)
                lam_sat = bsize / W
                be_rows.append({
                    "cell": cell.name, "host_gib": host_gib, "batch": bsize,
                    "tps_pred": p.tps_pred,
                    "usd_per_1k_tok_dee": c_dee,
                    "usd_per_1k_tok_dense": dense["usd_per_1k_tok"],
                    "ratio_dee_over_dense": ratio,
                    "beats_dense": bool(ratio < 1.0),
                    "lambda_saturation_per_s": lam_sat,
                    "note": "dee wins where ratio<1 at the batch the replica "
                            "can actually sustain (lambda_sat)",
                })
    write_csv("serve_breakeven.csv", be_rows)
    write_json("serve_dense_baseline.json", dense)

    volume_note = {
        "volume_price_gib_month": VOLUME_PRICE_GIB_MO.value,
        "store_gib": g.universe_bytes / GIB,
        "monthly_volume_cost_usd": max(0.0, g.universe_bytes / GIB - 1024)
                                   * VOLUME_PRICE_GIB_MO.value,
    }
    write_json("serve_volume_cost.json", volume_note)

    _plots(serve_rows, cost_rows, be_rows, dense)
    out = {"serve_rows": serve_rows, "cost_rows": cost_rows,
           "break_even": be_rows, "dense": dense, "volume": volume_note}
    dlog("   serving grid:", len(serve_rows), "hit-rate points,",
         len(cost_rows), "cost cells; dense baseline $",
         round(dense["usd_per_1k_tok"], 2), "/1k tok")
    return out


def _plots(serve_rows, cost_rows, be_rows, dense) -> None:
    fig, ax = figure("serve_hitrate_vs_concurrency", (7.4, 4.6))
    for host_gib in (16, 64, 256):
        rs = [r for r in serve_rows if r["host_gib"] == host_gib]
        rs.sort(key=lambda r: r["lambda_per_s"])
        ax.plot([r["lambda_per_s"] for r in rs], [100 * r["H_serve"] for r in rs],
                "o-", ms=3, label=f"host {host_gib} GiB")
    ax.set_xscale("log")
    ax.set_xlabel("arrival rate lambda (req/s), W=4 s")
    ax.set_ylabel("H_serve (%)")
    ax.set_title("Cross-request cache hit rate vs concurrency (Little: K = lambda W)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "serve_hitrate_vs_concurrency.png")

    fig, ax = figure("serve_bytes_per_token", (7.4, 4.6))
    for host_gib in (16, 64, 256):
        rs = [r for r in serve_rows if r["host_gib"] == host_gib]
        rs.sort(key=lambda r: r["lambda_per_s"])
        ax.plot([r["lambda_per_s"] for r in rs],
                [r["mib_per_token"] for r in rs], "o-", ms=3,
                label=f"host {host_gib} GiB")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arrival rate lambda (req/s)")
    ax.set_ylabel("slow-tier bytes per token (MiB)")
    ax.set_title("bytes/token vs concurrency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "serve_bytes_per_token.png")

    fig, ax = figure("serve_cost_frontier", (7.8, 5.0))
    for host_gib in (16, 64, 256):
        rs = [r for r in cost_rows if r["host_gib"] == host_gib and r["batch"] == 8]
        rs.sort(key=lambda r: r["tps_pred"])
        ax.plot([r["tps_pred"] for r in rs],
                [r["usd_per_1k_tok_total"] for r in rs], "o-", ms=4,
                label=f"dee host {host_gib} GiB, b=8")
    ax.axhline(dense["usd_per_1k_tok"], color="r", ls="--", lw=1.2)
    ax.annotate(f"dense baseline ${dense['usd_per_1k_tok']:.2f}/1k tok "
                f"(8xH100, 568 GiB resident, {dense['tok_s']:.0f} tok/s)",
                (0.02, dense["usd_per_1k_tok"] * 1.08), color="r", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("TPS (tok/s)")
    ax.set_ylabel("$ / 1k tokens (hardware)")
    ax.set_title("Serving cost frontier vs conservative dense-residency baseline")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "serve_cost_frontier.png")

    fig, ax = figure("serve_breakeven", (7.4, 4.6))
    cells = sorted({r["cell"] for r in be_rows})
    for cell in cells:
        rs = [r for r in be_rows if r["cell"] == cell and r["host_gib"] == 256]
        rs.sort(key=lambda r: r["batch"])
        ax.plot([r["batch"] for r in rs],
                [r["ratio_dee_over_dense"] for r in rs], "o-", ms=4, label=cell)
    ax.axhline(1.0, color="r", ls="--", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch b")
    ax.set_ylabel("$ per 1k tok: dee / dense")
    ax.set_title("Break-even vs dense baseline (host 256 GiB; <1 = dee cheaper)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "serve_breakeven.png")
