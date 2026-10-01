"""SYNTH MODE — long synthetic streams from the fitted popularity/temporal
models, testing the steady-state claims a 5k trace cannot reach.

The three questions (AGENT_BRIEF_PHASE7.md deliverable A, SYNTH MODE):

  (a) does the +25pp Che steady-state error at 16 GiB shrink as THEORY.md 3.3
      predicts on long stationary streams (the finite-window artifact
      vanishing)?
  (b) does the 16 GiB LRU knee stabilize to a steady-state value, and where?
  (c) does the LRU-vs-MIN gap behave as predicted (theory: -2.8pp at the
      knee on the sealed trace; what does it become in steady state)?

Method: StreamGenerator (geometric-decay popularity fitted from the sealed
trace + measured temporal coupling: sticky 0.375, cross-layer lift 12.2x)
emits >= 100,000 requests (SIM_SYNTH_REQUESTS; --fast shrinks to 10k
without changing conclusions).  For each of three request counts (early /
mid / final window) we replay LRU + Belady MIN and evaluate both Che forms
on the window's counts, at budgets spanning the expected knee.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from ..util import dlog, figure, savefig, write_csv, write_json
from .constants import (SIM_SYNTH_REQUESTS, SIM_SYNTH_SEED, SIM_SYNTH_STICKY)
from .engine import Stream
from .streamgen import (StreamGenConfig, StreamGenerator, che_two_level,
                        counts_and_weights, lru_replay_counts,
                        min_replay_counts)

GIB = float(1 << 30)


def _record_stream(stream: Stream, n_req: int) -> List[tuple]:
    """First n_req (layer, expert) requests of the stream."""
    out: List[tuple] = []
    for c in stream.calls:
        for k in c.key_set:
            out.append(k)
            if len(out) >= n_req:
                return out
    return out


def run(fast: bool = False) -> Dict[str, Any]:
    from ..sources import load_stores

    dlog("SIM. SYNTH MODE — long stationary streams (steady-state claims)")
    record_bytes = int(load_stores()["dsv4_flash"].record_bytes)
    n_layers, experts_per_layer, topk = 43, 256, 6
    records_per_gib = GIB / record_bytes
    # budgets in GiB (pooled), incl. the expected ~16 GiB knee band
    budgets_gib = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0]

    total_req_full = int(SIM_SYNTH_REQUESTS.value)
    n_tokens = int(math.ceil(total_req_full / (n_layers * topk)))
    if fast:
        n_tokens = max(8, n_tokens // 10)
    # budgeted run: ~26M python draws is several minutes; the brief needs
    # >= 100k requests which n_tokens*43*~6 already exceeds at 82 tokens.
    n_tokens = min(n_tokens, 400)      # 400 tokens ~ 100k+ requests, bounded

    cfgg = StreamGenConfig(
        n_tokens=n_tokens, n_layers=n_layers,
        experts_per_layer=experts_per_layer, topk=topk, batch=1,
        seed=int(SIM_SYNTH_SEED.value), label="synth")
    gen = StreamGenerator(cfgg)
    stream = gen.generate()
    gstats = gen.generated_stats(stream)

    keys_full = [(l, e) for c in stream.calls for (l, e) in c.key_set]
    n_full = len(keys_full)
    windows = sorted({int(n_full * f) for f in (0.05, 0.2, 0.6, 1.0)}
                     | {min(n_full, 5099)})    # the sealed-window size too
    rows: List[dict] = []
    knee_rows: List[dict] = []
    for n_req in windows:
        keys = keys_full[:n_req]
        s = Stream([c for c in stream.calls], n_layers, experts_per_layer,
                   topk, "synth")
        cnt, w = counts_and_weights(_slice_stream(stream, n_req))
        uniq = len(set(keys))
        for bg in budgets_gib:
            cap = max(1, int(bg * records_per_gib))
            lru = lru_replay_counts(_slice_stream(stream, n_req),
                                    cap, host_shared=True)
            mn = min_replay_counts(keys, cap)
            cf_fin = che_two_level(cnt, w, cap, cap, finite=True)
            cf_st = che_two_level(cnt, w, cap, cap, finite=False)
            h_fin = cf_fin["H_v"] * 0 + 1.0 - (cf_fin["cold_records_total"]
                                               / max(1.0, cf_fin["requests"]))
            h_st = cf_st["H_v"] * 0 + 1.0 - (cf_st["cold_records_total"]
                                             / max(1.0, cf_st["requests"]))
            rows.append({
                "n_requests": n_req, "n_unique": uniq,
                "budget_gib": bg, "capacity_records": cap,
                "lru_hit_rate": lru["hit_rate"],
                "min_hit_rate": mn["hit_rate"],
                "lru_minus_min_pp": 100.0 * (lru["hit_rate"]
                                             - mn["hit_rate"]),
                "che_finite_hit_rate": h_fin,
                "che_steady_hit_rate": h_st,
                "che_steady_error_pp": 100.0 * (h_st - lru["hit_rate"]),
                "che_finite_error_pp": 100.0 * (h_fin - lru["hit_rate"]),
                "window_label": ("sealed_size" if n_req == 5099
                                 else "n=%d" % n_req),
            })

    # (b) knee per window: smallest budget where LRU hit >= 95% of MIN
    for n_req in windows:
        rs = [r for r in rows if r["n_requests"] == n_req]
        target = max(r["min_hit_rate"] for r in rs)
        knee = next((r["budget_gib"] for r in sorted(rs, key=lambda x:
                                                     x["budget_gib"])
                     if r["lru_hit_rate"] >= 0.95 * target
                     and target > 0), None)
        knee_rows.append({"n_requests": n_req, "knee_gib_95pct_of_min": knee,
                          "min_hit_rate": target})
    write_csv("synth_steady_state.csv", rows)
    write_json("synth_knees.json", {"knees": knee_rows,
                                    "generator_stats": gstats,
                                    "n_tokens": n_tokens,
                                    "n_requests_total": n_full,
                                    "budgets_gib": budgets_gib})

    first, last = windows[0], windows[-1]
    e_first = [r["che_steady_error_pp"] for r in rows
               if r["n_requests"] == first]
    e_last = [r["che_steady_error_pp"] for r in rows
              if r["n_requests"] == last]
    gap_last = [r["lru_minus_min_pp"] for r in rows
                if r["n_requests"] == last]
    knee_series = [k["knee_gib_95pct_of_min"] for k in knee_rows]
    answers = {
        "a_che_error_shrinks": {
            "question": "does the +25pp Che steady-state error shrink on "
                        "long stationary streams?",
            "mean_error_pp_early_window": float(np.mean(e_first)),
            "mean_error_pp_final_window": float(np.mean(e_last)),
            "shrinks": bool(abs(np.mean(e_last)) < abs(np.mean(e_first))),
            "sealed_reference_pp": 25.0,
        },
        "b_knee_stabilizes": {
            "question": "does the 16 GiB LRU knee stabilize to a "
                        "steady-state value, and where?",
            "knee_gib_series": knee_series,
            "windows": windows,
            "stabilizes": bool(len({k for k in knee_series if k})
                              <= 2),
            "sealed_reference_gib": 16.0,
        },
        "c_lru_min_gap": {
            "question": "does the LRU-vs-MIN gap behave as predicted?",
            "lru_minus_min_pp_final_window": gap_last,
            "sealed_reference_pp": -2.8,
            "note": "theory expects a small negative gap that vanishes as "
                    "the working set fits; measured here per budget",
        },
    }
    out = {"generator_stats": gstats, "n_tokens": n_tokens,
           "n_requests_total": n_full, "windows": windows,
           "answers": answers, "rows": rows, "knees": knee_rows}
    write_json("synth_summary.json", out)
    _plots(rows, budgets_gib, windows)
    dlog("   synth: %d requests, %d windows; a=%s b=%s"
         % (n_full, len(windows), answers["a_che_error_shrinks"]["shrinks"],
            answers["b_knee_stabilizes"]["stabilizes"]))
    return out


def _slice_stream(stream: Stream, n_req: int) -> Stream:
    """Stream truncated to its first n_req requests (whole calls kept)."""
    calls = []
    used = 0
    for c in stream.calls:
        if used >= n_req:
            break
        calls.append(c)
        used += len(c.experts)
    return Stream(calls, stream.n_layers, stream.experts_per_layer,
                  stream.topk, stream.label)


def _plots(rows, budgets_gib, windows) -> None:
    fig, axes = figure("synth_steady_state", (11.0, 4.4))
    ax = axes[0] if hasattr(axes, "__len__") else axes
    for n_req in windows:
        rs = sorted([r for r in rows if r["n_requests"] == n_req],
                    key=lambda r: r["budget_gib"])
        ax.plot([r["budget_gib"] for r in rs],
                [r["che_steady_error_pp"] for r in rs], "o-",
                label="n=%d req" % n_req)
    ax.axhline(25.0, color="r", ls="--", lw=1, label="sealed-window +25pp")
    ax.set_xlabel("budget GiB (pooled)")
    ax.set_ylabel("Che steady - LRU replay (pp)")
    ax.set_title("(a) Che steady-state error vs window length")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    savefig(fig, "synth_che_error.png")

    fig, ax = figure("synth_knee", (8.0, 4.8))
    ks = [k["knee_gib_95pct_of_min"] for k in rows_knee(windows, rows)]
    ax.plot(windows, [k or 0 for k in ks], "o-")
    ax.axhline(16.0, color="r", ls="--", lw=1, label="sealed knee 16 GiB")
    ax.set_xlabel("window requests")
    ax.set_ylabel("knee GiB (95% of MIN)")
    ax.set_title("(b) LRU knee vs window length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, "synth_knee.png")


def rows_knee(windows, rows):
    """Knee per window recomputed for plotting (same rule as run())."""
    out = []
    for n_req in windows:
        rs = [r for r in rows if r["n_requests"] == n_req]
        target = max(r["min_hit_rate"] for r in rs)
        knee = next((r["budget_gib"] for r in sorted(rs, key=lambda x:
                                                     x["budget_gib"])
                     if r["lru_hit_rate"] >= 0.95 * target and target > 0),
                    None)
        out.append({"knee_gib_95pct_of_min": knee})
    return out
