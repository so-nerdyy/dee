"""Predictor lab orchestration: fit every arm on TRAIN, evaluate on TEST under
two budget semantics, calibrate, do the byte accounting, emit every artifact.

BUDGET SEMANTICS (both reported; pre-registered before any test number):

  * BUDGET-SET  ("m announced records per source CALL", budgeted max-coverage
    semantics of THEORY.md section 7): the ranked candidate list is taken from
    the full-source context score ``scores(ex)`` and truncated to m records.
    This is the semantics the P10 kill criterion ("precision > 0.30 at m=8")
    maps onto literally.

  * BUDGET-SRC  ("m candidates per source EXPERT", the semantics the
    FALSIFICATION P9/P10 ranges were DERIVED in -- theory/prefetch.py's
    predict_candidates top-m-per-source-expert): for each source expert i in
    S_l, announce its top-m under the single-source score ``pair_scores(ex)``,
    unioned (duplicate candidates deduplicated).  |S_l| = 8 on T2, so m=8 here
    means up to 64 announced records per call (48 on T1 with top-6).

The two agree only at small m; reporting both is required to say whether P9
and P10 are CONFIRMED or KILLED without moving their goalposts.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from ..util import dlog, figure, rng, savefig, write_csv, write_json
from .metrics import BASELINE_SCENARIOS, calibration_bins, evaluate_arm

Rec = Tuple[int, int]
ARMS_ORDER = ["popularity", "cond_pooled", "cond_perlayer",
              "logreg_pairwise", "mlp_set", "xlayer_bag"]
TASKS = ("next_layer", "next_step")


def run(fast: bool = False) -> Dict[str, object]:
    from . import (PRED_BOOT, PRED_M_SWEEP, PRED_MIN_PAIR_SUPPORT, PRED_SEED)
    from .arms import build_arms
    from .data import load_datasets

    dlog("B(pred). predictor lab")
    seed = int(PRED_SEED.value)
    n_boot = 0 if fast else int(PRED_BOOT.value)
    datasets = load_datasets()
    meta = {"seed": seed, "min_support": int(PRED_MIN_PAIR_SUPPORT.value),
            "n_layers": 256}

    metric_rows: List[dict] = []
    byte_rows: List[dict] = []
    calib_rows: List[dict] = []
    calib_summary: List[dict] = []

    for tname, ds in datasets.items():
        for task in TASKS:
            train_ex = ds.splits["train"][task]
            test_ex = ds.splits["test"][task]
            if not test_ex:
                continue
            m = dict(meta, n_layers=ds.n_layers)
            for arm in build_arms():
                dlog("   fit+eval %s / %s / %s (train %d, test %d)"
                     % (tname, task, arm.name, len(train_ex), len(test_ex)))
                arm.fit(train_ex, m)
                for sem in ("set", "src"):
                    ranked = _rank_all(arm, test_ex, sem)
                    rows, brows, per_call = evaluate_arm(
                        test_ex, ranked, ds.first_touch, n_boot=n_boot, seed=seed)
                    for r in rows:
                        r.update({"trace": tname, "task": task, "arm": arm.name,
                                  "budget_semantics": sem,
                                  "announced_bytes_per_call":
                                      r["announced_per_call"] * ds.record_bytes})
                        metric_rows.append(r)
                    for r in brows:
                        r.update({"trace": tname, "task": task, "arm": arm.name,
                                  "budget_semantics": sem,
                                  "record_bytes": ds.record_bytes,
                                  "saved_bytes": r["saved_cold_records"] * ds.record_bytes,
                                  "stale_bytes": r["stale_records"] * ds.record_bytes,
                                  "wasted_bytes": r["wasted_records"] * ds.record_bytes,
                                  "consumed_bytes": r["announced_total"] * ds.record_bytes,
                                  "saved_bytes_per_call":
                                      r["saved_cold_records"] * ds.record_bytes / max(1, r["n_calls"]),
                                  "consumed_bytes_per_call":
                                      r["announced_total"] * ds.record_bytes / max(1, r["n_calls"]),
                                  "saved_bytes_per_token":
                                      r["saved_cold_records"] * ds.record_bytes
                                      / max(1, len({c["cluster"] for c in per_call})),
                                  "consumed_bytes_per_token":
                                      r["announced_total"] * ds.record_bytes
                                      / max(1, len({c["cluster"] for c in per_call})),
                                  "net_bytes_per_token":
                                      (r["saved_cold_records"] - r["announced_total"])
                                      * ds.record_bytes
                                      / max(1, len({c["cluster"] for c in per_call})),
                                  "n_tokens": len({c["cluster"] for c in per_call}),
                                  })
                        byte_rows.append(r)
                    if sem == "set":
                        crows, ece, npairs = calibration_bins(per_call, 8)
                        for cr in crows:
                            cr.update({"trace": tname, "task": task,
                                       "arm": arm.name, "m": 8,
                                       "ece": ece, "n_pairs_total": npairs})
                            calib_rows.append(cr)
                        calib_summary.append({"trace": tname, "task": task,
                                              "arm": arm.name, "m": 8,
                                              "ece": ece, "n_pairs": npairs})

    write_csv("pred_metrics.csv", metric_rows)
    write_csv("pred_byte_value.csv", byte_rows)
    write_csv("pred_calibration.csv", calib_rows)
    _leak_diagnostic(datasets, metric_rows)
    _write_splits(datasets, metric_rows)
    _plots(metric_rows, byte_rows, calib_rows, calib_summary)
    out = {"metrics": metric_rows, "bytes": byte_rows,
           "calibration": calib_rows, "calib_summary": calib_summary}
    dlog("   pred: %d metric rows, %d byte rows, %d calibration rows"
         % (len(metric_rows), len(byte_rows), len(calib_rows)))
    return out


def _rank_all(arm, test_ex, sem: str) -> Dict[int, Dict[int, List[Tuple[Rec, float]]]]:
    """Announced candidate lists per test example and hint budget m.

    set: top-m records of the full-context ranking (|list| = m);
    src: union over source experts of their top-m single-source candidates,
         re-ranked by full-context score (|list| <= |S_src| x m).
    """
    ranked: Dict[int, Dict[int, List[Tuple[Rec, float]]]] = {}
    m_sweep = (1, 2, 4, 8, 16, 32)
    for idx, ex in enumerate(test_ex):
        s = arm.scores(ex)
        order = np.lexsort((np.arange(256), -s))
        by_m: Dict[int, List[Tuple[Rec, float]]] = {}
        if sem == "set":
            for m in m_sweep:
                by_m[m] = [((ex.dst_layer, int(j)), float(s[j])) for j in order[:m]]
        else:
            ps = arm.pair_scores(ex)
            for m in m_sweep:
                cand: Set[int] = set()
                for k in range(ps.shape[0]):
                    o = np.lexsort((np.arange(256), -ps[k]))[:m]
                    cand.update(int(j) for j in o)
                by_m[m] = sorted(
                    (((ex.dst_layer, j), float(s[j])) for j in cand),
                    key=lambda t: (-t[1], t[0]))
        ranked[idx] = by_m
    return ranked


def _leak_diagnostic(datasets, metric_rows) -> None:
    """data/pred_leak_diagnostic.csv: (a) how much the PREVIOUS run's in-sample
    protocol (fit conditionals on the whole trace, evaluate on the same pairs,
    theory/prefetch.py / THEORY.md section 7) overstates recall vs this stage's
    time-ordered out-of-sample protocol, and (b) leave-one-cluster-out
    robustness of the thin-cluster headline numbers.

    The in-sample rows are DIAGNOSTIC ONLY -- they are leaky by construction
    and are never used as an evaluation result.
    """
    from .arms import CondPooledArm
    from . import PRED_MIN_PAIR_SUPPORT, PRED_SEED

    meta = {"seed": int(PRED_SEED.value),
            "min_support": int(PRED_MIN_PAIR_SUPPORT.value), "n_layers": 256}
    rows: List[dict] = []
    for tname, ds in datasets.items():
        for task in TASKS:
            allx = ds.splits["train"][task] + ds.splits["test"][task]
            arm = CondPooledArm()
            arm.fit(allx, meta)                       # leaky in-sample fit
            for m in (8, 16):
                ranked = _rank_all(arm, allx, "src")
                mrows, _, per_call = evaluate_arm(allx, ranked, ds.first_touch,
                                                  n_boot=0)
                r = next(x for x in mrows if x["m"] == m)
                rows.append({"trace": tname, "task": task, "m": m,
                             "protocol": "in_sample_all_pairs (OLD prefetch.py "
                                         "protocol, leaky, diagnostic only)",
                             "n_calls": r["n_calls"],
                             "announced_per_call": r["announced_per_call"],
                             "recall": r["recall"], "precision": r["precision"],
                             "recall_lo": "", "recall_hi": "",
                             "cluster_lo": "", "cluster_hi": ""})
                oos = next((x for x in metric_rows
                            if x["trace"] == tname and x["task"] == task
                            and x["arm"] == "cond_pooled" and x["m"] == m
                            and x["budget_semantics"] == "src"), None)
                if oos:
                    rows.append({"trace": tname, "task": task, "m": m,
                                 "protocol": "out_of_sample_test (time-ordered, "
                                             "honest)",
                                 "n_calls": oos["n_calls"],
                                 "announced_per_call": oos["announced_per_call"],
                                 "recall": oos["recall"],
                                 "precision": oos["precision"],
                                 "recall_lo": oos["recall_lo"],
                                 "recall_hi": oos["recall_hi"],
                                 "cluster_lo": "", "cluster_hi": ""})
                # leave-one-cluster-out jackknife of the pooled precision/recall
                by_c: Dict[int, List[int]] = {}
                for c in per_call:
                    ann = [rr for rr, _s in c["ann_by_m"].get(m, [])]
                    h = len(set(ann) & c["tgt"])
                    cs = by_c.setdefault(c["cluster"], [0, 0, 0])
                    cs[0] += len(ann); cs[1] += c["n_tgt"]; cs[2] += h
                precs, recs = [], []
                for drop in list(by_c):
                    a = sum(v[0] for k, v in by_c.items() if k != drop)
                    t = sum(v[1] for k, v in by_c.items() if k != drop)
                    h = sum(v[2] for k, v in by_c.items() if k != drop)
                    precs.append(h / a if a else 0.0)
                    recs.append(h / t if t else 0.0)
                rows.append({"trace": tname, "task": task, "m": m,
                             "protocol": "leave_one_cluster_out (robustness of "
                                         "the in-sample run)",
                             "n_calls": len(allx),
                             "announced_per_call": "",
                             "recall": "", "precision": "",
                             "recall_lo": min(recs) if recs else "",
                             "recall_hi": max(recs) if recs else "",
                             "cluster_lo": min(precs) if precs else "",
                             "cluster_hi": max(precs) if precs else ""})
    write_csv("pred_leak_diagnostic.csv", rows)


def _write_splits(datasets, metric_rows) -> None:
    """data/pred_splits.json: exact split definition + counts."""
    from . import PRED_SPLIT_T1, PRED_SPLIT_T2

    out = {
        "fixed_before_test_evaluation": True,
        "note": ("Split fixed in theory/pred/__init__.py (PRED_SPLIT_T2 / "
                 "PRED_SPLIT_T1) before any test-set number was computed. "
                 "Leak prevention: run-level routing-stream identity was "
                 "checked first; byte-identical streams are grouped into one "
                 "split side. Vocabulary is fixed by geometry (256 experts/layer), "
                 "no test-derived vocabulary or features; hyperparameters fixed "
                 "a priori in PRED_LR / PRED_MLP / PRED_XLAYER."),
        "ornith_t2": {
            "definition": PRED_SPLIT_T2.value,
            "stream_identity_evidence": {
                "byte_identical_run_groups": [
                    ["dual-cold-primary", "dual-warm-profiled", "dual-warm-reference-present"],
                    ["dual-cache-disabled", "dual-cache-capacity-4", "single-t4-warm"],
                ],
                "prefix_run": "dual-one-token (strict prefix of group 1; dropped)",
                "train_test_jaccard_max": 0.081,
            },
        },
        "sealed_t1": {"definition": PRED_SPLIT_T1.value},
        "counts": {},
    }
    for r in metric_rows:
        if r["m"] != 8 or r["budget_semantics"] != "set":
            continue
        key = "%s/%s" % (r["trace"], r["task"])
        out["counts"].setdefault(key, {})[r["arm"]] = {
            "n_calls": r["n_calls"], "n_clusters": r["n_clusters"],
            "n_target_total": r["n_target_total"],
        }
    out["counts_summary"] = {}
    for key, arms in out["counts"].items():
        a0 = next(iter(arms.values()))
        out["counts_summary"][key] = {
            "test_calls": a0["n_calls"],
            "test_tokens": a0["n_clusters"],
            "test_target_records": a0["n_target_total"],
        }
    write_json("pred_splits.json", out)


def _plots(metric_rows, byte_rows, calib_rows, calib_summary) -> None:
    # ---- precision/recall vs m ------------------------------------------
    fig, ax = figure("pred_precision_recall", (8.4, 5.2))
    styles = {"set": "-", "src": "--"}
    colors = {"popularity": "gray", "cond_pooled": "tab:blue",
              "cond_perlayer": "tab:cyan", "logreg_pairwise": "tab:orange",
              "mlp_set": "tab:green", "xlayer_bag": "tab:red"}
    for task in TASKS:
        for arm_name in ARMS_ORDER:
            for sem in ("set", "src"):
                rs = [r for r in metric_rows
                      if r["task"] == task and r["arm"] == arm_name
                      and r["budget_semantics"] == sem and r["trace"] == "ornith_t2"]
                if not rs:
                    continue
                rs.sort(key=lambda r: r["m"])
                lab = f"{arm_name} {task[:12]}{'*' if sem=='set' else ''}"
                ax.plot([r["m"] for r in rs], [r["precision"] for r in rs],
                        styles[sem], color=colors[arm_name], lw=1.0,
                        label=lab if task == "next_layer" else None, alpha=0.85)
    ax.axhline(0.30, color="k", ls=":", lw=1.2)
    ax.annotate("P10 kill threshold 0.30", (1.1, 0.31), fontsize=8)
    ax.axhline(0.18, color="tab:purple", ls=":", lw=0.8)
    ax.axhline(0.10, color="tab:purple", ls=":", lw=0.8)
    ax.annotate("P10 predicted band 0.10-0.18", (1.1, 0.19), fontsize=7, color="tab:purple")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m (candidates; solid = per call, dashed = per source expert)")
    ax.set_ylabel("precision (Ornith T2)")
    ax.set_title("Predictor lab: precision vs m (P10 band + kill threshold)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    savefig(fig, "pred_precision_recall.png")

    # ---- recall panel ----------------------------------------------------
    fig, ax = figure("pred_recall", (8.4, 5.2))
    for task in TASKS:
        for arm_name in ARMS_ORDER:
            for sem in ("set", "src"):
                rs = [r for r in metric_rows
                      if r["task"] == task and r["arm"] == arm_name
                      and r["budget_semantics"] == sem and r["trace"] == "ornith_t2"]
                if not rs:
                    continue
                rs.sort(key=lambda r: r["m"])
                ax.plot([r["m"] for r in rs], [r["recall"] for r in rs],
                        styles[sem], color=colors[arm_name], lw=1.0, alpha=0.85)
    ax.axhspan(0.65, 0.80, color="tab:purple", alpha=0.12)
    ax.annotate("P9 predicted band 0.65-0.80", (1.1, 0.81), fontsize=7, color="tab:purple")
    ax.axhline(0.55, color="tab:purple", ls=":", lw=0.8)
    ax.axhline(0.88, color="tab:purple", ls=":", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("m (solid = per call, dashed = per source expert)")
    ax.set_ylabel("recall (Ornith T2)")
    ax.set_title("Predictor lab: recall vs m (P9 band + kill bounds)")
    ax.grid(True, alpha=0.3)
    savefig(fig, "pred_recall.png")

    # ---- calibration ------------------------------------------------------
    fig, ax = figure("pred_calibration", (6.4, 5.6))
    for arm_name in ARMS_ORDER:
        rs = [r for r in calib_rows
              if r["arm"] == arm_name and r["task"] == "next_layer"
              and r["trace"] == "ornith_t2"]
        if not rs:
            continue
        rs.sort(key=lambda r: r["bin"])
        ax.plot([r["mean_predicted"] for r in rs],
                [r["observed_freq"] for r in rs], "o-", ms=3,
                color=colors[arm_name], label=arm_name)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("predicted probability (bin mean)")
    ax.set_ylabel("observed landing frequency")
    ax.set_title("Calibration: next_layer, Ornith T2, m=8 (per-call budget)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    savefig(fig, "pred_calibration.png")

    # ---- byte value -------------------------------------------------------
    fig, ax = figure("pred_byte_value", (8.4, 5.2))
    arms_present = [a for a in ARMS_ORDER
                    if any(r["arm"] == a and r["task"] == "next_layer"
                           and r["budget_semantics"] == "set"
                           and r["scenario"] == "vram281_host682"
                           and r["trace"] == "ornith_t2" for r in byte_rows)]
    xs = np.arange(len(arms_present))
    width = 0.38
    for k, (m, hatch) in enumerate(((8, None), (16, "//"))):
        saved = [next((r["saved_bytes_per_token"] for r in byte_rows
                       if r["arm"] == a and r["task"] == "next_layer"
                       and r["budget_semantics"] == "set"
                       and r["scenario"] == "vram281_host682"
                       and r["trace"] == "ornith_t2" and r["m"] == m), 0.0)
                 for a in arms_present]
        consumed = [next((r["consumed_bytes_per_token"] for r in byte_rows
                          if r["arm"] == a and r["task"] == "next_layer"
                          and r["budget_semantics"] == "set"
                          and r["scenario"] == "vram281_host682"
                          and r["trace"] == "ornith_t2" and r["m"] == m), 0.0)
                    for a in arms_present]
        b1 = ax.bar(xs + (k - 0.5) * width, [v / 1e6 for v in saved], width * 0.9,
                    label=f"saved cold MB/tok (m={m})", hatch=hatch, alpha=0.8)
        b2 = ax.bar(xs + (k - 0.5) * width, [-v / 1e6 for v in consumed], width * 0.9,
                    label=f"consumed MB/tok (m={m})", hatch=hatch, alpha=0.35)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(arms_present, rotation=30, fontsize=8)
    ax.set_ylabel("MB per token (up = saved cold, down = consumed)")
    ax.set_title("Byte value of hints: saved cold bytes vs consumed bytes\n"
                 "(Ornith T2 next_layer, per-call budget, baseline 281+682 slots)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    savefig(fig, "pred_byte_value.png")
