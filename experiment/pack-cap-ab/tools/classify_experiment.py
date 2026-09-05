#!/usr/bin/env python3
"""Apply the pre-registered decision rule (experiment-contract.json) to the
fetched session evidence of the pack-cap A/B.

Reads:  results/live/<session>-<arm>/native-generate-result.json (+ memory.json)
Writes: results/comparison-sessionN.json, results/combined-comparison.json,
        results/memory-validation.json, results/miss-validation.json

This tool NEVER rewrites thresholds; it loads them from the contract and
applies them mechanically. Every output field is labeled MEASURED / DERIVED /
SIMULATED. It does not grant formal ACCEPT_PERFORMANCE (campaign policy).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
CONTRACT = PKG / "experiment-contract.json"


def load_json(p: Path):
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def arm_metrics(arm_dir: Path) -> dict:
    r = load_json(arm_dir / "native-generate-result.json")
    m = load_json(arm_dir / "memory.json")
    if r is None:
        return {"status": "MISSING", "dir": str(arm_dir)}
    hp = r.get("host_pack") or {}
    miss = {g: (hp.get(g) or {}).get("misses") for g in ("cuda0", "cuda1")}
    hits = {g: (hp.get(g) or {}).get("hits") for g in ("cuda0", "cuda1")}
    evic = {g: (hp.get(g) or {}).get("evictions") for g in ("cuda0", "cuda1")}
    eng = r.get("engine_stats") or {}
    ba = r.get("byte_accounting") or {}
    peaks = (m or {}).get("process_final_and_peak_gib") or {}
    out = {
        "status": "OK" if r.get("classification") == "ACCEPT_CORRECTNESS"
                  else f"INVALID ({r.get('classification')})",
        "label": "MEASURED",
        "dir": str(arm_dir),
        "run_id": r.get("run_id"),
        "engine_commit": r.get("commit"),
        "model_revision": r.get("model_revision"),
        "classification": r.get("classification"),
        "performance_eligible": r.get("performance_eligible"),
        "hardware": r.get("hardware_classification"),
        "decode_wall_s": r.get("decode_wall_s"),
        "decode_tok_s": r.get("decode_tok_s"),
        "prefill_ms": r.get("prefill_ms"),
        "total_wall_seconds": r.get("total_wall_seconds"),
        "token_latency_p50_ms": ((r.get("inter_token_latency_ms") or {}).get("p50")),
        "token_latency_p95_ms": ((r.get("inter_token_latency_ms") or {}).get("p95")),
        "host_pack_budget_gib": r.get("host_pack_budget_gib"),
        "host_pack_hits": hits, "host_pack_misses": miss,
        "host_pack_evictions": evic,
        "storage_requests_total": {
            g: (eng.get(g) or {}).get("cache_loads") for g in ("cuda0", "cuda1")},
        "h2d_bytes": {g: (eng.get(g) or {}).get("h2d_bytes")
                      for g in ("cuda0", "cuda1")},
        "storage_bytes_total": ba.get("storage_bytes_total"),
        "memory": {
            "label": "MEASURED",
            "vmhwm_gib": peaks.get("VmHWM"),
            "vmrss_gib": peaks.get("VmRSS"),
            "vmdata_gib": peaks.get("VmData"),
            "mem_total_gib": ((m or {}).get("system_final_gib") or {}).get("MemTotal"),
            "min_checkpoint_mem_available_gib":
                (m or {}).get("minimum_checkpoint_host_mem_available_gib"),
            "host_pack_budget_bytes": (m or {}).get("host_pack_budget_bytes"),
            "checkpoint_records": (m or {}).get("checkpoint_records"),
        },
    }
    return out


def pair_delta(a: dict, b: dict) -> dict:
    """b_minus_a deltas for one matched pair (DERIVED)."""
    d = {"label": "DERIVED", "delta_decode_wall_s": None,
         "delta_misses_total": None, "delta_misses_pct": None}
    wa, wb = a.get("decode_wall_s"), b.get("decode_wall_s")
    if isinstance(wa, (int, float)) and isinstance(wb, (int, float)):
        d["delta_decode_wall_s"] = round(wb - wa, 3)
        d["delta_decode_wall_pct"] = round(100.0 * (wb - wa) / wa, 3)
    ma = a.get("host_pack_misses") or {}
    mb = b.get("host_pack_misses") or {}
    if all(isinstance(ma.get(g), int) for g in ("cuda0", "cuda1")) and \
       all(isinstance(mb.get(g), int) for g in ("cuda0", "cuda1")):
        ta = ma["cuda0"] + ma["cuda1"]
        tb = mb["cuda0"] + mb["cuda1"]
        d["misses_A_total"] = ta
        d["misses_B_total"] = tb
        d["delta_misses_total"] = tb - ta
        if ta:
            d["delta_misses_pct"] = round(100.0 * (tb - ta) / ta, 2)
    return d


def apply_decision(combined: dict, contract: dict) -> dict:
    rule = contract["decision_rule"]
    perf = rule["performance"]
    out: dict = {"label": "DERIVED (mechanical application of pre-registered rule)",
                 "informative_positive": False,
                 "classification": "INVALID_EXPERIMENT",
                 "case": None, "order": None, "checks": {}}
    s1 = combined.get("sessions", {}).get("session1") or {}
    s2 = combined.get("sessions", {}).get("session2") or {}
    p1, p2 = s1.get("pair_delta") or {}, s2.get("pair_delta") or {}
    d1 = p1.get("delta_decode_wall_s")
    d2 = p2.get("delta_decode_wall_s")
    ok = lambda v: isinstance(v, (int, float))
    out["checks"]["both_valid"] = (
        s1.get("pair_delta") is not None and s2.get("pair_delta") is not None
        and all((s.get("a") or {}).get("classification") == "ACCEPT_CORRECTNESS"
                and (s.get("b") or {}).get("classification") == "ACCEPT_CORRECTNESS"
                for s in (s1, s2)))
    if not out["checks"]["both_valid"]:
        out["classification"] = "INVALID_EXPERIMENT"
        out["case"] = "arm missing or correctness gate failed"
        return out
    both_negative = ok(d1) and ok(d2) and d1 < 0 and d2 < 0
    mean_delta = statistics.mean([d1, d2]) if ok(d1) and ok(d2) else None
    out["mean_delta_s"] = mean_delta
    thr = perf["meaningful_noise_threshold_s"]
    out["informative_positive"] = bool(
        both_negative and mean_delta is not None and mean_delta <= -thr)
    # order test: "improves" means beyond the pre-registered noise threshold
    # (a -0.1 s noise wiggle must not manufacture an order confound).
    s1_order, s2_order = s1.get("order"), s2.get("order")
    imp1 = ok(d1) and d1 < -thr
    imp2 = ok(d2) and d2 < -thr
    confounded = (imp1 != imp2) and (
        (imp1 and s1_order == ["A", "B"]) or (imp2 and s2_order == ["B", "A"]))
    out["order"] = {
        "session1_delta": d1, "session2_delta": d2,
        "order_session1": s1_order, "order_session2": s2_order,
        "confounded": bool(confounded),
        "rule": rule["order_effect"]["order_confounded_if"]}
    # causal cases (miss-based)
    m1, m2 = p1.get("delta_misses_total"), p2.get("delta_misses_total")
    mean_dm = statistics.mean([m for m in (m1, m2)
                               if isinstance(m, int)]) \
        if isinstance(m1, int) and isinstance(m2, int) else None
    out["mean_delta_misses"] = mean_dm
    if mean_dm is not None and mean_dm > -10:
        out["case"] = "CASE_1_replay_failed"
    elif mean_dm is not None and mean_dm <= -30 and not out["informative_positive"]:
        out["case"] = "CASE_2_service_model_failed"
    elif out["informative_positive"] and mean_dm is not None and mean_dm <= -30:
        out["case"] = "CASE_3_supported"
    else:
        out["case"] = "NULL_UNRESOLVED"
    # final classification
    if confounded:
        out["classification"] = "PACK_CAP_ORDER_CONFOUNDED"
    elif out["informative_positive"] and out["case"] == "CASE_3_supported":
        out["classification"] = "PACK_CAP_SUPPORTED"
    elif out["case"] == "CASE_1_replay_failed":
        out["classification"] = "PACK_CAP_NULL"
    else:
        out["classification"] = "PACK_CAP_NULL"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-dir", type=Path, required=True,
                    help="directory containing sessionN-<arm>/ evidence "
                         "(directly, or under per-kernel subdirs)")
    ap.add_argument("--out-dir", type=Path, default=PKG / "results")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    sessions = {}
    for sname, order in (("session1", ["A", "B"]), ("session2", ["B", "A"])):
        arms = {}
        present = False
        for arm in order:
            d = args.live_dir / f"{sname}-{arm}"
            if not d.is_dir():
                # also accept per-kernel fetch subdirs (results/live/<slug>)
                for sub in sorted(args.live_dir.iterdir()):
                    if sub.is_dir() and (sub / f"{sname}-{arm}").is_dir():
                        d = sub / f"{sname}-{arm}"
                        break
            if d.is_dir():
                present = True
            arms[arm] = arm_metrics(d)
        if not present:
            continue
        # Deltas are ALWAYS candidate(B) minus baseline(A) by arm label,
        # never by session position (session 2 runs B first).
        a, b = arms["A"], arms["B"]
        sess = {"order": order, "a": a, "b": b,
                "pair_delta": pair_delta(a, b)}
        sessions[sname] = sess
    if not sessions:
        raise SystemExit(f"no session evidence found under {args.live_dir}")

    combined = {"schema": "pack-cap-ab/combined-comparison-v1",
                "sessions": sessions,
                "prediction_NOT_results": contract["prediction_NOT_results"]}
    decision = apply_decision(combined, contract)
    combined["decision"] = decision
    (args.out_dir / "combined-comparison.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8")

    for sname, sess in sessions.items():
        (args.out_dir / f"comparison-{sname}.json").write_text(
            json.dumps(sess, indent=2) + "\n", encoding="utf-8")

    # memory validation: projected (contract) vs measured
    proj = {"B": {"label": "SIMULATED/PROJECTED (ce6cb81 model)",
                  "projected_total_gib": 27.227,
                  "projected_total_decimal_gb": 29.23,
                  "projected_min_mem_available_gib": 4.12}}
    mv = {"schema": "pack-cap-ab/memory-validation-v1", "projected": proj,
          "measured": {}, "deltas": {"label": "DERIVED"}}
    for sname, sess in sessions.items():
        bmem = (sess["b"].get("memory") or {})
        mv["measured"][sname] = bmem
        tot = None
        mt = bmem.get("mem_total_gib")
        av = bmem.get("min_checkpoint_mem_available_gib")
        if isinstance(mt, (int, float)) and isinstance(av, (int, float)):
            tot = round(mt - av, 3)
            mv["deltas"][f"{sname}_measured_total_gib"] = tot
            mv["deltas"][f"{sname}_measured_minus_projected_gib"] = \
                round(tot - proj["B"]["projected_total_gib"], 3)
        if isinstance(av, (int, float)):
            mv["deltas"][f"{sname}_min_avail_measured_minus_projected_gib"] = \
                round(av - proj["B"]["projected_min_mem_available_gib"], 3)
    (args.out_dir / "memory-validation.json").write_text(
        json.dumps(mv, indent=2) + "\n", encoding="utf-8")

    # miss validation: replay prediction vs measured
    replay = {"label": "SIMULATED (pack replay, ce6cb81)",
              "predicted_delta_misses_total": -51}
    mv2 = {"schema": "pack-cap-ab/miss-validation-v1",
           "predicted": replay, "measured": {}, "deltas": {"label": "DERIVED"}}
    for sname, sess in sessions.items():
        dm = sess["pair_delta"].get("delta_misses_total")
        mv2["measured"][sname] = sess["pair_delta"]
        if isinstance(dm, int):
            mv2["deltas"][f"{sname}_measured_minus_predicted_misses"] = \
                dm - replay["predicted_delta_misses_total"]
    (args.out_dir / "miss-validation.json").write_text(
        json.dumps(mv2, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"decision": decision}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
