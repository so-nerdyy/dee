"""Regenerate every figure, table and CSV of the theory layer.

Usage (from ``dee.cpp/theory/``):

    python -m theory.run_all            # everything (~5-10 min)
    python -m theory.run_all --fast     # skip the bootstrap CIs and long sims
    python -m theory.run_all A B D      # named stages only

Stages: provenance, anchor, popularity, cache, temporal, roofline, serving,
prefetch, sensitivity, sim (Phase-7 digital twin), pred (predictor lab),
solve (provisioning solver), predecl (Phase-6 pre-registered matrix).

Phase-7 note: ``anchor`` must run (or have run) before ``sim`` — the sim's
REPLAY MODE consumes ``data/anchor_gate.json`` and is GATED on it.  The
``sim``, ``pred``, ``solve`` and ``predecl`` stages are independent of each
other and run in any order.
"""
from __future__ import annotations

import sys
import time

from .util import dlog


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    fast = "--fast" in argv
    argv = [a for a in argv if not a.startswith("-")]
    wanted = set(a.lower() for a in argv) or {
        "provenance", "anchor", "popularity", "cache", "temporal", "roofline",
        "serving", "prefetch", "sensitivity",
        "sim", "pred", "solve", "predecl"}
    alias = {
        "a": "popularity", "b": "cache", "c": "temporal", "d": "roofline",
        "e": "serving", "f": "prefetch", "g": "sensitivity",
        "0": "provenance", "h": "sim", "i": "pred", "j": "solve",
        "k": "predecl",
    }
    wanted = {alias.get(w, w) for w in wanted}

    t_all = time.time()
    results = {}
    if "provenance" in wanted:
        from . import sources
        t = time.time()
        sources.write_provenance()
        dlog("provenance ledger ->", sources.paths.DATA_DIR / "provenance.csv",
             "(%.1fs)" % (time.time() - t))
    if "anchor" in wanted:
        from . import anchor_gate
        t = time.time()
        results["ANCHOR"] = anchor_gate.run(fast=fast)
        dlog("stage anchor done in %.1fs" % (time.time() - t))
    if "popularity" in wanted:
        from . import popularity
        t = time.time()
        results["A"] = popularity.run(fast=fast)
        dlog("stage A done in %.1fs" % (time.time() - t))
    if "cache" in wanted:
        from . import cache
        t = time.time()
        results["B"] = cache.run(results.get("A"))
        dlog("stage B done in %.1fs" % (time.time() - t))
    if "temporal" in wanted:
        from . import temporal
        t = time.time()
        results["C"] = temporal.run()
        dlog("stage C done in %.1fs" % (time.time() - t))
    if "roofline" in wanted:
        from . import roofline
        t = time.time()
        results["D"] = roofline.run(results.get("B"))
        dlog("stage D done in %.1fs" % (time.time() - t))
    if "serving" in wanted:
        from . import serving
        t = time.time()
        results["E"] = serving.run(results.get("D"), results.get("B"))
        dlog("stage E done in %.1fs" % (time.time() - t))
    if "prefetch" in wanted:
        from . import prefetch
        t = time.time()
        results["F"] = prefetch.run(results.get("C"))
        dlog("stage F done in %.1fs" % (time.time() - t))
    if "sensitivity" in wanted:
        from . import sensitivity
        t = time.time()
        results["G"] = sensitivity.run(results.get("D"))
        dlog("stage G done in %.1fs" % (time.time() - t))
    for stage, modname, key in (("sim", "sim", "SIM"), ("pred", "pred", "PRED"),
                                ("solve", "solve", "SOLVE"),
                                ("predecl", "predecl", "PREDECL")):
        if stage not in wanted:
            continue
        try:
            mod = __import__("%s.%s" % (__package__, modname),
                             fromlist=[modname])
        except ImportError as exc:
            dlog("stage %s NOT IMPLEMENTED YET (Phase-7 deliverable): %s"
                 % (stage, exc))
            continue
        if not callable(getattr(mod, "run", None)):
            dlog("stage %s NOT IMPLEMENTED YET (no run(); in-flight): skipping"
                 % stage)
            continue
        t = time.time()
        if stage == "sim":
            anchor = results.get("ANCHOR")
            if anchor is None:
                from . import anchor_gate
                anchor = anchor_gate.run(fast=fast)
                results["ANCHOR"] = anchor
                dlog("stage anchor (auto, sim prerequisite) done")
            results[key] = mod.run(anchor, fast=fast)
        elif stage == "solve":
            results[key] = mod.run(results.get("D"), results.get("B"))
        elif stage == "predecl":
            results[key] = mod.run(results.get("SIM"), results.get("SOLVE"))
        else:
            results[key] = mod.run(fast=fast)
        dlog("stage %s done in %.1fs" % (stage, time.time() - t))

    dlog("all requested stages complete in %.1fs" % (time.time() - t_all))
    _summary(results)
    return 0


def _summary(results) -> None:
    from . import paths

    if "ANCHOR" in results:
        ag = results["ANCHOR"]["anchor_gate"]
        dlog("ANCHOR GATE TARGET: decode storage_requests mean %.1f "
             "(range %d-%d), host share pooled %.3f (definitions: %d), "
             "decode wall mean %.0f ms" % (
                 sum(ag["decode_storage_requests"])
                 / len(ag["decode_storage_requests"]),
                 min(ag["decode_storage_requests"]),
                 max(ag["decode_storage_requests"]),
                 ag["host_share"]["definitions"]["pooled_dedup"]["value"],
                 len(ag["host_share"]["definitions"]),
                 ag["decode_wall_shape_stats"]["mean_ms"]))
    if "SIM" in results and isinstance(results["SIM"], dict):
        dlog("SIM GATE: %s" % results["SIM"].get("gate_status", "?"))
    if "D" in results:
        a = results["D"]["anchor"]
        dlog("ANCHOR: measured %.3f tok/s vs predicted %s -> ratio %s (within 2x: %s)" % (
            a["measured_tps"], [round(v, 3) for v in a["predicted_tps_pred"]],
            [round(v, 2) for v in a["ratio_measured_over_pred"]], a["within_2x"]))
    if "B" in results:
        lm = results["B"]["landmarks"]
        dlog("CACHE KNEE: recomputed %.1f GiB vs measured %.1f GiB; "
             "LRU-MIN %.2f pp vs %.1f pp measured; repeat %d/%d (matches)" % (
                 lm["recomputed_knee_gib"], lm["measured_knee_gib"],
                 lm["recomputed_lru_minus_min_pp_at_knee"],
                 -abs(lm["measured_lru_minus_min_pp"]),
                 lm["recomputed_repeat_records"], lm["unique_records"]))
    dlog("outputs in", paths.DATA_DIR, "and", paths.FIGS_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
