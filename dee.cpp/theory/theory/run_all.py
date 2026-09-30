"""Regenerate every figure, table and CSV of the theory layer.

Usage (from ``dee.cpp/theory/``):

    python -m theory.run_all            # everything (~3-5 min)
    python -m theory.run_all --fast     # skip the bootstrap CIs
    python -m theory.run_all A B D      # named stages only

Stages: provenance, popularity, cache, temporal, roofline, serving,
prefetch, sensitivity.
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
        "provenance", "popularity", "cache", "temporal", "roofline",
        "serving", "prefetch", "sensitivity"}
    alias = {
        "a": "popularity", "b": "cache", "c": "temporal", "d": "roofline",
        "e": "serving", "f": "prefetch", "g": "sensitivity",
        "0": "provenance",
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

    dlog("all requested stages complete in %.1fs" % (time.time() - t_all))
    _summary(results)
    return 0


def _summary(results) -> None:
    from . import paths

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
