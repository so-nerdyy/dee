#!/usr/bin/env python3
"""Grade run p6m-l4-r6 against PREDICTIONS_PHASE6.md Addendum 5 (A0-A10).

Usage (from dee.cpp/):
    python aws/phase6/addendum5.py runs/p6m-l4-r6-b8-h16-{lru,nu,nu-prev,nu-router} [--json]

Pass the four evidence dirs in any order; each one's role comes from its run-id suffix
(-lru, -nu, -nu-prev, -nu-router). A0 (the --placement-exact gate, lru as baseline) runs
first. A row that fails A0 has every other line voided, as the addendum says.

Every count is a delta over the last 32 decode steps, read from the cohort checkpoint
(generated_checkpoint-c*.jsonl), the same window score.py uses.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

ROLES = ("lru", "nu", "nu-prev", "nu-router")
INF = math.inf
EPS = 1e-12

# id: (row, metric key, nominal, lo, hi, kill (lo, hi) - outside is KILLED)
BANDS = {
    "A1": ("lru", "host_hit", 0.000, 0.0, 0.02, (-INF, 0.05)),
    "A2": ("nu", "host_hit", 0.42, 0.33, 0.52, (0.25, INF)),
    "A3": ("nu", "cold_rec_per_tok", 115.0, 95.0, 140.0, (-INF, 160.0)),
    "A4": ("nu", "vram_hit", 0.12, 0.07, 0.18, (0.03, INF)),
    "A5": ("nu-prev", "host_hit", 0.87, 0.72, 0.93, (0.55, INF)),
    "A6": ("nu-prev", "reads_vs_nu", 1.00, 0.90, 1.20, (-INF, 1.35)),
    "A7": ("nu-prev", "hint_hit_per_read", 0.77, 0.60, 0.92, (0.45, INF)),
    "A8": ("nu", "tps_vs_lru", 1.27, 1.08, 1.50, (1.00 + EPS, INF)),
    "A9": ("nu-prev", "tps_vs_nu", 1.24, 1.05, 1.45, (1.00 + EPS, INF)),
    "A10": ("nu-prev", "decode_tps", 0.92, 0.70, 1.10, (-INF, INF)),
}


def role_of(root: Path) -> str:
    name = root.name.rstrip("/")
    for role in sorted(ROLES, key=len, reverse=True):
        if name.endswith("-" + role):
            return role
    raise SystemExit(f"{root}: run id must end in one of -{', -'.join(ROLES)}")


def window_counts(ev: score.Evidence) -> dict:
    """Deltas over the last B8_WINDOW_STEPS decode steps."""
    r = ev.result or {}
    keys = score.engine_keys(r)
    cp = ev.checkpoint
    w = score.B8_WINDOW_STEPS
    if len(cp) < w + 1:
        return {"error": f"only {len(cp)} checkpoint records, need {w + 1}"}
    start, end = cp[-w - 1], cp[-1]

    def delta(section: str, fld: str) -> float:
        return (score._sum(end.get(section) or {}, keys, fld)
                - score._sum(start.get(section) or {}, keys, fld))

    def delta_engine(fld: str) -> float:
        a, b = start.get("engine_stats") or {}, end.get("engine_stats") or {}
        return float(sum(float((b.get(k) or {}).get(fld, 0) or 0)
                         - float((a.get(k) or {}).get(fld, 0) or 0) for k in keys))

    out = {"from_step": start.get("step"), "to_step": end.get("step"),
           "source_reads": delta("expert_store", "source_reads")}
    for fld in ("hits", "misses", "hint_hits", "hint_reads", "hints_submitted",
                "hints_rejected", "hints_cancelled", "hints_wasted", "hint_waits",
                "hint_failures"):
        out["host_" + fld] = delta("host_pack", fld)
    for fld in ("resident_hits", "inflight_hits", "cache_hits", "cache_loads", "cold_loads"):
        out["vram_" + fld] = delta_engine(fld)
    return out


def row_metrics(ev: score.Evidence) -> dict:
    r = ev.result or {}
    b = score.batch_of(r)
    m = score.measure(ev, usd_per_hour=0.0)
    w = window_counts(ev)
    out = {"b": b, "window": w, "decode_tps": m.values.get("decode_tps"),
           "placement": r.get("placement"), "lookahead": r.get("prefetch_lookahead")}
    if "error" in w:
        return out
    hm = w["host_hits"] + w["host_misses"]
    out["host_hit"] = w["host_hits"] / hm if hm else None
    out["host_hit_retention_only"] = ((w["host_hits"] - w["host_hint_hits"]) / hm) if hm else None
    out["cold_rec_per_tok"] = w["source_reads"] / (b * score.B8_WINDOW_STEPS)
    # requests = resident hits + in-flight hits + cold loads (engine invariant);
    # cache_hits == resident_hits and cache_loads == cold_loads.
    req = w["vram_resident_hits"] + w["vram_inflight_hits"] + w["vram_cache_loads"]
    out["vram_hit"] = w["vram_resident_hits"] / req if req else None
    out["hint_hit_per_read"] = (w["host_hint_hits"] / w["host_hint_reads"]
                                if w["host_hint_reads"] else None)
    return out


def grade_all(evs: dict[str, score.Evidence]) -> dict:
    missing = [r for r in ROLES if r not in evs]
    if missing:
        raise SystemExit(f"missing rows: {missing}")
    base = evs["lru"]
    a0 = {role: score.verify_placement_pair(base, evs[role]) for role in ROLES if role != "lru"}
    a0["lru"] = [] if base.result is not None else ["no result.json"]
    rows = {role: row_metrics(ev) for role, ev in evs.items()}
    nu, prev, lru = rows["nu"], rows["nu-prev"], rows["lru"]

    def ratio(a, b):
        return a / b if a is not None and b else None

    nu["tps_vs_lru"] = ratio(nu.get("decode_tps"), lru.get("decode_tps"))
    prev["tps_vs_nu"] = ratio(prev.get("decode_tps"), nu.get("decode_tps"))
    prev["reads_vs_nu"] = ratio(prev.get("cold_rec_per_tok"), nu.get("cold_rec_per_tok"))

    bands = {}
    void = {role for role, why in a0.items() if why}
    # A8 and A9 compare two rows: void if either fails A0.
    pair = {"A8": ("nu", "lru"), "A9": ("nu-prev", "nu"), "A6": ("nu-prev", "nu")}
    for bid, (row, key, nom, lo, hi, kill) in BANDS.items():
        involved = pair.get(bid, (row,))
        x = rows[row].get(key)
        if any(r in void for r in involved):
            g = "VOID (A0 failed)"
        elif bid == "A10":
            g = score.UNTESTABLE if x is None else (score.PASS if lo <= x <= hi else score.WEAKENED)
        else:
            g = score.grade(x, nom, lo, hi, kill)
        bands[bid] = {"row": row, "metric": key, "measured": x, "nominal": nom,
                      "range": [lo, hi], "kill": list(kill), "grade": g}

    router = rows["nu-router"]
    la = router.get("lookahead") or {}
    rh, ph = router.get("host_hit"), prev.get("host_hit")
    router_check = {
        "recall": la.get("recall"), "precision": la.get("precision"),
        "host_hit": rh, "host_hit_minus_nu_prev": (rh - ph) if rh is not None and ph is not None else None,
        "registered_expectation": "host hit not below nu-prev by more than 0.03",
    }
    if "nu-router" in void:
        router_check["grade"] = "VOID (A0 failed)"
    elif router_check["host_hit_minus_nu_prev"] is None:
        router_check["grade"] = score.UNTESTABLE
    else:
        router_check["grade"] = (score.PASS if router_check["host_hit_minus_nu_prev"] >= -0.03
                                 else "MISSED")
    gate = "PASS" if not void else "FAIL: " + ", ".join(sorted(void))
    scorable = [b["grade"] for b in bands.values() if not b["grade"].startswith("VOID")]
    return {"A0": {"gate": gate, "reasons": a0}, "bands": bands,
            "overall": score.overall(scorable) if scorable else "VOID (A0 failed)",
            "nu_router": router_check, "rows": rows}


def _f(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"
    return str(x)


def render_md(card: dict) -> str:
    lines = [f"**A0 exactness gate: {card['A0']['gate']}**", ""]
    for role, why in card["A0"]["reasons"].items():
        if why:
            lines.append(f"- {role}: " + "; ".join(why))
    lines += ["| id | row | metric | measured | nominal [range] | grade |",
              "|---|---|---|---|---|---|"]
    for bid, b in card["bands"].items():
        lines.append(f"| {bid} | {b['row']} | {b['metric']} | {_f(b['measured'])} | "
                     f"{_f(b['nominal'])} [{_f(b['range'][0])}, {_f(b['range'][1])}] | {b['grade']} |")
    rc = card["nu_router"]
    lines += ["", f"Overall (A1-A10): **{card['overall']}**", "",
              f"nu-router: recall {_f(rc['recall'])}, precision {_f(rc['precision'])}, "
              f"host hit {_f(rc['host_hit'])} ({_f(rc['host_hit_minus_nu_prev'])} vs nu-prev): "
              f"{rc['grade']}", "",
              "| row | decode tok/s | host hit | retention-only hit | rec/tok | VRAM hit | hint hit/read |",
              "|---|---|---|---|---|---|---|"]
    for role in ROLES:
        r = card["rows"][role]
        lines.append(f"| {role} | {_f(r.get('decode_tps'))} | {_f(r.get('host_hit'))} | "
                     f"{_f(r.get('host_hit_retention_only'))} | {_f(r.get('cold_rec_per_tok'))} | "
                     f"{_f(r.get('vram_hit'))} | {_f(r.get('hint_hit_per_read'))} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("evidence", nargs=4, type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    evs = {}
    for root in a.evidence:
        role = role_of(root)
        if role in evs:
            raise SystemExit(f"two dirs for row {role}")
        evs[role] = score.load_evidence(root)
    card = grade_all(evs)
    print(json.dumps(card, indent=2, default=str) if a.json else render_md(card))
    return 0 if card["A0"]["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
