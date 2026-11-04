"""Grade one Phase-6 AWS run against the registered prediction.

The registration of record is PREDICTIONS_PHASE6.md Addendum 2 plus
`theory/predecl_aws.py`: the prediction for a run is
`predecl_aws.predict(cell, b, host, B_ssd)` evaluated at the run's own measured
B_SSD (`summary.json measured_b_ssd_gib_s x 2^30`). This script reads one
evidence directory (the S3 `evidence/<run_id>/` prefix synced locally),
measures the four metrics the way the matrix defines them, and grades each one
PASS / WEAKENED / KILLED / UNTESTABLE under the frozen sections 0 and 4.

Nothing here edits a frozen range. The scorer only reads.

Usage (from this directory):

    aws s3 sync s3://<bucket>/evidence/<run_id>/ runs/<run_id>/
    python score.py runs/<run_id>                 # print the scorecard
    python score.py runs/<run_id> --write         # also write scorecard.{json,md}
    python score.py runs/*  --ledger SCOREBOARD.jsonl

Measurement definitions (each one cites the frozen section it implements):

- decode tok/s, b=1: `result.decode_tok_s` = decode tokens / decode wall
  (Addendum 2 run protocol; the anchor's decode_wall_s convention).
- decode tok/s, b=8: AGGREGATE over the last 32 decode steps,
  `b * 32 / sum(last 32 decode_timings_ms)` (Addendum 2: steady rows are
  scored on the last 32 of >= 64 decode steps).
- host-tier hit rate: pooled_dedup = sum(hits) / sum(hits + misses) over the
  run's distinct engines (section 4; `data/anchor_gate.json` definitions).
  b=1 uses the prompt-scoped totals; b=8 uses the same 32-step window.
- cold records/token: storage source reads per emitted token (section 4,
  `byte_accounting.storage_requests_per_generated_token`, the basis the
  162-vs-155.06 anchor check was calibrated on). b=8 uses the window.
- $/1k tok: `1000 * rate_s / measured decode tok/s` with rate_s the cell's
  registered on-demand price (section 4 cross-check form; AWS has no per-run
  billing receipt, and spot runs are scored at on-demand, Addendum 2 (iv)).

Registered-ID tightening applied: P2's short-window kill band 0.45-0.57 on the
host-hit metric at host 16 GiB (section 4: "the intersection governs").
P1/P1b/P14/P15b/P15c are frozen against Modal cells and their B_SSD
assumptions, so they are not applied to AWS rows; the scorecard says so.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

METRICS = ("decode_tps", "host_hit_rate", "cold_records_per_tok", "usd_per_1k_tok")
GIB = 1 << 30
B8_MIN_DECODE_STEPS = 64
B8_WINDOW_STEPS = 32
B1_MIN_TOKENS = 16
P2_KILL = (0.45, 0.57)
ROW_RESULT = re.compile(r"-r\d+\.json$")

PASS, WEAKENED, KILLED, UNTESTABLE = "PASS", "WEAKENED", "KILLED", "UNTESTABLE"


# ---------------------------------------------------------------------------
# Registry: the frozen prediction function and cell table (injectable for tests)
# ---------------------------------------------------------------------------

@dataclass
class Registry:
    # label -> (frozen_gpu_cell, instance, usd_per_hour, vram_budget_mib, host budgets)
    cells: dict
    predict: Callable[[str, int, int, float], dict]
    source: str = "theory.predecl_aws"


def load_registry() -> Registry:
    """Import the frozen predictor from dee.cpp/theory (needs numpy + scipy)."""
    theory_root = Path(__file__).resolve().parents[2] / "theory"
    if str(theory_root) not in sys.path:
        sys.path.insert(0, str(theory_root))
    from theory import predecl_aws  # noqa: PLC0415  (deferred: heavy deps)
    return Registry(cells=dict(predecl_aws.AWS_CELLS.value), predict=predecl_aws.predict)


def cell_label(gpu: str, instance_type: str) -> str:
    return f"aws-{gpu}-{instance_type}"


# ---------------------------------------------------------------------------
# Evidence loading
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    root: Path
    summary: dict
    result: dict | None
    result_name: str | None
    checkpoint: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_evidence(root: Path) -> Evidence:
    root = Path(root)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{root}: no summary.json (not a Phase-6 evidence dir)")
    ev = Evidence(root=root, summary=_read_json(summary_path), result=None, result_name=None)
    candidates = sorted(p for p in root.glob("result*.json") if not ROW_RESULT.search(p.name))
    if not candidates:
        ev.problems.append("no runner result*.json in evidence")
        return ev
    if len(candidates) > 1:
        # The runner publishes a RUNNING placeholder result.json at start; in cohort
        # (and multi-prompt) mode the real result lands under a suffix beside it.
        finished = [p for p in candidates if _read_json(p).get("status") != "RUNNING"]
        if finished:
            candidates = finished
    if len(candidates) > 1:
        ev.problems.append("more than one runner result (" + ", ".join(p.name for p in candidates)
                           + "); a scored run is one prompt (b=1) or one cohort (b=8)")
        return ev
    ev.result_name = candidates[0].name
    ev.result = _read_json(candidates[0])
    if ev.result.get("mode") == "cohort":
        suffix = ev.result_name[len("result"):-len(".json")]
        cp = root / f"generated_checkpoint{suffix}.jsonl"
        if cp.is_file():
            ev.checkpoint = [json.loads(ln) for ln in cp.read_text(encoding="utf-8").splitlines()
                             if ln.strip()]
    return ev


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def engine_keys(result: dict) -> tuple[str, ...]:
    """Distinct engines. Single-GPU runs alias cuda1 to cuda0, so count cuda0 once."""
    env = result.get("gpu_environment") or {}
    if env.get("single_gpu_mode") or int(env.get("gpu_count", 1) or 1) == 1:
        return ("cuda0",)
    return ("cuda0", "cuda1")


def _sum(section: dict, keys: tuple[str, ...], fld: str) -> float:
    return float(sum(float((section.get(k) or {}).get(fld, 0) or 0) for k in keys))


def batch_of(result: dict) -> int:
    return int(result.get("cohort_k", 1)) if result.get("mode") == "cohort" else 1


@dataclass
class Measured:
    b: int
    values: dict                                   # metric -> float | None
    notes: dict = field(default_factory=dict)      # metric -> why None / basis
    window: dict = field(default_factory=dict)


def measure(ev: Evidence, usd_per_hour: float) -> Measured:
    r = ev.result or {}
    keys = engine_keys(r)
    b = batch_of(r)
    vals: dict = {m: None for m in METRICS}
    notes: dict = {}
    window: dict = {}
    if b == 1:
        tps = r.get("decode_tok_s")
        if tps is None and r.get("decode_wall_s"):
            tps = r.get("decode_tokens", 0) / r["decode_wall_s"]
        vals["decode_tps"] = float(tps) if tps else None
        notes["decode_tps"] = "decode tokens / decode wall"
        hp = r.get("host_pack") or {}
        hits, misses = _sum(hp, keys, "hits"), _sum(hp, keys, "misses")
        if hits + misses > 0:
            vals["host_hit_rate"] = hits / (hits + misses)
            notes["host_hit_rate"] = f"pooled_dedup, prompt-scoped ({int(hits)}/{int(hits + misses)})"
        else:
            notes["host_hit_rate"] = "no host_pack hits+misses in result"
        ba = r.get("byte_accounting") or {}
        cold = ba.get("storage_requests_per_generated_token")
        if cold is not None:
            vals["cold_records_per_tok"] = float(cold)
            notes["cold_records_per_tok"] = "storage_requests_per_generated_token"
            dec = [row for row in r.get("per_token_accounting") or [] if row.get("phase") == "decode"]
            if dec:
                window["decode_only_cold_records_per_tok"] = (
                    sum(row.get("storage_requests", 0) for row in dec) / len(dec))
        else:
            notes["cold_records_per_tok"] = "byte_accounting.storage_requests_per_generated_token absent"
    else:
        timings = [float(t) for t in r.get("decode_timings_ms") or []]
        window["decode_steps"] = len(timings)
        if len(timings) >= B8_WINDOW_STEPS:
            last = timings[-B8_WINDOW_STEPS:]
            vals["decode_tps"] = b * B8_WINDOW_STEPS / (sum(last) / 1000.0)
            notes["decode_tps"] = f"aggregate, last {B8_WINDOW_STEPS} of {len(timings)} decode steps"
        else:
            notes["decode_tps"] = f"only {len(timings)} decode steps"
        cp = ev.checkpoint
        if len(cp) >= B8_WINDOW_STEPS + 1:
            start, end = cp[-B8_WINDOW_STEPS - 1], cp[-1]

            def delta(section: str, fld: str) -> float:
                return (_sum(end.get(section) or {}, keys, fld)
                        - _sum(start.get(section) or {}, keys, fld))
            reads = delta("expert_store", "source_reads")
            hits, misses = delta("host_pack", "hits"), delta("host_pack", "misses")
            window.update({"from_step": start.get("step"), "to_step": end.get("step"),
                           "source_reads": reads, "host_hits": hits, "host_misses": misses})
            vals["cold_records_per_tok"] = reads / (b * B8_WINDOW_STEPS)
            notes["cold_records_per_tok"] = f"window source reads / ({b} x {B8_WINDOW_STEPS})"
            if hits + misses > 0:
                vals["host_hit_rate"] = hits / (hits + misses)
                notes["host_hit_rate"] = f"pooled_dedup over the window ({int(hits)}/{int(hits + misses)})"
        else:
            notes["cold_records_per_tok"] = ("no per-step cohort checkpoint covering the window "
                                             "(generated_checkpoint-c*.jsonl)")
        if vals["host_hit_rate"] is None:
            hp = r.get("host_pack") or {}
            hits, misses = _sum(hp, keys, "hits"), _sum(hp, keys, "misses")
            if hits + misses > 0:
                vals["host_hit_rate"] = hits / (hits + misses)
                notes["host_hit_rate"] = "pooled_dedup, whole cohort (no window counters)"
    if vals["decode_tps"]:
        vals["usd_per_1k_tok"] = 1000.0 * (usd_per_hour / 3600.0) / vals["decode_tps"]
        notes["usd_per_1k_tok"] = f"1000 x ({usd_per_hour} USD/h / 3600) / decode tok/s"
    else:
        notes["usd_per_1k_tok"] = "no decode tok/s"
    return Measured(b=b, values=vals, notes=notes, window=window)


# ---------------------------------------------------------------------------
# Prerequisites (section 0, adapted by Addendum 2) and grading (section 4)
# ---------------------------------------------------------------------------

def prerequisites(ev: Evidence, reg: Registry) -> tuple[str | None, list[str]]:
    """Return (cell label or None, failures). Any failure makes the run UNTESTABLE."""
    s, r = ev.summary, ev.result or {}
    fails = list(ev.problems)
    label = cell_label(str(s.get("gpu", "")), str(s.get("instance_type", "")))
    host = s.get("host_gib_requested")
    if s.get("model") != "dsv4-flash":
        fails.append(f"model {s.get('model')!r}: Addendum 2 registers DSv4-Flash only")
    if label not in reg.cells:
        fails.append(f"{label} is not a registered AWS cell")
        label_ok = None
    else:
        label_ok = label
        cell = reg.cells[label]
        if host is None or float(host) not in [float(h) for h in cell[4]]:
            fails.append(f"{label} is registered for host {list(cell[4])} GiB, run used {host}")
        if int(s.get("budget_mib") or 0) != int(cell[3]):
            fails.append(f"VRAM budget {s.get('budget_mib')} MiB != registered {cell[3]} MiB")
    if s.get("host_gib_effective") != host:
        fails.append("host_gib_effective != host_gib_requested")
    if s.get("host_budget_honored") is not True:
        fails.append("host budget not honored by the runner (host_budget_honored != true)")
    if not (s.get("measured_b_ssd_gib_s") or 0) > 0:
        fails.append("no measured B_SSD (fio randread probe)")
    if s.get("exit_code") not in (0, None) or (ev.result is not None and r.get("status") != "COMPLETE"):
        fails.append(f"runner did not complete (exit_code={s.get('exit_code')}, "
                     f"status={r.get('status')})")
    if ev.result is not None:
        if not r.get("stage_profile"):
            fails.append("no stage profile in result (section 0 prerequisite 4)")
        b = batch_of(r)
        if b == 1:
            n = len(r.get("generated_token_ids") or [])
            if n < B1_MIN_TOKENS:
                fails.append(f"b=1 generated {n} tokens (< {B1_MIN_TOKENS})")
        elif b == 8:
            steps = len(r.get("decode_timings_ms") or [])
            if steps < B8_MIN_DECODE_STEPS:
                fails.append(f"b=8 ran {steps} decode steps (< {B8_MIN_DECODE_STEPS})")
            n_fwd = r.get("n_forward_steps")
            if n_fwd is not None and r.get("emitted_tokens") != b * n_fwd:
                fails.append(f"b=8 emitted {r.get('emitted_tokens')} tokens != {b} x {n_fwd} "
                             "forwards (a row stopped at EOS; launch with --ignore-eos)")
        else:
            fails.append(f"batch {b} is not registered (b in {{1, 8}})")
    return label_ok, fails


def grade(x: float | None, nominal: float, lo: float, hi: float,
          kill: tuple[float, float]) -> str:
    if x is None:
        return UNTESTABLE
    if x < kill[0] or x > kill[1]:
        return KILLED
    if lo <= x <= hi:
        return PASS
    return WEAKENED


def overall(grades: list[str]) -> str:
    if KILLED in grades:
        return KILLED
    if WEAKENED in grades:
        return WEAKENED
    if UNTESTABLE in grades:
        return "PARTIAL" if any(g == PASS for g in grades) else UNTESTABLE
    return PASS


def score_run(root: Path, reg: Registry | None = None) -> dict:
    ev = load_evidence(Path(root))
    reg = reg or load_registry()
    s = ev.summary
    label, fails = prerequisites(ev, reg)
    card: dict = {
        "run_id": s.get("run_id"), "evidence_dir": str(ev.root), "cell": label,
        "gpu": s.get("gpu"), "instance_type": s.get("instance_type"),
        "host_gib": s.get("host_gib_requested"), "host_basis": "TOTAL",
        "spot": bool(s.get("spot")), "commit": s.get("commit"),
        "measured_b_ssd_gib_s": s.get("measured_b_ssd_gib_s"),
        "est_cost_usd": s.get("est_cost_usd"), "result_file": ev.result_name,
        "prerequisite_failures": fails, "registry": reg.source,
        "registered_id_tightening": "P2 kill 0.45-0.57 on host hit at host 16 GiB",
        "not_applied": "P1/P1b/P14/P15b/P15c (frozen against Modal cells), P5/P8b/P16/P17 "
                       "(need a stage-profile bucket analysis; read the profile by hand)",
    }
    if label is None or ev.result is None:
        card.update({"b": None, "verdict": UNTESTABLE, "metrics": {}})
        return card
    usd_h = float(reg.cells[label][2])
    m = measure(ev, usd_h)
    card["b"] = m.b
    card["window"] = m.window
    if fails:
        card.update({"verdict": UNTESTABLE, "metrics": {
            k: {"measured": v, "basis": m.notes.get(k), "grade": UNTESTABLE}
            for k, v in m.values.items()}})
        return card
    bssd = float(s["measured_b_ssd_gib_s"]) * GIB
    pred = reg.predict(label, m.b, int(float(s["host_gib_requested"])), bssd)
    card["predicted_limiter"] = pred.get("limiter")
    metrics: dict = {}
    for k in METRICS:
        nominal, lo, hi, kill = pred[k]
        nominal, lo, hi = float(nominal), float(lo), float(hi)
        kill = (float(kill[0]), float(kill[1]))
        if k == "host_hit_rate" and int(float(s["host_gib_requested"])) == 16:
            kill = (max(kill[0], P2_KILL[0]), min(kill[1], P2_KILL[1]))
        x = m.values[k]
        metrics[k] = {"measured": x, "basis": m.notes.get(k), "nominal": nominal,
                      "lo": lo, "hi": hi, "kill_lo": kill[0], "kill_hi": kill[1],
                      "grade": grade(x, nominal, lo, hi, kill)}
    if m.b > 1:
        # Addendum 2 scorability gate: steady-state TPS/$ rows only count if the
        # window's cold rate sits inside the row's cold kill band.
        c = metrics["cold_records_per_tok"]
        in_band = c["measured"] is not None and c["kill_lo"] <= c["measured"] <= c["kill_hi"]
        card["b8_steady_gate"] = "OPEN" if in_band else "CLOSED"
        if not in_band:
            for k in ("decode_tps", "usd_per_1k_tok"):
                metrics[k]["grade"] = UNTESTABLE
                metrics[k]["basis"] = (str(metrics[k]["basis"]) + "; UNTESTABLE: window cold "
                                       "rate outside the cold kill band (regime not reached)")
    card["metrics"] = metrics
    card["verdict"] = overall([v["grade"] for v in metrics.values()])
    return card


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(x) -> str:
    if x is None:
        return "-"
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.3f}"
    return f"{x:.4g}"


def render_md(card: dict) -> str:
    lines = [f"# Phase-6 scorecard: {card.get('run_id')}", "",
             f"**Verdict: {card['verdict']}**  ",
             f"cell `{card.get('cell')}` · b={card.get('b')} · host {card.get('host_gib')} GiB "
             f"({card.get('host_basis')} basis) · measured B_SSD {_fmt(card.get('measured_b_ssd_gib_s'))} "
             f"GiB/s · commit `{(card.get('commit') or '-')[:12]}` · est. cost "
             f"${card.get('est_cost_usd')}" + (" · spot (scored at on-demand)" if card.get("spot") else ""),
             ""]
    if card.get("prerequisite_failures"):
        lines += ["## Prerequisites not met (run is UNTESTABLE)", ""]
        lines += [f"- {f}" for f in card["prerequisite_failures"]] + [""]
    if card.get("metrics"):
        lines += ["| metric | measured | nominal | range | kill band | grade | basis |",
                  "|---|---|---|---|---|---|---|"]
        for k in METRICS:
            v = card["metrics"].get(k)
            if not v:
                continue
            rng = f"[{_fmt(v.get('lo'))}, {_fmt(v.get('hi'))}]" if "lo" in v else "-"
            kb = f"<{_fmt(v.get('kill_lo'))} \\| >{_fmt(v.get('kill_hi'))}" if "kill_lo" in v else "-"
            lines.append(f"| {k} | {_fmt(v.get('measured'))} | {_fmt(v.get('nominal'))} | {rng} "
                         f"| {kb} | {v['grade']} | {v.get('basis') or ''} |")
        lines.append("")
    if card.get("b8_steady_gate"):
        lines.append(f"b=8 steady-state gate: **{card['b8_steady_gate']}**  ")
    if card.get("predicted_limiter"):
        lines.append(f"Predicted limiter: `{card['predicted_limiter']}` (check against the stage profile)  ")
    lines += [f"ID tightening applied: {card['registered_id_tightening']}  ",
              f"Not applied here: {card['not_applied']}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("evidence", nargs="+", type=Path, help="evidence dir(s) holding summary.json")
    ap.add_argument("--write", action="store_true", help="write scorecard.json/.md into each dir")
    ap.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    ap.add_argument("--ledger", type=Path, help="append one JSON line per run to this file")
    args = ap.parse_args(argv)
    reg = load_registry()
    for root in args.evidence:
        card = score_run(root, reg)
        md = render_md(card)
        print(json.dumps(card, indent=2, default=float) if args.json else md)
        if args.write:
            (root / "scorecard.json").write_text(json.dumps(card, indent=2, default=float) + "\n",
                                                 encoding="utf-8")
            (root / "scorecard.md").write_text(md, encoding="utf-8")
        if args.ledger:
            with args.ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({k: card.get(k) for k in (
                    "run_id", "cell", "b", "host_gib", "measured_b_ssd_gib_s", "verdict",
                    "commit")} | {"grades": {k: v["grade"] for k, v in card["metrics"].items()},
                                  "measured": {k: v["measured"] for k, v in card["metrics"].items()}},
                    default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
