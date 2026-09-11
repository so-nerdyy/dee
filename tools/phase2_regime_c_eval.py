#!/usr/bin/env python3
"""Regime-C (prewarmed cross-request residency) contract evaluation.

Standalone re-derivation and reconciliation for the phase2-regime-c contract
memo (research/phase2-regime-c/CONTRACT.md). Stdlib only. No engine changes.

WHAT THIS CHECKS (three contract questions flagged in the ws-policy v4 study,
research/phase2-ws-policy @ 95dfe0d):

 1. The "never-used distance" artifact. The committed
    tools/phase2_ws_policy_sim_v4.py initializes the prewarmed-MIN oracle
    (belady_same_state_prewarm, contract C2) with each resident entry's
    TRUE first future use; entries absent from the stream get 10**12.
    The prose docs quote different numbers (56.83/59.11/70.68/83.02/91.14 %
    at 8/12/16/20/24 GiB) that match a draft which initialized ALL resident
    entries with 10**12 ("never used") until first real access -- evicting
    soon-needed prewarmed records on the first faults and understating the
    C2 bound. This tool reproduces BOTH inits and diffs them against the
    committed sim rows so the drift is mechanical, not asserted.

 2. Contract identity. C1 (static top-N pin, reject-admission/bypass) and
    C2 (fetch-on-fault MIN given the same initial state) are DIFFERENT
    contracts. This tool reports each bound on its own and tests whether
    C2 >= C1 holds here (it does on this sealed window: strictly at 8 GiB,
    exact ties at >=12 GiB). That ordering is trace-specific, NOT a theorem:
    C1 - C2 misses = (non-pin repeats) - (capacity misses), whose sign is
    workload-dependent. The docs' "ties or beats" phrasing must never be
    read as the two contracts being interchangeable bounds.

 3. Why the tie: on this window only 935 of 2,364 records ever repeat
    (freq >= 2); 1,429 are single-use. Once the pin capacity >= 935 records
    (~11.63 GiB), the top-N set contains every repeater and BOTH contracts
    sit on the compulsory-miss floor (one miss per non-resident single-use
    record) -- the tie is structural, not a policy result.

Inputs (committed):
  - sealed v50 route journal (in this branch):
    dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/
      v50-evidence-20260829T195940Z/routed_experts.jsonl
    sha256 665aac3e... is verified before any number is computed.
  - research/phase2-regime-c/inputs/sim_rows_v4.csv
    (byte copy of research/phase2-ws-policy @ 95dfe0d
    results/sim_rows_v4.csv; used to confirm the committed sim agrees with
    the exact-init bound re-derived here).

Output:
  - research/phase2-regime-c/results/regime_c_eval.json
  - stdout table, same content.

Run from the repository root:  python tools/phase2_regime_c_eval.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOURNAL = (REPO / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
           "v50-evidence-20260829T195940Z/routed_experts.jsonl")
SIM_ROWS = REPO / "research/phase2-regime-c/inputs/sim_rows_v4.csv"
OUT_DIR = REPO / "research/phase2-regime-c/results"

JOURNAL_SHA256 = "665aac3e8db570237c6dc6acaf08dc39f2af890e8a04e400ce7154f1a858dae1"
RECORD_BYTES = 13_369_344          # 12.75 MiB DEE4 packed FP4 record
GIB = 1 << 30
N_TOKENS = 16                    # sealed window: 1 prefill + 15 decode
BUDGETS_GIB = [8, 12, 16, 20, 24, 32]
BW_BANK = (0.29, 0.37)           # Phase-1 measured /tmp bank, GiB/s

# Figures quoted in the ws-policy v4 prose docs for belady_same_state_prewarm.
# They match the never-used-init draft, not the committed sim.
DOC_QUOTED_C2_HITRATES = {8: 56.83, 12: 59.11, 16: 70.68, 20: 83.02, 24: 91.14}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_stream(path: Path):
    """Engine-truth stream: per (forward_step, layer) call the engine
    deduplicates route IDs and stages unique experts in ascending id order."""
    recs = [json.loads(line) for line in open(path) if line.strip()]
    recs.sort(key=lambda r: r["record_index"])
    keys = []
    steps = set()
    for r in recs:
        uniq = sorted({e for row in r["expert_ids_rank_order"] for e in row})
        keys.extend((r["layer"], e) for e in uniq)
        steps.add(r["forward_step"])
    return keys, len(steps)


def next_use_index(keys):
    nxt = [10 ** 12] * len(keys)
    seen = {}
    for i in range(len(keys) - 1, -1, -1):
        nxt[i] = seen.get(keys[i], 10 ** 12)
        seen[keys[i]] = i
    return nxt


def first_use_of(keys):
    fu = {}
    for i, k in enumerate(keys):
        if k not in fu:
            fu[k] = i
    return fu


def c1_static_pin_misses(keys, freq, slots):
    """C1 bypass contract: top-slots resident at t=0; non-pins never enter.
    misses = requests outside the pin set."""
    pin = {k for k, _ in freq.most_common(slots)} if slots > 0 else set()
    return sum(1 for k in keys if k not in pin)


def c2_min_misses(keys, freq, slots, nxt, first_use, exact_init=True):
    """C2 fetch-on-fault MIN given the same initial resident set (top-slots
    by frequency). Every miss is admitted; eviction is furthest-next-use.

    exact_init=True: resident distance = true first future use (10**12 iff
    the record never occurs). This is the committed sim's semantic and is
    the only init that is exact MIN given the initial state.
    exact_init=False: the flagged artifact -- every resident starts at
    10**12 ("never used") regardless of its real first use; reproduced here
    only to explain the stale doc figures."""
    init = {k for k, _ in freq.most_common(slots)} if slots > 0 else set()
    if exact_init:
        resident = {k: first_use.get(k, 10 ** 12) for k in init}
    else:
        resident = {k: 10 ** 12 for k in init}
    misses = 0
    for i, k in enumerate(keys):
        nu = nxt[i]
        if k in resident:
            resident[k] = nu
            continue
        misses += 1
        if len(resident) >= slots:
            far = max(resident, key=lambda kk: resident[kk])
            del resident[far]
        resident[k] = nu
    return misses


def c2_lru_prewarm_misses(keys, freq, slots):
    """C2 control: plain LRU given the same top-slots prewarm (rank order)."""
    order = [k for k, _ in freq.most_common(slots)] if slots > 0 else []
    pos = set(order)
    lru = list(order)             # index 0 = most recent (rank order)
    misses = 0
    for k in keys:
        if k in pos:
            lru.remove(k)
            lru.insert(0, k)
            continue
        misses += 1
        if slots <= 0:
            continue
        if len(lru) >= slots:
            pos.discard(lru.pop())
        lru.insert(0, k)
        pos.add(k)
    return misses


def main():
    if not JOURNAL.exists():
        print(f"missing journal: {JOURNAL}", file=sys.stderr)
        return 1
    jsha = sha256_of(JOURNAL)
    if jsha != JOURNAL_SHA256:
        print(f"journal sha256 mismatch: {jsha}", file=sys.stderr)
        return 1

    keys, n_steps = load_stream(JOURNAL)
    freq = Counter(keys)
    nxt = next_use_index(keys)
    fu = first_use_of(keys)
    n = len(keys)
    repeaters = sum(1 for c in freq.values() if c >= 2)
    single_use = sum(1 for c in freq.values() if c == 1)
    assert n == 5099 and len(freq) == 2364 and n_steps == 16, "unexpected stream shape"

    committed = {}
    if SIM_ROWS.exists():
        for r in csv.DictReader(open(SIM_ROWS)):
            if r["scope"] == "full" and r["regime"] == "C":
                committed[(r["policy"], float(r["budget_gib"]), r["pin_frac"])] = r

    rows = []
    checks = {"sim_csv_agrees_exact_init": True,
              "artifact_reproduces_doc_figures": True,
              "c2_ge_c1_every_budget": True}
    for g in BUDGETS_GIB:
        slots = int(g * GIB // RECORD_BYTES)
        init_gib = round(slots * RECORD_BYTES / GIB, 3)
        m_c1 = c1_static_pin_misses(keys, freq, slots)
        m_c2 = c2_min_misses(keys, freq, slots, nxt, fu, exact_init=True)
        m_c2bad = c2_min_misses(keys, freq, slots, nxt, fu, exact_init=False)
        m_lru = c2_lru_prewarm_misses(keys, freq, slots)

        row = {
            "budget_gib": g, "slots": slots, "prewarm_gib": init_gib,
            "coverage_pct": round((n - m_c1) / n * 100, 2),
            "c1_pin_misses": m_c1, "c1_hit_rate_pct": round((n - m_c1) / n * 100, 2),
            "c2_min_misses": m_c2, "c2_hit_rate_pct": round((n - m_c2) / n * 100, 2),
            "c2_min_inf_init_misses": m_c2bad,
            "c2_inf_init_hit_rate_pct": round((n - m_c2bad) / n * 100, 2),
            "c2_lru_prewarm_misses": m_lru,
            "c2_lru_hit_rate_pct": round((n - m_lru) / n * 100, 2),
            "c1_ssd_MiB_per_token": round(m_c1 * RECORD_BYTES / (1 << 20) / N_TOKENS, 1),
            "prewarm_cost_s_at_bank": {
                str(bw): round(init_gib / bw, 1) for bw in BW_BANK},
            "c2_ge_c1": m_c2 <= m_c1,
        }
        rows.append(row)

        # cross-check the committed v4 sim rows (same sealed inputs)
        for pol, misses in (("static_freq_prewarm", m_c1),
                            ("belady_same_state_prewarm", m_c2),
                            ("lru_prewarm_topN", m_lru)):
            r = committed.get((pol, float(g), ""))
            if r is not None and int(r["misses"]) != misses:
                checks["sim_csv_agrees_exact_init"] = False
        if g in DOC_QUOTED_C2_HITRATES:
            if abs((n - m_c2bad) / n * 100 - DOC_QUOTED_C2_HITRATES[g]) > 0.05:
                checks["artifact_reproduces_doc_figures"] = False
        if m_c2 > m_c1:
            checks["c2_ge_c1_every_budget"] = False

    # tie boundary: once slots >= number of repeating records, the top-N set
    # holds every repeater and both contracts sit on the compulsory floor
    tie_gib = round(repeaters * RECORD_BYTES / GIB, 2)
    structure = {
        "requests": n, "unique_records": len(freq), "forward_steps": n_steps,
        "records_freq_ge2": repeaters, "records_freq_eq1": single_use,
        "tie_boundary_gib": tie_gib,
        "explanation": ("slots >= records_freq_ge2 -> top-N set contains every "
                        "repeating record; all remaining misses are compulsory "
                        "first touches of single-use records under BOTH C1 "
                        "and C2 -> exact tie. Below the boundary C2-MIN can "
                        "drop late-use pins and admit soon-repeating non-pins; "
                        "C1 cannot, so C2 > C1 (observed at 8 GiB)."),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "journal_sha256": jsha,
        "record_bytes": RECORD_BYTES,
        "structure": structure,
        "contracts": {
            "C1_bypass_fixed_set": ("static top-N prewarm; non-pinned faults "
                                    "are never admitted (reject-admission)"),
            "C2_fetch_on_fault": ("same prewarm state, every fault admitted; "
                                  "bound = exact MIN given the initial state"),
            "C3_pin_plus_remainder": ("protected pin + dynamic remainder; "
                                      "evaluated in the v4 sim, bounds "
                                      "belady_pin_same_state(pf)"),
        },
        "rows": rows,
        "checks": checks,
        "doc_figures_explained_by_inf_init": DOC_QUOTED_C2_HITRATES,
        "standing_rules": [
            "regime-C rows are never scored against cold-start rows",
            "the one-time prewarm read (prewarm_gib GiB) is charged outside "
            "the measured window",
            "24 GiB pooled projects ~29-30 GiB peak RSS vs the 31.35 GiB "
            "host limit (memory gate)",
        ],
    }
    with open(OUT_DIR / "regime_c_eval.json", "w") as f:
        json.dump(out, f, indent=1)

    hdr = (f"{'GiB':>4} {'slots':>6} {'C1 pin':>8} {'C2 MIN':>8} "
           f"{'C2 inf-init':>11} {'C2 LRU':>8} {'MiB/tok(C1)':>11}")
    print(f"journal sha256 {jsha[:12]}... requests {n} "
          f"unique {len(freq)} repeaters {repeaters} single-use {single_use}")
    print(f"tie boundary: {repeaters} records = {tie_gib} GiB")
    print(hdr)
    for r in rows:
        print(f"{r['budget_gib']:>4} {r['slots']:>6} "
              f"{r['c1_hit_rate_pct']:>7.2f}% {r['c2_hit_rate_pct']:>7.2f}% "
              f"{r['c2_inf_init_hit_rate_pct']:>10.2f}% "
              f"{r['c2_lru_hit_rate_pct']:>7.2f}% "
              f"{r['c1_ssd_MiB_per_token']:>10.1f}")
    print("\nchecks:")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\nwrote {OUT_DIR / 'regime_c_eval.json'}")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
