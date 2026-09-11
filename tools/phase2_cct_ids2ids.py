"""R2 OPEN-UNKNOWN #2 — journal-only ids->ids cross-layer test.

Question: does the committed expert set E(L) (free at the row boundary)
carry information about E(L+1) beyond chance? This is the ids-space hint
channel; if it is dead, only hidden-state gate_trace hints remain.

Method: expert-keyed CCT (predict the INDEX+layer's expert identity set —
note a k-lead hint must name experts on L+k, so we learn
P(e' on L+1 | e on L) per adjacent layer pair, keyed by expert index).

Train/test split over decode steps (train = first half, test = second
half; prefill step 0 kept separate — its routing is hash-layer mixed).
Metrics per layer pair:
  - same_index_full: mean P(i in E(L+1) | i in E(L)) over observed i
  - cct_top1/top3: given E(L) at test time, take each active expert's
    most-likely successors, union, measure |hit|/|E(L+1)|
  - baseline_chance: |E(L+1)|/256 (chance hit rate for any k-subset)
Mutual information I(E_L; E_{L+1}) is also reported (bits/expert-row).

Local-only analysis: pure CPU, sealed journal, zero remote spend.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_legal_prefetch_eval import JOURNAL, load_batches  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "research" / "prior-art" / "r02-results"


def main():
    batches = load_batches(JOURNAL)
    # index: (step, layer) -> expert set
    by = {(b["step"], b["layer"]): set(b["experts"]) for b in batches}
    steps = sorted({b["step"] for b in batches})
    layers = sorted({b["layer"] for b in batches})
    decode_steps = [s for s in steps if s > 0]
    train_steps = set(decode_steps[: len(decode_steps) // 2])
    test_steps = set(decode_steps[len(decode_steps) // 2:])
    print(f"steps={steps}  train={sorted(train_steps)} test={sorted(test_steps)}")

    # --- train CCT per layer pair: P(e' on L+1 | e on L) ----------------
    # counts[layer][e][e'] = # train steps with e in E(L) and e' in E(L+1)
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    denom = defaultdict(lambda: defaultdict(int))
    same_idx_hits = defaultdict(lambda: [0, 0])
    for s in train_steps:
        for L in layers[:-1]:
            el, el1 = by.get((s, L)), by.get((s, L + 1))
            if el is None or el1 is None:
                continue
            for e in el:
                denom[L][e] += 1
                for e2 in el1:
                    counts[L][e][e2] += 1
                if e in el1:
                    same_idx_hits[L][0] += 1
                same_idx_hits[L][1] += 1

    # --- evaluate on test steps ------------------------------------------
    res = []
    for L in layers[:-1]:
        top1_hits = top3_hits = union_top3_hits = 0
        n_rows = 0
        tot_tgt = 0
        chance_union = 0.0
        mi_num = mi_den = 0.0
        for s in sorted(test_steps):
            el, el1 = by.get((s, L)), by.get((s, L + 1))
            if el is None or el1 is None:
                continue
            n_rows += 1
            tot_tgt += len(el1)
            pass
            # per-active-expert argmax successor
            union_pred = set()
            for e in el:
                c = counts[L].get(e, {})
                if not c:
                    continue
                d = denom[L][e]
                ranked = sorted(c.items(), key=lambda kv: -kv[1])
                top1_hits += 1 if ranked[0][0] in el1 else 0
                top3_hits += sum(1 for e2, _ in ranked[:3] if e2 in el1)
                union_pred |= {e2 for e2, _ in ranked[:3]}
                # MI contribution: sum over e' of p(e'|e) log ratio vs marginal
                for e2, n in c.items():
                    p = n / d
                    if p > 0 and e2 in el1:
                        mi_num += math.log2(p / (len(el1) / 256.0))
                        mi_den += 1
            union_top3_hits += len(union_pred & el1)
            # fair chance for a |union_pred|-sized random guess set:
            # E[hits] = |pred| * |E(L+1)|/256 -> recall = |pred|/256
            chance_union += len(union_pred) / 256.0
        si_h, si_n = same_idx_hits[L]
        res.append({
            "layer_pair": f"{L}->{L+1}",
            "test_rows": n_rows,
            "same_index_p": round(si_h / si_n, 4) if si_n else None,
            "top1_acc_per_active": round(top1_hits / max(n_rows, 1) / 1, 4),
            "union_top3_recall": round(union_top3_hits / tot_tgt, 4)
                                 if tot_tgt else None,
            "chance_union_recall": round(chance_union / tot_tgt, 4)
                                   if tot_tgt else None,
            "mi_bits_per_active": round(mi_num / mi_den, 4) if mi_den else 0.0,
        })

    agg_same = sum(r["same_index_p"] or 0 for r in res) / len(res)
    agg_u3 = sum(r["union_top3_recall"] or 0 for r in res) / len(res)
    agg_ch = sum(r["chance_union_recall"] or 0 for r in res) / len(res)
    agg_mi = sum(r["mi_bits_per_active"] for r in res) / len(res)
    summary = {
        "journal": str(JOURNAL),
        "train_steps": sorted(train_steps), "test_steps": sorted(test_steps),
        "mean_same_index_p": round(agg_same, 4),
        "mean_union_top3_recall": round(agg_u3, 4),
        "mean_chance_union_recall": round(agg_ch, 4),
        "mean_mi_bits": round(agg_mi, 4),
        "verdict": (
            "ids->ids carries signal" if agg_u3 > 1.5 * agg_ch
            else "ids->ids at/near chance — hidden-state channel required"),
        "per_pair": res,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "cct_ids2ids.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_pair"},
                     indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
