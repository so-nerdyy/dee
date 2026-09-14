#!/usr/bin/env python3
"""Access-pattern fragmentation analysis for expert-record reads.

Reads a route journal (routed_experts.jsonl rows with layer +
expert_ids_rank_order) and reports the pattern structure that governs
storage behavior: unique working set, per-layer regions, reuse distances,
batch compositions, and — given a (layer,expert)->record_index map —
seek-distance distributions in fill order.

The bank layout map is OPTIONAL. Without it, record order is ASSUMED
layer-major (all of one layer's records contiguous, layers in order) and
every map-dependent metric is labeled ASSUMED. Live mincore residency from
the instrumented run is ground truth; this analysis bounds the question
structurally. Never confuse the two tiers.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

RECORD_BYTES_DEFAULT = 13369344


def load_journal(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def expert_sequence(rows: list[dict]) -> list[tuple[int, int]]:
    """Flattened (layer, expert) uses in journal order (decode phase only
    unless --include-prefill)."""
    seq = []
    for row in rows:
        for rank in row.get("expert_ids_rank_order", []):
            for expert in rank:
                seq.append((row["layer"], expert))
    return seq


def pattern_metrics(rows: list[dict], record_bytes: int = RECORD_BYTES_DEFAULT,
                    index_map: dict | None = None) -> dict:
    seq = expert_sequence(rows)
    unique = sorted(set(seq))
    per_layer: dict[int, set] = collections.defaultdict(set)
    for layer, expert in seq:
        per_layer[layer].add(expert)
    # Reuse distances over the use sequence.
    last_seen: dict[tuple[int, int], int] = {}
    reuse = []
    for i, key in enumerate(seq):
        if key in last_seen:
            reuse.append(i - last_seen[key])
        last_seen[key] = i
    # Batch composition: rows are per (token, layer) calls; count experts.
    batch_sizes = [sum(len(rank) for rank in r.get("expert_ids_rank_order", []))
                   for r in rows]
    out: dict = {
        "uses": len(seq),
        "unique_experts": len(unique),
        "unique_bytes": len(unique) * record_bytes,
        "layers": len(per_layer),
        "per_layer_unique": {str(k): len(v) for k, v in sorted(per_layer.items())},
        "reuse_distance": {
            "n": len(reuse),
            "p50": statistics.median(reuse) if reuse else None,
            "p95": (sorted(reuse)[int(0.95 * (len(reuse) - 1))] if reuse else None),
        },
        "batch_size": {"min": min(batch_sizes), "max": max(batch_sizes),
                       "mean": round(sum(batch_sizes) / len(batch_sizes), 3)},
        "layout": "measured-map" if index_map else "ASSUMED-layer-major",
    }
    if index_map is not None:
        order = [index_map[f"{layer},{expert}"] for layer, expert in seq]
    else:
        # ASSUMED: records grouped by layer in layer order; within a layer
        # by first-seen expert order. Labels every derived seek metric.
        layer_base: dict[int, int] = {}
        cursor = 0
        for layer in sorted(per_layer):
            layer_base[layer] = cursor
            cursor += len(per_layer[layer])
        seen: dict[tuple[int, int], int] = {}
        order = []
        for layer, expert in seq:
            if (layer, expert) not in seen:
                seen[(layer, expert)] = layer_base[layer] + len(
                    [k for k in seen if k[0] == layer])
            order.append(seen[(layer, expert)])
    # Fill order within a batch is record-sorted (source_order); model that
    # by sorting each row's indices, then measure consecutive seek gaps.
    # Inter-batch jumps (last record of batch N -> first of batch N+1) are
    # tracked separately: same-layer locality does not hide cross-layer seeks.
    gaps = []
    inter_jumps = []
    per_row = collections.defaultdict(list)
    for (layer, _), rec in zip(seq, order):
        per_row[layer].append(rec)
    # NOTE: rows interleave layers in journal order; reconstruct per-call
    # batches from the journal row order (each row = one call).
    pos = 0
    prev_last = None
    for row in rows:
        n = sum(len(rank) for rank in row.get("expert_ids_rank_order", []))
        batch = sorted(order[pos:pos + n])
        if batch:
            if prev_last is not None:
                inter_jumps.append(abs(batch[0] - prev_last) * record_bytes)
            prev_last = batch[-1]
            for a, b in zip(batch, batch[1:]):
                gaps.append((b - a) * record_bytes)
        pos += n
    if gaps:
        out["seek_gap_bytes"] = {
            "p50": statistics.median(gaps),
            "p95": sorted(gaps)[int(0.95 * (len(gaps) - 1))],
            "max": max(gaps),
            "sequential_64k": sum(1 for g in gaps if g <= 65536) / len(gaps),
        }
    if inter_jumps:
        out["inter_batch_jump_bytes"] = {
            "p50": statistics.median(inter_jumps),
            "p95": sorted(inter_jumps)[int(0.95 * (len(inter_jumps) - 1))],
            "max": max(inter_jumps),
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--index-map", type=Path, default=None,
                        help='JSON {"layer,expert": record_index}')
    parser.add_argument("--record-bytes", type=int, default=RECORD_BYTES_DEFAULT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = [r for r in load_journal(args.journal)
            if r.get("phase", "decode") == "decode"]
    index_map = (json.loads(args.index_map.read_text(encoding="utf-8"))
                 if args.index_map else None)
    metrics = pattern_metrics(rows, args.record_bytes, index_map)
    text = json.dumps(metrics, indent=2)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
