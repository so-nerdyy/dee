#!/usr/bin/env python3
"""Prototype A: hash-layer L0-L2 early expert staging (isolated, exact).

Causality contract (mirrors deepseek_v4_layer_common.router_select, hash
branch): expert IDs = tid2eid[input_ids] — a pure table lookup depending
ONLY on (routing table, token id). Routing WEIGHTS need hidden states and
are NEVER produced or consumed here; consume() refuses until
mark_weights_ready(layer) is called with hidden-state evidence.

Legal early operations (IDs alone suffice): host-pack lookup, SSD read,
packed-buffer preparation, H2D of packed bytes, VRAM residency insertion.
Forbidden: routed compute before true weights exist (no compute path
exists in this module at all).

Model-faithfulness contract (per-layer tables): official DeepSeek-V4
creates tid2eid as a parameter of EACH Gate instance, so hash layers have
layer-specific tables. The contract is explicitly:

    ids_L0 = tid2eid_L0[input_id]
    ids_L1 = tid2eid_L1[input_id]
    ids_L2 = tid2eid_L2[input_id]

The six IDs are NEVER assumed identical across layers. Expert identity is
(layer, expert_id): expert #42 in L0 and #42 in L1 are different records.
Duplicate suppression operates on (layer, expert_id) tuples only.

Staleness: expert records are immutable, so a staged-but-unused expert is
merely wasted bandwidth, never wrong. invalidate() cancels pending submits
at token boundaries for accounting; resident bytes stay valid.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field

HASH_LAYERS = (0, 1, 2)
TOPK_DEFAULT = 6
EXPERT_BYTES_DEFAULT = 13369344  # 12.75 MiB packed MXFP4 record
TID2EID_NAME = "layers.{layer}.ffn.gate.tid2eid"  # I64 [vocab, topk]


def resolve_hash_ids(tid2eid: dict[int, list[int]], input_id: int,
                     topk: int = TOPK_DEFAULT) -> list[int]:
    """One layer's IDs: tid2eid_L[input_id]. Pure table lookup depending on
    (that layer's table, token id) only — never hidden, never another layer.

    Raises KeyError for unknown token ids (fail-closed: no guessed routes).
    """
    ids = tid2eid[input_id]
    if len(ids) < topk:
        raise ValueError(f"table row for token {input_id} has "
                         f"{len(ids)} ids, need topk={topk}")
    return list(ids[:topk])


def resolve_all_hash_layers(tables: dict[int, dict[int, list[int]]],
                            input_id: int,
                            topk: int = TOPK_DEFAULT) -> dict[int, list[int]]:
    """ids_L = tables[L][input_id] for each hash layer. Every layer table
    independently required: a missing layer table raises (fail-closed),
    never falls back to another layer's row."""
    out: dict[int, list[int]] = {}
    for layer in HASH_LAYERS:
        if layer not in tables:
            raise KeyError(f"missing tid2eid table for hash layer {layer}; "
                           "refusing to reuse another layer's IDs")
        out[layer] = resolve_hash_ids(tables[layer], input_id, topk)
    return out


def load_tid2eid_rows(shard_path: str, input_id: int,
                      layers: tuple[int, ...] = HASH_LAYERS,
                      topk: int = TOPK_DEFAULT) -> dict:
    """Real-fixture loader: header-only safetensors parse, then raw reads of
    layers.{L}.ffn.gate.tid2eid (I64 [vocab, topk]) and struct decode of the
    input_id row per layer. Returns per-layer {tensor, sha256, ids,
    duplicate_within_layer} plus cross-layer numeric-ID comparison. Needs
    only the shard(s) holding the three small tables, never the checkpoint.
    """
    with open(shard_path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        result: dict = {"shard": shard_path, "input_id": input_id,
                        "layers": {}}
        for layer in layers:
            name = TID2EID_NAME.format(layer=layer)
            entry = header.get(name)
            if not entry:
                raise KeyError(f"tensor {name} not in {shard_path}")
            begin, end = entry["data_offsets"]
            row_off = data_start + begin + input_id * topk * 8
            if not (data_start + begin <= row_off
                    and row_off + topk * 8 <= data_start + end):
                raise ValueError(f"input_id {input_id} out of range for {name}")
            fh.seek(row_off)
            raw = fh.read(topk * 8)
            ids = list(struct.unpack("<" + "q" * topk, raw))
            result["layers"][layer] = {
                "tensor": name,
                "table_sha256": hashlib.sha256(
                    f"{name}:{begin}:{end}".encode()).hexdigest()[:16],
                "ids": ids,
                "duplicate_within_layer": len(set(ids)) != len(ids),
            }
        seen: dict[int, list[int]] = {}
        for layer, info in result["layers"].items():
            for e in info["ids"]:
                seen.setdefault(e, []).append(layer)
        result["numeric_id_shared_across_layers"] = {
            str(e): ls for e, ls in seen.items() if len(ls) > 1}
        return result


@dataclass
class TelemetryRow:
    layer: int
    expert: int
    id_known_t: float
    table_tag: str = ""  # identity of the layer table that produced the ID
    stage_submit_t: float | None = None
    stage_complete_t: float | None = None
    weight_ready_t: float | None = None
    compute_needed_t: float | None = None
    ready_before_demand: bool | None = None
    suppressed_as_duplicate: bool = False
    cancelled: bool = False


@dataclass
class HashStagePlanner:
    topk: int = TOPK_DEFAULT
    expert_bytes: int = EXPERT_BYTES_DEFAULT
    h2d_ms_per_expert: float = 1.0  # ASSUMED transfer model; calibrate live
    resident: set = field(default_factory=set)      # (layer, expert) in VRAM
    hostpacked: set = field(default_factory=set)    # (layer, expert) pinned
    weights_ready: set = field(default_factory=set)  # layers w/ hidden-state weights
    telemetry: list = field(default_factory=list)   # TelemetryRow
    pending: dict = field(default_factory=dict)     # (l,e) -> submit_t
    now_ms: float = 0.0

    def _key(self, layer: int, expert: int) -> tuple[int, int]:
        return (layer, expert)

    def token_start(self, tables: dict[int, dict[int, list[int]]],
                    input_id: int, t_ms: float = 0.0,
                    table_tags: dict[int, str] | None = None) -> list[tuple[int, int]]:
        """Resolve each hash layer's OWN six IDs and submit staging for L0-L2
        misses. Returns the submitted [(layer, expert)] list.
        Duplicate-suppresses (layer, expert) records already resident,
        host-packed, or pending — the same numeric expert ID in different
        layers is a DIFFERENT record and is never suppressed. Score layers
        (L3+) are never touched: no official IDs exist for them at token
        start and none are guessed."""
        self.now_ms = t_ms
        submitted: list[tuple[int, int]] = []
        per_layer = resolve_all_hash_layers(tables, input_id, self.topk)
        for layer in HASH_LAYERS:
            tag = (table_tags or {}).get(layer, f"layer-{layer}-table")
            for expert in per_layer[layer]:
                key = self._key(layer, expert)
                row = TelemetryRow(layer=layer, expert=expert,
                                   id_known_t=t_ms, table_tag=tag)
                if key in self.resident or key in self.hostpacked or key in self.pending:
                    row.suppressed_as_duplicate = True
                    row.stage_submit_t = None
                    self.telemetry.append(row)
                    continue
                row.stage_submit_t = t_ms
                row.stage_complete_t = t_ms + self.h2d_ms_per_expert
                self.pending[key] = t_ms
                self.telemetry.append(row)
                submitted.append(key)
        return submitted

    def complete_due(self, t_ms: float) -> list[tuple[int, int]]:
        """Advance the transfer model: returns newly completed keys."""
        self.now_ms = t_ms
        done = [k for k, s in self.pending.items()
                if s + self.h2d_ms_per_expert <= t_ms]
        return done

    def mark_weights_ready(self, layer: int, t_ms: float,
                           evidence: str) -> None:
        """Record that true routing weights exist for a layer. `evidence`
        must name the hidden-state source (e.g. 'h_0 post-residual'); empty
        evidence raises — weights are never assumed."""
        if not evidence:
            raise ValueError("weights-ready requires hidden-state evidence")
        self.weights_ready.add(layer)
        self.now_ms = t_ms
        for row in self.telemetry:
            if row.layer == layer and row.weight_ready_t is None:
                row.weight_ready_t = t_ms

    def consume(self, layer: int, expert: int, t_ms: float) -> TelemetryRow:
        """Demand one expert's contribution. Refuses (raises) unless weights
        are ready AND staging completed: routed compute before true weights
        is forbidden, staged or not."""
        if layer not in self.weights_ready:
            raise ValueError(f"layer {layer}: consume before weights ready "
                             "(routed compute before true weights forbidden)")
        key = self._key(layer, expert)
        if key in self.pending:
            submit = self.pending.pop(key)
            if submit + self.h2d_ms_per_expert > t_ms:
                raise ValueError(f"expert {key}: staging incomplete at demand")
            self.hostpacked.add(key)
        if key not in self.resident and key not in self.hostpacked:
            raise ValueError(f"expert {key}: never staged (score-layer path "
                             "must stage at route-known; see sim.py)")
        for row in reversed(self.telemetry):
            if row.layer == layer and row.expert == expert and not row.cancelled:
                row.compute_needed_t = t_ms
                row.ready_before_demand = (
                    row.stage_complete_t is not None
                    and row.stage_complete_t <= t_ms)
                return row
        raise ValueError(f"expert {key}: no telemetry (IDs were never resolved)")

    def invalidate(self, t_ms: float) -> list[tuple[int, int]]:
        """Token boundary: cancel pending submits (accounting only; resident
        bytes stay valid because records are immutable)."""
        self.now_ms = t_ms
        cancelled = list(self.pending)
        for key in cancelled:
            del self.pending[key]
            for row in reversed(self.telemetry):
                if (row.layer, row.expert) == key and row.compute_needed_t is None:
                    row.cancelled = True
                    break
        self.weights_ready.clear()
        return cancelled

    def bytes_summary(self) -> dict:
        staged = [r for r in self.telemetry if r.stage_submit_t is not None]
        dupes = [r for r in self.telemetry if r.suppressed_as_duplicate]
        return {
            "submitted_reads": len(staged),
            "submitted_bytes": len(staged) * self.expert_bytes,
            "suppressed_duplicates": len(dupes),
            "suppressed_bytes": len(dupes) * self.expert_bytes,
            "max_early_reads": len(HASH_LAYERS) * self.topk,
            "max_early_bytes": len(HASH_LAYERS) * self.topk * self.expert_bytes,
        }
