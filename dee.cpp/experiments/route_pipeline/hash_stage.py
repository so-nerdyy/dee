#!/usr/bin/env python3
"""Prototype A: hash-layer L0-L2 early expert staging (isolated, exact).

Causality contract (mirrors deepseek_v4_layer_common.router_select, hash
branch): expert IDs = tid2eid[input_ids] — a pure table lookup depending
ONLY on (routing table, token id). Routing WEIGHTS need hidden states and
are NEVER produced or consumed here; consume() refuses until
mark_weights_ready(layer) is called with hidden-state evidence.

Legal early operations (IDs alone suffice): host-pack lookup, SSD read,
packed-buffer preparation, H2D of packed bytes, VRAM residency insertion.
Forbidden: routed contribution before true weights exist (no compute path
exists in this module at all).

Staleness: expert records are immutable, so a staged-but-unused expert is
merely wasted bandwidth, never wrong. invalidate() cancels pending submits
at token boundaries for accounting; resident bytes stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HASH_LAYERS = (0, 1, 2)
TOPK_DEFAULT = 6
EXPERT_BYTES_DEFAULT = 13369344  # 12.75 MiB packed MXFP4 record


def resolve_hash_ids(tid2eid: dict[int, list[int]], input_id: int,
                     topk: int = TOPK_DEFAULT) -> list[int]:
    """Pure table lookup. Depends on (table, token id) only — never hidden.

    Raises KeyError for unknown token ids (fail-closed: no guessed routes).
    """
    ids = tid2eid[input_id]
    if len(ids) < topk:
        raise ValueError(f"table row for token {input_id} has "
                         f"{len(ids)} ids, need topk={topk}")
    return list(ids[:topk])


@dataclass
class TelemetryRow:
    layer: int
    expert: int
    id_known_t: float
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

    def token_start(self, tid2eid: dict[int, list[int]], input_id: int,
                    t_ms: float = 0.0) -> list[tuple[int, int]]:
        """Resolve hash IDs and submit staging for L0-L2 misses. Returns the
        submitted [(layer, expert)] list. Duplicate-suppresses resident and
        host-packed experts (no redundant staging)."""
        self.now_ms = t_ms
        submitted: list[tuple[int, int]] = []
        ids = resolve_hash_ids(tid2eid, input_id, self.topk)
        for layer in HASH_LAYERS:
            for expert in ids:
                key = self._key(layer, expert)
                row = TelemetryRow(layer=layer, expert=expert, id_known_t=t_ms)
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
