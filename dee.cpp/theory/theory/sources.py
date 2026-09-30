"""Read-only loaders for every in-repo artifact the theory layer consumes.

Each loader returns plain dicts/arrays and, where useful, emits the exact
source line that a constant cites.  No loader mutates anything.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import paths
from .constants import CONSTS, Const


# ==========================================================================
# Trace A: Ornith milestone-2.5 50k-event router trace (multi-model, 4 layers)
# ==========================================================================
@dataclass
class RouterStream:
    """Per-run authoritative routing stream (layer-call sequences)."""
    run_id: str
    n_layers: int
    experts_per_layer: int
    topk: int
    n_calls: int                       # layer-calls
    n_slots: int                       # rank slots (events)
    keys: List[Tuple[int, int]]        # (token_index, layer) in emission order
    sets: Dict[Tuple[int, int], List[int]]   # dedup expert ids per layer-call, rank order
    raw: Dict[Tuple[int, int], List[List[int]]]  # per-row top-k lists (rows = batch positions)
    phases: Dict[Tuple[int, int], str] = field(default_factory=dict)

    def stream(self) -> List[Tuple[int, int]]:
        """Deduplicated cache-request stream in emission order."""
        out: List[Tuple[int, int]] = []
        for k in self.keys:
            out.extend((k[1], e) for e in self.sets[k])
        return out

    def raw_stream(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for k in self.keys:
            for row in self.raw[k]:
                out.extend((k[1], e) for e in row)
        return out


# Ornith model geometry (milestone-2.5 trace): 40 MoE layers x 256 experts,
# top-8 routed + shared.  Verified from the trace itself (route_selection
# rows carry routing_rank 0..7 over logical_layer 0..39, expert_id 0..255).
ORNITH_LAYERS = 40
ORNITH_EXPERTS = 256
ORNITH_TOPK = 8


def load_ornith_streams() -> Dict[str, RouterStream]:
    """Rebuild authoritative per-run routing streams from route_selection rows.

    Only ``event_type == route_selection`` rows are authoritative routing
    (per the AGENTS.md exactness contract: authoritative routing = the
    router's own selection).  ``expert_request`` rows describe cache
    behaviour and are consumed separately by temporal.py / cache.py.

    Token index = (step, sequence_token) ordered as emitted; rows within a
    token-layer call are grouped by routing_rank into one top-k row.
    """
    rows: Dict[str, List[dict]] = {}
    with gzip.open(paths.TRACE_JSONL_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("event_type") != "route_selection":
                continue
            rows.setdefault(d["run_id"], []).append(d)

    streams: Dict[str, RouterStream] = {}
    for run_id, rs in rows.items():
        # token index: order of first appearance of each (step, sequence_token)
        tok_index: Dict[Tuple[int, int], int] = {}
        calls: Dict[Tuple[int, int], List[List[int]]] = {}
        by_rank: Dict[Tuple[int, int], Dict[int, int]] = {}
        order: List[Tuple[int, int]] = []
        phases: Dict[Tuple[int, int], str] = {}
        for d in rs:
            tok = (int(d["step"]), int(d["sequence_token"]))
            if tok not in tok_index:
                tok_index[tok] = len(tok_index)
            t = tok_index[tok]
            layer = int(d["layer"])
            k = (t, layer)
            if k not in calls:
                calls[k] = []
                by_rank[k] = {}
                order.append(k)
                phases[k] = d.get("phase", "")
            by_rank[k][int(d["routing_rank"])] = int(d["expert"])
        for k, ranks in by_rank.items():
            calls[k] = [[ranks[r] for r in sorted(ranks)]]
        sets = {}
        for k, rowsk in calls.items():
            seq: List[int] = []
            for row in rowsk:
                for e in row:
                    if e not in seq:
                        seq.append(e)
            sets[k] = seq
        streams[run_id] = RouterStream(
            run_id=run_id,
            n_layers=ORNITH_LAYERS,
            experts_per_layer=ORNITH_EXPERTS,
            topk=ORNITH_TOPK,
            n_calls=len(order),
            n_slots=sum(1 for k in order for row in calls[k] for _e in row),
            keys=order,
            sets=sets,
            raw=calls,
            phases=phases,
        )
    return streams


def load_ornith_cache_rows() -> List[dict]:
    """expert_request rows (cache hits/misses + reuse distances)."""
    out = []
    with gzip.open(paths.TRACE_JSONL_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("event_type") == "expert_request":
                out.append(d)
    return out


# ==========================================================================
# Trace B: sealed DSv4-Flash 16-token journal (43 layers x 256 experts)
# ==========================================================================
def load_sealed_stream() -> RouterStream:
    """Per-layer-call deduplicated stream over routed_experts.jsonl."""
    keys: List[Tuple[int, int]] = []
    raw: Dict[Tuple[int, int], List[List[int]]] = {}
    phases: Dict[Tuple[int, int], str] = {}
    with open(paths.SEALED_ROUTED, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            k = (int(d["forward_step"]), int(d["layer"]))
            if k not in raw:
                raw[k] = []
                keys.append(k)
                phases[k] = d.get("phase", "")
            raw[k].extend([int(e) for e in row] for row in d["expert_ids_rank_order"])
    sets = {}
    for k, rows in raw.items():
        seq: List[int] = []
        for row in rows:
            for e in row:
                if e not in seq:
                    seq.append(e)
        sets[k] = seq
    return RouterStream(
        run_id="fill-live-t4x2-20260909",
        n_layers=43,
        experts_per_layer=256,
        topk=6,
        n_calls=len(keys),
        n_slots=sum(1 for k in keys for row in raw[k] for _e in row),
        keys=keys,
        sets=sets,
        raw=raw,
        phases=phases,
    )


def load_sealed_result() -> dict:
    with open(paths.SEALED_RESULT, encoding="utf-8") as f:
        return json.load(f)


def load_sealed_profile() -> dict:
    with open(paths.SEALED_PROFILE, encoding="utf-8") as f:
        return json.load(f)


# ==========================================================================
# Store specs (Phase-3 tooling) + DSv4 geometry
# ==========================================================================
@dataclass(frozen=True)
class Store:
    name: str
    label: str
    layers: int                 # cached MoE buckets
    experts_per_layer: int
    record_bytes: int
    topk: int
    source: str

    @property
    def universe_records(self) -> int:
        return self.layers * self.experts_per_layer

    @property
    def universe_bytes(self) -> int:
        return self.universe_records * self.record_bytes


def load_stores() -> Dict[str, Store]:
    from .constants import (D4_EXPERTS, D4_LAYERS, D4_RECORD, D4_TOPK,
                            D4_UNIVERSE_RECORDS)

    stores = {
        "dsv4_flash": Store(
            "dsv4_flash", "DeepSeek-V4-Flash-0731",
            46, 256, 13369344, 6,
            "geometry: dee.cpp/experiments/route_pipeline/fill-live-t4x2-20260909/result.json "
            "engine_config + AGENTS.md CPU-4/10 (46 buckets incl. 3 MTP = 11,776 records)"),
    }
    for key, p in paths.SPECS.items():
        s = json.loads(p.read_text(encoding="utf-8"))
        n_layers = int(s["n_layers"]) + int(s.get("n_mtp_buckets", 0))
        stores[key] = Store(
            key,
            s["model"],
            n_layers,
            int(s["experts_per_layer"]),
            int(s["record_bytes"]),
            6,
            f"{paths.rel(p)} (revision {s['revision']})",
        )
    return stores


# ==========================================================================
# Provenance ledger
# ==========================================================================
def write_provenance() -> None:
    from .util import write_csv

    rows = []
    for name, c in CONSTS.items():
        val = c.value
        if isinstance(val, (dict, list, tuple)):
            val = json.dumps(val, sort_keys=True)
        rows.append({
            "name": name,
            "value": val,
            "unit": c.unit,
            "tag": c.tag,
            "source": c.source,
            "note": c.note,
            "uncertainty": "" if c.uncertainty is None else f"{c.uncertainty[0]}..{c.uncertainty[1]}",
        })
    write_csv("provenance.csv", rows,
              ["name", "value", "unit", "tag", "source", "note", "uncertainty"])
