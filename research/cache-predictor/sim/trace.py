"""Trace loader for DSV4 routed_experts.jsonl journals.

Converts sealed route journals into a canonical demand sequence used by
the offline cache-policy tournament and the cross-layer predictor.

The OFFICIAL router is authoritative: this module only reads what the
runtime actually routed, it never re-routes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Canonical per-record byte size observed in dee4-v3-trace metadata
# (record_bytes = 13,369,344 for the DSV4 flash codec). Kept as a
# per-record default so variable-size codecs work in the future.
DEFAULT_RECORD_BYTES = 13_369_344

DEFAULT_TRACE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "dynamic_expert_eviction" / "dee.cpp" / "tmp"
)


def trace_root() -> Path:
    return Path(os.environ.get("DEE_TRACE_ROOT", str(DEFAULT_TRACE_ROOT)))


@dataclass(frozen=True)
class Demand:
    """One routed expert selection, in execution order.

    layer_scope keys the cache: residency is per (layer, expert) pair,
    exactly like dee's per-layer VRAM expert cache.
    """

    seq: int          # global sequence number (execution order)
    step: int         # forward step (token index)
    layer: int        # transformer layer
    expert: int       # expert id within the layer
    scope: int        # cache key: layer * 100000 + expert
    nbytes: int       # record byte size (13,369,344 today)
    phase: str        # "prefill" | "decode"
    token_row: int    # which token row within the batch requested it
    rank: int         # top-k rank of this expert for that token row


def load_trace(path: Path) -> list[Demand]:
    """Read routed_experts.jsonl into an ordered demand list (read-only)."""
    demands: list[Demand] = []
    seq = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = rec["forward_step"]
            layer = rec["layer"]
            phase = rec.get("phase", "?")
            start_pos = rec.get("start_pos", 0)
            ids = rec["expert_ids_rank_order"]
            for token_row, row in enumerate(ids):
                for rank, expert in enumerate(row):
                    demands.append(Demand(
                        seq=seq,
                        step=step,
                        layer=layer,
                        expert=expert,
                        scope=layer * 100_000 + expert,
                        nbytes=DEFAULT_RECORD_BYTES,
                        phase=phase,
                        token_row=start_pos + token_row,
                        rank=rank,
                    ))
                    seq += 1
    return demands


def load_default_trace() -> tuple[list[Demand], Path]:
    """Load the newest sealed DSV4 trace (v65 terminal fetch)."""
    path = (trace_root()
            / "v65-terminal-fetch-20260903T0202Z"
            / "routed_experts.jsonl")
    return load_trace(path), path
