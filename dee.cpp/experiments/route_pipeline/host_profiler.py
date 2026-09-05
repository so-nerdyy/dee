#!/usr/bin/env python3
"""Host/sync profiler schema, closure accounting, and decision rules.

Pure-Python companion to the C++ HostSpan records (profiling.h) and the
DEE_HOST_PROFILE rows (deepseek_v4_layer_candidate.py). No torch/CUDA needed.

Provenance labels: HOST_WALL, CUDA_EVENT, COUNTER, DERIVED, UNKNOWN, NESTED.
Closure rule: accounted = sum of non-nested, non-null spans + device work
NOT already covered by a host wait on it; NESTED spans are never summed;
unknown = max(0, wall - accounted); closure_fraction = accounted / wall.
UNKNOWN is carried, never force-filled: 85% honest closure beats fake 100%.
"""

from __future__ import annotations

PROVENANCE = ("HOST_WALL", "CUDA_EVENT", "COUNTER", "DERIVED", "UNKNOWN", "NESTED")

# Span fields of the per token/layer record (Phase I schema).
SPAN_FIELDS = (
    "route_d2h_host_wait_ms",
    "native_call_wall_ms",
    "source_wait_ms",
    "fill_wait_ms",
    "h2d_ms",
    "decode_ms",
    "expert_compute_ms",
    "native_output_sync_wait_ms",
    "shared_expert_ms",
    "combine_ms",
    "orchestration_ms",
    "handoff_ms",
)
REQUIRED_META = ("token", "layer", "device")

# Spans nested inside native_call_wall_ms (excluded from closure sums).
NESTED_IN_NATIVE_CALL = frozenset((
    "source_wait_ms",
    "fill_wait_ms",
    "h2d_ms",
    "decode_ms",
    "expert_compute_ms",
    "native_output_sync_wait_ms",
))


class SchemaError(ValueError):
    pass


def validate_record(record: dict) -> dict:
    """Fail-closed validation. Returns a normalized copy. Malformed input
    raises SchemaError (never silently repaired)."""
    if not isinstance(record, dict):
        raise SchemaError("record must be an object")
    for key in REQUIRED_META:
        if key not in record:
            raise SchemaError(f"missing required field: {key}")
    out = {k: record[k] for k in REQUIRED_META}
    prov = record.get("provenance", {})
    if not isinstance(prov, dict):
        raise SchemaError("provenance must be an object")
    for field in SPAN_FIELDS:
        value = record.get(field)
        if value is None:
            out[field] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(f"{field}: must be a number or null")
        if value < 0:
            raise SchemaError(f"{field}: negative time")
        out[field] = float(value)
    out["provenance"] = {}
    for field in SPAN_FIELDS:
        label = prov.get(field, "UNKNOWN" if out[field] is None else "HOST_WALL")
        if label not in PROVENANCE:
            raise SchemaError(f"{field}: bad provenance {label!r}")
        if out[field] is None and label not in ("UNKNOWN", "NESTED"):
            raise SchemaError(f"{field}: null value needs UNKNOWN/NESTED")
        out["provenance"][field] = label
    wall = record.get("layer_wall_ms")
    if wall is not None:
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall < 0:
            raise SchemaError("layer_wall_ms: must be a non-negative number or null")
        out["layer_wall_ms"] = float(wall)
    else:
        out["layer_wall_ms"] = None
    return out


def _is_nested(field: str, provenance: dict) -> bool:
    return provenance.get(field) == "NESTED" or field in NESTED_IN_NATIVE_CALL


def compute_closure(records: list[dict], decode_wall_ms: float | None = None) -> dict:
    """Reconcile decode wall against attributable spans.

    accounted = route_d2h + native_call_wall (inclusive parent) + shared +
    combine + orchestration + handoff. Children nested inside native_call are
    reported under by_field_nested_ms and NEVER added (they are already
    inside the parent wall). Device-only work overlapped by a host wait is
    not double-counted: native_call_wall is host wall covering it.
    unknown = max(0, wall - accounted); closure = accounted / wall.
    """
    if decode_wall_ms is not None and (decode_wall_ms < 0):
        raise SchemaError("decode_wall_ms must be non-negative")
    TOP_LEVEL = ("route_d2h_host_wait_ms", "native_call_wall_ms",
                 "shared_expert_ms", "combine_ms", "orchestration_ms", "handoff_ms")
    accounted = 0.0
    by_field: dict[str, float] = {}
    nested_detail: dict[str, float] = {}
    unknown_fields = 0
    for record in records:
        rec = validate_record(record)
        for field in TOP_LEVEL:
            value = rec[field]
            if value is None:
                unknown_fields += 1
                continue
            accounted += value
            by_field[field] = by_field.get(field, 0.0) + value
        for field in NESTED_IN_NATIVE_CALL:
            value = rec[field]
            if value is None:
                continue
            nested_detail[field] = nested_detail.get(field, 0.0) + value
    wall = decode_wall_ms
    if wall is None:
        wall = accounted  # no independent wall: closure undefined, report 1.0 iff complete
    unknown = max(0.0, wall - accounted)
    closure = (accounted / wall) if wall > 0 else 1.0
    return {"accounted_ms": round(accounted, 6),
            "unknown_ms": round(unknown, 6),
            "closure_fraction": round(min(closure, 1.0), 6),
            "wall_ms": round(wall, 6),
            "by_field_ms": {k: round(v, 6) for k, v in by_field.items()},
            "by_field_nested_ms": {k: round(v, 6) for k, v in nested_detail.items()},
            "unmeasured_top_level_fields": unknown_fields,
            "nested_excluded": sorted(NESTED_IN_NATIVE_CALL),
            "forced": False}


DEFAULT_THRESHOLDS = {
    # All fractions refer to measured (non-simulated) profile data.
    "shared_material_fraction": 0.05,   # shared_expert_ms / wall
    "sync_material_fraction": 0.05,     # native_output_sync_wait_ms / wall
    "d2h_material_fraction": 0.02,      # route_d2h excess over copy floor
    "min_closure_fraction": 0.85,       # closure required before promotion
}


def evaluate_decisions(closure: dict, thresholds: dict | None = None) -> dict:
    """Automatic promotion rules evaluated on REAL profile data. Every
    candidate defaults to HOLD; promotion requires its evidence gate. No
    winner is selected here — the function only maps measurements to ranks."""
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    wall = closure.get("wall_ms") or 0.0
    by_field = closure.get("by_field_ms", {})
    nested = closure.get("by_field_nested_ms", {})
    out: dict = {"closure_ok": closure.get("closure_fraction", 0.0) >= th["min_closure_fraction"],
                 "candidates": {}}

    def frac(field: str) -> float | None:
        return (by_field[field] / wall) if wall > 0 and field in by_field else None

    def nested_frac(field: str) -> float | None:
        return (nested[field] / wall) if wall > 0 and field in nested else None

    shared = frac("shared_expert_ms")
    out["candidates"]["SHARED_OVERLAP"] = {
        "rank": ("PROMOTE" if (out["closure_ok"] and shared is not None
                               and shared >= th["shared_material_fraction"])
                 else "HOLD"),
        "reason": f"shared_expert fraction={shared} (need >={th['shared_material_fraction']} with closure>={th['min_closure_fraction']})",
    }
    sync = nested_frac("native_output_sync_wait_ms")
    out["candidates"]["EVENT_HANDOFF"] = {
        "rank": ("PROMOTE" if (out["closure_ok"] and sync is not None
                               and sync >= th["sync_material_fraction"])
                 else "HOLD"),
        "reason": f"output-sync fraction={sync} (need >={th['sync_material_fraction']})",
    }
    d2h = frac("route_d2h_host_wait_ms")
    out["candidates"]["ROUTE_D2H_NARROW"] = {
        "rank": ("PROMOTE" if (out["closure_ok"] and d2h is not None
                               and d2h >= th["d2h_material_fraction"])
                 else "HOLD"),
        "reason": f"route-d2h fraction={d2h} (need >={th['d2h_material_fraction']}); copy floor still applies",
    }
    out["candidates"]["HASH_EARLY_STAGING"] = {
        "rank": "PROMOTE_IF_MECHANICS",
        "reason": "legality closed host-side; promote on runner case-A hidden ms, not on closure fractions",
    }
    out["candidates"]["C1_C2"] = {
        "rank": "HOLD",
        "reason": "return only when decode/materialization or expert compute is measured on the critical path after storage improves",
    }
    return out
