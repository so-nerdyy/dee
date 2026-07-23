#!/usr/bin/env python3
"""Analyze Milestone 2.5 per-expert JSONL traces without inventing data.

The preferred input is one JSON object per line with ``event_type`` equal to
``expert_request``, ``expert_transfer``, or ``expert_eviction``.  A request may
embed its transfer and eviction information.  The reader also accepts the
field names emitted by dee.cpp's older ``RequestTraceRecord`` JSON (for
example, ``logical_layer``, ``resolved_shard_layer``, ``expert``, and
``kind``).  Optional fields may be absent; every partial calculation is paired
with coverage metadata and a limitation is emitted for evidence that cannot be
established from the input.

Preferred request fields:

    run_id, event_index, token_step, token_phase, logical_layer,
    resolved_shard_layer, expert_id, routing_rank, routing_weight,
    source_checkpoint_shard, expert_bytes, cache_state_before, cache_result,
    gpu_destination, transfer_id, duplicated_on_both_gpus

Preferred transfer fields:

    run_id, transfer_id, token_step, token_phase, logical_layer,
    resolved_shard_layer, expert_id, gpu_destination, direction, bytes,
    transfer_start_ns, transfer_completion_ns

Preferred eviction fields:

    run_id, token_step, logical_layer, resolved_shard_layer, expert_id,
    gpu_destination, eviction_reason

Only an actual transfer byte field contributes to H2D totals.  ``expert_bytes``
describes the requested payload but is never assumed to have been transferred.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
UNKNOWN_RUN = "__default__"
REQUEST_TYPES = {
    "expert_request", "request", "cache_request", "expert_cache_request",
    "expert_lookup", "cache_hit", "cache_miss",
}
TRANSFER_TYPES = {
    "expert_transfer", "tensor_transfer", "transfer", "h2d", "host_to_device", "d2d",
    "device_to_device", "cpu_copy", "mmap_to_pinned", "h2d_transfer",
    "d2d_transfer", "d2h_transfer",
}
EVICTION_TYPES = {"expert_eviction", "eviction", "cache_eviction", "evict"}


def _pick(record: Mapping[str, Any], names: Sequence[str], sections: Sequence[str] = ()) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    for section in sections:
        nested = record.get(section)
        if isinstance(nested, Mapping):
            for name in names:
                if name in nested and nested[name] is not None:
                    return nested[name]
    return None


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_integer(value: Any) -> int | None:
    result = _integer(value)
    return result if result is not None and result >= 0 else None


def _run_id(record: Mapping[str, Any]) -> str:
    value = _pick(record, ("run_id", "request_id", "sequence_id", "generation_id"))
    return str(value) if value is not None else UNKNOWN_RUN


def _gpu(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return f"cuda:{value}"
    text = str(value).strip().lower()
    if not text:
        return None
    if text.isdigit():
        return f"cuda:{text}"
    if text.startswith("gpu:") and text[4:].isdigit():
        return f"cuda:{text[4:]}"
    if text.startswith("gpu") and text[3:].isdigit():
        return f"cuda:{text[3:]}"
    if text.startswith("cuda") and ":" not in text and text[4:].isdigit():
        return f"cuda:{text[4:]}"
    return text


def _direction(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "h2d": "h2d", "h2d_transfer": "h2d", "htod": "h2d", "host_to_device": "h2d",
        "cpu_to_gpu": "h2d", "d2d": "d2d", "dtod": "d2d",
        "device_to_device": "d2d", "gpu_to_gpu": "d2d", "d2d_transfer": "d2d",
        "d2h": "d2h", "d2h_transfer": "d2h", "dtoh": "d2h", "device_to_host": "d2h",
        "gpu_to_cpu": "d2h", "cpu_copy": "cpu_copy",
        "pageable_to_pinned": "pageable_to_pinned",
        "mmap_to_pinned": "mmap_to_pinned",
    }
    return aliases.get(text, text or None)


def _event_type(record: Mapping[str, Any]) -> str:
    value = _pick(record, ("event_type", "type", "event", "record_type"))
    if value is not None:
        return str(value).strip().lower().replace("-", "_")
    kind = str(record.get("kind", "")).strip().lower().replace("-", "_")
    if kind in TRANSFER_TYPES or kind in EVICTION_TYPES or kind in REQUEST_TYPES:
        return kind
    if _pick(record, ("direction", "transfer_direction", "copy_kind")) is not None:
        return "expert_transfer"
    if _pick(record, ("evicted_expert", "evicted_expert_id")) is not None:
        return "expert_request"
    if _pick(record, ("expert", "expert_id")) is not None and _pick(
        record, ("logical_layer", "layer", "layer_idx", "resolved_shard_layer")
    ) is not None:
        return "expert_request"
    return "unknown"


def _cache_result(record: Mapping[str, Any]) -> str | None:
    explicit_type = str(_pick(record, ("event_type", "type", "event", "record_type")) or "")
    explicit_type = explicit_type.strip().lower().replace("-", "_")
    if explicit_type == "cache_hit":
        return "hit"
    if explicit_type == "cache_miss":
        return "miss"
    hit = _pick(record, ("hit", "cache_hit"), ("cache",))
    if isinstance(hit, bool):
        return "hit" if hit else "miss"
    miss = _pick(record, ("miss", "cache_miss"), ("cache",))
    if isinstance(miss, bool):
        return "miss" if miss else "hit"
    value = _pick(
        record,
        ("cache_result", "cache_kind", "request_kind", "kind", "result"),
        ("cache",),
    )
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    if text in {"resident", "resident_hit", "inflight", "inflight_hit", "hit", "cache_hit"}:
        return "hit"
    if text in {"cold", "cold_load", "load", "miss", "cache_miss"}:
        return "miss"
    return None


def _duration_seconds(record: Mapping[str, Any]) -> float | None:
    sections = ("transfer", "timing")
    for names, scale in (
        (("transfer_duration_ns", "duration_ns"), 1e-9),
        (("transfer_duration_us", "duration_us"), 1e-6),
        (("transfer_duration_ms", "duration_ms"), 1e-3),
        (("transfer_duration_seconds", "duration_seconds", "duration_sec"), 1.0),
    ):
        value = _number(_pick(record, names, sections))
        if value is not None and value >= 0:
            return value * scale
    for starts, ends, scale in (
        (("transfer_start_ns", "start_ns"), ("transfer_completion_ns", "transfer_end_ns", "end_ns"), 1e-9),
        (("transfer_start_us", "start_us"), ("transfer_completion_us", "transfer_end_us", "end_us"), 1e-6),
        (("transfer_start_ms", "start_ms"), ("transfer_completion_ms", "transfer_end_ms", "end_ms"), 1e-3),
        (("transfer_start_seconds", "start_seconds"), ("transfer_completion_seconds", "transfer_end_seconds", "end_seconds"), 1.0),
    ):
        start = _number(_pick(record, starts, sections))
        end = _number(_pick(record, ends, sections))
        if start is not None and end is not None and end >= start:
            return (end - start) * scale
    return None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _distribution(values: Iterable[float | int]) -> dict[str, Any]:
    materialized = list(values)
    if not materialized:
        return {
            "count": 0, "minimum": None, "mean": None, "median": None,
            "p95": None, "maximum": None,
        }
    return {
        "count": len(materialized),
        "minimum": min(materialized),
        "mean": statistics.fmean(materialized),
        "median": _percentile(materialized, 0.5),
        "p95": _percentile(materialized, 0.95),
        "maximum": max(materialized),
    }


def _stable(value: Any) -> tuple[str, str]:
    return (type(value).__name__, "" if value is None else str(value))


def _identity(record: Mapping[str, Any]) -> tuple[Any, ...] | None:
    expert = record.get("expert_id")
    layer = record.get("physical_layer")
    if expert is None or layer is None:
        return None
    return (record["run_id"], layer, expert)


def _cache_identity(record: Mapping[str, Any]) -> tuple[Any, ...] | None:
    source = _identity(record)
    return (*source, record.get("gpu_destination")) if source is not None else None


def _identity_json(key: tuple[Any, ...]) -> dict[str, Any]:
    return {"run_id": key[0], "physical_layer": key[1], "expert_id": key[2]}


def _normalize_common(raw: Mapping[str, Any], line: int) -> dict[str, Any]:
    explicit_logical = _integer(_pick(raw, ("logical_layer", "layer", "layer_idx"), ("source",)))
    resolved = _integer(
        _pick(raw, ("resolved_shard_layer", "resolved_layer", "physical_layer", "source_layer"), ("source",))
    )
    logical = explicit_logical if explicit_logical is not None else resolved
    return {
        "line": line,
        "run_id": _run_id(raw),
        "event_index": _integer(_pick(raw, ("event_index", "sequence_index", "request_index", "index"))),
        "token_step": _integer(_pick(raw, ("token_step", "token", "token_index", "step"))),
        "token_phase": _pick(raw, ("token_phase", "generation_phase", "phase")),
        "logical_layer": logical,
        "logical_layer_explicit": explicit_logical is not None,
        "physical_layer": resolved if resolved is not None else logical,
        "physical_layer_explicit": resolved is not None,
        "expert_id": _integer(_pick(raw, ("expert_id", "expert"), ("source",))),
        "gpu_destination": _gpu(
            _pick(raw, ("gpu_destination", "destination_gpu", "gpu", "device", "destination"), ("transfer", "cache"))
        ),
        "source_checkpoint_shard": _pick(
            raw, ("source_checkpoint_shard", "checkpoint_shard", "source_shard", "shard"), ("source",)
        ),
    }


def _normalize_request(raw: Mapping[str, Any], line: int) -> dict[str, Any] | None:
    item = _normalize_common(raw, line)
    if item["logical_layer"] is None or item["expert_id"] is None:
        return None
    item.update({
        "routing_rank": _integer(_pick(raw, ("routing_rank", "rank", "topk_rank"), ("routing",))),
        "routing_weight": _number(_pick(raw, ("routing_weight", "weight"), ("routing",))),
        "expert_bytes": _nonnegative_integer(
            _pick(raw, ("expert_bytes", "expert_byte_size", "payload_bytes"), ("source",))
        ),
        "cache_state_before": _pick(raw, ("cache_state_before", "cache_before"), ("cache",)),
        "cache_result": _cache_result(raw),
        "transfer_id": _pick(raw, ("transfer_id",), ("transfer",)),
        "duplicated_on_both_gpus": _pick(
            raw, ("duplicated_on_both_gpus", "duplicate_on_both_gpus"), ("cache",)
        ),
        "reported_previous_reuse_distance": _integer(
            _pick(raw, ("reuse_distance", "previous_reuse_distance"))
        ),
        "reported_distinct_reuse_distance": _integer(
            _pick(raw, ("distinct_reuse_distance", "previous_distinct_reuse_distance"))
        ),
    })
    return item


def _transfer_bytes(raw: Mapping[str, Any], event_type: str) -> tuple[int | None, str | None]:
    h2d = _nonnegative_integer(_pick(raw, ("h2d_bytes",), ("transfer",)))
    if h2d is not None:
        return h2d, "h2d"
    value = _nonnegative_integer(_pick(raw, ("transfer_bytes", "bytes", "byte_count", "size_bytes"), ("transfer",)))
    direction = _direction(_pick(raw, ("direction", "transfer_direction", "copy_kind"), ("transfer",)))
    if direction is None and event_type in {
        "h2d", "h2d_transfer", "host_to_device", "d2d", "d2d_transfer",
        "d2h_transfer", "device_to_device", "cpu_copy", "mmap_to_pinned",
    }:
        direction = _direction(event_type)
    return value, direction


def _normalize_transfer(raw: Mapping[str, Any], line: int, event_type: str,
                        embedded: bool = False) -> dict[str, Any]:
    item = _normalize_common(raw, line)
    byte_count, direction = _transfer_bytes(raw, event_type)
    overlap_ms = _number(_pick(raw, ("overlap_with_compute_ms",), ("transfer",)))
    serialized_ms = _number(_pick(raw, ("serialized_transfer_ms",), ("transfer",)))
    item.update({
        "transfer_id": _pick(raw, ("transfer_id", "copy_id"), ("transfer",)),
        "direction": direction,
        "bytes": byte_count,
        "duration_seconds": _duration_seconds(raw),
        "component": _pick(
            raw, ("component", "transfer_component", "purpose"), ("transfer",)
        ),
        "overlapped_compute": _pick(raw, ("overlapped_compute",), ("transfer",)),
        "overlap_with_compute_seconds": (
            overlap_ms / 1000.0 if overlap_ms is not None else None
        ),
        "serialized_transfer_seconds": (
            serialized_ms / 1000.0 if serialized_ms is not None else None
        ),
        "embedded_in_request": embedded,
    })
    return item


def _embedded_transfer_present(raw: Mapping[str, Any]) -> bool:
    nested = raw.get("transfer")
    return (
        isinstance(nested, Mapping)
        or _pick(raw, ("h2d_bytes", "transfer_bytes", "transfer_duration_ns", "transfer_duration_us",
                    "transfer_duration_ms", "transfer_duration_seconds")) is not None
    )


def _embedded_eviction_present(raw: Mapping[str, Any]) -> bool:
    return isinstance(raw.get("eviction"), Mapping) or _pick(
        raw, ("evicted_expert_id", "evicted_expert")
    ) is not None


def _normalize_eviction(raw: Mapping[str, Any], line: int,
                        parent: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    nested = raw.get("eviction")
    source: Mapping[str, Any] = nested if isinstance(nested, Mapping) else raw
    expert = _integer(_pick(source, ("evicted_expert_id", "evicted_expert", "expert_id", "expert")))
    layer = _integer(_pick(source, (
        "evicted_resolved_shard_layer", "evicted_resolved_layer", "evicted_layer",
        "resolved_shard_layer", "resolved_layer", "physical_layer", "layer",
    )))
    if expert is None or expert < 0 or layer is None or layer < 0:
        return None
    parent = parent or {}
    return {
        "line": line,
        "run_id": str(source.get("run_id", parent.get("run_id", _run_id(raw)))),
        "event_index": _integer(_pick(source, ("event_index", "sequence_index", "index")))
            or parent.get("event_index"),
        "token_step": _integer(_pick(source, ("token_step", "token", "token_index", "step")))
            if _pick(source, ("token_step", "token", "token_index", "step")) is not None
            else parent.get("token_step"),
        "logical_layer": _integer(_pick(source, ("logical_layer", "layer"))),
        "physical_layer": layer,
        "expert_id": expert,
        "gpu_destination": _gpu(
            _pick(source, ("evicted_gpu", "gpu_destination", "destination_gpu", "gpu", "device"))
        ) or parent.get("gpu_destination"),
        "eviction_reason": _pick(source, ("eviction_reason", "reason")),
    }


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    records: list[tuple[int, dict[str, Any]]] = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL record must be an object")
            records.append((line_no, value))
    return records


def _deduplicate_transfers(transfers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    result: list[dict[str, Any]] = []
    by_id: dict[tuple[str, str], int] = {}
    duplicate_count = 0
    for transfer in transfers:
        transfer_id = transfer.get("transfer_id")
        if transfer_id is None:
            result.append(transfer)
            continue
        key = (transfer["run_id"], str(transfer_id))
        previous_index = by_id.get(key)
        if previous_index is None:
            by_id[key] = len(result)
            result.append(transfer)
            continue
        duplicate_count += 1
        previous = result[previous_index]
        # Merge only missing fields.  A standalone record wins over an embedded
        # summary because it normally carries CUDA-event timing.
        if previous.get("embedded_in_request") and not transfer.get("embedded_in_request"):
            primary, secondary = transfer.copy(), previous
        else:
            primary, secondary = previous.copy(), transfer
        for name, value in secondary.items():
            if primary.get(name) is None and value is not None:
                primary[name] = value
        result[previous_index] = primary
    return result, duplicate_count


def normalize(records: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    evictions: list[dict[str, Any]] = []
    ignored_types: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    for line_no, raw in records:
        event_type = _event_type(raw)
        if event_type in REQUEST_TYPES:
            request = _normalize_request(raw, line_no)
            if request is None:
                invalid.append({"line": line_no, "event_type": event_type,
                                "reason": "request requires logical layer and expert ID"})
                continue
            requests.append(request)
            if _embedded_transfer_present(raw):
                transfers.append(_normalize_transfer(raw, line_no, event_type, embedded=True))
            if _embedded_eviction_present(raw):
                eviction = _normalize_eviction(raw, line_no, request)
                if eviction is not None:
                    evictions.append(eviction)
        elif event_type in TRANSFER_TYPES:
            transfers.append(_normalize_transfer(raw, line_no, event_type))
        elif event_type in EVICTION_TYPES:
            eviction = _normalize_eviction(raw, line_no)
            if eviction is None:
                invalid.append({"line": line_no, "event_type": event_type,
                                "reason": "eviction requires layer and expert ID"})
            else:
                evictions.append(eviction)
        else:
            ignored_types[event_type] += 1
    transfers, deduplicated = _deduplicate_transfers(transfers)
    for ordinal, request in enumerate(requests):
        request["request_ordinal"] = ordinal
    return {
        "requests": requests,
        "transfers": transfers,
        "evictions": evictions,
        "ignored_types": dict(sorted(ignored_types.items())),
        "invalid_events": invalid,
        "deduplicated_transfer_records": deduplicated,
    }


def _field_coverage(items: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    total = len(items)
    result: dict[str, Any] = {"records": total}
    for field in fields:
        count = sum(item.get(field) is not None for item in items)
        result[field] = {"present": count, "fraction": count / total if total else None}
    return result


def _new_request_group() -> dict[str, Any]:
    return {
        "requests": 0, "hits": 0, "misses": 0, "cache_result_unknown": 0,
        "expert_bytes_known_requests": 0, "expert_bytes_requested": 0,
    }


def _request_group_rows(requests: Sequence[Mapping[str, Any]], dimensions: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for request in requests:
        key = tuple(request.get(name) for name in dimensions)
        group = groups.setdefault(key, _new_request_group())
        group["requests"] += 1
        result = request.get("cache_result")
        if result == "hit":
            group["hits"] += 1
        elif result == "miss":
            group["misses"] += 1
        else:
            group["cache_result_unknown"] += 1
        byte_count = request.get("expert_bytes")
        if byte_count is not None:
            group["expert_bytes_known_requests"] += 1
            group["expert_bytes_requested"] += byte_count
    rows = []
    for key, group in groups.items():
        row = {name: value for name, value in zip(dimensions, key)}
        known = group["hits"] + group["misses"]
        group["hit_rate_for_known_results"] = group["hits"] / known if known else None
        group["miss_rate_for_known_results"] = group["misses"] / known if known else None
        row.update(group)
        rows.append(row)
    rows.sort(key=lambda row: tuple(_stable(row.get(name)) for name in dimensions))
    return rows


def _new_transfer_group() -> dict[str, Any]:
    return {
        "transfers": 0, "bytes_known_transfers": 0, "measured_bytes": 0,
        "duration_known_transfers": 0, "measured_duration_seconds": 0.0,
        "bytes_and_duration_known_transfers": 0,
        "paired_bytes": 0, "paired_duration_seconds": 0.0,
        "overlap_known_transfers": 0, "overlapped_compute_transfers": 0,
        "measured_overlap_with_compute_seconds": 0.0,
        "measured_serialized_transfer_seconds": 0.0,
    }


def _transfer_group_rows(transfers: Sequence[Mapping[str, Any]], dimensions: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for transfer in transfers:
        key = tuple(transfer.get(name) for name in dimensions)
        group = groups.setdefault(key, _new_transfer_group())
        group["transfers"] += 1
        byte_count = transfer.get("bytes")
        duration = transfer.get("duration_seconds")
        if byte_count is not None:
            group["bytes_known_transfers"] += 1
            group["measured_bytes"] += byte_count
        if duration is not None:
            group["duration_known_transfers"] += 1
            group["measured_duration_seconds"] += duration
        if byte_count is not None and duration is not None:
            group["bytes_and_duration_known_transfers"] += 1
            group["paired_bytes"] += byte_count
            group["paired_duration_seconds"] += duration
        if transfer.get("overlapped_compute") is not None:
            group["overlap_known_transfers"] += 1
            group["overlapped_compute_transfers"] += int(
                bool(transfer.get("overlapped_compute"))
            )
        if transfer.get("overlap_with_compute_seconds") is not None:
            group["measured_overlap_with_compute_seconds"] += float(
                transfer["overlap_with_compute_seconds"]
            )
        if transfer.get("serialized_transfer_seconds") is not None:
            group["measured_serialized_transfer_seconds"] += float(
                transfer["serialized_transfer_seconds"]
            )
    rows = []
    for key, group in groups.items():
        row = {name: value for name, value in zip(dimensions, key)}
        paired_seconds = group["paired_duration_seconds"]
        group["effective_bandwidth_bytes_per_second_for_paired_transfers"] = (
            group["paired_bytes"] / paired_seconds if paired_seconds > 0 else None
        )
        row.update(group)
        rows.append(row)
    rows.sort(key=lambda row: tuple(_stable(row.get(name)) for name in dimensions))
    return rows


def _h2d_bytes_per_miss_rows(requests: Sequence[Mapping[str, Any]],
                             transfers: Sequence[Mapping[str, Any]],
                             dimensions: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for request in requests:
        key = tuple(request.get(name) for name in dimensions)
        group = groups.setdefault(key, {
            "known_cache_results": 0, "cache_misses": 0, "h2d_transfers": 0,
            "h2d_bytes_known_transfers": 0, "measured_h2d_bytes": 0,
        })
        if request.get("cache_result") in {"hit", "miss"}:
            group["known_cache_results"] += 1
        if request.get("cache_result") == "miss":
            group["cache_misses"] += 1
    for transfer in transfers:
        if transfer.get("direction") != "h2d" or transfer.get("component") not in {
            None, "expert_weight"
        }:
            continue
        key = tuple(transfer.get(name) for name in dimensions)
        group = groups.setdefault(key, {
            "known_cache_results": 0, "cache_misses": 0, "h2d_transfers": 0,
            "h2d_bytes_known_transfers": 0, "measured_h2d_bytes": 0,
        })
        group["h2d_transfers"] += 1
        if transfer.get("bytes") is not None:
            group["h2d_bytes_known_transfers"] += 1
            group["measured_h2d_bytes"] += transfer["bytes"]
    rows = []
    for key, group in groups.items():
        row = {name: value for name, value in zip(dimensions, key)}
        misses = group["cache_misses"]
        group["measured_h2d_bytes_per_known_cache_miss"] = (
            group["measured_h2d_bytes"] / misses if misses else None
        )
        row.update(group)
        rows.append(row)
    rows.sort(key=lambda row: tuple(_stable(row.get(name)) for name in dimensions))
    return rows


def _compute_reuse(requests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    last: dict[tuple[Any, ...], int] = {}
    # A Fenwick tree holding the latest position of every cache identity gives
    # an O(n log n) exact count of distinct identities between reuses.
    tree = [0] * (len(requests) + 1)

    def add(index: int, delta: int) -> None:
        index += 1
        while index < len(tree):
            tree[index] += delta
            index += index & -index

    def prefix(index: int) -> int:
        total = 0
        index += 1
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    def range_sum(begin: int, end: int) -> int:
        return 0 if end < begin else prefix(end) - (prefix(begin - 1) if begin else 0)

    for request in requests:
        request.update({
            "previous_request_distance": None, "next_request_distance": None,
            "previous_distinct_expert_distance": None,
            "next_distinct_expert_distance": None,
            "previous_token_distance": None, "next_token_distance": None,
        })
    for index, request in enumerate(requests):
        key = _cache_identity(request)
        if key is None:
            continue
        previous = last.get(key)
        if previous is not None:
            distance = index - previous - 1
            distinct = range_sum(previous + 1, index - 1)
            request["previous_request_distance"] = distance
            request["previous_distinct_expert_distance"] = distinct
            requests[previous]["next_request_distance"] = distance
            requests[previous]["next_distinct_expert_distance"] = distinct
            previous_token = requests[previous].get("token_step")
            current_token = request.get("token_step")
            if previous_token is not None and current_token is not None:
                token_distance = current_token - previous_token
                request["previous_token_distance"] = token_distance
                requests[previous]["next_token_distance"] = token_distance
            add(previous, -1)
        add(index, 1)
        last[key] = index

    details = []
    reported_comparisons = 0
    reported_mismatches = 0
    for request in requests:
        reported = request.get("reported_previous_reuse_distance")
        computed = request.get("previous_request_distance")
        if reported is not None and reported >= 0 and computed is not None:
            reported_comparisons += 1
            if reported != computed:
                reported_mismatches += 1
        details.append({
            name: request.get(name) for name in (
                "line", "request_ordinal", "run_id", "event_index", "token_step",
                "token_phase", "logical_layer", "physical_layer", "expert_id",
                "gpu_destination", "cache_result", "routing_rank", "routing_weight",
                "source_checkpoint_shard", "reported_previous_reuse_distance",
                "reported_distinct_reuse_distance", "previous_request_distance",
                "next_request_distance", "previous_distinct_expert_distance",
                "next_distinct_expert_distance", "previous_token_distance",
                "next_token_distance",
            )
        })
    summary = {
        "previous_request_distance": _distribution(
            r["previous_request_distance"] for r in requests
            if r.get("previous_request_distance") is not None
        ),
        "next_request_distance": _distribution(
            r["next_request_distance"] for r in requests
            if r.get("next_request_distance") is not None
        ),
        "previous_distinct_expert_distance": _distribution(
            r["previous_distinct_expert_distance"] for r in requests
            if r.get("previous_distinct_expert_distance") is not None
        ),
        "next_distinct_expert_distance": _distribution(
            r["next_distinct_expert_distance"] for r in requests
            if r.get("next_distinct_expert_distance") is not None
        ),
        "previous_token_distance": _distribution(
            r["previous_token_distance"] for r in requests
            if r.get("previous_token_distance") is not None
        ),
        "next_token_distance": _distribution(
            r["next_token_distance"] for r in requests
            if r.get("next_token_distance") is not None
        ),
        "reported_previous_distance_comparisons": reported_comparisons,
        "reported_previous_distance_mismatches": reported_mismatches,
    }
    return details, summary


def _same_token_repeats(items: Sequence[Mapping[str, Any]], transfer: bool = False) -> dict[str, Any]:
    groups: Counter[tuple[Any, ...]] = Counter()
    known_byte_counts: Counter[tuple[Any, ...]] = Counter()
    measured_bytes: Counter[tuple[Any, ...]] = Counter()
    for item in items:
        identity = _cache_identity(item)
        token = item.get("token_step")
        if identity is not None and token is not None:
            key = (item["run_id"], token, item.get("token_phase"), *identity[1:])
            if transfer:
                key = (*key, item.get("direction"))
            groups[key] += 1
            if transfer and item.get("bytes") is not None:
                known_byte_counts[key] += 1
                measured_bytes[key] += item["bytes"]
    duplicates = [(key, count) for key, count in groups.items() if count > 1]
    result = {
        "groups": len(duplicates),
        "repeated_events": sum(count - 1 for _, count in duplicates),
        "records": [],
    }
    for key, count in sorted(duplicates, key=lambda pair: tuple(_stable(v) for v in pair[0])):
        record = {
            "run_id": key[0], "token_step": key[1], "token_phase": key[2],
            "physical_layer": key[3], "expert_id": key[4],
            "gpu_destination": key[5], "count": count,
        }
        if transfer:
            record["direction"] = key[6]
            record["measured_bytes"] = measured_bytes[key]
            record["bytes_known_transfers"] = known_byte_counts[key]
        result["records"].append(record)
    return result


def _expert_request_rows(requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for request in requests:
        key = _identity(request)
        if key is None:
            continue
        group = groups.setdefault(key, {
            **_identity_json(key), "requests": 0, "hits": 0, "misses": 0,
            "cache_result_unknown": 0, "known_expert_bytes": [], "gpus": set(),
            "source_checkpoint_shards": set(),
        })
        group["requests"] += 1
        result = request.get("cache_result")
        if result == "hit": group["hits"] += 1
        elif result == "miss": group["misses"] += 1
        else: group["cache_result_unknown"] += 1
        if request.get("expert_bytes") is not None:
            group["known_expert_bytes"].append(request["expert_bytes"])
        if request.get("gpu_destination") is not None:
            group["gpus"].add(request["gpu_destination"])
        if request.get("source_checkpoint_shard") is not None:
            group["source_checkpoint_shards"].add(str(request["source_checkpoint_shard"]))
    rows = []
    for group in groups.values():
        sizes = group.pop("known_expert_bytes")
        group["expert_bytes"] = sizes[0] if sizes and len(set(sizes)) == 1 else None
        group["observed_expert_byte_sizes"] = sorted(set(sizes))
        group["gpus"] = sorted(group["gpus"])
        group["source_checkpoint_shards"] = sorted(group["source_checkpoint_shards"])
        group["reused"] = group["requests"] > 1
        rows.append(group)
    rows.sort(key=lambda row: (-row["requests"], _stable(row["run_id"]), row["physical_layer"], row["expert_id"]))
    return rows


def _expert_transfer_rows(transfers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for transfer in transfers:
        key = _identity(transfer)
        if key is None:
            continue
        group = groups.setdefault(key, {
            **_identity_json(key), "transfers": 0, "bytes_known_transfers": 0,
            "measured_bytes": 0, "gpus": set(), "directions": set(),
            "source_checkpoint_shards": set(),
        })
        group["transfers"] += 1
        if transfer.get("bytes") is not None:
            group["bytes_known_transfers"] += 1
            group["measured_bytes"] += transfer["bytes"]
        if transfer.get("gpu_destination") is not None:
            group["gpus"].add(transfer["gpu_destination"])
        if transfer.get("direction") is not None:
            group["directions"].add(transfer["direction"])
        if transfer.get("source_checkpoint_shard") is not None:
            group["source_checkpoint_shards"].add(str(transfer["source_checkpoint_shard"]))
    rows = []
    for group in groups.values():
        group["gpus"] = sorted(group["gpus"])
        group["directions"] = sorted(group["directions"])
        group["source_checkpoint_shards"] = sorted(group["source_checkpoint_shards"])
        rows.append(group)
    rows.sort(key=lambda row: (-row["measured_bytes"], -row["transfers"],
                               _stable(row["run_id"]), row["physical_layer"], row["expert_id"]))
    return rows


def _match_evictions(evictions: Sequence[Mapping[str, Any]], requests: Sequence[Mapping[str, Any]],
                     short_distance: int) -> dict[str, Any]:
    exact: dict[tuple[Any, ...], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    source: dict[tuple[Any, ...], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for request in requests:
        identity = _identity(request)
        cache_identity = _cache_identity(request)
        if identity is not None:
            source[identity].append((request["line"], request))
        if cache_identity is not None:
            exact[cache_identity].append((request["line"], request))
    exact_lines = {key: [line for line, _ in values] for key, values in exact.items()}
    source_lines = {key: [line for line, _ in values] for key, values in source.items()}
    request_lines = [request["line"] for request in requests]
    records = []
    no_future_reuse = 0
    for eviction in evictions:
        identity = _identity(eviction)
        cache_identity = _cache_identity(eviction)
        if eviction.get("gpu_destination") is not None:
            candidates = exact.get(cache_identity, [])
            lines = exact_lines.get(cache_identity, [])
        else:
            candidates = source.get(identity, [])
            lines = source_lines.get(identity, [])
        position = bisect.bisect_right(lines, eviction["line"])
        if position >= len(candidates):
            no_future_reuse += 1
            continue
        next_request = candidates[position][1]
        requests_before_or_at_eviction = bisect.bisect_right(request_lines, eviction["line"])
        distance = next_request["request_ordinal"] - requests_before_or_at_eviction
        token_distance = None
        if eviction.get("token_step") is not None and next_request.get("token_step") is not None:
            token_distance = next_request["token_step"] - eviction["token_step"]
        records.append({
            "eviction_line": eviction["line"], "run_id": eviction["run_id"],
            "physical_layer": eviction["physical_layer"], "expert_id": eviction["expert_id"],
            "gpu_destination": eviction.get("gpu_destination"),
            "eviction_reason": eviction.get("eviction_reason"),
            "next_request_line": next_request["line"],
            "next_request_distance": distance,
            "next_token_distance": token_distance,
            "next_request_cache_result": next_request.get("cache_result"),
            "gpu_match_basis": "exact" if eviction.get("gpu_destination") is not None else "gpu_unavailable",
            "shortly_before_reuse": distance <= short_distance,
        })
    short = [record for record in records if record["shortly_before_reuse"]]
    return {
        "threshold_intervening_requests": short_distance,
        "evictions": len(evictions),
        "evictions_with_future_reuse": len(records),
        "evictions_without_future_reuse": no_future_reuse,
        "shortly_before_reuse_count": len(short),
        "shortly_before_reuse_then_miss_count": sum(
            record["next_request_cache_result"] == "miss" for record in short
        ),
        "reuse_distance": _distribution(record["next_request_distance"] for record in records),
        "records": short,
        "all_future_reuse_records": records,
    }


def _thrashing_by_layer(requests: Sequence[Mapping[str, Any]],
                         eviction_analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any], dict[str, Any]] = {}
    for request in requests:
        key = (request["run_id"], request.get("logical_layer"))
        group = groups.setdefault(key, {
            "run_id": key[0], "logical_layer": key[1], "requests": 0,
            "misses": 0, "repeated_requests": 0, "repeated_misses": 0,
            "repeated_miss_expert_bytes_known": 0,
            "repeated_miss_expert_bytes": 0,
            "short_post_eviction_reuses": 0,
            "short_post_eviction_misses": 0,
        })
        group["requests"] += 1
        if request.get("cache_result") == "miss":
            group["misses"] += 1
        if request.get("previous_request_distance") is not None:
            group["repeated_requests"] += 1
            if request.get("cache_result") == "miss":
                group["repeated_misses"] += 1
                if request.get("expert_bytes") is not None:
                    group["repeated_miss_expert_bytes_known"] += 1
                    group["repeated_miss_expert_bytes"] += request["expert_bytes"]
    request_by_line = {request["line"]: request for request in requests}
    for record in eviction_analysis["records"]:
        request = request_by_line.get(record["next_request_line"])
        if request is None:
            continue
        group = groups[(request["run_id"], request.get("logical_layer"))]
        group["short_post_eviction_reuses"] += 1
        if request.get("cache_result") == "miss":
            group["short_post_eviction_misses"] += 1
    rows = []
    for group in groups.values():
        repeated = group["repeated_requests"]
        group["repeated_request_miss_rate"] = group["repeated_misses"] / repeated if repeated else None
        rows.append(group)
    rows.sort(key=lambda row: (_stable(row["run_id"]), _stable(row["logical_layer"])))
    return rows


def _request_byte_accounting(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seen: set[tuple[Any, ...]] = set()
    unique_bytes = 0
    repeated_bytes = 0
    unattributed = 0
    known = 0
    for request in requests:
        byte_count = request.get("expert_bytes")
        if byte_count is None:
            continue
        known += 1
        key = _identity(request)
        if key is None:
            unattributed += byte_count
        elif key in seen:
            repeated_bytes += byte_count
        else:
            seen.add(key)
            unique_bytes += byte_count
    return {
        "requests": len(requests), "expert_bytes_known_requests": known,
        "measured_requested_bytes": unique_bytes + repeated_bytes + unattributed,
        "first_request_bytes_by_physical_expert": unique_bytes,
        "repeat_request_bytes_by_physical_expert": repeated_bytes,
        "identity_unavailable_bytes": unattributed,
    }


def _transfer_byte_accounting(transfers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seen_source: set[tuple[Any, ...]] = set()
    seen_destination: set[tuple[Any, ...]] = set()
    unique_source = repeated_source = 0
    first_destination = repeated_destination = 0
    source_unknown = destination_unknown = 0
    known = total = 0
    for transfer in transfers:
        byte_count = transfer.get("bytes")
        if byte_count is None:
            continue
        known += 1
        total += byte_count
        source = _identity(transfer)
        destination = _cache_identity(transfer) if transfer.get("gpu_destination") is not None else None
        if source is None:
            source_unknown += byte_count
        elif source in seen_source:
            repeated_source += byte_count
        else:
            seen_source.add(source)
            unique_source += byte_count
        if destination is None:
            destination_unknown += byte_count
        elif destination in seen_destination:
            repeated_destination += byte_count
        else:
            seen_destination.add(destination)
            first_destination += byte_count
    return {
        "transfers": len(transfers), "bytes_known_transfers": known,
        "measured_cumulative_bytes": total,
        "first_transfer_bytes_by_physical_expert": unique_source,
        "repeat_transfer_bytes_by_physical_expert": repeated_source,
        "physical_expert_identity_unavailable_bytes": source_unknown,
        "first_transfer_bytes_by_expert_and_gpu": first_destination,
        "repeat_transfer_bytes_to_same_expert_and_gpu": repeated_destination,
        "gpu_destination_unavailable_bytes": destination_unknown,
    }


def analyze(records: list[tuple[int, dict[str, Any]]], source: str,
            short_reuse_distance: int = 32, top_n: int = 20) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = normalize(records)
    requests = normalized["requests"]
    transfers = normalized["transfers"]
    evictions = normalized["evictions"]
    reuse_details, reuse_summary = _compute_reuse(requests)
    eviction_analysis = _match_evictions(evictions, requests, short_reuse_distance)
    expert_requests = _expert_request_rows(requests)
    expert_transfers = _expert_transfer_rows(transfers)
    h2d_expert_transfers = _expert_transfer_rows(
        [transfer for transfer in transfers if transfer.get("direction") == "h2d"]
    )
    thrashing_layers = _thrashing_by_layer(requests, eviction_analysis)
    request_coverage = _field_coverage(requests, (
        "token_step", "token_phase", "logical_layer", "physical_layer", "expert_id",
        "routing_rank", "routing_weight", "source_checkpoint_shard", "expert_bytes",
        "cache_state_before", "cache_result", "gpu_destination", "transfer_id",
        "duplicated_on_both_gpus",
    ))
    transfer_coverage = _field_coverage(transfers, (
        "token_step", "token_phase", "logical_layer", "physical_layer", "expert_id",
        "gpu_destination", "direction", "component", "bytes", "duration_seconds",
        "transfer_id", "overlapped_compute",
    ))
    request_coverage["logical_layer_explicit"] = {
        "present": sum(bool(request.get("logical_layer_explicit")) for request in requests),
        "fraction": (
            sum(bool(request.get("logical_layer_explicit")) for request in requests) / len(requests)
            if requests else None
        ),
    }
    request_coverage["resolved_shard_layer_explicit"] = {
        "present": sum(bool(request.get("physical_layer_explicit")) for request in requests),
        "fraction": (
            sum(bool(request.get("physical_layer_explicit")) for request in requests) / len(requests)
            if requests else None
        ),
    }
    limitations = []
    if request_coverage["gpu_destination"]["present"] < len(requests):
        limitations.append("GPU-specific cache identity and reuse are incomplete because some requests lack gpu_destination.")
    if request_coverage["expert_bytes"]["present"] < len(requests):
        limitations.append("Requested expert-byte totals cover only requests with an explicit expert byte size.")
    if transfer_coverage["bytes"]["present"] < len(transfers):
        limitations.append("Transfer-byte totals cover only transfer records with explicit bytes; expert_bytes was not substituted.")
    if not transfers:
        limitations.append("No transfer events were available; request expert_bytes was not substituted for measured transfer traffic.")
    if transfer_coverage["duration_seconds"]["present"] < len(transfers):
        limitations.append("Bandwidth covers only transfers with explicit duration or unit-qualified start/completion timestamps.")
    if transfer_coverage["direction"]["present"] < len(transfers):
        limitations.append("Direction-specific transfer totals are incomplete because some transfers lack direction.")
    if not evictions:
        limitations.append("No eviction events were available, so shortly-before-reuse eviction evidence is unavailable.")
    if request_coverage["resolved_shard_layer_explicit"]["present"] < len(requests):
        limitations.append("Some requests lack resolved_shard_layer; logical_layer is used only as an identity fallback and is disclosed in coverage.")
    if request_coverage["duplicated_on_both_gpus"]["present"] < len(requests):
        limitations.append("Simultaneous cross-GPU residency cannot be established without duplicated_on_both_gpus; observed destinations show traffic only.")

    runs = sorted({item["run_id"] for item in (*requests, *transfers, *evictions)})
    cache_analysis = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "expert-cache-analysis",
        "source": source,
        "order_basis": "nonblank JSONL input line order",
        "runs": runs,
        "input_summary": {
            "jsonl_records": len(records), "request_events": len(requests),
            "eviction_events": len(evictions), "transfer_events": len(transfers),
            "ignored_event_types": normalized["ignored_types"],
            "invalid_events": normalized["invalid_events"],
            "deduplicated_transfer_records": normalized["deduplicated_transfer_records"],
        },
        "coverage": request_coverage,
        "overall": _request_group_rows(requests, ())[0] if requests else _new_request_group(),
        "request_byte_accounting": _request_byte_accounting(requests),
        "by_layer": _request_group_rows(requests, ("run_id", "logical_layer")),
        "by_token": _request_group_rows(requests, ("run_id", "token_phase", "token_step")),
        "by_gpu": _request_group_rows(requests, ("run_id", "gpu_destination")),
        "by_layer_token_gpu": _request_group_rows(
            requests, ("run_id", "token_phase", "token_step", "logical_layer", "gpu_destination")
        ),
        "reuse_distance": reuse_summary,
        "request_reuse_records": reuse_details,
        "same_token_repeats": _same_token_repeats(requests),
        "evictions_shortly_before_reuse": eviction_analysis,
        "cache_thrashing": {
            "definition": (
                "A repeated miss is a miss for an expert previously requested on the same "
                "GPU identity. A confirmed short post-eviction miss also has a traced eviction "
                "within the configured intervening-request threshold."
            ),
            "repeated_requests": sum(row["repeated_requests"] for row in thrashing_layers),
            "repeated_misses": sum(row["repeated_misses"] for row in thrashing_layers),
            "short_post_eviction_reuses": sum(
                row["short_post_eviction_reuses"] for row in thrashing_layers
            ),
            "short_post_eviction_misses": sum(
                row["short_post_eviction_misses"] for row in thrashing_layers
            ),
        },
        "cache_thrashing_by_layer": thrashing_layers,
        "never_reused_experts": [row for row in expert_requests if not row["reused"]],
        "experts": expert_requests,
        "most_requested_experts": expert_requests[:top_n],
        "limitations": limitations,
    }

    known_sizes = [transfer["bytes"] for transfer in transfers if transfer.get("bytes") is not None]
    bandwidths = [
        transfer["bytes"] / transfer["duration_seconds"]
        for transfer in transfers
        if transfer.get("bytes") is not None and transfer.get("duration_seconds") is not None
        and transfer["duration_seconds"] > 0
    ]
    h2d_all = [transfer for transfer in transfers if transfer.get("direction") == "h2d"]
    h2d = [transfer for transfer in h2d_all if transfer.get("component") in {
        None, "expert_weight"
    }]
    misses = sum(request.get("cache_result") == "miss" for request in requests)
    h2d_bytes = sum(transfer["bytes"] for transfer in h2d if transfer.get("bytes") is not None)
    multi_gpu = [row for row in expert_transfers if len(row["gpus"]) > 1]
    transfers_by_direction: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for transfer in transfers:
        transfers_by_direction[transfer.get("direction") or "unknown"].append(transfer)
    transfer_analysis = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "transfer-analysis",
        "source": source,
        "order_basis": "nonblank JSONL input line order",
        "runs": runs,
        "input_summary": cache_analysis["input_summary"],
        "coverage": transfer_coverage,
        "overall": _transfer_group_rows(transfers, ())[0] if transfers else _new_transfer_group(),
        "all_h2d": (
            _transfer_group_rows(h2d_all, ())[0] if h2d_all
            else _new_transfer_group()
        ),
        "h2d": {
            "transfers": len(h2d),
            "bytes_known_transfers": sum(t.get("bytes") is not None for t in h2d),
            "measured_bytes": h2d_bytes,
            "known_cache_misses": misses,
            "measured_h2d_bytes_per_known_cache_miss": h2d_bytes / misses if misses else None,
            "scope": "expert weight transfers; legacy records without component are included",
        },
        "byte_accounting": _transfer_byte_accounting(transfers),
        "byte_accounting_by_direction": {
            direction: _transfer_byte_accounting(items)
            for direction, items in sorted(transfers_by_direction.items())
        },
        "h2d_byte_accounting": _transfer_byte_accounting(h2d),
        "h2d_bytes_per_cache_miss": {
            "overall": _h2d_bytes_per_miss_rows(requests, transfers, ())[0]
                if requests or h2d else {},
            "by_layer": _h2d_bytes_per_miss_rows(
                requests, transfers, ("run_id", "logical_layer")
            ),
            "by_token": _h2d_bytes_per_miss_rows(
                requests, transfers, ("run_id", "token_phase", "token_step")
            ),
            "by_gpu": _h2d_bytes_per_miss_rows(
                requests, transfers, ("run_id", "gpu_destination")
            ),
            "by_layer_token_gpu": _h2d_bytes_per_miss_rows(
                requests, transfers,
                ("run_id", "token_phase", "token_step", "logical_layer", "gpu_destination"),
            ),
        },
        "transfer_size_bytes": _distribution(known_sizes),
        "per_transfer_bandwidth_bytes_per_second": _distribution(bandwidths),
        "by_direction": _transfer_group_rows(transfers, ("run_id", "direction")),
        "by_component": _transfer_group_rows(
            transfers, ("run_id", "component", "direction")
        ),
        "by_layer": _transfer_group_rows(transfers, ("run_id", "logical_layer", "direction")),
        "by_token": _transfer_group_rows(
            transfers, ("run_id", "token_phase", "token_step", "direction")
        ),
        "by_gpu": _transfer_group_rows(transfers, ("run_id", "gpu_destination", "direction")),
        "by_layer_token_gpu": _transfer_group_rows(
            transfers,
            ("run_id", "token_phase", "token_step", "logical_layer", "gpu_destination", "direction"),
        ),
        "same_token_repeat_transfers": _same_token_repeats(transfers, transfer=True),
        "experts": expert_transfers,
        "most_transferred_experts": expert_transfers[:top_n],
        "most_h2d_transferred_experts": h2d_expert_transfers[:top_n],
        "experts_observed_transferred_to_multiple_gpus": multi_gpu,
        "multi_gpu_observation_note": (
            "Multiple transfer destinations prove duplicate traffic, not simultaneous live residency."
        ),
        "limitations": limitations,
    }
    return cache_analysis, transfer_analysis


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="expert-trace.jsonl or .jsonl.gz")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory for expert-cache-analysis.json and transfer-analysis.json")
    parser.add_argument("--short-reuse-distance", type=int, default=32,
                        help="maximum intervening requests for a shortly-before-reuse eviction")
    parser.add_argument("--top", type=int, default=20, help="number of top experts to highlight")
    args = parser.parse_args(argv)
    if args.short_reuse_distance < 0:
        parser.error("--short-reuse-distance must be non-negative")
    if args.top <= 0:
        parser.error("--top must be positive")
    records = read_jsonl(args.trace)
    cache, transfer = analyze(
        records, str(args.trace), args.short_reuse_distance, args.top
    )
    _write_json(args.output_dir / "expert-cache-analysis.json", cache)
    _write_json(args.output_dir / "transfer-analysis.json", transfer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
