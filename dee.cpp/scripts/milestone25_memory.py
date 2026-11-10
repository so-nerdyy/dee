"""Bounded host, PyTorch, and CUDA memory measurements for Milestone 2.5.

The public collectors deliberately return JSON-compatible summaries instead of
raw tensors, allocator snapshots, or ``/proc`` contents.  ``MemoryProbe`` keeps
only those summaries, so passing model/cache tensors to ``checkpoint`` does not
extend their lifetime.
"""

from __future__ import annotations

import csv
import datetime as _datetime
import heapq
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
_PROC_MAP_RE = re.compile(
    r"^([0-9a-fA-F]+)-([0-9a-fA-F]+)\s+([rwxps-]{4})\s+"
    r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+:[0-9a-fA-F]+)\s+(\d+)\s*(.*)$"
)
_UNSET = object()


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _bounded_text(value: Any, limit: int = 512) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def json_safe(
    value: Any,
    *,
    max_depth: int = 6,
    max_items: int = 256,
    max_string: int = 2048,
    _depth: int = 0,
) -> Any:
    """Convert arbitrary metadata to a bounded JSON-compatible value."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, Path)):
        return _bounded_text(value, max_string)
    if _depth >= max_depth:
        return _bounded_text(value, max_string)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                output["__truncated_items__"] = (
                    len(value) - max_items if hasattr(value, "__len__") else True
                )
                break
            output[_bounded_text(key, 256)] = json_safe(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                _depth=_depth + 1,
            )
        return output
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "byte_length": len(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        output = [
            json_safe(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                _depth=_depth + 1,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            output.append({"__truncated_items__": len(value) - max_items})
        return output
    return _bounded_text(value, max_string)


def _parse_scaled_integer(parts: list[str]) -> int | str:
    try:
        value = int(parts[0])
    except (IndexError, ValueError):
        return " ".join(parts)
    if len(parts) == 1:
        return value
    unit = parts[1].lower()
    scales = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}
    return value * scales[unit] if unit in scales else " ".join(parts)


def parse_proc_key_values(text: str) -> dict[str, Any]:
    """Parse ``/proc`` colon-delimited counters, normalizing byte units."""
    result: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        value = _parse_scaled_integer(parts)
        output_key = (
            f"{key}_bytes"
            if len(parts) > 1 and parts[1].lower() in {"b", "kb", "mb", "gb"}
            else key
        )
        result[output_key] = value
    return result


def _mapping_category(path: str) -> str:
    lowered = path.lower()
    if ".safetensors" in lowered or re.search(
        r"(?:^|/)pytorch_model.*\.bin(?:\s|$)", lowered
    ):
        return "checkpoint"
    if (
        not path
        or path.startswith("[anon")
        or path in {"[heap]", "[stack]"}
        or path.startswith("[stack:")
    ):
        return "anonymous"
    if "/dev/shm" in lowered or "memfd:" in lowered or "[shmem" in lowered:
        return "shared_memory"
    if (
        "libtorch" in lowered
        or "libc10" in lowered
        or "/torch/" in lowered
        or "\\torch\\" in lowered
    ):
        return "pytorch"
    if any(
        token in lowered
        for token in ("libcuda", "libcud", "libnv", "nvidia", "libnccl")
    ):
        return "cuda"
    if "python" in lowered:
        return "python"
    if lowered.startswith("/tmp/") or "/.cache/" in lowered or "\\temp\\" in lowered:
        return "temporary_file"
    if path.startswith("["):
        return "kernel_special"
    return "other_file_backed"


def _new_category() -> dict[str, int]:
    return {
        "region_count": 0,
        "virtual_bytes": 0,
        "rss_bytes": 0,
        "pss_bytes": 0,
        "anonymous_bytes": 0,
        "shared_clean_bytes": 0,
        "shared_dirty_bytes": 0,
        "private_clean_bytes": 0,
        "private_dirty_bytes": 0,
        "locked_bytes": 0,
        "swap_bytes": 0,
    }


_SMAPS_FIELDS = {
    "Rss": "rss_bytes",
    "Pss": "pss_bytes",
    "Anonymous": "anonymous_bytes",
    "Shared_Clean": "shared_clean_bytes",
    "Shared_Dirty": "shared_dirty_bytes",
    "Private_Clean": "private_clean_bytes",
    "Private_Dirty": "private_dirty_bytes",
    "Locked": "locked_bytes",
    "Swap": "swap_bytes",
}


def summarize_smaps(lines: Iterable[str], *, max_regions: int = 64) -> dict[str, Any]:
    """Attribute a smaps stream without retaining its potentially huge text."""
    categories: dict[str, dict[str, int]] = {}
    top: list[tuple[int, int, dict[str, Any]]] = []
    region_count = 0
    sequence = 0
    current: dict[str, Any] | None = None

    def finish(region: dict[str, Any] | None) -> None:
        nonlocal region_count, sequence
        if region is None:
            return
        region_count += 1
        category = region["category"]
        totals = categories.setdefault(category, _new_category())
        totals["region_count"] += 1
        totals["virtual_bytes"] += region["virtual_bytes"]
        for field in _SMAPS_FIELDS.values():
            totals[field] += region.get(field, 0)
        detail = {
            "path": _bounded_text(region["path"], 512),
            "category": category,
            "permissions": region["permissions"],
            "virtual_bytes": region["virtual_bytes"],
            "rss_bytes": region.get("rss_bytes", 0),
            "pss_bytes": region.get("pss_bytes", 0),
            "anonymous_bytes": region.get("anonymous_bytes", 0),
            "locked_bytes": region.get("locked_bytes", 0),
        }
        candidate = (detail["rss_bytes"], sequence, detail)
        sequence += 1
        if max_regions > 0:
            if len(top) < max_regions:
                heapq.heappush(top, candidate)
            elif candidate[:2] > top[0][:2]:
                heapq.heapreplace(top, candidate)

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        match = _PROC_MAP_RE.match(line)
        if match:
            finish(current)
            start, end, permissions, _offset, _device, _inode, path = match.groups()
            path = path.strip()
            current = {
                "path": path,
                "category": _mapping_category(path),
                "permissions": permissions,
                "virtual_bytes": int(end, 16) - int(start, 16),
            }
            continue
        if current is None or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        if key in _SMAPS_FIELDS:
            parsed = _parse_scaled_integer(raw.strip().split())
            if isinstance(parsed, int):
                current[_SMAPS_FIELDS[key]] = parsed
    finish(current)
    return {
        "region_count": region_count,
        "categories": categories,
        "top_regions_by_rss": [item[2] for item in sorted(top, reverse=True)],
        "top_regions_truncated": region_count > len(top),
    }


def summarize_maps(lines: Iterable[str], *, max_paths: int = 64) -> dict[str, Any]:
    """Summarize ``/proc/<pid>/maps`` virtual regions and bounded top paths."""
    categories: dict[str, dict[str, int]] = {}
    path_totals: dict[str, int] = {}
    region_count = 0
    for raw_line in lines:
        match = _PROC_MAP_RE.match(raw_line.rstrip("\n"))
        if not match:
            continue
        start, end, _permissions, _offset, _device, _inode, path = match.groups()
        path = path.strip()
        size = int(end, 16) - int(start, 16)
        category = categories.setdefault(
            _mapping_category(path), {"region_count": 0, "virtual_bytes": 0}
        )
        category["region_count"] += 1
        category["virtual_bytes"] += size
        path_key = _bounded_text(path or "[anonymous]", 512)
        path_totals[path_key] = path_totals.get(path_key, 0) + size
        region_count += 1
    ranked = sorted(path_totals.items(), key=lambda item: (-item[1], item[0]))[
        :max_paths
    ]
    return {
        "region_count": region_count,
        "categories": categories,
        "top_paths_by_virtual_bytes": [
            {"path": path, "virtual_bytes": size} for path, size in ranked
        ],
        "top_paths_truncated": len(path_totals) > len(ranked),
    }


def _read_proc_stat(text: str) -> dict[str, int]:
    # comm is parenthesized and may itself contain spaces or parentheses.
    end = text.rfind(")")
    if end < 0:
        raise ValueError("malformed proc stat: missing command terminator")
    fields = text[end + 2 :].split()
    if len(fields) < 22:
        raise ValueError("malformed proc stat: too few fields")
    return {
        "minor_faults": int(fields[7]),
        "children_minor_faults": int(fields[8]),
        "major_faults": int(fields[9]),
        "children_major_faults": int(fields[10]),
    }


def collect_proc_memory(
    pid: int | None = None,
    *,
    proc_root: os.PathLike[str] | str = "/proc",
    include_maps: bool = False,
    include_smaps: bool = False,
    max_regions: int = 64,
) -> dict[str, Any]:
    """Collect Linux process memory, I/O, faults, and optional map attribution."""
    target_pid = int(pid or os.getpid())
    root = Path(proc_root) / str(target_pid)
    result: dict[str, Any] = {
        "available": root.is_dir(),
        "pid": target_pid,
        "errors": [],
    }
    if not root.is_dir():
        result["reason"] = f"{root} is unavailable"
        return result

    for filename, output_name in (
        ("smaps_rollup", "smaps_rollup"),
        ("status", "status"),
        ("io", "io"),
    ):
        try:
            result[output_name] = parse_proc_key_values(
                (root / filename).read_text(encoding="utf-8", errors="replace")
            )
        except (OSError, ValueError) as exc:
            result["errors"].append({"source": filename, "error": _bounded_text(exc)})
    try:
        result["page_faults"] = _read_proc_stat(
            (root / "stat").read_text(encoding="utf-8", errors="replace")
        )
    except (OSError, ValueError) as exc:
        result["errors"].append({"source": "stat", "error": _bounded_text(exc)})
    if include_maps:
        try:
            with (root / "maps").open(
                "r", encoding="utf-8", errors="replace"
            ) as stream:
                result["maps"] = summarize_maps(stream, max_paths=max_regions)
        except OSError as exc:
            result["errors"].append({"source": "maps", "error": _bounded_text(exc)})
    if include_smaps:
        try:
            with (root / "smaps").open(
                "r", encoding="utf-8", errors="replace"
            ) as stream:
                result["smaps_attribution"] = summarize_smaps(
                    stream, max_regions=max_regions
                )
        except OSError as exc:
            result["errors"].append({"source": "smaps", "error": _bounded_text(exc)})
    return result


def _namedtuple_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "_asdict"):
        return {str(key): json_safe(item) for key, item in value._asdict().items()}
    output: dict[str, Any] = {}
    for name in (
        "rss",
        "vms",
        "shared",
        "text",
        "lib",
        "data",
        "dirty",
        "uss",
        "pss",
        "swap",
        "pfaults",
        "pageins",
    ):
        if hasattr(value, name):
            output[name] = json_safe(getattr(value, name))
    return output


def collect_psutil_memory(
    pid: int | None = None, *, psutil_module: Any = _UNSET
) -> dict[str, Any]:
    """Collect psutil RSS/PSS/USS data when psutil is installed and permitted."""
    target_pid = int(pid or os.getpid())
    if psutil_module is _UNSET:
        try:
            psutil_module = importlib.import_module("psutil")
        except ImportError:
            psutil_module = None
    if psutil_module is None:
        return {
            "available": False,
            "pid": target_pid,
            "reason": "psutil is not installed",
        }
    try:
        process = psutil_module.Process(target_pid)
        result = {
            "available": True,
            "pid": target_pid,
            "memory_info": _namedtuple_dict(process.memory_info()),
            "memory_percent": float(process.memory_percent()),
        }
        try:
            result["memory_full_info"] = _namedtuple_dict(process.memory_full_info())
        except (
            Exception
        ) as exc:  # psutil.AccessDenied is not a built-in PermissionError.
            result["full_info_error"] = _bounded_text(exc)
        return result
    except Exception as exc:  # psutil uses platform-specific exception classes.
        return {"available": False, "pid": target_pid, "reason": _bounded_text(exc)}


def collect_page_faults(
    pid: int | None = None,
    *,
    proc_root: os.PathLike[str] | str = "/proc",
) -> dict[str, Any]:
    """Collect absolute Linux proc faults plus self-process resource counters."""
    target_pid = int(pid or os.getpid())
    result: dict[str, Any] = {"pid": target_pid, "sources": {}, "errors": []}
    try:
        stat_path = Path(proc_root) / str(target_pid) / "stat"
        result["sources"]["proc_stat"] = _read_proc_stat(
            stat_path.read_text(encoding="utf-8", errors="replace")
        )
    except (OSError, ValueError) as exc:
        result["errors"].append({"source": "proc_stat", "error": _bounded_text(exc)})
    if target_pid == os.getpid():
        try:
            resource = importlib.import_module("resource")
            usage = resource.getrusage(resource.RUSAGE_SELF)
            result["sources"]["resource"] = {
                "minor_faults": int(usage.ru_minflt),
                "major_faults": int(usage.ru_majflt),
                "max_rss_platform_units": int(usage.ru_maxrss),
                "input_blocks": int(usage.ru_inblock),
                "output_blocks": int(usage.ru_oublock),
            }
        except (ImportError, AttributeError, OSError) as exc:
            result["errors"].append({"source": "resource", "error": _bounded_text(exc)})
    result["available"] = bool(result["sources"])
    return result


def _walk_tensors(
    value: Any,
    torch_module: Any,
    *,
    path: str,
    seen: set[int],
    state: dict[str, int | bool],
    max_nodes: int,
) -> Iterable[tuple[str, Any]]:
    if int(state["nodes"]) >= max_nodes:
        state["truncated"] = True
        return
    state["nodes"] = int(state["nodes"]) + 1
    if isinstance(value, torch_module.Tensor):
        yield path, value
        return
    if isinstance(value, (str, bytes, bytearray, memoryview, Path)) or value is None:
        return
    container_id = id(value)
    if container_id in seen:
        return

    if hasattr(value, "named_parameters") and hasattr(value, "named_buffers"):
        seen.add(container_id)
        try:
            for name, tensor in value.named_parameters(recurse=True):
                yield from _walk_tensors(
                    tensor,
                    torch_module,
                    path=f"{path}.parameter.{name}",
                    seen=seen,
                    state=state,
                    max_nodes=max_nodes,
                )
            for name, tensor in value.named_buffers(recurse=True):
                yield from _walk_tensors(
                    tensor,
                    torch_module,
                    path=f"{path}.buffer.{name}",
                    seen=seen,
                    state=state,
                    max_nodes=max_nodes,
                )
        finally:
            seen.discard(container_id)
        return

    if isinstance(value, Mapping):
        seen.add(container_id)
        try:
            for key, item in value.items():
                yield from _walk_tensors(
                    item,
                    torch_module,
                    path=f"{path}.{_bounded_text(key, 96)}",
                    seen=seen,
                    state=state,
                    max_nodes=max_nodes,
                )
                if bool(state["truncated"]):
                    break
        finally:
            seen.discard(container_id)
        return

    if isinstance(value, Iterable):
        seen.add(container_id)
        try:
            for index, item in enumerate(value):
                yield from _walk_tensors(
                    item,
                    torch_module,
                    path=f"{path}[{index}]",
                    seen=seen,
                    state=state,
                    max_nodes=max_nodes,
                )
                if bool(state["truncated"]):
                    break
        except TypeError:
            return
        finally:
            seen.discard(container_id)


def _tensor_facts(tensor: Any) -> dict[str, Any]:
    device = getattr(tensor, "device", "unknown")
    device_type = str(getattr(device, "type", device)).split(":", 1)[0]
    numel = int(tensor.numel())
    element_size = int(tensor.element_size())
    logical_bytes = numel * element_size
    storage = (
        tensor.untyped_storage()
        if hasattr(tensor, "untyped_storage")
        else tensor.storage()
    )
    storage_bytes = (
        int(storage.nbytes())
        if hasattr(storage, "nbytes")
        else int(storage.size()) * element_size
    )
    pointer = int(storage.data_ptr()) if hasattr(storage, "data_ptr") else id(storage)
    if pointer == 0:
        pointer = id(storage)
    pinned = False
    if device_type == "cpu" and hasattr(tensor, "is_pinned"):
        try:
            pinned = bool(tensor.is_pinned())
        except RuntimeError:
            pinned = False
    shape = [int(dimension) for dimension in list(getattr(tensor, "shape", ()))[:32]]
    return {
        "device": str(device),
        "device_type": device_type,
        "dtype": str(getattr(tensor, "dtype", "unknown")),
        "shape": shape,
        "shape_truncated": len(getattr(tensor, "shape", ())) > len(shape),
        "numel": numel,
        "logical_bytes": logical_bytes,
        "storage_bytes": storage_bytes,
        "storage_key": f"{device}:{pointer:#x}:{storage_bytes}",
        "pinned": pinned,
        "storage_offset": int(tensor.storage_offset())
        if hasattr(tensor, "storage_offset")
        else 0,
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }


def inventory_tensors(
    owners: Mapping[str, Any] | None,
    *,
    device_type: str | None = None,
    owner_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    torch_module: Any = _UNSET,
    max_owners: int = 128,
    max_tensors: int = 100_000,
    max_nodes: int = 250_000,
    max_tensor_details: int = 256,
    max_duplicate_storages: int = 128,
) -> dict[str, Any]:
    """Inventory caller-labeled tensors without retaining any tensor objects.

    Values may be tensors, modules, mappings, or iterables.  Logical tensor
    bytes and unique storage bytes are both reported, exposing aliasing across
    labels.  ``owner_metadata`` can record purpose/source/lifetime/copy status.
    """
    if torch_module is _UNSET:
        try:
            torch_module = importlib.import_module("torch")
        except ImportError:
            torch_module = None
    if torch_module is None:
        return {"available": False, "reason": "PyTorch is not installed"}
    owners = owners or {}
    metadata = owner_metadata or {}
    storage_records: dict[str, dict[str, Any]] = {}
    group_records: dict[tuple[str, str, bool], dict[str, Any]] = {}
    details: list[dict[str, Any]] = []
    owner_rows: dict[str, dict[str, Any]] = {}
    total_tensors = 0
    skipped_device_tensors = 0
    scan_truncated = False

    for owner_index, (owner_raw, value) in enumerate(owners.items()):
        if owner_index >= max_owners:
            scan_truncated = True
            break
        owner = _bounded_text(owner_raw, 256)
        row: dict[str, Any] = {
            "tensor_count": 0,
            "logical_bytes": 0,
            "unique_storage_bytes": 0,
            "pinned_tensor_count": 0,
            "pinned_logical_bytes": 0,
            "metadata": json_safe(
                metadata.get(owner_raw, {}), max_depth=4, max_items=32
            ),
        }
        owner_storage_keys: set[str] = set()
        walk_state: dict[str, int | bool] = {"nodes": 0, "truncated": False}
        for name, tensor in _walk_tensors(
            value,
            torch_module,
            path=owner,
            seen=set(),
            state=walk_state,
            max_nodes=max_nodes,
        ):
            if total_tensors >= max_tensors:
                scan_truncated = True
                break
            try:
                facts = _tensor_facts(tensor)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                row.setdefault("inspection_errors", []).append(_bounded_text(exc))
                row["inspection_errors"] = row["inspection_errors"][:16]
                continue
            if device_type is not None and facts["device_type"] != device_type:
                skipped_device_tensors += 1
                continue
            total_tensors += 1
            row["tensor_count"] += 1
            row["logical_bytes"] += facts["logical_bytes"]
            if facts["pinned"]:
                row["pinned_tensor_count"] += 1
                row["pinned_logical_bytes"] += facts["logical_bytes"]
            storage_key = facts["storage_key"]
            if storage_key not in owner_storage_keys:
                owner_storage_keys.add(storage_key)
                row["unique_storage_bytes"] += facts["storage_bytes"]
            record = storage_records.setdefault(
                storage_key,
                {
                    "storage_key": storage_key,
                    "storage_bytes": facts["storage_bytes"],
                    "occurrences": 0,
                    "owners": set(),
                    "sample_names": [],
                },
            )
            record["occurrences"] += 1
            record["owners"].add(owner)
            if len(record["sample_names"]) < 8:
                record["sample_names"].append(_bounded_text(name, 384))
            group_key = (facts["device"], facts["dtype"], facts["pinned"])
            group = group_records.setdefault(
                group_key,
                {
                    "device": facts["device"],
                    "dtype": facts["dtype"],
                    "pinned": facts["pinned"],
                    "tensor_count": 0,
                    "logical_bytes": 0,
                },
            )
            group["tensor_count"] += 1
            group["logical_bytes"] += facts["logical_bytes"]
            if len(details) < max_tensor_details:
                details.append(
                    {"owner": owner, "name": _bounded_text(name, 384), **facts}
                )
        if bool(walk_state["truncated"]):
            row["walk_truncated"] = True
            scan_truncated = True
        owner_rows[owner] = row
        if total_tensors >= max_tensors:
            break

    duplicates = []
    for record in storage_records.values():
        if record["occurrences"] > 1 or len(record["owners"]) > 1:
            duplicates.append(
                {
                    **{key: value for key, value in record.items() if key != "owners"},
                    "owners": sorted(record["owners"]),
                }
            )
    duplicates.sort(key=lambda item: (-item["storage_bytes"], item["storage_key"]))
    unique_storage_bytes = sum(
        record["storage_bytes"] for record in storage_records.values()
    )
    referenced_storage_bytes = sum(
        record["storage_bytes"] * record["occurrences"]
        for record in storage_records.values()
    )
    return {
        "available": True,
        "device_filter": device_type,
        "owner_count": len(owner_rows),
        "tensor_count": total_tensors,
        "logical_bytes": sum(row["logical_bytes"] for row in owner_rows.values()),
        "unique_storage_count": len(storage_records),
        "unique_storage_bytes": unique_storage_bytes,
        "referenced_storage_bytes": referenced_storage_bytes,
        "alias_storage_overcount_bytes": max(
            0, referenced_storage_bytes - unique_storage_bytes
        ),
        "pinned_logical_bytes": sum(
            row["pinned_logical_bytes"] for row in owner_rows.values()
        ),
        "skipped_device_tensors": skipped_device_tensors,
        "owners": owner_rows,
        "groups": sorted(
            group_records.values(),
            key=lambda item: (-item["logical_bytes"], item["device"], item["dtype"]),
        ),
        "duplicate_storages": duplicates[:max_duplicate_storages],
        "duplicate_storages_truncated": len(duplicates) > max_duplicate_storages,
        "tensor_details": details,
        "tensor_details_truncated": total_tensors > len(details),
        "scan_truncated": scan_truncated,
    }


def inventory_cpu_tensors(
    owners: Mapping[str, Any] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper for a caller-labeled live CPU tensor inventory."""
    return inventory_tensors(owners, device_type="cpu", **kwargs)


def _summarize_cuda_snapshot(snapshot: Any) -> dict[str, Any]:
    devices: dict[str, dict[str, Any]] = {}
    for segment in snapshot or []:
        device = str(segment.get("device", "unknown"))
        row = devices.setdefault(
            device,
            {
                "segment_count": 0,
                "total_size_bytes": 0,
                "allocated_size_bytes": 0,
                "active_size_bytes": 0,
                "requested_size_bytes": 0,
                "block_states": {},
            },
        )
        row["segment_count"] += 1
        for source, target in (
            ("total_size", "total_size_bytes"),
            ("allocated_size", "allocated_size_bytes"),
            ("active_size", "active_size_bytes"),
            ("requested_size", "requested_size_bytes"),
        ):
            row[target] += int(segment.get(source, 0) or 0)
        for block in segment.get("blocks", ()):
            state = _bounded_text(block.get("state", "unknown"), 64)
            state_row = row["block_states"].setdefault(
                state, {"block_count": 0, "size_bytes": 0, "requested_size_bytes": 0}
            )
            state_row["block_count"] += 1
            state_row["size_bytes"] += int(block.get("size", 0) or 0)
            state_row["requested_size_bytes"] += int(
                block.get("requested_size", 0) or 0
            )
    return {"devices": devices, "raw_snapshot_omitted": True}


def collect_cuda_memory(
    *,
    torch_module: Any = _UNSET,
    devices: Iterable[int] | None = None,
    include_snapshot: bool = False,
    allow_cuda_initialization: bool = False,
) -> dict[str, Any]:
    """Collect allocator counters without synchronizing or retaining tensors.

    By default this will not initialize CUDA merely to observe it. That keeps a
    pre-context checkpoint from changing the workload it is intended to measure.
    Set ``allow_cuda_initialization`` only for an explicitly intrusive probe.
    """
    if torch_module is _UNSET:
        try:
            torch_module = importlib.import_module("torch")
        except ImportError:
            torch_module = None
    if torch_module is None:
        return {"available": False, "reason": "PyTorch is not installed", "devices": []}
    try:
        is_initialized = getattr(torch_module.cuda, "is_initialized", None)
        if (
            callable(is_initialized)
            and not bool(is_initialized())
            and not allow_cuda_initialization
        ):
            return {
                "available": False,
                "initialized": False,
                "reason": "CUDA is not initialized; skipped to avoid perturbing the workload",
                "torch_version": str(getattr(torch_module, "__version__", "unknown")),
                "devices": [],
            }
        if not bool(torch_module.cuda.is_available()):
            return {
                "available": False,
                "reason": "torch.cuda.is_available() is false",
                "torch_version": str(getattr(torch_module, "__version__", "unknown")),
                "devices": [],
            }
        count = int(torch_module.cuda.device_count())
        device_ids = list(devices) if devices is not None else list(range(count))
        output_devices: list[dict[str, Any]] = []
        stat_keys = (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "active_bytes.all.current",
            "active_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "inactive_split_bytes.all.current",
            "inactive_split_bytes.all.peak",
            "num_alloc_retries",
            "num_ooms",
        )
        for device in device_ids:
            try:
                stats = torch_module.cuda.memory_stats(device)
                properties = torch_module.cuda.get_device_properties(device)
                row: dict[str, Any] = {
                    "index": int(device),
                    "name": _bounded_text(
                        getattr(properties, "name", f"cuda:{device}"), 256
                    ),
                    "total_memory_bytes": int(getattr(properties, "total_memory", 0)),
                    "allocated_bytes": int(torch_module.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch_module.cuda.memory_reserved(device)),
                    "peak_allocated_bytes": int(
                        torch_module.cuda.max_memory_allocated(device)
                    ),
                    "peak_reserved_bytes": int(
                        torch_module.cuda.max_memory_reserved(device)
                    ),
                    "memory_stats": {key: int(stats.get(key, 0)) for key in stat_keys},
                }
                if hasattr(torch_module.cuda, "mem_get_info"):
                    free, total = torch_module.cuda.mem_get_info(device)
                    row["driver_free_bytes"] = int(free)
                    row["driver_total_bytes"] = int(total)
                output_devices.append(row)
            except (RuntimeError, ValueError, AssertionError) as exc:
                output_devices.append(
                    {
                        "index": int(device),
                        "available": False,
                        "error": _bounded_text(exc),
                    }
                )
        result: dict[str, Any] = {
            "available": True,
            "torch_version": str(getattr(torch_module, "__version__", "unknown")),
            "cuda_version": str(
                getattr(getattr(torch_module, "version", None), "cuda", "unknown")
            ),
            "device_count": count,
            "devices": output_devices,
        }
        if include_snapshot:
            try:
                result["allocator_snapshot_summary"] = _summarize_cuda_snapshot(
                    torch_module.cuda.memory_snapshot()
                )
            except (AttributeError, RuntimeError) as exc:
                result["allocator_snapshot_error"] = _bounded_text(exc)
        return result
    except (RuntimeError, AssertionError, AttributeError) as exc:
        return {"available": False, "reason": _bounded_text(exc), "devices": []}


def _decode_nvml(value: Any) -> str:
    return (
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
    )


def _collect_pynvml(module: Any, pid: int) -> dict[str, Any]:
    module.nvmlInit()
    try:
        devices = []
        count = int(module.nvmlDeviceGetCount())
        for index in range(count):
            handle = module.nvmlDeviceGetHandleByIndex(index)
            memory = module.nvmlDeviceGetMemoryInfo(handle)
            row: dict[str, Any] = {
                "index": index,
                "name": _decode_nvml(module.nvmlDeviceGetName(handle)),
                "uuid": _decode_nvml(module.nvmlDeviceGetUUID(handle)),
                "memory_total_bytes": int(memory.total),
                "memory_used_bytes": int(memory.used),
                "memory_free_bytes": int(memory.free),
            }
            try:
                pci = module.nvmlDeviceGetPciInfo(handle)
                row["pci_bus_id"] = _decode_nvml(getattr(pci, "busId", "unknown"))
            except Exception as exc:
                row["pci_error"] = _bounded_text(exc)
            try:
                utilization = module.nvmlDeviceGetUtilizationRates(handle)
                row["gpu_utilization_percent"] = int(utilization.gpu)
                row["memory_utilization_percent"] = int(utilization.memory)
            except Exception as exc:
                row["utilization_error"] = _bounded_text(exc)
            process_samples: list[int] = []
            compute_functions = (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses",
            )
            for function_name in compute_functions:
                function = getattr(module, function_name, None)
                if function is None:
                    continue
                try:
                    for process in function(handle):
                        if int(process.pid) == pid:
                            used = getattr(process, "usedGpuMemory", 0)
                            if isinstance(used, int) and 0 <= used <= int(memory.total):
                                process_samples.append(used)
                except Exception:
                    continue
                # v3 and legacy accessors expose the same compute-process table.
                break
            graphics_function = getattr(
                module, "nvmlDeviceGetGraphicsRunningProcesses", None
            )
            if graphics_function is not None:
                try:
                    for process in graphics_function(handle):
                        if int(process.pid) == pid:
                            used = getattr(process, "usedGpuMemory", 0)
                            if isinstance(used, int) and 0 <= used <= int(memory.total):
                                process_samples.append(used)
                except Exception:
                    pass
            # A process can appear in both compute and graphics tables; max avoids
            # double-counting the same per-device residency.
            row["process_used_bytes"] = (
                max(process_samples) if process_samples else None
            )
            devices.append(row)
        try:
            driver_version = _decode_nvml(module.nvmlSystemGetDriverVersion())
        except Exception:
            driver_version = "unknown"
        return {
            "available": True,
            "method": "pynvml",
            "driver_version": driver_version,
            "devices": devices,
        }
    finally:
        module.nvmlShutdown()


def _parse_int_field(value: str) -> int | None:
    stripped = value.strip()
    return int(stripped) if stripped and stripped not in {"N/A", "[N/A]", "-"} else None


def _collect_nvidia_smi(path: str, pid: int, timeout: float) -> dict[str, Any]:
    fields = [
        "index",
        "name",
        "uuid",
        "pci.bus_id",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "utilization.memory",
        "driver_version",
    ]
    command = [path, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            _bounded_text(
                completed.stderr.strip() or f"nvidia-smi exited {completed.returncode}"
            )
        )
    devices = []
    for values in csv.reader(completed.stdout.splitlines()):
        if len(values) != len(fields):
            continue
        total, used, free = (_parse_int_field(values[index]) for index in (4, 5, 6))
        devices.append(
            {
                "index": _parse_int_field(values[0]),
                "name": values[1].strip(),
                "uuid": values[2].strip(),
                "pci_bus_id": values[3].strip(),
                "memory_total_bytes": total * 1024**2 if total is not None else None,
                "memory_used_bytes": used * 1024**2 if used is not None else None,
                "memory_free_bytes": free * 1024**2 if free is not None else None,
                "gpu_utilization_percent": _parse_int_field(values[7]),
                "memory_utilization_percent": _parse_int_field(values[8]),
                "driver_version": values[9].strip(),
                "process_used_bytes": None,
            }
        )
    by_uuid = {row["uuid"]: row for row in devices}
    process_command = [
        path,
        "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    process_result = subprocess.run(
        process_command, capture_output=True, text=True, timeout=timeout, check=False
    )
    if process_result.returncode == 0:
        for values in csv.reader(process_result.stdout.splitlines()):
            if len(values) != 3 or _parse_int_field(values[0]) != pid:
                continue
            used = _parse_int_field(values[2])
            if values[1].strip() in by_uuid and used is not None:
                by_uuid[values[1].strip()]["process_used_bytes"] = used * 1024**2
    return {"available": True, "method": "nvidia-smi", "devices": devices}


def collect_nvml_memory(
    pid: int | None = None,
    *,
    pynvml_module: Any = _UNSET,
    nvidia_smi_path: str | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Collect device/global and current-process memory via NVML or nvidia-smi."""
    target_pid = int(pid or os.getpid())
    errors: list[dict[str, str]] = []
    if pynvml_module is _UNSET:
        try:
            pynvml_module = importlib.import_module("pynvml")
        except ImportError:
            pynvml_module = None
    if pynvml_module is not None:
        try:
            result = _collect_pynvml(pynvml_module, target_pid)
            result["pid"] = target_pid
            return result
        except Exception as exc:
            errors.append({"source": "pynvml", "error": _bounded_text(exc)})
    executable = (
        nvidia_smi_path if nvidia_smi_path is not None else shutil.which("nvidia-smi")
    )
    if executable:
        try:
            result = _collect_nvidia_smi(executable, target_pid, timeout)
            result["pid"] = target_pid
            result["errors"] = errors
            return result
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            errors.append({"source": "nvidia-smi", "error": _bounded_text(exc)})
    return {
        "available": False,
        "pid": target_pid,
        "reason": "NVML and nvidia-smi are unavailable",
        "errors": errors,
        "devices": [],
    }


def collect_memory_snapshot(
    *,
    pid: int | None = None,
    tensor_owners: Mapping[str, Any] | None = None,
    tensor_owner_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    include_maps: bool = False,
    include_smaps: bool = False,
    include_cuda: bool = True,
    include_cuda_snapshot: bool = False,
    allow_cuda_initialization: bool = False,
    include_nvml: bool = True,
    max_regions: int = 64,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Take one timestamped, JSON-serializable memory checkpoint."""
    target_pid = int(pid or os.getpid())
    snapshot: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": _utc_now(),
        "timestamp_unix_ns": time.time_ns(),
        "pid": target_pid,
        "proc": collect_proc_memory(
            target_pid,
            include_maps=include_maps,
            include_smaps=include_smaps,
            max_regions=max_regions,
        ),
        "psutil": collect_psutil_memory(target_pid),
        "page_faults": collect_page_faults(target_pid),
        "metadata": json_safe(metadata or {}, max_depth=5, max_items=128),
    }
    if tensor_owners is not None:
        snapshot["cpu_tensors"] = inventory_cpu_tensors(
            tensor_owners, owner_metadata=tensor_owner_metadata, max_tensor_details=256
        )
    if include_cuda:
        snapshot["cuda"] = collect_cuda_memory(
            include_snapshot=include_cuda_snapshot,
            allow_cuda_initialization=allow_cuda_initialization,
        )
    if include_nvml:
        snapshot["nvml"] = collect_nvml_memory(target_pid)
    # Assert the contract now, close to whichever optional collector violated it.
    json.dumps(snapshot, allow_nan=False)
    return snapshot


class MemoryProbe:
    """A bounded phase-checkpoint timeline safe to keep beside a live model."""

    def __init__(
        self,
        *,
        pid: int | None = None,
        max_checkpoints: int = 512,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if max_checkpoints < 2:
            raise ValueError("max_checkpoints must be at least 2")
        self.pid = int(pid or os.getpid())
        self.max_checkpoints = int(max_checkpoints)
        self._clock = clock
        self._start = clock()
        self._started_utc = _utc_now()
        self._checkpoints: list[dict[str, Any]] = []
        self._dropped = 0
        self._lock = threading.Lock()

    def checkpoint(
        self,
        label: str,
        *,
        tensor_owners: Mapping[str, Any] | None = None,
        tensor_owner_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        include_maps: bool = False,
        include_smaps: bool = False,
        include_cuda: bool = True,
        include_cuda_snapshot: bool = False,
        allow_cuda_initialization: bool = False,
        include_nvml: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not str(label).strip():
            raise ValueError("checkpoint label must not be empty")
        collection_start = self._clock()
        entry = collect_memory_snapshot(
            pid=self.pid,
            tensor_owners=tensor_owners,
            tensor_owner_metadata=tensor_owner_metadata,
            include_maps=include_maps,
            include_smaps=include_smaps,
            include_cuda=include_cuda,
            include_cuda_snapshot=include_cuda_snapshot,
            allow_cuda_initialization=allow_cuda_initialization,
            include_nvml=include_nvml,
            metadata=metadata,
        )
        collection_end = self._clock()
        entry["label"] = _bounded_text(label, 256)
        entry["elapsed_seconds"] = float(collection_start - self._start)
        entry["collection_seconds"] = float(collection_end - collection_start)
        with self._lock:
            if len(self._checkpoints) >= self.max_checkpoints:
                # Preserve the process-start checkpoint and the newest window.
                del self._checkpoints[1]
                self._dropped += 1
            self._checkpoints.append(entry)
        return entry

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            # JSON round-trip gives callers an isolated copy and enforces the API contract.
            checkpoints = json.loads(json.dumps(self._checkpoints, allow_nan=False))
            dropped = self._dropped
        return {
            "schema_version": SCHEMA_VERSION,
            "pid": self.pid,
            "started_utc": self._started_utc,
            "checkpoint_count": len(checkpoints),
            "dropped_checkpoint_count": dropped,
            "checkpoints": checkpoints,
        }

    def write_json(self, path: os.PathLike[str] | str, *, indent: int = 2) -> Path:
        """Write the bounded timeline atomically and return its path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self.as_dict(), indent=indent, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target


__all__ = [
    "MemoryProbe",
    "SCHEMA_VERSION",
    "collect_cuda_memory",
    "collect_memory_snapshot",
    "collect_nvml_memory",
    "collect_page_faults",
    "collect_proc_memory",
    "collect_psutil_memory",
    "inventory_cpu_tensors",
    "inventory_tensors",
    "json_safe",
    "parse_proc_key_values",
    "summarize_maps",
    "summarize_smaps",
]
