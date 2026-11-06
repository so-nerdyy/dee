#!/usr/bin/env python3
"""M3 v6 trace_alloc diagnostic analyzer.

Strict no-fix analyzer that:
- Walks every `logs/matrix-driver.log` (and any other *.log) under the
  downloaded evidence directory.
- Parses DEE_TA_* stderr markers line-by-line:
    [ta_alloc] id=N ptr=0x... kind=... size=N owner=... alloc=... file=...:L tid=...
    [ta_free]  id=N ptr=0x... kind=... alloc=... owner=... file=...:L
    [ta_insert] map=... key=... ptr=0x... origin=... file=...:L (alloc_rec=match)
    [ta_insert_USE_AFTER_FREE_ABORT]    ptr=0x... alloc_rec.alive=false  site=...
    [ta_insert_MISMATCH_ABORT]          ptr=0x... alloc_rec.kind=K alloc_rec.allocator=A ...
    [ta_free_DOUBLE_FREE_ABORT]         ptr=0x... original_id=N original_site=...:L ...
    [ta_free_UNALLOC_ABORT]             ptr=0x... (ptr was never registered)
- Tracks the FIRST `_ABORT` token (in chronological per-file order) and the
  FIRST glibc heap-check token ("double free or corruption (!prev)").
- Reconstructs the COMPLETE LIFETIME of the failing pointer: allocation ->
  -> ownership transfers -> container insertions -> frees -> invalid op.
- Writes the report under --output-dir:
    lifetime_report.json     -- structured first-failing-pointer record
    timeline.txt             -- chronological human-readable timeline
    markers_index.json       -- per-log-file grep-hit counts (audit trail)
    markers_index.txt        -- same, as plain text
    abort_summary.txt        -- first abort region + ALL subsequent abort
                                sites across the matrix (not just the first)
- ZERO fix recommendations. Per the user directive, evidence only.

Usage:
  python m3_analyze_trace.py --evidence-dir <path> [--output-dir <path>]
                              [--log-globs '*.log']
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


# ------------------------------------------------------------------
# Per-line marker regexes. Each has a small named-group shape so that
# the parser can pull out the fields without ambiguity. Note: stderr
# from glibc uses unstructured formats; the markers are matched last.
# ------------------------------------------------------------------
RE_TA_ALLOC = re.compile(
    r"\[ta_alloc\]\s+id=(?P<id>\d+)\s+ptr=(?P<ptr>0x[0-9a-fA-F]+|\w+)\s+"
    r"kind=(?P<kind>\w+)\s+size=(?P<size>\d+)\s+owner=(?P<owner>\S+)\s+"
    r"alloc=(?P<alloc>\S+)\s+file=(?P<file>[^:]+):(?P<line>\d+)\s+"
    r"tid=(?P<tid>-?\d+)"
)
RE_TA_FREE = re.compile(
    r"\[ta_free\]\s+id=(?P<id>\d+)\s+ptr=(?P<ptr>0x[0-9a-fA-F]+|\w+)\s+"
    r"kind=(?P<kind>\w+)\s+alloc=(?P<alloc>\S+)\s+owner=(?P<owner>\S+)\s+"
    r"file=(?P<file>[^:]+):(?P<line>\d+)"
)
RE_TA_INSERT = re.compile(
    r"\[ta_insert\]\s+map=(?P<map>\S+)\s+key=(?P<key>\d+)\s+"
    r"ptr=(?P<ptr>0x[0-9a-fA-F]+|\w+)\s+origin=(?P<origin>\S+)\s+"
    r"file=(?P<file>[^:]+):(?P<line>\d+)(?:\s+\(alloc_rec=match\))?"
)
RE_TA_INSERT_UAF = re.compile(
    r"\[ta_insert_USE_AFTER_FREE_ABORT\]\s+map=(?P<map>\S+)\s+"
    r"key=(?P<key>\d+)\s+ptr=(?P<ptr>0x[0-9a-fA-F]+|\w+)\s+"
    r"alloc_rec\.id=(?P<alloc_id>\d+)\s+alloc_rec\.kind=(?P<alloc_kind>\w+)\s+"
    r"alloc_rec\.allocator=(?P<alloc_allocator>\S+)\s+"
    r"alloc_rec\.site=(?P<alloc_site>[^:]+):(?P<alloc_line>\d+)\s+"
    r"alloc_rec\.alive=false\s+insert_site=(?P<insert_site>[^:]+):(?P<insert_line>\d+)"
)
RE_TA_INSERT_MISMATCH = re.compile(
    r"\[ta_insert_MISMATCH_ABORT\]\s+map=(?P<map>\S+)\s+"
    r"key=(?P<key>\d+)\s+ptr=(?P<ptr>0x[0-9a-fA-F]+|\w+)\s+"
    r"alloc_rec\.kind=(?P<alloc_kind>\w+)\s+"
    r"alloc_rec\.allocator=(?P<alloc_allocator>\S+)\s+"
    r"alloc_rec\.id=(?P<alloc_id>\d+)\s+"
    r"alloc_rec\.site=(?P<alloc_site>[^:]+):(?P<alloc_line>\d+)\s+"
    r"claimed_origin=(?P<claimed_origin>\S+)\s+"
    r"insert_site=(?P<insert_site>[^:]+):(?P<insert_line>\d+)"
)
RE_TA_FREE_DOUBLE = re.compile(
    r"\[ta_free_DOUBLE_FREE_ABORT\]\s+owner=(?P<owner>\S+)\s+"
    r"ptr=(?P<ptr>0x[0-9a-fA-F]+|\w+)\s+"
    r"file=(?P<file>[^:]+):(?P<line>\d+)\s+alloc=(?P<alloc>\S+)\s+"
    r"original_id=(?P<original_id>\d+)\s+"
    r"original_site=(?P<original_site>[^:]+):(?P<original_line>\d+)\s+"
    r"lifetime=(?P<lifetime>[^:]+):(?P<lifetime_line>\d+)"
)
RE_TA_FREE_UNALLOC = re.compile(
    r"\[ta_free_UNALLOC_ABORT\]\s+owner=(?P<owner>\S+)\s+"
    r"ptr=(?P<ptr>0x[0-9a-fA-F]+|\w+)\s+file=(?P<file>[^:]+):(?P<line>\d+)"
)
RE_GLIBC_DOUBLE_FREE = re.compile(
    # glibc emits several shapes across toolchain versions; cover all of
    # them so the analyzer doesn't miss the v4 (!prev) signature.
    r"double free or corruption.+\(!prev\)|"
    r"double free or corruption \(!prev\)|"
    r"double free or corruption \(\s*!\s*prev\s*\)|"     # ( !prev )
    r"\(!prev\)"                                          # bare in dump
)
RE_GLIBC_GENERIC = re.compile(
    r"double free or corruption(?: \(!prev\))?|"
    r"munmap_chunk\(\): invalid pointer|"
    r"free\(\): invalid pointer|"
    r"free\(\): invalid size"
)
RE_ADDRESS_SANITIZER = re.compile(
    r"AddressSanitizer|compute-sanitizer"
)


@dataclass
class AllocRecord:
    """Reconstructed AllocRec mirrored from trace_alloc.cpp's struct."""
    id: int
    ptr: str
    kind: str
    alloc: str
    owner: str
    file: str
    line: int
    tid: int = 0
    size: int = 0
    freed_at: tuple[str, int] | None = None  # file, line of matching free
    inserts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InvalidOp:
    """An abort event surfaced by stderr."""
    op_type: str                    # "USE_AFTER_FREE_ABORT" / "MISMATCH_ABORT" ...
    ptr_repr: str
    raw_text: str
    source_log: str
    source_log_relative: str
    line_number: int
    pointer: str
    alloc_id: int | None
    alloc_kind: str | None
    alloc_allocator: str | None
    alloc_site: str | None
    alloc_line: int | None
    insert_site: str | None
    insert_line: int | None
    map_name: str | None = None
    key: int | None = None
    claimed_origin: str | None = None
    original_id: int | None = None
    original_site: str | None = None
    original_line: int | None = None
    lifetime_line: int | None = None
    owner: str | None = None


@dataclass
class FileStats:
    log_path: str
    log_relative: str
    bytes: int
    line_count: int
    counts_by_marker: dict[str, int] = field(default_factory=dict)
    first_invalid_op: InvalidOp | None = None
    glibc_first_hit: dict[str, Any] | None = None


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_marker_line(line: str) -> dict[str, Any] | None:
    """Return parsed marker dict or None if line isn't a recognized marker."""
    for name, regex in (
        ("alloc",         RE_TA_ALLOC),
        ("free",          RE_TA_FREE),
        ("insert",        RE_TA_INSERT),
        ("insert_uaf",    RE_TA_INSERT_UAF),
        ("insert_mismatch", RE_TA_INSERT_MISMATCH),
        ("free_double",   RE_TA_FREE_DOUBLE),
        ("free_unalloc",  RE_TA_FREE_UNALLOC),
    ):
        m = regex.search(line)
        if m:
            return {"name": name, "groups": m.groupdict()}
    return None


def yield_log_lines(log_path: Path) -> Iterator[tuple[int, str]]:
    """Memory-bounded line generator that also surfaces line numbers."""
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            yield lineno, raw.rstrip("\r\n")


def build_invalid_op(
    parser_name: str, m: re.Match, log_path: Path, line_number: int,
    raw_text: str,
) -> InvalidOp:
    """Construct an InvalidOp from a regex Match for any abort marker."""
    g = m.groupdict()
    base = dict(
        source_log=str(log_path),
        source_log_relative=str(log_path),
        line_number=line_number,
        pointer=g.get("ptr"),
    )
    if parser_name == "insert_uaf":
        return InvalidOp(
            op_type="USE_AFTER_FREE_ABORT",
            ptr_repr=g["ptr"],
            raw_text=raw_text,
            **base,
            alloc_id=int(g["alloc_id"]),
            alloc_kind=g.get("alloc_kind"),
            alloc_allocator=g.get("alloc_allocator"),
            alloc_site=g.get("alloc_site"),
            alloc_line=int(g["alloc_line"]),
            insert_site=g.get("insert_site"),
            insert_line=int(g["insert_line"]),
            map_name=g.get("map"),
            key=int(g["key"]),
        )
    if parser_name == "insert_mismatch":
        return InvalidOp(
            op_type="MISMATCH_ABORT",
            ptr_repr=g["ptr"],
            raw_text=raw_text,
            **base,
            alloc_id=int(g["alloc_id"]),
            alloc_kind=g.get("alloc_kind"),
            alloc_allocator=g.get("alloc_allocator"),
            alloc_site=g.get("alloc_site"),
            alloc_line=int(g["alloc_line"]),
            insert_site=g.get("insert_site"),
            insert_line=int(g["insert_line"]),
            map_name=g.get("map"),
            key=int(g["key"]),
            claimed_origin=g.get("claimed_origin"),
        )
    if parser_name == "free_double":
        return InvalidOp(
            op_type="DOUBLE_FREE_ABORT",
            ptr_repr=g["ptr"],
            raw_text=raw_text,
            **base,
            owner=g.get("owner"),
            alloc_id=int(g["original_id"]),
            alloc_allocator=g.get("alloc"),
            alloc_site=g.get("original_site"),
            alloc_line=int(g["original_line"]),
            insert_site=None,
            insert_line=int(g["lifetime_line"]),
            original_id=int(g["original_id"]),
            original_site=g.get("original_site"),
            original_line=int(g["original_line"]),
            lifetime_line=int(g["lifetime_line"]),
        )
    if parser_name == "free_unalloc":
        return InvalidOp(
            op_type="UNALLOC_ABORT",
            ptr_repr=g["ptr"],
            raw_text=raw_text,
            **base,
            owner=g.get("owner"),
        )
    raise ValueError(f"unexpected parser_name={parser_name}")


def analyze_log_file(log_path: Path, evidence_root: Path) -> tuple[
    FileStats,
    dict[str, AllocRecord],
    list[InvalidOp],
]:
    """Parse one *.log file; return stats + the alloc registry keyed by
    pointer + ALL invalid ops in chronological order."""
    stats = FileStats(
        log_path=str(log_path),
        log_relative=str(log_path.relative_to(evidence_root)) if log_path.is_relative_to(evidence_root)
                     else str(log_path),
        bytes=log_path.stat().st_size,
        line_count=0,
    )
    registry: dict[str, AllocRecord] = {}
    invalid_ops: list[InvalidOp] = []

    for lineno, raw in yield_log_lines(log_path):
        stats.line_count += 1

        parsed = parse_marker_line(raw)
        if parsed is not None:
            name = parsed["name"]
            stats.counts_by_marker[name] = stats.counts_by_marker.get(name, 0) + 1
            g = parsed["groups"]
            if name == "alloc":
                ptr = g["ptr"]
                registry[ptr] = AllocRecord(
                    id=int(g["id"]),
                    ptr=ptr,
                    kind=g["kind"],
                    alloc=g["alloc"],
                    owner=g["owner"],
                    file=g["file"],
                    line=int(g["line"]),
                    tid=int(g["tid"]),
                    size=int(g["size"]),
                )
            elif name == "free":
                ptr = g["ptr"]
                rec = registry.get(ptr)
                if rec is not None:
                    rec.freed_at = (g["file"], int(g["line"]))
            elif name == "insert":
                ptr = g["ptr"]
                rec = registry.get(ptr)
                insert_event = {
                    "map": g["map"],
                    "key": int(g["key"]),
                    "origin": g["origin"],
                    "site": f"{g['file']}:{g['line']}",
                    "file": g["file"],
                    "line": int(g["line"]),
                }
                if rec is not None:
                    rec.inserts.append(insert_event)
            elif name in ("insert_uaf", "insert_mismatch",
                          "free_double", "free_unalloc"):
                inv = build_invalid_op(name, re_match:=_match(raw, name), log_path, lineno, raw)
                invalid_ops.append(inv)
                if stats.first_invalid_op is None:
                    stats.first_invalid_op = inv
            continue  # no need for glibc regex after a recognized marker

        if RE_GLIBC_DOUBLE_FREE.search(raw):
            stats.counts_by_marker["glibc_double_free_!prev"] = \
                stats.counts_by_marker.get("glibc_double_free_!prev", 0) + 1
            if stats.glibc_first_hit is None:
                stats.glibc_first_hit = {
                    "line_number": lineno,
                    "raw_text": raw,
                    "context_before": _context_lines(log_path, lineno, before=3),
                    "context_after": _context_lines(log_path, lineno, before=0, after=10),
                }
        elif RE_GLIBC_GENERIC.search(raw):
            stats.counts_by_marker["glibc_generic_corruption"] = \
                stats.counts_by_marker.get("glibc_generic_corruption", 0) + 1
        elif RE_ADDRESS_SANITIZER.search(raw):
            stats.counts_by_marker["address_sanitizer"] = \
                stats.counts_by_marker.get("address_sanitizer", 0) + 1
    return stats, registry, invalid_ops


def _match(line: str, parser_name: str) -> re.Match:
    """Capture an already-known matched group for build_invalid_op."""
    if parser_name == "insert_uaf":
        return RE_TA_INSERT_UAF.search(line)  # type: ignore
    if parser_name == "insert_mismatch":
        return RE_TA_INSERT_MISMATCH.search(line)  # type: ignore
    if parser_name == "free_double":
        return RE_TA_FREE_DOUBLE.search(line)  # type: ignore
    if parser_name == "free_unalloc":
        return RE_TA_FREE_UNALLOC.search(line)  # type: ignore
    raise ValueError(f"unexpected parser_name={parser_name}")


def _context_lines(log_path: Path, anchor_lineno: int,
                   before: int = 0, after: int = 6) -> list[str]:
    """Pull `before` lines BEFORE and `after` lines AFTER anchor_lineno."""
    out: list[str] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if anchor_lineno - before <= lineno <= anchor_lineno + after:
                out.append(f"L{lineno}: {raw.rstrip()}")
    return out


def reconstruct_lifetime(
    op: InvalidOp, registry: dict[str, AllocRecord],
    invalid_ops_all: list[InvalidOp],
) -> dict[str, Any]:
    """Build a structured lifetime record for `op`."""
    alloc_rec = registry.get(op.pointer)
    events: list[dict[str, Any]] = []

    if alloc_rec is not None:
        events.append({
            "phase": "ALLOC",
            "id": alloc_rec.id,
            "ptr": alloc_rec.ptr,
            "kind": alloc_rec.kind,
            "allocator": alloc_rec.alloc,
            "owner": alloc_rec.owner,
            "site": f"{alloc_rec.file}:{alloc_rec.line}",
            "tid": alloc_rec.tid,
            "size": alloc_rec.size,
        })
        for ins in alloc_rec.inserts:
            events.append({
                "phase": "INSERT",
                "map": ins["map"],
                "key": ins["key"],
                "claimed_origin": ins["origin"],
                "site": ins["site"],
            })
        if alloc_rec.freed_at is not None:
            f_file, f_line = alloc_rec.freed_at
            events.append({
                "phase": "FREE",
                "id": alloc_rec.id,
                "ptr": alloc_rec.ptr,
                "site": f"{f_file}:{f_line}",
            })
    else:
        events.append({
            "phase": "ALLOC",
            "id": None,
            "ptr": op.pointer,
            "note": "no ta_alloc record for this pointer (UNALLOC or wrapper bypass)",
        })
    events.append({
        "phase": "FATAL",
        "op_type": op.op_type,
        "ptr": op.pointer,
        "site": (f"{op.insert_site}:{op.insert_line}"
                 if op.insert_site else op.source_log),
        "claimed_origin": op.claimed_origin,
        "alloc_id": op.alloc_id,
        "alloc_kind": op.alloc_kind,
        "alloc_allocator": op.alloc_allocator,
        "alloc_site": (f"{op.alloc_site}:{op.alloc_line}"
                       if op.alloc_site else None),
        "raw_line": op.raw_text,
    })
    # Sanity: total count of subsequent invalid ops that touch the same
    # pointer, to expose cascade noise.
    same_ptr_cascade = [o for o in invalid_ops_all
                        if o.pointer == op.pointer and o is not op]
    return {
        "target_pointer": op.pointer,
        "op_type": op.op_type,
        "source_log": op.source_log_relative,
        "source_line": op.line_number,
        "events": events,
        "same_pointer_cascade_count": len(same_ptr_cascade),
        "registered": alloc_rec is not None,
        "alloc_id_when_known": (None if alloc_rec is None else alloc_rec.id),
    }


def write_outputs(
    out_dir: Path,
    per_file: list[tuple[FileStats, dict[str, AllocRecord], list[InvalidOp]]],
    evidence_root: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- markers_index.json / .txt -----------------------------------
    markers_index = {
        "schema_version": 1,
        "generated_at": stamp(),
        "log_files": [
            {
                "log_relative": stats.log_relative,
                "log_path": stats.log_path,
                "bytes": stats.bytes,
                "line_count": stats.line_count,
                "counts_by_marker": stats.counts_by_marker,
                "first_invalid_op": (
                    None if stats.first_invalid_op is None else
                    asdict(stats.first_invalid_op)
                ),
                "glibc_first_hit": stats.glibc_first_hit,
            } for stats, _, _ in per_file
        ],
    }
    (out_dir / "markers_index.json").write_text(
        json.dumps(markers_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines: list[str] = []
    lines.append(f"=== {stamp()} markers_index (M3 v6) ===")
    for stats, _, inv in per_file:
        lines.append(f"-- {stats.log_relative}: "
                     f"bytes={stats.bytes} lines={stats.line_count}"
                     f"  markers={stats.counts_by_marker}"
                     f"  invalid_ops_in_file={len(inv)}")
    (out_dir / "markers_index.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )

    # ---- lifetime_report.json / timeline.txt / abort_summary.txt ------
    # Find the FIRST invalid op across all files (chronological union is
    # hard; with per-file marker counts we tiebreak by lexically smallest
    # log path so the run order matches the matrix order).
    first_op: tuple[InvalidOp, dict[str, AllocRecord], list[InvalidOp]] | None = None
    for stats, registry, inv in per_file:
        for op in inv:
            if op is stats.first_invalid_op:
                if first_op is None or stats.log_relative < first_op[0].source_log_relative:
                    first_op = (op, registry, inv)
    all_ops_unrolled: list[tuple[str, InvalidOp]] = [
        (stats.log_relative, op)
        for stats, _, ops in per_file for op in ops
    ]
    abort_summary_lines: list[str] = [
        f"=== {stamp()} abort_summary (M3 v6) ===",
        f"total_invalid_ops={len(all_ops_unrolled)}",
        f"first_pointer={first_op[0].pointer if first_op else None}",
    ]
    for source_log, op in all_ops_unrolled:
        abort_summary_lines.append(
            f"{source_log}:L{op.line_number}  {op.op_type}  "
            f"ptr={op.pointer}  site="
            f"{(op.insert_site or op.source_log)}:{op.insert_line or op.line_number}"
        )
    (out_dir / "abort_summary.txt").write_text(
        "\n".join(abort_summary_lines) + "\n", encoding="utf-8",
    )

    if first_op is None:
        # No abort markers anywhere; emit a clean no-abort report.
        report = {
            "schema_version": 1,
            "generated_at": stamp(),
            "result": "NO_TRACE_ABORT",
            "notes": [
                "No DEE_TA_*_ABORT markers in any downloaded log.",
                "Either the v5 sentinel table never caught an invalid free",
                "(the bug slipped past our hooks), OR glibc fired first and",
                "the kernel aborted before our post-mortem dump could run.",
            ],
            "glibc_double_free_anywhere": any(
                stat.counts_by_marker.get("glibc_double_free_!prev", 0) > 0
                for stat, _, _ in per_file
            ),
            "glibc_generic_corruption_anywhere": any(
                stat.counts_by_marker.get("glibc_generic_corruption", 0) > 0
                for stat, _, _ in per_file
            ),
            "address_sanitizer_anywhere": any(
                stat.counts_by_marker.get("address_sanitizer", 0) > 0
                for stat, _, _ in per_file
            ),
            "per_file_marker_counts": [s.counts_by_marker for s, _, _ in per_file],
        }
        (out_dir / "lifetime_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / "timeline.txt").write_text(
            "=== M3 v6 LIFETIME RECONSTRUCTION ===\n"
            f"RESULT: NO_TRACE_ABORT (no DEE_TA_*_ABORT markers found)\n"
            f"GLIBC_DOUBLE_FREE: {report['glibc_double_free_anywhere']}\n"
            f"GLIBC_GENERIC:     {report['glibc_generic_corruption_anywhere']}\n"
            f"ADDRESS_SANITIZER: {report['address_sanitizer_anywhere']}\n"
            "Per user directive: no fix recommendations are produced until\n"
            "the invalid pointer + allocator mismatch (or other root cause)\n"
            "is proven from runtime evidence.\n", encoding="utf-8")
        return

    op, registry, invalid_ops = first_op
    lifetime = reconstruct_lifetime(op, registry, invalid_ops_all=invalid_ops)

    report = {
        "schema_version": 1,
        "generated_at": stamp(),
        "result": "FIRST_TRACE_ABORT",
        "first_invalid_op": asdict(op),
        "lifetime_reconstruction": lifetime,
    }
    (out_dir / "lifetime_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Plain-text timeline suitable for terminal paste into M3 v6 report.
    timeline_lines = [
        "=== M3 v6 LIFETIME RECONSTRUCTION (terminal-stage download) ===",
        f"Source log       : {op.source_log_relative}",
        f"Source L#        : {op.line_number}",
        f"Target pointer   : {op.pointer}",
        f"Op type          : {op.op_type}",
        f"Registered in DEE_TA table: {'YES' if lifetime['registered'] else 'NO'}",
        f"Alloc ID (when known): {lifetime['alloc_id_when_known']}",
        f"Same-pointer cascade count (later attempts on same ptr): "
        f"{lifetime['same_pointer_cascade_count']}",
        "",
        "CHRONOLOGICAL EVENTS:",
    ]
    for ev in lifetime["events"]:
        tag = ev.pop("phase")
        line = f"  [{tag}] " + " ".join(
            f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}"
            for k, v in ev.items())
        timeline_lines.append(line)
    timeline_lines.extend([
        "",
        "EVIDENCE-ONLY CONCLUSION (no fix recommendations per directive):",
        f"  Pointer {op.pointer} produced a {op.op_type} event in",
        f"  source log {op.source_log_relative} at L{op.line_number}.",
        f"  This is the FIRST invalid operation surfaced by the v5 sentinel",
        f"  table across the full M3 v6 forensic matrix.  Subsequent",
        f"  invalid operations targeting other pointers were observed but are",
        f"  suppressed here per the user's 'first pointer' directive.  Any",
        f"  fix or repair MUST wait until this exact allocation ID + pointer",
        f"  + allocator chain is reviewed in the post-mortem dump",
        f"  (TRACE_ALLOC POST-MORTEM DUMP) that trace_alloc.cpp emits before",
        f"  std::abort().",
    ])
    (out_dir / "timeline.txt").write_text(
        "\n".join(timeline_lines) + "\n", encoding="utf-8",
    )


def chronologically_order_log_files(
    log_files: list[Path], matrix_summary_path: Path | None,
) -> list[Path]:
    """Return `log_files` sorted in the matrix's run order.

    If `matrix-summary.json` is supplied we use its `runs` array to get the
    declared execution order; otherwise we fall back to file mtime
    ascending (which approximates chronological order on a stable FS).

    Files whose stem does not appear in the matrix summary are sorted
    AFTER the listed runs (treated as out-of-matrix).  Within each group
    we tiebreak by mtime so ties are broken deterministically.
    """
    if not matrix_summary_path or not matrix_summary_path.exists():
        return sorted(log_files, key=lambda p: p.stat().st_mtime)
    summary = json.loads(matrix_summary_path.read_text())
    runs = summary.get("runs", [])
    run_order: dict[str, int] = {}
    for i, entry in enumerate(runs):
        run_order[str(entry.get("id",
                                 entry.get("name",
                                            entry.get("label", i))))] = i
    out_of_matrix = 10 ** 6
    def sort_key(p: Path) -> tuple[int, float]:
        return (run_order.get(p.stem, out_of_matrix), p.stat().st_mtime)
    return sorted(log_files, key=sort_key)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--matrix-summary", type=Path, default=None,
                        help="path to matrix-summary.json; if supplied,"
                             " log files are ordered by the matrix's run"
                             " declaration rather than lexically.")
    parser.add_argument("--log-globs", nargs="*",
                        default=["*.log"],
                        help="file globs to scan inside --evidence-dir"
                             " (default: matrix-driver.log + dual-*.log).")
    args = parser.parse_args(argv)
    evidence_dir: Path = args.evidence_dir
    output_dir: Path = args.output_dir or (evidence_dir / "analysis")
    #  Auto-discover matrix-summary.json so Kaggle output ordering follows
    #  the matrix's run declaration rather than alphabetic fall-back.
    #  The mtime fallback on a downloaded tar.gz can collapse to ~single-
    #  timestamp (ks not preserving mtimes through extraction); the matrix-
    #  summary path bypasses this entirely when present.
    if not args.matrix_summary:
        for cand in (evidence_dir.parent / "matrix-summary.json",
                      evidence_dir / "matrix-summary.json"):
            if cand.exists():
                args.matrix_summary = cand
                break
    if not evidence_dir.exists():
        sys.stderr.write(
            f"error: evidence-dir {evidence_dir} does not exist\n")
        return 2

    log_files: list[Path] = []
    for pat in args.log_globs:
        log_files.extend(sorted(evidence_dir.rglob(pat)))
    log_files = [p for p in log_files if p.is_file()]
    #  MILESTONE 3 v6 / FIX 2: chronological order tiebreak.  Lexical
    #  sort != matrix run order on Kaggle (per-run logs may appear in any
    #  order); use --matrix-summary when available, else fall back to
    #  file mtime.  Either way, write_outputs() then sees the FIRST
    #  pointer in chronological order.
    log_files = chronologically_order_log_files(
        log_files, args.matrix_summary,
    )
    if not log_files:
        sys.stderr.write(
            f"warning: no log files matched under {evidence_dir}\n")

    per_file: list[tuple[FileStats, dict[str, AllocRecord], list[InvalidOp]]] = []

    # Globals for cross-file marker counts.
    aggregate_counter: Counter[str] = Counter()
    total_invalid_ops = 0
    total_glibc_hits = 0

    for lf in log_files:
        stats, registry, invalid_ops = analyze_log_file(lf, evidence_dir)
        per_file.append((stats, registry, invalid_ops))
        for name, count in stats.counts_by_marker.items():
            aggregate_counter[name] += count
        total_invalid_ops += len(invalid_ops)
        total_glibc_hits += stats.counts_by_marker.get(
            "glibc_double_free_!prev", 0)

    write_outputs(output_dir, per_file, evidence_dir)

    summary = {
        "log_files_scanned": [str(p) for p in log_files],
        "aggregate_marker_counts": dict(aggregate_counter),
        "total_invalid_ops_seen": total_invalid_ops,
        "total_glibc_double_free_hits": total_glibc_hits,
        "output_dir": str(output_dir),
        "fatal_first_pointer": (
            None
            if not any(s.first_invalid_op for s, _, _ in per_file)
            else next(s.first_invalid_op.pointer
                      for s, _, _ in per_file if s.first_invalid_op is not None)
        ),
    }
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
