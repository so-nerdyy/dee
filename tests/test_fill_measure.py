#!/usr/bin/env python3
"""Tests for the fill-path instrumentation + replay + timeline (no CUDA).

Covers: fill timeline occupancy math (worker sums never counted as wall),
fragmentation metrics on synthetic journals, kernel fill-flavor build
(patch applies to 217a333, additive except const), driver contents, and
static presence of the C++ instrumentation. Live T4 values are asserted
nowhere here.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "dee.cpp" / "experiments" / "route_pipeline"
sys.path.insert(0, str(EXP))

from fill_timeline import build_series, render_html  # noqa: E402

ENGINE_217 = "217a33359b06a0453444a698ec52e4078b77e388"
FILL_KERNEL = EXP / "kernel_fill_measure"


def _driver_source() -> str:
    return (FILL_KERNEL / "session-driver.py").read_text(encoding="utf-8")


def _embedded_fill_patch() -> str:
    src = _driver_source()
    m = re.search("FILL_PATCH_B64 = \"([A-Za-z0-9+/=]+)\"", src)
    assert m, "embedded fill patch missing"
    return base64.b64decode(m.group(1)).decode("utf-8")


def test_fill_kernel_dir_complete():
    assert (FILL_KERNEL / "session-driver.py").is_file()
    meta = json.loads((FILL_KERNEL / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == ["nivind/deepseek-v4-flash-0731-shards"]
    assert meta["id"].startswith("nivind/")
    assert "fill" in meta["id"]
    proof = json.loads((FILL_KERNEL / "BUILD-PROOF.json").read_text(encoding="utf-8"))
    assert proof["flavor"] == "fill"
    ast.parse(_driver_source())


def _committed_fill_patch() -> str:
    return (ROOT / "dee.cpp" / "experiments" / "route_pipeline" / "patches"
            / "fill-extra-217a333.patch").read_text(encoding="utf-8")


def _check_hunks(patch: str, rev: str) -> int:
    """Every hunk's context+removed lines must exist in the rev target."""
    current_file, hunk_body, checked = None, [], 0

    def blob(path: str) -> list[str]:
        proc = subprocess.run(["git", "show", f"{rev}:{path}"],
                              capture_output=True, cwd=str(ROOT))
        assert proc.returncode == 0, proc.stderr.decode()[-300:]
        return proc.stdout.replace(b"\r\n", b"\n").decode("utf-8").splitlines()

    def check(path: str, body: list[str]) -> None:
        nonlocal checked
        want = [ln[1:] for ln in body if ln[:1] in (" ", "-")]
        if not want:
            return
        if path == "/dev/null":
            return
        target = blob(path)
        assert any(target[i:i + len(want)] == want for i in range(len(target))), \
            f"hunk not present in {rev}:{path}: {want[0][:80]}"
        checked += 1

    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            if current_file and hunk_body:
                check(current_file, hunk_body)
            current_file, hunk_body = line[6:], []
        elif line.startswith("@@"):
            if current_file and hunk_body:
                check(current_file, hunk_body)
            hunk_body = []
        elif line.startswith("--- "):
            continue  # --- a/ and --- /dev/null headers are not content
        elif current_file is not None and line[:1] in (" ", "-", "+"):
            hunk_body.append(line)
    if current_file and hunk_body:
        check(current_file, hunk_body)
    return checked


def test_embedded_fill_patch_matches_committed_file():
    """The builder embeds profile(generated)+fill-extra(committed); the tail
    must equal the committed file bytes exactly."""
    src = _driver_source()
    m = re.search("FILL_PATCH_B64 = \"([A-Za-z0-9+/=]+)\"", src)
    assert m, "embedded fill patch missing"
    import base64 as _b64
    embedded = _b64.b64decode(m.group(1)).decode("utf-8")
    committed = _committed_fill_patch()
    assert embedded.endswith(committed), "embedded tail != committed fill-extra"
    head = embedded[: -len(committed)]
    assert "HostSpanGuard" in head  # generated profile part present


def test_fill_patch_applies_to_217a333():
    patch = _committed_fill_patch()
    assert _check_hunks(patch, ENGINE_217) > 10


def test_fill_patch_files_and_additivity():
    patch = _committed_fill_patch()
    files = sorted(l[6:] for l in patch.splitlines() if l.startswith("+++ b/"))
    assert files == [
        "dee.cpp/CMakeLists.txt",
        "dee.cpp/include/dee/expert_store.h",
        "dee.cpp/include/dee/host_pack_cache.h",
        "dee.cpp/src/expert_store.cpp",
        "dee.cpp/src/host_pack_cache.cpp",
        "dee.cpp/tools/fill_replay.cpp",
    ], files
    removed, added = set(), set()
    for line in patch.splitlines():
        if line.startswith("-") and not line.startswith("---"):
            removed.add(line[1:].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            added.add(line[1:].strip())
    vanished = removed - added
    assert vanished <= {
        "const bool computed = d_blob && swiglu_expert_batch_fp16_cuda("}, vanished


def test_branch_markers_covered_by_patch():
    """Every fill-instrumentation marker in branch source must appear in the
    committed execution patch (no silent drift between review material and
    what runs on T4)."""
    patch = _committed_fill_patch()
    markers = (
        # host_pack_cache phases
        "fill_start_offset_ms", "set_fill_profiler", "set_fill_context",
        "phase_t0", "active_batch_begin_", "note_fill_batch",
        # expert_store read-service split + residency probe
        "note_pread_service", "read_telemetry", "mincore",
        "pread_service_ms", "mincore_resident_bytes",
        # replay tool + build
        "fill_replay", "DEE_BUILD_FILL_REPLAY",
    )
    for marker in markers:
        assert marker in patch, marker


def test_occupancy_never_sums_workers_as_wall():
    batches = [
        {"batch_id": 1, "misses": 3, "batch_wall_ms": 10.0,
         "requests": [
             {"key": 1, "start_offset_ms": 0.0, "service_ms": 9.0,
              "nbytes": 100, "success": True},
             {"key": 2, "start_offset_ms": 0.5, "service_ms": 9.0,
              "nbytes": 100, "success": True},
             {"key": 3, "start_offset_ms": 1.0, "service_ms": 9.0,
              "nbytes": 100, "success": True}]},
        {"batch_id": 2, "misses": 1, "batch_wall_ms": 4.0,
         "requests": [
             {"key": 4, "start_offset_ms": 12.0, "service_ms": 3.0,
              "nbytes": 100, "success": True}]},
    ]
    series = build_series(batches, bucket_ms=1.0)
    summary = series["summary"]
    # Batch 2 submits at cursor t=10; its request runs [22,25). Busy is the
    # UNION of service windows ([0,10) + [22,25) = 13 ms), never the
    # worker sum (30 ms) and never clipped to batch walls.
    assert summary["window_ms"] == 25.0
    assert summary["ssd_busy_ms"] == 13.0
    assert summary["ssd_busy_fraction"] == 0.52
    assert summary["max_reads_outstanding"] == 3
    assert summary["starvation_gaps"] == [[10.0, 22.0]]


def test_occupancy_gap_detection():
    batches = [{"batch_id": 1, "misses": 1, "batch_wall_ms": 2.0,
                "submit_offset_ms": 0.0,
                "requests": [{"key": 1, "start_offset_ms": 0.0,
                              "service_ms": 2.0, "nbytes": 10,
                              "success": True}]},
               {"batch_id": 2, "misses": 1, "batch_wall_ms": 2.0,
                "submit_offset_ms": 10.0,
                "requests": [{"key": 2, "start_offset_ms": 0.0,
                              "service_ms": 2.0, "nbytes": 10,
                              "success": True}]}]
    series = build_series(batches, bucket_ms=1.0)
    summary = series["summary"]
    assert summary["window_ms"] == 12.0
    assert summary["ssd_busy_ms"] == 4.0
    assert summary["starvation_gaps"] == [[2.0, 10.0]]
    html = render_html(series, title="t")
    assert "<svg" in html and "SSD busy" in html


def test_fragmentation_metrics_synthetic(tmp_path):
    from analyze_fragmentation import pattern_metrics
    rows = [{"layer": 0,
             "expert_ids_rank_order": [[e] for e in range(6)]},
            {"layer": 0,
             "expert_ids_rank_order": [[e] for e in range(6)]},
            {"layer": 1,
             "expert_ids_rank_order": [[e] for e in range(6)]}]
    # Same-shape rows: second L0 row reuses -> duplicates within batch order.
    metrics = pattern_metrics(rows, 100)
    assert metrics["uses"] == 18
    assert metrics["unique_experts"] == 12
    assert metrics["batch_size"] == {"min": 6, "max": 6, "mean": 6.0}
    assert metrics["layout"].startswith("ASSUMED") or metrics["layout"] == "measured-map"
    assert metrics["seek_gap_bytes"]["p50"] == 100  # adjacent under layer-major


def test_fragmentation_with_real_map():
    from analyze_fragmentation import pattern_metrics
    rows = [{"layer": 0, "expert_ids_rank_order": [[5, 6]]},
            {"layer": 1, "expert_ids_rank_order": [[5, 6]]}]
    index_map = {"0,5": 10, "0,6": 11, "1,5": 100, "1,6": 101}
    metrics = pattern_metrics(rows, 1000, index_map)
    assert metrics["layout"] == "measured-map"
    assert metrics["seek_gap_bytes"]["max"] == 1000  # within-batch, adjacent
    assert metrics["inter_batch_jump_bytes"]["max"] == 89000  # cross-layer jump


def test_instrumentation_present_in_source():
    engine = (ROOT / "dee.cpp" / "src" / "engine.cpp").read_text(encoding="utf-8")
    assert "set_fill_context(current_token_, source_layer" in engine
    assert "pack_cache_.set_fill_profiler(" in engine
    cache = (ROOT / "dee.cpp" / "src" / "host_pack_cache.cpp").read_text(encoding="utf-8")
    assert "note_fill_batch(record)" in cache
    assert "fill_start_offset_ms" in cache
    store = (ROOT / "dee.cpp" / "src" / "expert_store.cpp").read_text(encoding="utf-8")
    assert "mincore" in store and "note_pread_service" in store
    assert "DEE_BUILD_FILL_REPLAY" in (ROOT / "dee.cpp" / "CMakeLists.txt").read_text(
        encoding="utf-8")
    assert "fill_replay" in (ROOT / "dee.cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
