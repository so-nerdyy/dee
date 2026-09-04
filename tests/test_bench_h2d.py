#!/usr/bin/env python3
"""Tests for tools/bench_h2d.py (no GPU required; graceful UNKNOWN paths)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import bench_h2d  # noqa: E402
from bench_h2d import parse_size_list, run_bench, unknown_results  # noqa: E402


def test_parse_size_list_defaults():
    assert parse_size_list("1,4,8,13,16,32") == [1, 4, 8, 13, 16, 32]
    assert parse_size_list("1MiB,4MiB") == [1024 ** 2, 4 * 1024 ** 2]


def test_unknown_results_shape():
    sizes = [1024 ** 2, 13 * 1024 ** 2]
    res = unknown_results(sizes, "no CUDA")
    for key in ("pageable_h2d", "pinned_h2d", "d2h"):
        assert len(res[key]) == 2
        assert all(e["status"] == "unknown" and e["reason"] == "no CUDA"
                   for e in res[key])


def test_run_bench_without_cuda_reports_unknown(monkeypatch):
    monkeypatch.setattr(bench_h2d, "torch_cuda_info",
                        lambda: {"available": False, "reason": "fake-no-cuda"})
    report = run_bench([1, 13], iterations=1, warmup=0)
    assert report["cuda"]["available"] is False
    for key in ("pageable_h2d", "pinned_h2d", "d2h"):
        assert len(report["results"][key]) == 2
        assert all(e["status"] == "unknown" for e in report["results"][key])
    assert "note" in report
    json.dumps(report)


def test_run_bench_includes_expert_size_13mib(monkeypatch):
    monkeypatch.setattr(bench_h2d, "torch_cuda_info",
                        lambda: {"available": False, "reason": "fake-no-cuda"})
    report = run_bench(None, iterations=1, warmup=0)  # defaults
    sizes_mib = [e["size_mib"] for e in report["results"]["pinned_h2d"]]
    assert 13 in sizes_mib  # ~packed MXFP4 expert sweep point
    assert sizes_mib == [1, 4, 8, 13, 16, 32]
