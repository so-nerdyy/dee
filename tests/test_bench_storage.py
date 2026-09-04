#!/usr/bin/env python3
"""Tests for tools/bench_storage.py helpers (small real I/O + pure math)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from bench_storage import (  # noqa: E402
    bench_storage,
    detect_filesystem,
    disk_stats,
    parse_block_list,
    parse_bytes,
)


def test_parse_block_list():
    assert parse_block_list("256KiB,1MiB,4MiB") == [256 * 1024, 1024 ** 2, 4 * 1024 ** 2]
    with pytest.raises(ValueError):
        parse_block_list("  , ")


def test_parse_bytes_units():
    assert parse_bytes("1MiB") == 1024 ** 2
    assert parse_bytes("4MiB") == 4 * 1024 ** 2


def test_disk_stats_measured(tmp_path):
    stats = disk_stats(tmp_path)
    assert stats["status"] == "measured"
    assert stats["free_bytes"] > 0 and stats["total_bytes"] >= stats["free_bytes"]


def test_detect_filesystem_does_not_fabricate(tmp_path):
    fs = detect_filesystem(tmp_path)
    assert fs["filesystem"]  # string, possibly UNKNOWN
    assert fs["status"] in ("measured", "unknown")
    if fs["status"] == "unknown":
        assert "reason" in fs or "method" in fs


def test_bench_storage_tiny_roundtrip(tmp_path):
    report = bench_storage(tmp_path, file_mib=1, blocks=[256 * 1024],
                           iterations=2, random_ops=4)
    assert report["tool"] == "bench_storage"
    assert report["test_file_bytes"] == 1024 ** 2
    seq = report["sequential_read"][0]
    assert seq["throughput_bps"]["median"] > 0
    assert seq["cold_bps"] > 0
    assert seq["warm_bps_median"] is not None
    assert "cold_note" in seq and seq["cold_note"]
    rnd = report["random_read"][0]
    assert rnd["status"] == "measured" and rnd["throughput_bps"] > 0
    # Temp file cleaned up.
    leftovers = list(tmp_path.glob(".dee_storage_probe_*.bin"))
    assert leftovers == []
    json.dumps(report)  # serializable


def test_bench_storage_block_larger_than_file_skipped(tmp_path):
    report = bench_storage(tmp_path, file_mib=1, blocks=[4 * 1024 ** 2],
                           iterations=1, random_ops=4)
    assert report["random_read"][0]["status"] == "skipped"
