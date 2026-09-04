#!/usr/bin/env python3
"""Tests for tools/memory_budget.py (pure arithmetic, no hardware)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from memory_budget import (  # noqa: E402
    DSV4_FLASH_0731,
    EXPERT_RECORD_BYTES,
    compute_budget,
    format_bytes,
    parse_bytes,
    resolve_expert_bytes,
)


class _Args:
    def __init__(self, **kw):
        self.expert_bytes = kw.get("expert_bytes")
        self.expert_mib = kw.get("expert_mib")
        self.bits_per_param = kw.get("bits_per_param")
        self.expert_params = kw.get("expert_params")
        self.scale_overhead = kw.get("scale_overhead")
        self.codec = kw.get("codec", "mxfp4")


def test_canon_mxfp4_size():
    assert EXPERT_RECORD_BYTES["mxfp4"] == 13369344
    assert EXPERT_RECORD_BYTES["mxfp4"] == int(12.75 * 1024 * 1024)
    assert EXPERT_RECORD_BYTES["fp16"] == 50331648
    assert DSV4_FLASH_0731["params_per_routed_expert"] == 3 * 4096 * 2048


def test_parse_bytes():
    assert parse_bytes("16GiB") == 16 * 1024 ** 3
    assert parse_bytes("12.75MiB") == int(12.75 * 1024 * 1024)
    assert parse_bytes("8589934592") == 8589934592
    assert parse_bytes("8.84GB") == int(8.84 * 1000 ** 3)
    with pytest.raises(ValueError):
        parse_bytes("not-a-size")


def test_slots_16gib_mxfp4():
    vram = 16 * 1024 ** 3
    reserved = 8840000000
    out = compute_budget(vram, reserved, EXPERT_RECORD_BYTES["mxfp4"],
                         24 * 1024 ** 3, 11008, 6, 43)
    assert out["vram_slots"] == (vram - reserved) // 13369344
    assert out["vram_slots"] > 0
    assert out["ram_slots"] == (24 * 1024 ** 3) // 13369344
    assert out["cached_slots"] == out["vram_slots"] + out["ram_slots"]
    assert out["full_model_fits_cached"] is False  # eviction mandatory
    assert out["worst_case_fits_vram"] is True  # 3.2 GiB bound < ~7.8 GiB
    assert 0.0 < out["cache_coverage_fraction"] < 1.0


def test_custom_expert_size():
    out = compute_budget(16 * 1024 ** 3, 0, 8 * 1024 * 1024, 0, 100, 1, 1)
    assert out["vram_slots"] == (16 * 1024 ** 3) // (8 * 1024 * 1024)
    assert out["ram_slots"] == 0


def test_reserved_exceeds_vram_gives_zero_slots():
    out = compute_budget(8 * 1024 ** 3, 16 * 1024 ** 3, 13369344, 1024, 11008, 6, 43)
    assert out["vram_slots"] == 0
    assert out["feasible_bounded"] is False


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        compute_budget(0, 0, 100, 0, 10, 1, 1)
    with pytest.raises(ValueError):
        compute_budget(100, 0, 0, 0, 10, 1, 1)
    with pytest.raises(ValueError):
        compute_budget(100, -1, 10, 0, 10, 1, 1)


def test_resolve_arbitrary_bytes():
    nbytes, approx, _ = resolve_expert_bytes(_Args(expert_bytes="13MiB"))
    assert nbytes == 13 * 1024 * 1024 and approx is False
    nbytes, approx, _ = resolve_expert_bytes(_Args(expert_mib=12.75))
    assert nbytes == 13369344 and approx is False


def test_resolve_hypothetical_flagged_approximate():
    nbytes, approx, note = resolve_expert_bytes(
        _Args(codec="mxfp4", bits_per_param=2.06))
    assert approx is True
    assert "NO quality claim" in note
    expected = int(25165824 * 2.06 / 8.0) + 786432
    assert nbytes == expected


def test_resolve_named_hypothetical_requires_bpb():
    with pytest.raises(ValueError):
        resolve_expert_bytes(_Args(codec="iq2"))
    with pytest.raises(ValueError):
        resolve_expert_bytes(_Args(codec="nope"))


def test_json_serializable():
    out = compute_budget(16 * 1024 ** 3, 8840000000, 13369344,
                         8 * 1024 ** 3, 11008, 6, 43)
    json.dumps(out)


def test_format_bytes():
    assert format_bytes(16 * 1024 ** 3) == "16.00 GiB"
    assert format_bytes(13369344) == "12.75 MiB"
