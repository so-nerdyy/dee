"""CACHE1f staging-module tests (local, no GPU, no checkpoint download).

Covers:
1. ``build_staging_manifest``: deterministic structure from the committed
   shard headers + official config + sealed DS10 evidence; priority classes
   and full-set totals match the measured ledger.
2. ``select_staged_tensors``: budget-respecting greedy selection preserves
   priority order (must-stage dense/shared first, then routed by access).
3. ``_write_partial_shard``: writes a valid safetensors file readable by
   ``safe_open`` with identical tensor bytes.
4. ``HybridTensorSource`` local read path: a staged partial shard serves a
   tensor byte-identically to the committed header metadata without any
   HTTP (the remote fallback path is exercised by the harness remotely).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

from scripts.deepseek_v4_staging import (
    HybridTensorSource,
    build_staging_manifest,
    select_staged_tensors,
    _write_partial_shard,
)

REPORTS = Path("benchmark_reports/deepseek-v4-flash-0731-t4")
HEADERS = REPORTS / "shard-headers"
CONFIG = REPORTS / "official-source/inference/config.json"
EVIDENCE = REPORTS / "ds10-cache1d-kaggle-20260808T183416Z/ds10-evidence.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not HEADERS.is_dir() or not CONFIG.is_file() or not EVIDENCE.is_file():
        pytest.skip("committed shard headers / config / sealed evidence absent")
    return build_staging_manifest(HEADERS, CONFIG, EVIDENCE)


def test_manifest_structure(manifest: dict) -> None:
    assert manifest["schema_version"] == 1
    assert manifest["n_layers"] == 43
    assert manifest["tensor_count"] == 15754
    assert manifest["total_gib"] == pytest.approx(37.685, abs=0.01)
    # priority classes: 6 top-level, dense, 258 shared, routed
    from collections import Counter
    prio = Counter(t["priority"] for t in manifest["tensors"])
    assert prio[1] == 6
    assert prio[3] == 258
    assert prio[4] == 14190
    # every routed tensor has an access count from the sealed trace
    assert all(t["access_count"] > 0 for t in manifest["tensors"]
               if t["priority"] == 4)
    # byte accounting matches the committed headers
    assert all(t["length"] > 0 for t in manifest["tensors"])


def test_manifest_deterministic(manifest: dict) -> None:
    again = build_staging_manifest(HEADERS, CONFIG, EVIDENCE)
    assert again["tensors"] == manifest["tensors"]
    assert again["total_bytes"] == manifest["total_bytes"]


def test_select_respects_priority_and_budget(manifest: dict) -> None:
    # A budget of exactly the must-stage set keeps ONLY priorities 1-3.
    must_bytes = sum(t["length"] for t in manifest["tensors"]
                     if t["priority"] < 4)
    sel = select_staged_tensors(manifest, must_bytes)
    assert all(t["priority"] < 4 for t in sel)
    assert sum(t["length"] for t in sel) == must_bytes
    # A full budget selects everything.
    full = select_staged_tensors(manifest, manifest["total_bytes"])
    assert len(full) == manifest["tensor_count"]


def _fake_bytes(n: int) -> bytes:
    """Deterministic byte blob of length n with values in range(0,256)."""
    import itertools
    return bytes(itertools.islice(itertools.cycle(range(256)), n))


def test_partial_shard_roundtrip(tmp_path: Path) -> None:
    # shape [2,2] BF16 = 8 bytes; shape [2] F8_E8M0 = 2 bytes.
    tensors = {
        "a.weight": (_fake_bytes(8), "BF16", [2, 2]),
        "b.scale": (b"\x00\x80", "F8_E8M0", [2]),
    }
    out = tmp_path / "model-00042-of-00048.safetensors"
    _write_partial_shard(out, tensors)
    raw = out.read_bytes()
    hlen = struct.unpack("<Q", raw[:8])[0]
    assert hlen == len(raw) - 8 - sum(len(b) for b, _, _ in tensors.values())
    from safetensors import safe_open
    with safe_open(str(out), framework="pt", device="cpu") as fh:
        t = fh.get_slice("a.weight")[:]
        # Compare raw storage bytes (BF16 tensor) via a uint8 view.
        got = t.detach().cpu().view(torch.uint8).numpy().tobytes()
        assert got == _fake_bytes(8)
        s = fh.get_slice("b.scale")[:]
        assert s.detach().cpu().view(torch.uint8).numpy().tolist() == [0, 128]


def test_hybrid_local_read(tmp_path: Path) -> None:
    """A staged partial shard serves a real committed tensor locally.

    Uses ``norm.weight`` (a real tensor in the committed headers, tiny) with
    synthetic bytes; the hybrid must return exactly those bytes with
    local_hits == 1 and zero remote requests (no HTTP).
    """
    if not HEADERS.is_dir():
        pytest.skip("committed shard headers absent")
    from scripts.deepseek_v4_model import CommittedHeaderSource
    src = CommittedHeaderSource(HEADERS)
    name = "norm.weight"
    row = src.tensor_identity(name)
    # Write a partial shard containing ONLY norm.weight with fake bytes.
    fake = _fake_bytes(row["length"])
    shard = row["shard"]
    staged_dir = tmp_path / "staged"
    _write_partial_shard(staged_dir / shard,
                         {name: (fake, row["dtype"], row["shape"])})
    hybrid = HybridTensorSource(HEADERS, staged_dir)
    got = hybrid._fetch_bytes(name)
    assert got == fake
    assert hybrid._local_hits == 1
    assert hybrid._local_misses == 0
    assert hybrid.stats["requests"] == 0
    # A tensor NOT staged must NOT be served from the local shard: it either
    # falls back to the remote path (network) or raises.  Stub the remote so
    # no real HTTP happens; the local counters must still show the miss.
    def _boom(name: str) -> bytes:
        raise RuntimeError("remote path not exercised in unit test")
    hybrid.remote._fetch_bytes = _boom  # type: ignore[method-assign]
    # A tensor in the SAME staged shard but not present in the partial file
    # must fall back (stubbed remote raises) and be counted as a miss.
    other = "head.weight"
    assert src.tensor_identity(other)["shard"] == shard
    with pytest.raises(RuntimeError):
        hybrid._fetch_bytes(other)
    assert hybrid._local_misses == 1
    # A tensor whose shard was never staged at all also falls back.
    another = "embed.weight"
    assert src.tensor_identity(another)["shard"] != shard
    (staged_dir / shard).unlink()  # drop the only staged shard
    with pytest.raises(RuntimeError):
        hybrid._fetch_bytes(another)
    assert hybrid._local_misses == 2


def test_hybrid_retry_file_merge(tmp_path: Path) -> None:
    """A ``<shard>.retry`` partial file (from the staging retry pass) is
    merged into the hybrid source and served locally, with the main file
    untouched and no remote requests."""
    if not HEADERS.is_dir():
        pytest.skip("committed shard headers absent")
    from scripts.deepseek_v4_model import CommittedHeaderSource
    src = CommittedHeaderSource(HEADERS)
    name = "norm.weight"
    row = src.tensor_identity(name)
    shard = row["shard"]
    staged_dir = tmp_path / "staged"
    # Main file holds one tensor; retry file holds a DIFFERENT tensor in the
    # same shard (hc_head_scale, tiny), simulating the serial retry pass.
    main_blob = _fake_bytes(row["length"])
    _write_partial_shard(staged_dir / shard,
                         {name: (main_blob, row["dtype"], row["shape"])})
    other = "hc_head_scale"
    orow = src.tensor_identity(other)
    assert orow["shard"] == shard
    retry_blob = _fake_bytes(orow["length"])
    _write_partial_shard(staged_dir / f"{shard}.retry",
                         {other: (retry_blob, orow["dtype"], orow["shape"])})
    hybrid = HybridTensorSource(HEADERS, staged_dir)
    assert hybrid._fetch_bytes(name) == main_blob
    assert hybrid._fetch_bytes(other) == retry_blob
    assert hybrid._local_hits == 2
    assert hybrid._local_misses == 0
    assert hybrid.stats["requests"] == 0


def test_hybrid_ranged_read_matches_raw_bytes(tmp_path: Path) -> None:
    """The ranged read path returns byte-exact tensor data for a shard whose
    header length is non-trivial (offset arithmetic bug would corrupt the
    read).  Uses a real committed tensor identity with fake bytes."""
    if not HEADERS.is_dir():
        pytest.skip("committed shard headers absent")
    from scripts.deepseek_v4_model import CommittedHeaderSource
    src = CommittedHeaderSource(HEADERS)
    name = "norm.weight"
    row = src.tensor_identity(name)
    shard = row["shard"]
    staged_dir = tmp_path / "staged"
    # Pad the header with many extra keys so 8 + hlen + d0 is not trivially
    # equal to d0 (forces correct offset arithmetic).
    pad = {f"pad.{i}.weight": (_fake_bytes(64), "BF16", [4, 8]) for i in range(20)}
    fake = _fake_bytes(row["length"])
    pad[name] = (fake, row["dtype"], row["shape"])
    _write_partial_shard(staged_dir / shard, pad)
    hybrid = HybridTensorSource(HEADERS, staged_dir)
    assert hybrid._fetch_bytes(name) == fake
    assert hybrid._local_hits == 1
    assert hybrid._local_misses == 0
