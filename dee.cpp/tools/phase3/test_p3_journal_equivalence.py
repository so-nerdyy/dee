"""Tests for p3_journal_equivalence.

Builds a real (miniature) dee4-v4-segmented store with p3_builder, then checks
that the equivalence tool passes on clean bytes, fails on a flipped byte, and
localizes the damage to the right component.  Also exercises journal parsing
against the real RoutedExpertJournal schema, including a truncated tail.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p3_manifest  # noqa: E402
import p3_journal_equivalence as pje  # noqa: E402

TOOL = Path(__file__).resolve().parent / "p3_journal_equivalence.py"


# ---------------------------------------------------------------------------
# Journal parsing
# ---------------------------------------------------------------------------


def _journal_line(layer: int, rows: list[list[int]], step: int = 0) -> str:
    return json.dumps({
        "canonical_order": "forward_step,layer,token_row,topk_rank",
        "chain_sha256": "0" * 64,
        "device": "cuda:0",
        "expert_ids_rank_order": rows,
        "forward_step": step,
        "layer": layer,
        "phase": "decode",
        "previous_chain_sha256": "0" * 64,
        "record_index": step * 43 + layer,
        "run_id": "test",
        "schema_version": 1,
        "start_pos": 0,
        "token_rows": len(rows),
        "topk": len(rows[0]) if rows else 0,
    })


def test_journal_pairs_deduplicate(tmp_path):
    jp = tmp_path / "routed_experts-q0.jsonl"
    jp.write_text("\n".join([
        _journal_line(0, [[5, 9], [9, 5]]),
        _journal_line(1, [[5, 9]]),
    ]) + "\n")
    pairs, stats = pje.collect_pairs([jp])
    assert stats["total_expert_uses"] == 6
    assert set(pairs) == {(0, 5), (0, 9), (1, 5), (1, 9)}
    assert stats["distinct_pairs"] == 4


def test_journal_truncated_tail_is_salvaged(tmp_path, capsys):
    """A timed-out arm leaves a half-written final line (A0's q0 did)."""
    jp = tmp_path / "partial.jsonl"
    jp.write_text(_journal_line(0, [[1, 2]]) + "\n" + '{"layer": 1, "exp')
    pairs, stats = pje.collect_pairs([jp])
    assert set(pairs) == {(0, 1), (0, 2)}
    assert "truncated" in capsys.readouterr().err


def test_multiple_journals_union(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(_journal_line(0, [[1]]) + "\n")
    b.write_text(_journal_line(0, [[1]]) + "\n" + _journal_line(2, [[7]]) + "\n")
    pairs, stats = pje.collect_pairs([a, b])
    assert set(pairs) == {(0, 1), (2, 7)}
    assert stats["new_pairs_per_journal"] == {"a.jsonl": 1, "b.jsonl": 1}


# ---------------------------------------------------------------------------
# Store locator
# ---------------------------------------------------------------------------


def _mini_store(tmp_path, *, buckets=3, experts=4, record_bytes=96):
    """Write a dee4-v4-segmented store with deterministic per-record bytes."""
    import hashlib
    root = tmp_path / "store"
    (root / "segments").mkdir(parents=True)
    segments = []
    for b in range(buckets):
        blob = bytearray()
        for e in range(experts):
            blob += bytes([(b * 31 + e * 7 + i) & 0xFF
                           for i in range(record_bytes)])
        path = root / "segments" / f"experts-bucket-{b:02d}.dee4"
        path.write_bytes(bytes(blob))
        segments.append({
            "file": f"segments/experts-bucket-{b:02d}.dee4",
            "bucket": b,
            "domain": "main",
            "first_record": b * experts,
            "record_count": experts,
            "bytes": experts * record_bytes,
            "sha256": hashlib.sha256(bytes(blob)).hexdigest(),
        })
    (root / "metadata.json").write_text(json.dumps({
        "format": "dee4-v4-segmented",
        "data_file": None,
        "start_layer": 0,
        "num_layers": buckets,
        "experts_per_layer": experts,
        "total_experts": buckets * experts,
        "record_bytes": record_bytes,
        "segments": segments,
    }, indent=1))
    return root


def test_locator_and_structural_check(tmp_path):
    root = _mini_store(tmp_path)
    store = pje.SegmentedStore(root)
    try:
        assert store.structural_check()["ok"]
        assert store.locate(0, 0) == (0, 0)
        assert store.locate(0, 3) == (0, 3 * 96)
        assert store.locate(2, 1) == (2, 96)
        # Record content is addressed correctly across a segment boundary.
        assert store.read_record(1, 0)[:1] == bytes([31])
        with pytest.raises(IndexError):
            store.locate(0, 4)
        with pytest.raises(IndexError):
            store.locate(3, 0)
    finally:
        store.close()


def test_structural_check_flags_bad_size(tmp_path):
    root = _mini_store(tmp_path)
    seg = root / "segments" / "experts-bucket-01.dee4"
    seg.write_bytes(seg.read_bytes()[:-8])
    result = pje.SegmentedStore(root).structural_check()
    assert not result["ok"]
    assert any("size" in p for p in result["problems"])


def test_segment_table_digest_is_declared_seal_digest(tmp_path):
    import hashlib
    root = _mini_store(tmp_path)
    meta = json.loads((root / "metadata.json").read_text())
    expected = hashlib.sha256(
        "".join(s["sha256"] for s in meta["segments"]).encode("ascii")
    ).hexdigest()
    assert pje.SegmentedStore(root).segment_table_digest() == expected


# ---------------------------------------------------------------------------
# End-to-end against a real builder-produced store
# ---------------------------------------------------------------------------


HEADERS = (Path(__file__).resolve().parents[2] / "benchmark_reports"
           / "deepseek-v4-flash-0731-t4" / "shard-headers")


class _SyntheticShardSource:
    """Deterministic stand-in for real shards, keyed by (shard, offset).

    Uses sha256 rather than builtins.hash(): hash() is salted per interpreter
    (PYTHONHASHSEED), so a fixture built in one process and verified in
    another would spuriously mismatch.
    """

    def fetch(self, shard: str, data_offset: int, nbytes: int) -> bytes:
        import hashlib
        seed = hashlib.sha256(
            f"{shard}:{data_offset}".encode()).digest()[0]
        return bytes([(seed + i) & 0xFF for i in range(nbytes)])


@pytest.mark.skipif(not HEADERS.is_dir(), reason="committed headers absent")
def test_end_to_end_clean_then_corrupted(tmp_path):
    """Full path: build a store, prove it matches, corrupt it, prove it fails."""
    from p3_builder import build_segmented

    # build_manifest only accepts the real universe scopes (43 or 46).  Build
    # the real thing, then subset to 2 buckets x 2 experts so the fixture keeps
    # true record geometry (13,369,344 B, real data_offsets) while staying
    # small enough to build in a test.
    manifest = p3_manifest.build_manifest(HEADERS)
    manifest["_records"] = [
        r for r in manifest["_records"]
        if int(r["bucket"]) < 2 and int(r["expert"]) < 2]
    manifest["n_buckets"] = 2
    manifest["experts_per_layer"] = 2
    manifest["total_experts"] = len(manifest["_records"])
    for i, rec in enumerate(manifest["_records"]):
        rec["record_index"] = i

    store_dir = tmp_path / "store"
    source = _SyntheticShardSource()
    records = manifest["_records"]
    build_segmented(manifest, records, source, store_dir)

    journal = tmp_path / "routed.jsonl"
    journal.write_text("\n".join([
        _journal_line(0, [[0, 1]]),
        _journal_line(1, [[0, 1]]),
    ]) + "\n")

    report = tmp_path / "report.json"
    cmd = [sys.executable, str(TOOL), "--store", str(store_dir),
           "--source", "local", "--shards", str(tmp_path),
           "--headers", str(HEADERS), "--journal", str(journal),
           "--out", str(report)]

    # The tool's local source reads real files; inject the synthetic one by
    # running the comparison in-process instead.
    store = pje.SegmentedStore(store_dir)
    records_by_pair = {(int(r["bucket"]), int(r["expert"])): r
                       for r in manifest["_records"]}
    pairs, _ = pje.collect_pairs([journal])
    pairs = [p for p in pairs if p in records_by_pair]
    assert pairs, "journal pairs must exist in the manifest"

    clean = [pje.check_one((l, e, records_by_pair, store, source))
             for l, e in pairs]
    assert all(r["match"] for r in clean), [r for r in clean if not r["match"]]

    # Flip one byte inside w2.scale of (bucket 0, expert 1) and confirm the
    # tool both fails and names the right component.
    layer, expert = 0, 1
    seg_index, offset = store.locate(layer, expert)
    seg_path = store_dir / store.segments[seg_index]["file"]
    record = records_by_pair[(layer, expert)]
    target = next(r for r in record["ranges"] if r[0] == "w2.scale")
    flip_at = offset + int(target[3]) + 5
    raw = bytearray(seg_path.read_bytes())
    raw[flip_at] ^= 0xFF
    seg_path.write_bytes(bytes(raw))
    store.close()

    store2 = pje.SegmentedStore(store_dir)
    try:
        dirty = pje.check_one((layer, expert, records_by_pair, store2, source))
        assert not dirty["match"]
        assert [c["component"] for c in dirty["components"]] == ["w2.scale"]
        assert dirty["components"][0]["first_differing_byte_in_component"] == 5
        # An untouched record in the same segment still matches.
        other = pje.check_one((0, 0, records_by_pair, store2, source))
        assert other["match"]
    finally:
        store2.close()


@pytest.mark.skipif(not HEADERS.is_dir(), reason="committed headers absent")
def test_dry_run_cli(tmp_path):
    root = _mini_store(tmp_path)
    report = tmp_path / "dry.json"
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--store", str(root),
         "--shards", str(tmp_path), "--dry-run", "--out", str(report)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(report.read_text())
    assert payload["verdict"] == "DRY_RUN"
    assert payload["structural"]["ok"]
    assert len(payload["segment_table_digest"]) == 64
