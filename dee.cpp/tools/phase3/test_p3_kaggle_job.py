"""Phase 3 tests: the Kaggle/cloud build+publish driver (p3_kaggle_job).

Everything runs on the synthetic mini universe (3 buckets x 4 experts x
70 B records) — no network, no real checkpoint bytes.  The only
network-marked test is a wiring check of the pinned RemoteRangeSource,
which self-skips when huggingface.co is unreachable (the already-verified
2-record real fetch lives in test_p3_full_store.py and is not re-run
here).

Run:  python -m pytest dee.cpp/tools/phase3/test_p3_kaggle_job.py -q
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import p3_builder  # noqa: E402
import p3_kaggle_job as job  # noqa: E402
import p3_manifest  # noqa: E402
from p3_builder import _read_lines  # noqa: E402

# Reuse the proven synthetic-universe fixtures (same directory).
from test_p3_full_store import (  # noqa: E402
    HEADERS,
    MINI_EXPERTS,
    MINI_LAYERS,
    MINI_RECORD_BYTES,
    expected_record_blob,
    make_mini_universe,
)

SEG = job.SEGMENT_TEMPLATE
MINI_SEGMENT_BYTES = MINI_EXPERTS * MINI_RECORD_BYTES  # 280


def _mini(tmp_path: Path):
    """Build the synthetic mini universe (shards + manifest + records)."""
    return make_mini_universe(tmp_path)


def _local_pub(tmp_path: Path, manifest, name="published"):
    return job.LocalDirPublisher(
        tmp_path / name, "mini", manifest["universe_sha256"])


def _seg_blob(bucket: int) -> bytes:
    return b"".join(
        expected_record_blob(bucket, e) for e in range(MINI_EXPERTS))


# ---------------------------------------------------------------------------
# End-to-end: build -> per-segment push -> index push
# ---------------------------------------------------------------------------


def test_end_to_end_build_push_evict(tmp_path):
    shards_dir, manifest, records = _mini(tmp_path)
    pub = _local_pub(tmp_path, manifest)
    report = job.run_job(
        manifest, records, p3_builder.LocalShardSource(shards_dir),
        tmp_path / "store", publisher=pub, poll_seconds=0.01)

    assert report["success"], report
    assert report["pushed_buckets"] == [0, 1, 2]
    assert report["push_errors"] == []

    # Published "datasets" contain byte-exact segment content.
    for bucket in range(MINI_LAYERS):
        slug = job.segment_slug("mini", manifest["universe_sha256"], bucket)
        seg = tmp_path / "published" / slug / SEG.format(bucket=bucket)
        assert seg.read_bytes() == _seg_blob(bucket)
    # Index dataset carries metadata + integrity + journals + manifest.
    idx_slug = job.index_slug("mini", manifest["universe_sha256"])
    idx = tmp_path / "published" / idx_slug
    for name in ("metadata.json", "integrity.jsonl",
                 "build.journal.jsonl", "publish.journal.jsonl",
                 "p3_manifest.json", "p3_records.jsonl"):
        assert (idx / name).is_file(), name

    # Pushed segments were evicted to same-size sparse tombstones.
    for bucket in range(MINI_LAYERS):
        tomb = tmp_path / "store" / "segments" / SEG.format(bucket=bucket)
        assert tomb.stat().st_size == MINI_SEGMENT_BYTES
        assert tomb.read_bytes() == b"\x00" * MINI_SEGMENT_BYTES
    # Staging dir holds no segment afterwards.
    assert not list((tmp_path / "store" / "_staging").glob("*.dee4"))

    # Publish journal: one line per bucket + one index line, all fsync'd.
    pj = _read_lines(tmp_path / "store" / "publish.journal.jsonl")
    buckets = [e["bucket"] for e in pj if "bucket" in e]
    assert buckets == [0, 1, 2]
    assert all(e["ref"].startswith("local://") for e in pj if "bucket" in e)
    assert any(e.get("index") for e in pj)

    meta = json.loads((tmp_path / "store" / "metadata.json").read_text())
    assert meta["format"] == "dee4-v4-segmented"
    assert len(meta["segments"]) == MINI_LAYERS


def test_segments_are_never_copied_one_transient(tmp_path):
    """The staging contract: at most one staged file, moved not copied.

    Proven structurally: after the job, the staged area is empty, each
    segment exists exactly once (in the published dataset dir), and the
    build dir holds only tombstones — the sum of on-disk real bytes is
    one segment per published dataset, never a local copy too.
    """
    shards_dir, manifest, records = _mini(tmp_path)
    pub = _local_pub(tmp_path, manifest)
    report = job.run_job(
        manifest, records, p3_builder.LocalShardSource(shards_dir),
        tmp_path / "store", publisher=pub, poll_seconds=0.01)
    assert report["success"]
    # Real segment bytes exist ONLY under published/, not in the build dir.
    real = b""
    for bucket in range(MINI_LAYERS):
        seg = (tmp_path / "store" / "segments" / SEG.format(bucket=bucket))
        assert seg.read_bytes() == b"\x00" * MINI_SEGMENT_BYTES  # tombstone
        slug = job.segment_slug("mini", manifest["universe_sha256"], bucket)
        real += (tmp_path / "published" / slug
                 / SEG.format(bucket=bucket)).read_bytes()
    assert real == b"".join(_seg_blob(b) for b in range(MINI_LAYERS))


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


class _FailPublisher(job.LocalDirPublisher):
    """Local publisher that fails on one bucket for the first N calls."""

    def __init__(self, *a, fail_bucket=1, times=999, **kw):
        super().__init__(*a, **kw)
        self.fail_bucket = fail_bucket
        self.remaining = times

    def publish(self, staging_dir, seg_name, bucket, *, sha256, size):
        if bucket == self.fail_bucket and self.remaining > 0:
            self.remaining -= 1
            raise job.PublishError("injected transient upload failure")
        return super().publish(
            staging_dir, seg_name, bucket, sha256=sha256, size=size)


def test_resume_after_publish_failure(tmp_path):
    shards_dir, manifest, records = _mini(tmp_path)
    pub = _FailPublisher(
        tmp_path / "published", "mini", manifest["universe_sha256"],
        fail_bucket=1)
    report = job.run_job(
        manifest, records, p3_builder.LocalShardSource(shards_dir),
        tmp_path / "store", publisher=pub, poll_seconds=0.01)
    # Build completed; bucket 1's push kept failing -> job fails but every
    # other committed segment still shipped (pushes are independent).
    assert report["success"] is False
    assert report["push_errors"]
    assert report["pushed_buckets"] == [0, 2]
    committed = _read_lines(tmp_path / "store" / "build.journal.jsonl")
    assert len(committed) == MINI_LAYERS

    # Retry: same build dir, healthy publisher -> finishes without
    # re-pushing already-shipped buckets.
    pub2 = _local_pub(tmp_path, manifest)
    report2 = job.run_job(
        manifest, records, p3_builder.LocalShardSource(shards_dir),
        tmp_path / "store", publisher=pub2, poll_seconds=0.01)
    assert report2["success"], report2
    assert report2["pushed_buckets"] == [0, 1, 2]
    assert pub2.calls == [1]  # only the missing bucket was pushed
    for bucket in range(MINI_LAYERS):
        slug = job.segment_slug("mini", manifest["universe_sha256"], bucket)
        assert (tmp_path / "published" / slug
                / SEG.format(bucket=bucket)).read_bytes() == _seg_blob(bucket)


def test_cross_session_resume_via_seed_dir(tmp_path):
    """Simulate a new Kaggle session: only the previous run's small files
    survive (kernel output).  Journals + tombstones must let the builder
    skip everything and the publisher skip every shipped segment."""
    shards_dir, manifest, records = _mini(tmp_path)
    pub = _local_pub(tmp_path, manifest)
    store1 = tmp_path / "store1"
    report = job.run_job(
        manifest, records, p3_builder.LocalShardSource(shards_dir),
        store1, publisher=pub, poll_seconds=0.01)
    assert report["success"]

    # New session: seed dir = what /kaggle/working output carried over.
    seed = tmp_path / "seed"
    seed.mkdir()
    for name in ("build.journal.jsonl", "integrity.jsonl",
                 "publish.journal.jsonl", "metadata.json",
                 "p3_manifest.json", "p3_records.jsonl"):
        src = store1 / name
        if src.is_file():
            (seed / name).write_bytes(src.read_bytes())

    pub2 = _local_pub(tmp_path, manifest, name="published2")
    store2 = tmp_path / "store2"
    j = job.P3KaggleJob(
        manifest, records, p3_builder.LocalShardSource(shards_dir),
        store2, publisher=pub2, poll_seconds=0.01)
    j.seed_from(seed)
    report2 = j.run()
    assert report2["success"], report2
    # Nothing rebuilt, nothing re-published.
    assert report2["build"]["buckets_built"] == []
    assert sorted(report2["build"]["buckets_skipped"]) == [0, 1, 2]
    assert pub2.calls == []
    # Tombstones were recreated so the builder's file checks passed.
    for bucket in range(MINI_LAYERS):
        tomb = store2 / "segments" / SEG.format(bucket=bucket)
        assert tomb.stat().st_size == MINI_SEGMENT_BYTES


# ---------------------------------------------------------------------------
# Build-only and concat paths
# ---------------------------------------------------------------------------


def test_build_only_then_concat_v2_and_verify(tmp_path):
    shards_dir, manifest, records = _mini(tmp_path)
    store = tmp_path / "store"
    report = job.run_job(
        manifest, records, p3_builder.LocalShardSource(shards_dir),
        store, publisher=None)
    assert report["success"]
    assert p3_builder.verify_store(store)["success"]

    out = tmp_path / "v2"
    rep = job.concat_v2(store, out)
    assert rep["success"] and rep["verify"]["success"]
    data = (out / "experts.dee4").read_bytes()
    expected = b"".join(_seg_blob(b) for b in range(MINI_LAYERS))
    assert data == expected
    assert rep["data_sha256"] == hashlib.sha256(expected).hexdigest()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["format"] == "dee4-v2"
    assert meta["data_sha256"] == rep["data_sha256"]
    # The C++ reader contract fields are all present.
    for key in ("num_layers", "experts_per_layer", "total_experts",
                "record_bytes", "weight_offsets", "scale_offsets"):
        assert key in meta

    # Flat layout (segments downloaded per-dataset, no segments/ dir) also
    # resolves through the same code path.
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "metadata.json").write_text(
        (store / "metadata.json").read_text())
    for b in range(MINI_LAYERS):
        (flat / SEG.format(bucket=b)).write_bytes(
            (store / "segments" / SEG.format(bucket=b)).read_bytes())
    import shutil
    shutil.copyfile(store / "integrity.jsonl", flat / "integrity.jsonl")
    out2 = tmp_path / "v2flat"
    rep2 = job.concat_v2(flat, out2)
    assert rep2["data_sha256"] == rep["data_sha256"]


def test_concat_from_published_dataset_layout(tmp_path):
    """The real consumer shape: metadata/integrity live in the index
    dataset; each segment lives in its own dataset dir (e.g. mounted at
    /kaggle/input/<slug>/)."""
    shards_dir, manifest, records = _mini(tmp_path)
    pub = _local_pub(tmp_path, manifest)
    store = tmp_path / "store"
    report = job.run_job(
        manifest, records, p3_builder.LocalShardSource(shards_dir),
        store, publisher=pub, poll_seconds=0.01)
    assert report["success"]
    idx_dir = (tmp_path / "published"
               / job.index_slug("mini", manifest["universe_sha256"]))
    out = tmp_path / "v2"
    rep = job.concat_v2(idx_dir, out, dataset_root=tmp_path / "published")
    assert rep["success"] and rep["verify"]["success"]
    expected = b"".join(_seg_blob(b) for b in range(MINI_LAYERS))
    assert (out / "experts.dee4").read_bytes() == expected


def test_keep_segments_mode_leaves_real_files(tmp_path):
    shards_dir, manifest, records = _mini(tmp_path)
    pub = _local_pub(tmp_path, manifest)
    store = tmp_path / "store"
    report = job.run_job(
        manifest, records, p3_builder.LocalShardSource(shards_dir),
        store, publisher=pub, evict_after_push=False, poll_seconds=0.01)
    assert report["success"]
    for bucket in range(MINI_LAYERS):
        seg = store / "segments" / SEG.format(bucket=bucket)
        assert seg.read_bytes() == _seg_blob(bucket)  # real, not tombstone
    # The kept store still verifies end-to-end.
    assert p3_builder.verify_store(store)["success"]


def test_prefix_scope_build(tmp_path):
    shards_dir, manifest, records = _mini(tmp_path)
    m2, r2 = job.prefix_scope(manifest, records, 2)
    assert m2["n_buckets"] == 2 and m2["total_experts"] == 8
    assert len(r2) == 8
    report = job.run_job(
        m2, r2, p3_builder.LocalShardSource(shards_dir),
        tmp_path / "store", publisher=None)
    assert report["success"]
    assert report["build"]["records_written"] == 8
    meta = json.loads((tmp_path / "store" / "metadata.json").read_text())
    assert meta["num_layers"] == 2 and meta["total_experts"] == 8


# ---------------------------------------------------------------------------
# Source wrappers: gate + prefetch
# ---------------------------------------------------------------------------


class _CountingSource:
    """Deterministic source that records max in-flight concurrency."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = 0
        self.inflight = 0
        self.max_inflight = 0
        self._lock = threading.Lock()

    def fetch(self, shard: str, data_offset: int, nbytes: int) -> bytes:
        with self._lock:
            self.calls += 1
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.delay:
                time.sleep(self.delay)
            return b"\xAB" * nbytes
        finally:
            with self._lock:
                self.inflight -= 1


def test_push_gate_blocks_until_prior_bucket_pushed():
    inner = _CountingSource()
    gate = job.PushGate(inner, {"s0": 0, "s1": 1, "s2": 2}, pushed=set())
    # Bucket 0 never waits.
    assert gate.fetch("s0", 0, 4) == b"\xAB" * 4

    done = []

    def _fetch1():
        done.append(gate.fetch("s1", 0, 4))

    t = threading.Thread(target=_fetch1)
    t.start()
    time.sleep(0.2)
    assert not done  # blocked: bucket 0 not pushed
    gate.mark_pushed(0)
    t.join(timeout=5)
    assert done == [b"\xAB" * 4]

    # fail() releases every wait permanently.
    def _fetch2():
        done.append(gate.fetch("s2", 0, 4))

    t2 = threading.Thread(target=_fetch2)
    t2.start()
    time.sleep(0.2)
    gate.fail()
    t2.join(timeout=5)
    assert len(done) == 2


def test_prefetch_source_order_concurrency_and_resume():
    inner = _CountingSource(delay=0.02)
    requests = [(f"shard-{i % 3}", i * 16, 8) for i in range(24)]
    src = job.PrefetchRangeSource(inner, requests, workers=4, window=6)
    try:
        out = [src.fetch(*r) for r in requests]
        assert out == [b"\xAB" * 8] * 24
        assert src.stats["served"] == 24
        assert src.stats["direct"] == 0
        assert inner.max_inflight >= 2  # real concurrency happened
        assert src.stats["inflight_max"] <= 6

        # Out-of-order fetch (e.g. resumed build): falls through to direct
        # and realigns the request pointer.
        again = src.fetch(*requests[10])
        assert again == b"\xAB" * 8
        nxt = src.fetch(*requests[11])
        assert nxt == b"\xAB" * 8
    finally:
        src.close()


def test_prefetch_full_build_byte_exact(tmp_path):
    """A prefetched local source produces the identical store bytes."""
    shards_dir, manifest, records = _mini(tmp_path)
    source = job.make_source(
        "local", manifest, records, shards_dir=shards_dir,
        prefetch_workers=4, prefetch_window=8)
    assert isinstance(source, job.PrefetchRangeSource)
    report = job.run_job(manifest, records, source, tmp_path / "store",
                         publisher=None)
    assert report["success"]
    for bucket in range(MINI_LAYERS):
        seg = tmp_path / "store" / "segments" / SEG.format(bucket=bucket)
        assert seg.read_bytes() == _seg_blob(bucket)
    source.close()


def test_interleaved_push_never_overlaps_past_gate(tmp_path):
    """With the gate on, the builder cannot start bucket K until K-1 is
    pushed — publish order is strictly 0,1,2 and pushes all landed."""
    shards_dir, manifest, records = _mini(tmp_path)
    pub = _local_pub(tmp_path, manifest)
    report = job.run_job(
        manifest, records, p3_builder.LocalShardSource(shards_dir),
        tmp_path / "store", publisher=pub, interleave=True,
        poll_seconds=0.01)
    assert report["success"]
    assert pub.calls == sorted(pub.calls)  # pushed in bucket order
    assert pub.calls == [0, 1, 2]


# ---------------------------------------------------------------------------
# Universe loading + remote wiring (the only networked check, self-skipping)
# ---------------------------------------------------------------------------


def test_load_universe_from_committed_headers(tmp_path):
    manifest, records = job.load_universe(headers_dir=HEADERS)
    assert manifest["n_buckets"] == 46
    assert manifest["total_experts"] == 11_776
    assert len(records) == 11_776
    assert manifest["revision"] == job.PINNED_REVISION
    # Prefix scoping the real universe is a valid smaller universe.
    m2, r2 = job.prefix_scope(manifest, records, 43)
    assert m2["total_experts"] == 11_008
    assert m2["universe_sha256"] == p3_manifest.universe_sha256(43)
    assert len(r2) == 11_008


def test_load_universe_round_trip_files(tmp_path):
    manifest, records = job.load_universe(headers_dir=HEADERS)
    out = tmp_path / "u"
    m = dict(manifest)
    m["_records"] = records
    paths = p3_manifest.write_manifest(m, out)
    manifest2, records2 = job.load_universe(
        manifest_path=paths["manifest"], records_path=paths["records"])
    assert manifest2["universe_sha256"] == manifest["universe_sha256"]
    assert len(records2) == 11_776


@pytest.mark.network
def test_remote_source_is_pinned_and_reaches_hf():
    """Wiring check only: the remote source is bound to the pinned repo +
    revision, and a real 8-byte header probe + one small range succeed.
    (The byte-exact 2-record proof lives in test_p3_full_store.py.)"""
    manifest, records = job.load_universe(headers_dir=HEADERS)
    src = job.make_source("remote", manifest, records)
    assert isinstance(src, p3_builder.RemoteRangeSource)
    assert src.repository == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert src.revision == "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
    rec = records[0]
    # One real range: the smallest component of record 0 (a 256 KiB scale).
    comp, off, nb, _ro, tensor = min(rec["ranges"], key=lambda r: r[2])
    data = src.fetch(rec["shard"], off, nb)
    assert len(data) == nb == 262_144
    assert src.stats["requests"] >= 2  # header probe + range fetch


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_build_and_status(tmp_path):
    shards_dir, manifest, records = _mini(tmp_path)
    udir = tmp_path / "u"
    mpath = udir / "m.json"
    rpath = udir / "r.jsonl"
    udir.mkdir()
    mpath.write_text(json.dumps(manifest))
    with rpath.open("w", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    rc = job.main([
        "build", "--build-dir", str(tmp_path / "store"),
        "--manifest", str(mpath), "--records", str(rpath),
        "--source", "local", "--shards", str(shards_dir),
        "--publisher", "local",
        "--published-root", str(tmp_path / "published"),
        "--dataset-prefix", "mini", "--poll", "0.01"])
    assert rc == 0
    rc = job.main(["status", "--build-dir", str(tmp_path / "store")])
    assert rc == 0


def test_preflight_creates_index_dataset_first(tmp_path):
    _shards, manifest, records = _mini(tmp_path)
    # CLI path: local publisher, universe from files.
    udir = tmp_path / "u"
    udir.mkdir()
    (udir / "m.json").write_text(json.dumps(manifest))
    with (udir / "r.jsonl").open("w", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    rc = job.main([
        "preflight", "--work-dir", str(tmp_path / "pf"),
        "--manifest", str(udir / "m.json"), "--records",
        str(udir / "r.jsonl"), "--publisher", "local",
        "--published-root", str(tmp_path / "published"),
        "--dataset-prefix", "mini"])
    assert rc == 0
    idx = tmp_path / "published" / job.index_slug(
        "mini", manifest["universe_sha256"])
    assert (idx / "preflight.json").is_file()
