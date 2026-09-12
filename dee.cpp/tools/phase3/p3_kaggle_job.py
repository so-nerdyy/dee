"""Phase 3: executable build+publish driver for the full-universe store.

Wraps ``p3_builder.build_segmented`` (dee4-v4-segmented) on the pinned
checkpoint revision ``9e165c30e2704aec5d9d593cce3eebd58bbef1cb`` and adds
what the prototype does not have — a bounded-transient dataset push and a
cross-restart/cross-session resume:

* **One transient segment at a time.**  Each finished bucket segment
  (3,422,552,064 B on the real universe) is staged into a one-file upload
  directory by *hardlink* (zero-copy, atomic, never a second byte
  allocation), uploaded, journal-committed, then evicted.  Source bytes
  are only ever read (HTTP ranges or a read-only dataset mount), so the
  job never stages source+dest simultaneously; peak local disk is ~2
  segments (~6.4 GiB), comfortably inside Kaggle's 19.5 GiB working
  budget — the whole job can run in ``/kaggle/working`` if desired.

* **Bounded-disk backpressure.**  ``PushGate`` wraps the byte source and
  blocks the first fetch of bucket K's shard until bucket K-1 is pushed,
  so at most one built-but-unpushed segment can ever exist; upload
  slowness stalls the *build*, never overflows the disk.

* **Resume.**  ``build.journal.jsonl`` (written by build_segmented after
  each fsync'd segment publish) plus ``publish.journal.jsonl`` (one
  fsync'd line per shipped segment) fully describe job state.  Pushed
  buckets leave a same-size sparse *tombstone* behind so the builder's
  committed-bucket checks pass on restart; a crashed or preempted run
  resumes exactly where it stopped, including across Kaggle sessions via
  ``--seed-dir`` pointed at the previous run's kernel-output files.

* **Publish shape.**  A new Kaggle dataset *version* is a complete
  snapshot of the upload folder (files are NOT carried over between
  versions), so incremental publish uses ONE dataset per segment
  (``<prefix>-u<universe_sha256[:12]>-b<NN>``) plus a small ``-index``
  dataset carrying metadata.json / integrity.jsonl / the journals /
  manifest / records.  Every segment dataset name binds the store to its
  ``universe_sha256``.

Usage (Kaggle kernel cell or local shell)::

    python dee.cpp/tools/phase3/p3_kaggle_job.py build \
        --build-dir /kaggle/working/p3-store \
        --headers dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers \
        --source remote --prefetch 8 \
        --publisher kaggle --dataset-owner <user> --public

    # rehearsal on the synthetic mini universe (tests cover this path):
    python p3_kaggle_job.py build --build-dir out --manifest m.json \
        --records r.jsonl --source local --shards shards \
        --publisher local --published-root published

    # consumer side: stitch published segments into the dee4-v2 bank:
    python p3_kaggle_job.py concat --store <dir-with-metadata+segments> \
        --out <v2-store-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import p3_builder  # noqa: E402
import p3_manifest  # noqa: E402
from p3_builder import (  # noqa: E402
    INTEGRITY_FILE,
    JOURNAL_FILE,
    SEGMENT_TEMPLATE,
    RangeSource,
    _append_line,
    _read_lines,
    _write_json_atomic,
)

PINNED_REPOSITORY = p3_manifest.MODEL
PINNED_REVISION = p3_manifest.REVISION

PUBLISH_JOURNAL = "publish.journal.jsonl"
JOB_REPORT = "job_report.json"
STAGING_DIRNAME = "_staging"
INDEX_STAGING_DIRNAME = "_index_staging"
TOMBSTONE_SUFFIX = ".tomb"

# Dataset naming: <prefix>-u<universe_sha256[:12]>-b<NN> / ...-index
UNIVERSE_TAG_LEN = 12


class PublishError(RuntimeError):
    """A segment could not be shipped or failed its pre-ship check."""


# ---------------------------------------------------------------------------
# Universe loading / scoping
# ---------------------------------------------------------------------------


def load_universe(
    manifest_path: Path | str | None = None,
    records_path: Path | str | None = None,
    headers_dir: Path | str | None = None,
    n_buckets: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the (manifest, records) pair from JSON files or rebuild it
    deterministically from the committed shard headers.

    ``n_buckets`` restricts the build to the first N buckets (a prefix-scope
    smoke run); the emitted store is then a self-consistent smaller store
    whose ``universe_sha256`` covers exactly those pairs.
    """
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text("utf-8"))
        if records_path is None:
            raise ValueError("--records is required with --manifest")
        records = p3_manifest.load_records(records_path)
    elif headers_dir is not None:
        manifest = p3_manifest.build_manifest(headers_dir)
        records = manifest.pop("_records")
    else:
        raise ValueError("need --manifest/--records or --headers")

    if int(manifest["total_experts"]) != len(records):
        raise ValueError(
            f"manifest total_experts {manifest['total_experts']} != "
            f"records {len(records)}")
    for i, rec in enumerate(records):
        if int(rec["record_index"]) != i:
            raise ValueError(f"records[{i}] record_index drift")

    if n_buckets is not None:
        manifest, records = prefix_scope(manifest, records, n_buckets)
    return manifest, records


def _universe_sha256(n_buckets: int, experts_per_layer: int) -> str:
    """Same convention as p3_manifest.universe_sha256, parameterized by
    experts_per_layer so prefix/synthetic scopes hash identically."""
    pairs = [
        [bucket, expert]
        for bucket in range(n_buckets)
        for expert in range(experts_per_layer)
    ]
    return hashlib.sha256(p3_manifest.canonical_json_bytes(pairs)).hexdigest()


def prefix_scope(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    n_buckets: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a self-consistent manifest+records for the first N buckets.

    Buckets build strictly in order, so a prefix build is exactly a smaller
    universe; ``build_segmented`` needs no changes to consume it.
    """
    full = int(manifest["n_buckets"])
    if not 0 < n_buckets <= full:
        raise ValueError(f"n_buckets {n_buckets} outside (0, {full}]")
    epl = int(manifest["experts_per_layer"])
    record_bytes = int(manifest["record_bytes"])
    m = dict(manifest)
    m["n_buckets"] = n_buckets
    m["total_experts"] = n_buckets * epl
    m["store_bytes"] = n_buckets * epl * record_bytes
    m["store_gib"] = round(m["store_bytes"] / (1 << 30), 4)
    if "n_mtp_buckets" in m:
        m["n_mtp_buckets"] = max(0, n_buckets - int(m["n_layers"]))
    if "bucket_shards" in m:
        m["bucket_shards"] = {
            k: v for k, v in m["bucket_shards"].items() if int(k) < n_buckets
        }
    m["universe_sha256"] = _universe_sha256(n_buckets, epl)
    m["manifest_sha256"] = hashlib.sha256(
        p3_manifest.canonical_json_bytes(
            {k: v for k, v in m.items() if k != "manifest_sha256"})
    ).hexdigest()
    return m, records[: n_buckets * epl]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _synthetic_source() -> RangeSource:
    """Zero-IO deterministic source (pipeline smoke only — content is not
    the checkpoint)."""
    block = hashlib.sha256(b"p3-kaggle-synth").digest()

    class _Src:
        def fetch(self, shard: str, data_offset: int, nbytes: int) -> bytes:
            return (block * (nbytes // 32 + 1))[:nbytes]

    return _Src()


class PrefetchRangeSource:
    """Bounded look-ahead wrapper: keeps up to ``window`` range requests in
    flight on a thread pool so a remote build is bandwidth-bound, not
    request-latency-bound (~70k requests at 100-300 ms each would otherwise
    dominate wall time on a datacenter link).

    Deterministic contract: the builder consumes ranges in exactly
    ``requests`` order (record_index order); out-of-order fetches (resume /
    repair) fall through to a direct call and realign the window.
    """

    def __init__(
        self,
        inner: RangeSource,
        requests: list[tuple[str, int, int]],
        *,
        workers: int = 8,
        window: int | None = None,
    ):
        if workers < 1:
            raise ValueError("PrefetchRangeSource needs workers >= 1")
        self._inner = inner
        self._req = list(requests)
        self._window = window or workers * 3
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="p3-prefetch")
        self._inflight: dict[int, Any] = {}
        self._sched = 0   # next index to schedule
        self._next = 0    # expected next consume index
        self._lock = threading.Lock()
        self.stats = {"served": 0, "direct": 0, "prefetched": 0,
                      "inflight_max": 0}
        self._fill()

    def _fill(self) -> None:
        while (self._sched < len(self._req)
               and len(self._inflight) < self._window):
            i = self._sched
            self._sched += 1
            shard, off, nb = self._req[i]
            self._inflight[i] = self._pool.submit(
                self._inner.fetch, shard, off, nb)
        self.stats["inflight_max"] = max(
            len(self._inflight), self.stats.get("inflight_max", 0))

    def fetch(self, shard: str, data_offset: int, nbytes: int) -> bytes:
        key = (shard, int(data_offset), int(nbytes))
        with self._lock:
            if self._next < len(self._req) and self._req[self._next] == key:
                i = self._next
                self._next += 1
                fut = self._inflight.pop(i, None)
                if fut is None:
                    fut = self._pool.submit(self._inner.fetch, *key)
                    self.stats["direct"] += 1
                else:
                    self.stats["prefetched"] += 1
                self._fill()
                self.stats["served"] += 1
            else:
                fut = self._pool.submit(self._inner.fetch, *key)
                self.stats["direct"] += 1
                # Realign: find this key ahead of _next and drop stale work.
                for j in range(self._next, len(self._req)):
                    if self._req[j] == key:
                        for k in list(self._inflight):
                            if k < j:
                                self._inflight[k].cancel()
                                del self._inflight[k]
                        self._next = j + 1
                        if self._sched < self._next:
                            self._sched = self._next
                        break
                self._fill()
        return fut.result()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def make_source(
    kind: str,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    shards_dir: Path | str | None = None,
    prefetch_workers: int = 0,
    prefetch_window: int | None = None,
) -> RangeSource:
    """Build the byte source.  ``remote`` is the pinned-revision HF range
    fetcher (no local checkpoint needed); ``local`` reads mounted/local
    shards; ``synthetic`` is a zero-IO pipeline smoke."""
    if kind == "remote":
        source: RangeSource = p3_builder.RemoteRangeSource(
            repository=manifest.get("model", PINNED_REPOSITORY),
            revision=manifest.get("revision", PINNED_REVISION),
        )
    elif kind == "local":
        if shards_dir is None:
            raise ValueError("--source local requires --shards")
        source = p3_builder.LocalShardSource(shards_dir)
    elif kind == "synthetic":
        source = _synthetic_source()
    else:
        raise ValueError(f"unknown source {kind}")

    if prefetch_workers > 0:
        requests = [
            (rec["shard"], int(data_offset), int(nbytes))
            for rec in records
            for _comp, data_offset, nbytes, _roff, _t in rec["ranges"]
        ]
        source = PrefetchRangeSource(
            source, requests,
            workers=prefetch_workers, window=prefetch_window)
    return source


class PushGate:
    """Disk backpressure valve wrapping a source.

    Blocks the first fetch of bucket K's home shard until bucket K-1 has
    been pushed, so at most one built-but-unpushed segment can ever occupy
    local disk.  Upload slowness then stalls the build instead of piling
    up segments; ``fail()`` releases the gate permanently so a push error
    degrades to "build everything, push at the end" rather than deadlock.
    """

    def __init__(
        self,
        inner: RangeSource,
        shard_to_bucket: dict[str, int],
        pushed: set[int],
    ):
        self._inner = inner
        self._map = shard_to_bucket
        self._pushed = set(pushed)
        self._cond = threading.Condition()
        self._failed = False

    def fetch(self, shard: str, data_offset: int, nbytes: int) -> bytes:
        bucket = self._map.get(shard)
        if bucket is not None and bucket > 0:
            with self._cond:
                while not self._failed and (bucket - 1) not in self._pushed:
                    self._cond.wait(1.0)
        return self._inner.fetch(shard, data_offset, nbytes)

    def mark_pushed(self, bucket: int) -> None:
        with self._cond:
            self._pushed.add(bucket)
            self._cond.notify_all()

    def fail(self) -> None:
        with self._cond:
            self._failed = True
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# Publishers
# ---------------------------------------------------------------------------


def segment_slug(prefix: str, universe_sha256: str, bucket: int) -> str:
    return f"{prefix}-u{universe_sha256[:UNIVERSE_TAG_LEN]}-b{bucket:02d}"


def index_slug(prefix: str, universe_sha256: str) -> str:
    return f"{prefix}-u{universe_sha256[:UNIVERSE_TAG_LEN]}-index"


class SegmentPublisher(Protocol):
    """Uploads the contents of a staging dir as one published unit.

    The publisher must not remove the staged segment file — the driver
    owns its lifecycle (evict/tombstone or keep).  ``prepare`` runs once
    per push before the segment lands in staging.
    """

    def prepare(self, staging_dir: Path, bucket: int, seg_name: str) -> None:
        ...

    def publish(
        self,
        staging_dir: Path,
        seg_name: str,
        bucket: int,
        *,
        sha256: str,
        size: int,
    ) -> dict[str, Any]:
        """Ship the staged file; return a ref dict (must include 'ref')."""
        ...

    def publish_index(self, staging_dir: Path) -> dict[str, Any]:
        """Ship a staging dir of small metadata files; return a ref dict."""
        ...


class LocalDirPublisher:
    """Dry-run/test publisher: 'datasets' are directories under ``root``.

    Models the real flow exactly (staging dir -> published location) but
    copies the staged file instead of uploading; only used where segment
    sizes are small or a full local rehearsal is intended.
    """

    def __init__(self, root: Path | str, prefix: str, universe_sha256: str):
        self.root = Path(root)
        self.prefix = prefix
        self.universe_sha256 = universe_sha256
        self.calls: list[int] = []

    def prepare(self, staging_dir: Path, bucket: int, seg_name: str) -> None:
        staging_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(staging_dir / "dataset-metadata.json", {
            "title": segment_slug(self.prefix, self.universe_sha256, bucket),
            "id": f"local/{segment_slug(self.prefix, self.universe_sha256, bucket)}",
            "licenses": [{"name": "other"}],
        })

    def publish(
        self,
        staging_dir: Path,
        seg_name: str,
        bucket: int,
        *,
        sha256: str,
        size: int,
    ) -> dict[str, Any]:
        slug = segment_slug(self.prefix, self.universe_sha256, bucket)
        dest_dir = self.root / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / seg_name
        uploaded = True
        if dest.is_file() and dest.stat().st_size == size \
                and _sha256_file(dest) == sha256:
            uploaded = False  # idempotent re-push after a lost journal line
        else:
            shutil.copyfile(staging_dir / seg_name, dest)
        self.calls.append(bucket)
        return {
            "dataset": slug,
            "ref": f"local://{slug}/{seg_name}",
            "uploaded": uploaded,
        }

    def publish_index(self, staging_dir: Path) -> dict[str, Any]:
        slug = index_slug(self.prefix, self.universe_sha256)
        dest_dir = self.root / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(staging_dir.iterdir()):
            if path.is_file() and path.name != "dataset-metadata.json":
                shutil.copyfile(path, dest_dir / path.name)
        return {"dataset": slug, "ref": f"local://{slug}/", "uploaded": True}


class KagglePublisher:
    """Real publisher: one Kaggle dataset per segment + one index dataset.

    Dataset versions are complete snapshots (no file carry-over), so a
    per-segment dataset is the only incremental publish shape that fits
    the working budget.  Uploads stream the staged file; the kaggle
    package's ResumableUploadContext resumes a partial blob upload inside
    a session, and the driver-level publish journal resumes across
    sessions.  Requires KAGGLE_USERNAME/KAGGLE_KEY (kernel Secrets or
    ~/.kaggle/kaggle.json).
    """

    def __init__(
        self,
        owner: str,
        prefix: str,
        universe_sha256: str,
        *,
        public: bool = True,
        license_name: str = "deepseek",
        quiet: bool = False,
    ):
        self.owner = owner
        self.prefix = prefix
        self.universe_sha256 = universe_sha256
        self.public = public
        self.license_name = license_name
        self.quiet = quiet
        self._api = None

    def _client(self):
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()  # env vars / kaggle.json / kernel secrets
            self._api = api
        return self._api

    def prepare(self, staging_dir: Path, bucket: int, seg_name: str) -> None:
        staging_dir.mkdir(parents=True, exist_ok=True)
        slug = segment_slug(self.prefix, self.universe_sha256, bucket)
        _write_json_atomic(staging_dir / "dataset-metadata.json", {
            "title": slug,
            "id": f"{self.owner}/{slug}",
            "licenses": [{"name": self.license_name}],
        })

    def _remote_has_file(self, slug: str, name: str, size: int) -> bool:
        try:
            listing = self._client().dataset_list_files(
                f"{self.owner}/{slug}")
        except Exception:
            return False
        files = getattr(listing, "files", listing) or []
        for f in files:
            fname = getattr(f, "name", getattr(f, "file_name", ""))
            fsize = int(getattr(f, "total_bytes",
                               getattr(f, "totalBytes", -1)))
            if fname == name and fsize == size:
                return True
        return False

    def _create_or_version(self, staging_dir: Path, notes: str) -> None:
        api = self._client()
        try:
            resp = api.dataset_create_new(
                folder=str(staging_dir), public=self.public,
                quiet=self.quiet, convert_to_csv=False, dir_mode="skip")
        except Exception:
            resp = None
        if resp is not None and getattr(resp, "status", "") == "ok":
            return
        resp = api.dataset_create_version(
            str(staging_dir), notes, quiet=self.quiet,
            convert_to_csv=False, delete_old_versions=False,
            dir_mode="skip")
        if getattr(resp, "status", "") == "ok":
            return
        # Older in-kernel kaggle (2.0.x) reports an ambiguous/empty status
        # even when the version upload succeeded; verify remotely before
        # failing closed.
        try:
            meta = json.loads(
                (staging_dir / "dataset-metadata.json").read_text())
            staged = {p.name for p in staging_dir.iterdir()
                      if p.name != "dataset-metadata.json"}
            listing = api.dataset_list_files(meta["id"])
            remote = {getattr(f, "name", getattr(f, "file_name", ""))
                      for f in (getattr(listing, "files", listing) or [])}
            if staged and staged <= remote:
                return
        except Exception:
            pass
        raise PublishError(
            f"dataset version failed: {getattr(resp, 'error', resp)}")

    def publish(
        self,
        staging_dir: Path,
        seg_name: str,
        bucket: int,
        *,
        sha256: str,
        size: int,
    ) -> dict[str, Any]:
        slug = segment_slug(self.prefix, self.universe_sha256, bucket)
        full = f"{self.owner}/{slug}"
        if self._remote_has_file(slug, seg_name, size):
            return {"dataset": full,
                    "ref": f"kaggle://{full}/{seg_name}",
                    "uploaded": False}
        self._create_or_version(staging_dir, f"p3 bucket {bucket:02d}")
        return {"dataset": full,
                "ref": f"kaggle://{full}/{seg_name}",
                "uploaded": True}

    def publish_index(self, staging_dir: Path) -> dict[str, Any]:
        slug = index_slug(self.prefix, self.universe_sha256)
        full = f"{self.owner}/{slug}"
        _write_json_atomic(staging_dir / "dataset-metadata.json", {
            "title": slug,
            "id": full,
            "licenses": [{"name": "other"}],
        })
        self._create_or_version(staging_dir, "p3 store index")
        return {"dataset": full, "ref": f"kaggle://{full}/",
                "uploaded": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _make_sparse_or_truncated(path: Path, size: int) -> None:
    """Create a zero-content file of exactly ``size`` bytes, sparse where
    the filesystem supports it (POSIX truncate is sparse; NTFS needs
    ``fsutil sparse`` — see p3_sparse_store for why)."""
    if sys.platform == "win32":
        try:
            from p3_sparse_store import _make_sparse
            if _make_sparse(path, size):
                return
        except Exception:
            pass
    with path.open("wb") as fh:
        fh.truncate(size)


def _same_fs(dir_a: Path, dir_b: Path) -> bool:
    """True when a file can be hardlinked/renamed between the two dirs."""
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)
    probe = dir_a / ".p3_fsprobe"
    probe2 = dir_b / ".p3_fsprobe"
    try:
        probe.write_bytes(b"x")
        os.link(probe, probe2)
        os.replace(probe2, probe.with_suffix(".moved"))
        return True
    except OSError:
        return False
    finally:
        for p in (probe, probe2, probe.with_suffix(".moved")):
            try:
                p.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


class P3KaggleJob:
    """Build the segmented store and push each segment dataset-style.

    Parameters
    ----------
    manifest, records : universe + record range list (load_universe output)
    source            : RangeSource (make_source output; wrapped here with
                        PushGate when pushing interleaved)
    build_dir         : store dir (journal, integrity, segments/, staging)
    publisher         : SegmentPublisher or None for build-only dry run
    evict_after_push  : replace pushed segments with sparse tombstones
                        (keeps the builder's committed-file checks valid
                        while bounding disk); False keeps real segments
    verify_before_push: stream-rehash each segment against its journal
                        sha256 before shipping (read-back integrity gate)
    interleave        : push segments while the build runs (True) or only
                        after build_segmented returns (False)
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        records: list[dict[str, Any]],
        source: RangeSource,
        build_dir: Path | str,
        *,
        publisher: SegmentPublisher | None = None,
        staging_dir: Path | str | None = None,
        evict_after_push: bool = True,
        verify_before_push: bool = True,
        interleave: bool = True,
        poll_seconds: float = 0.5,
    ):
        self.manifest = manifest
        self.records = records
        self.source = source
        self.build_dir = Path(build_dir)
        self.segments_dir = self.build_dir / "segments"
        self.staging_dir = Path(
            staging_dir) if staging_dir else self.build_dir / STAGING_DIRNAME
        self.publisher = publisher
        self.evict = evict_after_push
        self.verify_before_push = verify_before_push
        self.interleave = interleave
        self.poll_seconds = poll_seconds

        self.journal_path = self.build_dir / JOURNAL_FILE
        self.integrity_path = self.build_dir / INTEGRITY_FILE
        self.publish_journal_path = self.build_dir / PUBLISH_JOURNAL
        self.record_bytes = int(manifest["record_bytes"])
        self.epl = int(manifest["experts_per_layer"])
        self.n_buckets = int(manifest.get("n_buckets", manifest["n_layers"]))
        self.segment_bytes = self.record_bytes * self.epl
        self.pushed: dict[int, dict[str, Any]] = {}
        self.push_errors: list[str] = []
        self._push_failed: set[int] = set()
        self.gate: PushGate | None = None

    # -- state ----------------------------------------------------------

    def _load_pushed(self) -> None:
        self.pushed = {}
        for e in _read_lines(self.publish_journal_path):
            if "bucket" in e:
                self.pushed[int(e["bucket"])] = e

    def _committed(self) -> dict[int, dict[str, Any]]:
        out = {}
        for e in _read_lines(self.journal_path):
            if "bucket" in e:
                out[int(e["bucket"])] = e
        return out

    def _committed_safe(self) -> dict[int, dict[str, Any]]:
        """Concurrent-read variant for the watch loop: a torn trailing
        line (journal append in flight) is ignored this poll — the next
        poll sees it complete.  Post-join code uses the strict reader."""
        try:
            return self._committed()
        except (json.JSONDecodeError, OSError):
            return {}

    # -- staging / eviction ----------------------------------------------

    def _stage(self, src: Path, seg_name: str) -> Path:
        """Expose ``src`` inside the staging dir as ``seg_name`` via
        hardlink (same inode, zero copy, atomic).  Falls back to a rename
        on filesystems without hardlinks."""
        staged = self.staging_dir / seg_name
        if src == staged:
            return staged
        try:
            os.link(src, staged)
        except OSError:
            os.replace(src, staged)
        return staged

    def _tombstone(self, seg_path: Path) -> None:
        """Atomically replace a pushed segment with a same-size sparse
        tombstone so the builder's committed-bucket checks keep passing."""
        tmp = seg_path.with_name(seg_path.name + TOMBSTONE_SUFFIX)
        _make_sparse_or_truncated(tmp, self.segment_bytes)
        os.replace(tmp, seg_path)

    def _unstage(self, staged: Path, seg_path: Path) -> None:
        """After a successful push: drop the staged name; evict or keep."""
        if self.evict:
            try:
                staged.unlink()
            except FileNotFoundError:
                pass
            self._tombstone(seg_path)
        else:
            if staged.exists():
                if seg_path.exists():
                    staged.unlink()          # hardlink path: drop 2nd name
                else:
                    os.replace(staged, seg_path)  # rename path: move back
            elif not seg_path.exists():
                self._tombstone(seg_path)    # defensive: keep builder happy

    # -- push -------------------------------------------------------------

    def _push_one(self, bucket: int, journal_entry: dict[str, Any]) -> None:
        if self.publisher is None:
            return
        seg_name = SEGMENT_TEMPLATE.format(bucket=bucket)
        seg_path = self.segments_dir / seg_name
        staged = self.staging_dir / seg_name
        if seg_path.is_file():
            src = seg_path
        elif staged.is_file():
            src = staged  # crashed between stage and journal commit
        else:
            raise PublishError(
                f"bucket {bucket}: journal commits a segment that is "
                "neither in segments/ nor staging")
        expected_sha = journal_entry["segment_sha256"]
        if self.verify_before_push:
            actual = _sha256_file(src)
            if actual != expected_sha:
                raise PublishError(
                    f"bucket {bucket}: segment sha256 {actual} != "
                    f"journal {expected_sha} (read-back mismatch; refusing "
                    "to publish corrupt bytes)")
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.publisher.prepare(self.staging_dir, bucket, seg_name)
        if staged.exists() and src != staged:
            staged.unlink()  # stale staged copy from a crashed prior push
        staged = self._stage(src, seg_name)
        ref = self.publisher.publish(
            self.staging_dir, seg_name, bucket,
            sha256=expected_sha, size=self.segment_bytes)
        entry = {
            "bucket": bucket,
            "segment": seg_name,
            "segment_sha256": expected_sha,
            "segment_bytes": self.segment_bytes,
            "dataset": ref.get("dataset"),
            "ref": ref.get("ref"),
            "uploaded": ref.get("uploaded", True),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _append_line(self.publish_journal_path, entry)
        self.pushed[bucket] = entry
        self._unstage(staged, seg_path)
        if self.gate is not None:
            self.gate.mark_pushed(bucket)

    def _push_new_committed(
        self, retry_failed: bool = False, safe: bool = False
    ) -> None:
        committed = self._committed_safe() if safe else self._committed()
        for bucket, entry in sorted(committed.items()):
            if bucket in self.pushed:
                continue
            if bucket in self._push_failed and not retry_failed:
                continue
            try:
                self._push_one(bucket, entry)
            except Exception as exc:  # noqa: BLE001 — record, keep building
                self._push_failed.add(bucket)
                self.push_errors.append(f"bucket {bucket}: {exc!r}")
                if self.gate is not None:
                    self.gate.fail()

    def _reconcile(self) -> None:
        """Restore on-disk state so the builder and the push loop agree.

        * committed+pushed but no local file -> tombstone (cross-session
          resume where the previous run's output only carried journals);
        * committed+pushed with a real file left behind -> evict it now;
        * committed+unpushed -> push before building further.
        """
        committed = self._committed()
        if self.segments_dir.is_dir():
            for leftover in self.segments_dir.glob(f"*{TOMBSTONE_SUFFIX}"):
                leftover.unlink()  # crash between tombstone create + replace
        for bucket in committed:
            seg_name = SEGMENT_TEMPLATE.format(bucket=bucket)
            seg_path = self.segments_dir / seg_name
            staged = self.staging_dir / seg_name
            if bucket in self.pushed:
                # Real file left by a crash, an existing tombstone, or a
                # missing file (cross-session resume where the prior run's
                # output carried only journals): all reduce to "ensure a
                # same-size tombstone stands in for the pushed segment".
                if self.evict or not seg_path.exists():
                    self._tombstone(seg_path)
                if staged.exists():
                    staged.unlink()
        # push committed-but-unpushed segments before building more
        self._push_new_committed(retry_failed=True)

    # -- index publish ------------------------------------------------------

    def _publish_index(self) -> dict[str, Any] | None:
        if self.publisher is None:
            return None
        idx = self.build_dir / INDEX_STAGING_DIRNAME
        if idx.exists():
            shutil.rmtree(idx)
        idx.mkdir(parents=True)
        names = [
            "metadata.json", INTEGRITY_FILE, JOURNAL_FILE,
            PUBLISH_JOURNAL, "build_report.json",
            "p3_manifest.json", "p3_records.jsonl", JOB_REPORT,
        ]
        copied = []
        for name in names:
            src = self.build_dir / name
            if src.is_file():
                shutil.copyfile(src, idx / name)
                copied.append(name)
        ref = self.publisher.publish_index(idx)
        entry = {
            "index": True,
            "files": copied,
            "dataset": ref.get("dataset"),
            "ref": ref.get("ref"),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _append_line(self.publish_journal_path, entry)
        return ref

    # -- main entry ---------------------------------------------------------

    def seed_from(self, seed_dir: Path | str) -> None:
        """Copy journal/integrity/metadata files from a previous run's
        saved output so this run resumes instead of restarting (Kaggle:
        attach the prior kernel output as an input dataset, pass its dir).
        """
        seed_dir = Path(seed_dir)
        for name in (JOURNAL_FILE, INTEGRITY_FILE, PUBLISH_JOURNAL,
                     "metadata.json", "build_report.json",
                     "p3_manifest.json", "p3_records.jsonl"):
            src = seed_dir / name
            dst = self.build_dir / name
            if src.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
        # An in-flight bucket's partial is worth restoring too (bounded to
        # one segment size; saves re-fetching already-committed records).
        for sub in ("segments", Path(STAGING_DIRNAME).name):
            src_dir = seed_dir / sub
            if src_dir.is_dir():
                for f in src_dir.iterdir():
                    dst = self.build_dir / sub / f.name
                    if f.is_file() and not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(f, dst)

    def run(self) -> dict[str, Any]:
        t0 = time.monotonic()
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        write_universe_files(self.manifest, self.records, self.build_dir)
        report: dict[str, Any] = {
            "universe_sha256": self.manifest["universe_sha256"],
            "n_buckets": self.n_buckets,
            "record_bytes": self.record_bytes,
            "segment_bytes": self.segment_bytes,
            "build_dir": str(self.build_dir),
            "publisher": type(self.publisher).__name__
            if self.publisher else None,
        }
        if self.publisher is not None:
            if not _same_fs(self.build_dir, self.staging_dir):
                raise PublishError(
                    f"staging dir {self.staging_dir} is not on the same "
                    f"filesystem as {self.build_dir}; staged segments are "
                    "hardlinked/moved, never copied")
            self._load_pushed()
            self._reconcile()
            shard_to_bucket = {
                rec["shard"]: int(rec["bucket"]) for rec in self.records}
            if self.interleave:
                self.gate = PushGate(
                    self.source, shard_to_bucket, set(self.pushed))

        build_error: list[BaseException] = []
        build_report: list[dict[str, Any]] = []
        src = self.gate if self.gate is not None else self.source

        def _build() -> None:
            try:
                build_report.append(p3_builder.build_segmented(
                    self.manifest, self.records, src, self.build_dir))
            except BaseException as exc:  # noqa: BLE001
                build_error.append(exc)

        if self.publisher is not None and self.interleave:
            thread = threading.Thread(
                target=_build, name="p3-build", daemon=True)
            thread.start()
            while thread.is_alive():
                self._push_new_committed(safe=True)
                time.sleep(self.poll_seconds)
            thread.join()
        else:
            _build()

        # Push whatever completed (even on build failure — finished
        # segments are valid artifacts worth shipping).
        self._push_new_committed(retry_failed=True)

        if build_error:
            report["success"] = False
            report["error"] = repr(build_error[0])
        else:
            report["build"] = build_report[0]
            report["success"] = build_report[0].get("success", False)
        # A build that succeeded but left segments unpublished is not a
        # completed job.
        if report["success"] and self.publisher is not None:
            missing = sorted(set(self._committed()) - set(self.pushed))
            if missing:
                report["success"] = False
                report["error"] = (
                    f"{len(missing)} committed segments not pushed: "
                    f"{missing[:8]}")
        report["pushed_buckets"] = sorted(self.pushed)
        report["push_errors"] = self.push_errors
        report["wall_seconds"] = round(time.monotonic() - t0, 3)

        if report["success"] and self.publisher is not None:
            try:
                report["index"] = self._publish_index()
            except Exception as exc:  # noqa: BLE001
                report["success"] = False
                report["error"] = f"index publish failed: {exc!r}"
        _write_json_atomic(self.build_dir / JOB_REPORT, report)
        return report


def run_job(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    source: RangeSource,
    build_dir: Path | str,
    **kwargs: Any,
) -> dict[str, Any]:
    return P3KaggleJob(
        manifest, records, source, build_dir, **kwargs).run()


def preflight(
    manifest: dict[str, Any],
    publisher: SegmentPublisher,
    work_dir: Path | str,
) -> dict[str, Any]:
    """Create the index dataset BEFORE any build time is spent.

    Validates credentials, quota, and slug availability up front: if the
    dataset push is going to be refused, it fails here in seconds rather
    than after a multi-hour build.  Ships a tiny probe payload only; the
    real index files land in the post-build ``publish_index`` call (a new
    version of the same dataset).
    """
    work_dir = Path(work_dir)
    staging = work_dir / "_preflight_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _write_json_atomic(staging / "preflight.json", {
        "purpose": "p3 full-store index dataset preflight",
        "universe_sha256": manifest["universe_sha256"],
        "source_repository": manifest.get("model", PINNED_REPOSITORY),
        "source_revision": manifest.get("revision", PINNED_REVISION),
        "n_buckets": manifest.get("n_buckets", manifest.get("n_layers")),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    ref = publisher.publish_index(staging)
    return {"success": True, "index": ref}


def write_universe_files(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    build_dir: Path | str,
) -> None:
    """Drop p3_manifest.json + p3_records.jsonl next to the build so the
    index dataset is self-contained (no-ops if already present)."""
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    mpath = build_dir / "p3_manifest.json"
    if not mpath.is_file():
        _write_json_atomic(mpath, manifest)
    rpath = build_dir / "p3_records.jsonl"
    if not rpath.is_file():
        with rpath.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in records:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# dee4-v4-segmented -> dee4-v2 concatenation (+ verify)
# ---------------------------------------------------------------------------


def _segment_path(
    store_dir: Path,
    seg: dict[str, Any],
    dataset_root: Path | None = None,
) -> Path:
    """Resolve a segment file in any supported layout: the builder's
    ``segments/`` subdir, a flat download dir, or the real consumer layout
    where each segment is its own mounted dataset (``dataset_root/<slug>/
    experts-bucket-NN.dee4``)."""
    primary = store_dir / seg["file"]
    if primary.is_file():
        return primary
    flat = store_dir / Path(seg["file"]).name
    if flat.is_file():
        return flat
    if dataset_root is not None:
        name = Path(seg["file"]).name
        for cand in sorted(Path(dataset_root).glob(f"*/{name}")):
            return cand
    raise FileNotFoundError(f"segment file not found: {seg['file']}")


def _manifest_from_segmented_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Adapter so p3_builder._store_metadata can emit the v2 metadata."""
    return {
        "n_buckets": int(meta["num_layers"]),
        "n_layers": int(meta["num_layers"]),
        "experts_per_layer": int(meta["experts_per_layer"]),
        "total_experts": int(meta["total_experts"]),
        "record_bytes": int(meta["record_bytes"]),
        "model": meta["source_repository"],
        "revision": meta["source_revision"],
        "universe_sha256": meta["universe_sha256"],
        "manifest_sha256": meta.get("manifest_sha256", "0" * 64),
        "components": [
            {
                "component": f"{c['projection']}.{c['kind']}",
                "dtype": c["dtype"],
                "shape": c["shape"],
                "record_offset": c["offset"],
                "nbytes": c["nbytes"],
            }
            for c in meta["components"]
        ],
    }


def concat_v2(
    store_dir: Path | str,
    out_dir: Path | str,
    *,
    verify: bool = True,
    dataset_root: Path | str | None = None,
) -> dict[str, Any]:
    """Concatenate a dee4-v4-segmented store into a single dee4-v2 file.

    Byte-exact by construction (a segment is the v2 file restricted to its
    256-record range, and the table is contiguous): streams each segment
    through a running whole-file sha256 while re-verifying each segment
    hash against the published table, then os.replace-publishes
    ``experts.dee4`` plus a dee4-v2 metadata.json that the shipped
    ``dee::Dee4ExpertStore`` reads unchanged.  ``verify=True`` then runs
    the full per-record p3_builder.verify_store pass.
    """
    store_dir = Path(store_dir)
    out_dir = Path(out_dir)
    dataset_root = Path(dataset_root) if dataset_root is not None else None
    meta = json.loads((store_dir / "metadata.json").read_text("utf-8"))
    if meta["format"] != "dee4-v4-segmented":
        raise ValueError(f"concat expects dee4-v4-segmented, got "
                         f"{meta['format']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    record_bytes = int(meta["record_bytes"])
    total = int(meta["total_experts"])

    partial = out_dir / "experts.dee4.partial"
    data_hash = hashlib.sha256()
    written = 0
    t0 = time.monotonic()
    with partial.open("wb", buffering=8 << 20) as out:
        for seg in meta["segments"]:
            seg_path = _segment_path(store_dir, seg, dataset_root)
            seg_hash = hashlib.sha256()
            with seg_path.open("rb") as fh:
                while True:
                    chunk = fh.read(64 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    data_hash.update(chunk)
                    seg_hash.update(chunk)
                    written += len(chunk)
            if seg_hash.hexdigest() != seg["sha256"]:
                raise ValueError(
                    f"segment {seg['file']}: sha256 mismatch vs table")
        out.flush()
        os.fsync(out.fileno())
    expected = total * record_bytes
    if written != expected:
        raise RuntimeError(f"concat wrote {written} != {expected}")
    os.replace(partial, out_dir / "experts.dee4")

    integrity_src = store_dir / INTEGRITY_FILE
    if integrity_src.is_file():
        shutil.copyfile(integrity_src, out_dir / INTEGRITY_FILE)

    v2 = p3_builder._store_metadata(
        _manifest_from_segmented_metadata(meta), "dee4-v2")
    v2["data_file"] = "experts.dee4"
    v2["integrity_file"] = INTEGRITY_FILE
    v2["data_sha256"] = data_hash.hexdigest()
    v2["total_bytes"] = written
    v2["assembled_from"] = {
        "format": "dee4-v4-segmented",
        "segments": len(meta["segments"]),
        "segment_sha256": [s["sha256"] for s in meta["segments"]],
    }
    _write_json_atomic(out_dir / "metadata.json", v2)

    report = {
        "success": True,
        "format": "dee4-v2",
        "data_sha256": data_hash.hexdigest(),
        "total_bytes": written,
        "segments": len(meta["segments"]),
        "wall_seconds": round(time.monotonic() - t0, 3),
    }
    if verify:
        report["verify"] = p3_builder.verify_store(out_dir)
    _write_json_atomic(out_dir / "concat_report.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, default=str))


def _make_publisher(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> SegmentPublisher | None:
    if args.publisher == "kaggle":
        if not args.dataset_owner:
            raise ValueError("--publisher kaggle requires --dataset-owner")
        return KagglePublisher(
            args.dataset_owner, args.dataset_prefix,
            manifest["universe_sha256"],
            public=args.public, license_name=args.dataset_license)
    if args.publisher == "local":
        if not args.published_root:
            raise ValueError("--publisher local requires --published-root")
        return LocalDirPublisher(
            args.published_root, args.dataset_prefix,
            manifest["universe_sha256"])
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build segments + push datasets")
    b.add_argument("--build-dir", required=True, type=Path)
    b.add_argument("--staging-dir", type=Path, default=None)
    b.add_argument("--manifest", type=Path, default=None)
    b.add_argument("--records", type=Path, default=None)
    b.add_argument("--headers", type=Path, default=None,
                   help="committed shard-headers dir (rebuilds manifest)")
    b.add_argument("--buckets", type=int, default=None,
                   help="prefix-scope build: first N buckets only")
    b.add_argument("--source", choices=["remote", "local", "synthetic"],
                   default="remote")
    b.add_argument("--shards", type=Path, default=None)
    b.add_argument("--prefetch", type=int, default=0,
                   help="remote prefetch workers (0 = serial proven path)")
    b.add_argument("--prefetch-window", type=int, default=None)
    b.add_argument("--publisher",
                   choices=["none", "local", "kaggle"], default="none")
    b.add_argument("--published-root", type=Path, default=None,
                   help="local publisher output root")
    b.add_argument("--dataset-owner", default=None)
    b.add_argument("--dataset-prefix", default="dee4-p3-full")
    b.add_argument("--dataset-license", default="deepseek")
    b.add_argument("--public", action="store_true", default=True)
    b.add_argument("--private", dest="public", action="store_false")
    b.add_argument("--keep-segments", dest="evict", action="store_false",
                   default=True, help="keep real segments after push")
    b.add_argument("--no-verify-before-push", dest="verify_before_push",
                   action="store_false", default=True)
    b.add_argument("--no-interleave", dest="interleave",
                   action="store_false", default=True)
    b.add_argument("--seed-dir", type=Path, default=None,
                   help="previous run's saved files for cross-session resume")
    b.add_argument("--poll", type=float, default=0.5)

    c = sub.add_parser("concat", help="v4-segmented -> v2 single file")
    c.add_argument("--store", required=True, type=Path,
                   help="dir holding metadata.json + segments (e.g. the "
                        "index dataset dir, or a built store dir)")
    c.add_argument("--dataset-root", type=Path, default=None,
                   help="dir containing per-segment dataset dirs "
                        "(e.g. /kaggle/input) when segments are not colocated")
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--no-verify", dest="verify", action="store_false",
                   default=True)

    s = sub.add_parser("status", help="summarize build/publish journals")
    s.add_argument("--build-dir", required=True, type=Path)

    pf = sub.add_parser(
        "preflight", help="create the index dataset first (fail-fast auth)")
    pf.add_argument("--work-dir", required=True, type=Path)
    pf.add_argument("--manifest", type=Path, default=None)
    pf.add_argument("--records", type=Path, default=None)
    pf.add_argument("--headers", type=Path, default=None)
    pf.add_argument("--buckets", type=int, default=None)
    pf.add_argument("--publisher", choices=["local", "kaggle"],
                    default="kaggle")
    pf.add_argument("--published-root", type=Path, default=None)
    pf.add_argument("--dataset-owner", default=None)
    pf.add_argument("--dataset-prefix", default="dee4-p3-full")
    pf.add_argument("--dataset-license", default="deepseek")
    pf.add_argument("--public", action="store_true", default=True)
    pf.add_argument("--private", dest="public", action="store_false")

    args = ap.parse_args(argv)

    if args.cmd == "status":
        journal = _read_lines(args.build_dir / JOURNAL_FILE)
        publish = _read_lines(args.build_dir / PUBLISH_JOURNAL)
        _print({
            "committed_buckets": sorted(
                int(e["bucket"]) for e in journal if "bucket" in e),
            "pushed_buckets": sorted(
                int(e["bucket"]) for e in publish if "bucket" in e),
            "index_published": any(e.get("index") for e in publish),
            "integrity_lines": sum(
                1 for e in _read_lines(args.build_dir / INTEGRITY_FILE)),
        })
        return 0

    if args.cmd == "concat":
        return 0 if concat_v2(
            args.store, args.out, verify=args.verify,
            dataset_root=args.dataset_root)["success"] else 1

    if args.cmd == "preflight":
        manifest, _records = load_universe(
            args.manifest, args.records, args.headers, args.buckets)
        publisher = _make_publisher(args, manifest)
        if publisher is None:
            ap.error("preflight requires --publisher local|kaggle")
        report = preflight(manifest, publisher, args.work_dir)
        _print(report)
        return 0

    # build
    manifest, records = load_universe(
        args.manifest, args.records, args.headers, args.buckets)
    write_universe_files(manifest, records, args.build_dir)
    source = make_source(
        args.source, manifest, records,
        shards_dir=args.shards,
        prefetch_workers=args.prefetch,
        prefetch_window=args.prefetch_window)
    publisher = _make_publisher(args, manifest)

    job = P3KaggleJob(
        manifest, records, source, args.build_dir,
        publisher=publisher, staging_dir=args.staging_dir,
        evict_after_push=args.evict,
        verify_before_push=args.verify_before_push,
        interleave=args.interleave, poll_seconds=args.poll)
    if args.seed_dir:
        job.seed_from(args.seed_dir)
    report = job.run()
    _print(report)
    if isinstance(source, PrefetchRangeSource):
        source.close()
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
