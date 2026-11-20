"""Phase 3: demand-paged full-universe expert store ("dee4-v5-lazy").

A ``dee4-v2``-shaped data file is pre-allocated to the full
147,169,738,752 bytes (sparse where the filesystem supports it; zero-filled
semantics otherwise) and records are materialized ON DEMAND: the first read
of a (layer, expert) fetches the six source ranges, writes them at the
fixed-stride record offset, fsyncs, appends the integrity line, and flips a
bit in ``lazy.state``.

Why this shape:
  * Convergence — every materialized byte is at its FINAL offset, so when
    all 11,008 records are present the file is byte-identical to a prebuilt
    dee4-v2 bank and carries the same ``data_sha256``.  Lazy and batch
    builds are the same artifact at different fill levels.
  * Bounded disk — only routed pairs ever materialized consume space (a
    sparse backing keeps untouched regions free); the residency bitmap is
    1,376 B for the whole universe.
  * Honest fail-closed — ``read`` for an unmaterialized pair either fills it
    synchronously from the configured source or raises; nothing is ever
    served from uninitialized (zero) bytes because the bitmap gates every
    read, not the file contents.

Recovery: a crash between data-write and bitmap-set loses at most the
in-flight record; the next read simply re-materializes it.  ``audit()``
re-hashes every materialized record against its integrity line so a torn
record that was committed is detected and re-fetched.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from p3_builder import (
    INTEGRITY_FILE,
    JOURNAL_FILE,
    RangeSource,
    _append_line,
    _integrity_line,
    _read_lines,
    _write_json_atomic,
    assemble_record,
)

DATA_FILE = "experts.dee4"
STATE_FILE = "lazy.state.json"
BITMAP_FILE = "lazy.bitmap"


class LazyFullStore:
    """Demand-paged dee4-v2 file + residency bitmap + integrity journal."""

    def __init__(self, store_dir: Path | str, manifest: dict[str, Any]):
        self.store_dir = Path(store_dir)
        self.manifest = manifest
        self.total = int(manifest["total_experts"])
        self.record_bytes = int(manifest["record_bytes"])
        self.data_path = self.store_dir / DATA_FILE
        self.state_path = self.store_dir / STATE_FILE
        self.bitmap_path = self.store_dir / BITMAP_FILE

    # -- lifecycle --------------------------------------------------------

    def create(self) -> None:
        """Allocate the full-size data file + empty bitmap (idempotent).

        On Windows the data file is created through ``fsutil sparse`` so the
        ~147 GiB logical size costs nothing until records land; plain
        truncate() allocates all clusters on this NTFS volume.
        """
        self.store_dir.mkdir(parents=True, exist_ok=True)
        if not self.data_path.is_file():
            size = self.total * self.record_bytes
            sparse_ok = False
            if sys.platform == "win32":
                from p3_sparse_store import _make_sparse
                sparse_ok = _make_sparse(self.data_path, size)
            if not sparse_ok:
                with self.data_path.open("wb") as fh:
                    fh.truncate(size)
        if not self.bitmap_path.is_file():
            self.bitmap_path.write_bytes(b"\x00" * ((self.total + 7) // 8))
        if not self.state_path.is_file():
            _write_json_atomic(self.state_path, {
                "format": "dee4-v5-lazy",
                "codec": "deepseek-fp4-e2m1-e8m0",
                "source_repository": self.manifest["model"],
                "source_revision": self.manifest["revision"],
                "universe_sha256": self.manifest["universe_sha256"],
                "manifest_sha256": self.manifest["manifest_sha256"],
                "total_experts": self.total,
                "record_bytes": self.record_bytes,
                "materialized": 0,
            })

    # -- bitmap ------------------------------------------------------------

    def _bitmap(self) -> bytearray:
        return bytearray(self.bitmap_path.read_bytes())

    def _is_materialized(self, index: int, bitmap: bytearray | None = None) -> bool:
        bm = bitmap if bitmap is not None else self._bitmap()
        return bool(bm[index >> 3] & (1 << (index & 7)))

    def _set_materialized(self, index: int) -> None:
        bm = self._bitmap()
        bm[index >> 3] |= 1 << (index & 7)
        tmp = self.bitmap_path.with_suffix(".tmp")
        tmp.write_bytes(bytes(bm))
        os.replace(tmp, self.bitmap_path)

    # -- read path ----------------------------------------------------------

    def read(
        self,
        layer: int,
        expert: int,
        source: RangeSource,
        records: list[dict[str, Any]],
        *,
        fill: bool = True,
    ) -> bytes:
        """Return the exact 13,369,344-byte record for (layer, expert).

        Miss + fill=True materializes synchronously from ``source``; the
        integrity line is committed BEFORE the bitmap bit so a torn record
        is always re-fetched (bitmap is the last word).
        """
        index = layer * int(self.manifest["experts_per_layer"]) + expert
        if index < 0 or index >= self.total:
            raise IndexError(f"({layer},{expert}) outside the universe")
        if not self._is_materialized(index):
            if not fill:
                raise KeyError(f"record ({layer},{expert}) not materialized")
            self._materialize(index, source, records[index])
        with self.data_path.open("rb") as fh:
            fh.seek(index * self.record_bytes)
            blob = fh.read(self.record_bytes)
        if len(blob) != self.record_bytes:
            raise IOError(f"short record ({layer},{expert})")
        return blob

    def _materialize(
        self, index: int, source: RangeSource, record: dict[str, Any]
    ) -> None:
        blob, comp_sha, srcs = assemble_record(source, record)
        record_sha = hashlib.sha256(blob).hexdigest()
        with self.data_path.open("r+b") as fh:
            fh.seek(index * self.record_bytes)
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        _append_line(self.store_dir / INTEGRITY_FILE, _integrity_line(
            record["bucket"], record["expert"], index,
            index * self.record_bytes, self.record_bytes,
            record_sha, comp_sha, srcs))
        self._set_materialized(index)
        _append_line(self.store_dir / JOURNAL_FILE, {
            "materialized": index,
            "bucket": record["bucket"],
            "domain": record.get("domain", "main"),
            "expert": record["expert"],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    # -- audit / finalize ---------------------------------------------------

    def materialized_count(self) -> int:
        return sum(bin(b).count("1") for b in self._bitmap())

    def audit(self, source: RangeSource | None = None,
              records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Re-hash every materialized record against integrity.jsonl.

        Optionally repairs mismatches via ``source`` when provided.
        """
        integrity = _read_lines(self.store_dir / INTEGRITY_FILE)
        by_index = {int(e["record_index"]): e for e in integrity}
        bm = self._bitmap()
        checked = repaired = 0
        bad: list[int] = []
        with self.data_path.open("rb") as fh:
            for index in range(self.total):
                if not (bm[index >> 3] & (1 << (index & 7))):
                    continue
                entry = by_index.get(index)
                fh.seek(index * self.record_bytes)
                blob = fh.read(self.record_bytes)
                sha = hashlib.sha256(blob).hexdigest()
                if entry is None or entry["record_sha256"] != sha:
                    bad.append(index)
                    if source is not None and records is not None:
                        self._materialize(index, source, records[index])
                        repaired += 1
                checked += 1
        return {
            "materialized": self.materialized_count(),
            "checked": checked,
            "corrupt": bad,
            "repaired": repaired,
            "complete": self.materialized_count() == self.total,
        }

    def finalize(self) -> dict[str, Any]:
        """Seal a fully materialized store as plain dee4-v2.

        Computes the whole-file data_sha256 and writes a v2 metadata.json so
        the existing C++ reader opens it unchanged.
        """
        if self.materialized_count() != self.total:
            raise RuntimeError(
                f"cannot finalize: {self.materialized_count()}/{self.total} "
                "records materialized")
        data_hash = hashlib.sha256()
        with self.data_path.open("rb") as fh:
            while True:
                chunk = fh.read(64 << 20)
                if not chunk:
                    break
                data_hash.update(chunk)
        from p3_builder import _store_metadata  # late import: same module set
        metadata = _store_metadata(self.manifest, "dee4-v2")
        metadata["data_file"] = DATA_FILE
        metadata["integrity_file"] = INTEGRITY_FILE
        metadata["data_sha256"] = data_hash.hexdigest()
        metadata["total_bytes"] = self.total * self.record_bytes
        _write_json_atomic(self.store_dir / "metadata.json", metadata)
        return {"data_sha256": data_hash.hexdigest(), "finalized": True}
