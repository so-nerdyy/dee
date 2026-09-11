"""Phase 3: resumable builder for the full-universe DEE4 expert store.

"Bucket" = one expert-bearing module: main-model layer L (buckets 0..42)
or MTP draft layer mtp.N (buckets 43..45).  The dee4 fixed-stride index is
``record_index = bucket * experts_per_layer + expert``.

Two output shapes:

* ``single``    -> one ``experts.dee4`` data file + ``metadata.json`` with
  ``format = "dee4-v2"`` and ``num_layers = n_buckets``.  Readable TODAY by
  ``dee::Dee4ExpertStore`` with zero code changes: dense arithmetic lookup
  covers the full universe (a 43-bucket bank answers every main-model
  (layer, expert); a 46-bucket bank additionally answers mtp draft layers,
  which the runtime may address as layer indices 43..45).  Resume
  granularity = one record (fixed stride -> the partial file is truncated
  to the last committed record boundary on restart).

* ``segmented`` -> one file per bucket ``segments/experts-bucket-LL.dee4``
  (256 records each, 3,422,552,064 B) + ``metadata.json`` with
  ``format = "dee4-v4-segmented"``.  This is the distribution-friendly
  shape: per-bucket atomic publish (partial -> fsync -> rename -> journal
  commit), Kaggle-dataset-sized files, and per-bucket independent repair.
  Needs a small reader extension (segmented mmap), NOT needed for single.

Sources are byte ranges inside the ORIGINAL safetensors shards, addressed by
(shard, data_offset, nbytes) from the Phase-3 manifest.  ``LocalShardSource``
reads local shard files; ``RemoteRangeSource`` fetches HTTP ranges from the
pinned HF revision (absolute offset = 8 + live header length + data_offset),
which is exactly how a Kaggle/Modal builder with a 20 GiB disk produces the
bank without ever holding the checkpoint.

Failure recovery contract (both modes):
  - bytes are only ever appended to ``*.partial`` files;
  - a record is committed iff its ``integrity.jsonl`` line is fsync'd;
  - on resume, a partially written tail beyond the last committed record is
    truncated (fixed stride makes truncation exact);
  - the last committed record's sha256 is re-verified before resuming, so a
    torn final write can never silently persist;
  - a segment is published by atomic rename only after all 256 records are
    committed and the segment sha256 is known.

Integrity identity conventions mirror the existing bank: per-record and
per-component sha256 in ``integrity.jsonl``, whole-store ``data_sha256`` in
``metadata.json`` (single mode) or per-segment ``sha256`` (segmented mode),
plus ``universe_sha256`` binding the store to the full-universe pair list.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import urllib.request
from pathlib import Path
from typing import Any, Protocol

RECORD_BYTES = 13_369_344
EXPERTS_PER_LAYER = 256

DATA_FILE = "experts.dee4"
SEGMENT_TEMPLATE = "experts-bucket-{bucket:02d}.dee4"
INTEGRITY_FILE = "integrity.jsonl"
JOURNAL_FILE = "build.journal.jsonl"

USER_AGENT = "dee-p3-full-store/1.0"


# ---------------------------------------------------------------------------
# Byte-range sources
# ---------------------------------------------------------------------------


class RangeSource(Protocol):
    """Returns ``nbytes`` of shard data-section bytes at ``data_offset``."""

    def fetch(self, shard: str, data_offset: int, nbytes: int) -> bytes:
        ...


class LocalShardSource:
    """Reads ranges from local safetensors shard files.

    Header length is read once per shard (8 bytes + header).
    """

    def __init__(self, shards_dir: Path | str):
        self.shards_dir = Path(shards_dir)
        self._header_len: dict[str, int] = {}

    def _hlen(self, shard: str) -> int:
        if shard not in self._header_len:
            with (self.shards_dir / shard).open("rb") as fh:
                raw = fh.read(8)
            if len(raw) != 8:
                raise IOError(f"short safetensors prefix: {shard}")
            self._header_len[shard] = struct.unpack("<Q", raw)[0]
        return self._header_len[shard]

    def fetch(self, shard: str, data_offset: int, nbytes: int) -> bytes:
        absolute = 8 + self._hlen(shard) + data_offset
        with (self.shards_dir / shard).open("rb") as fh:
            fh.seek(absolute)
            data = fh.read(nbytes)
        if len(data) != nbytes:
            raise IOError(
                f"short read {shard}@{data_offset}: {len(data)} != {nbytes}"
            )
        return data


class RemoteRangeSource:
    """HTTP Range fetches against the pinned HF revision.

    Reuses the exact convention of scripts/deepseek_v4_model.py
    RemoteTensorSource: absolute = 8 + header_length + data_offset, with the
    header length discovered by one 8-byte probe per shard.
    """

    def __init__(
        self,
        *,
        repository: str = "deepseek-ai/DeepSeek-V4-Flash-0731",
        revision: str = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
        max_attempts: int = 6,
    ):
        self.repository = repository
        self.revision = revision
        self.max_attempts = max_attempts
        self._header_len: dict[str, int] = {}
        self.stats = {"requests": 0, "bytes": 0, "retries": 0}

    def _url(self, shard: str) -> str:
        return (
            f"https://huggingface.co/{self.repository}/resolve/"
            f"{self.revision}/{shard}"
        )

    def _range(self, url: str, start: int, end: int) -> bytes:
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                req = urllib.request.Request(
                    url, headers={
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": USER_AGENT,
                    })
                with urllib.request.urlopen(req, timeout=300) as resp:
                    if resp.status != 206:
                        raise RuntimeError(
                            f"server did not honor Range (status {resp.status})")
                    data = resp.read()
                self.stats["requests"] += 1
                self.stats["bytes"] += len(data)
                return data
            except Exception as exc:  # noqa: BLE001
                last = exc
                self.stats["retries"] += 1
                time.sleep(1.5 * (2 ** attempt))
        raise ConnectionError(f"range fetch failed: {last!r}")

    def _hlen(self, shard: str) -> int:
        if shard not in self._header_len:
            raw = self._range(self._url(shard), 0, 7)
            if len(raw) != 8:
                raise RuntimeError(f"{shard}: bad prefix")
            hlen = int.from_bytes(raw, "little")
            if hlen <= 0 or hlen > (1 << 31):
                raise RuntimeError(f"{shard}: implausible header length {hlen}")
            self._header_len[shard] = hlen
        return self._header_len[shard]

    def fetch(self, shard: str, data_offset: int, nbytes: int) -> bytes:
        absolute = 8 + self._hlen(shard) + data_offset
        return self._range(self._url(shard), absolute, absolute + nbytes - 1)


# ---------------------------------------------------------------------------
# Journal / integrity helpers
# ---------------------------------------------------------------------------


def _append_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _truncate_lines(path: Path, keep: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(lines[:keep])
        fh.flush()
        os.fsync(fh.fileno())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via tmp-file + atomic replace (publish-once semantics)."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _integrity_line(
    bucket: int,
    expert: int,
    record_index: int,
    record_offset: int,
    record_bytes: int,
    record_sha256: str,
    component_sha256: dict[str, str],
    source_shards: list[str],
) -> dict[str, Any]:
    return {
        "bucket": bucket,
        # "layer" kept for tooling parity with the existing bank's
        # integrity.jsonl; for mtp buckets it is the mtp index (0..2).
        "layer": bucket if bucket < 43 else bucket - 43,
        "domain": "main" if bucket < 43 else "mtp",
        "expert": expert,
        "record_index": record_index,
        "record_offset": record_offset,
        "record_bytes": record_bytes,
        "record_sha256": record_sha256,
        "component_sha256": component_sha256,
        "source_shards": sorted(source_shards),
    }


def assemble_record(
    source: RangeSource, record: dict[str, Any]
) -> tuple[bytes, dict[str, str], list[str]]:
    """Fetch + concatenate the six components of one record.

    Returns (record_bytes_blob, component_sha256, source_shards).  The blob
    order is exactly the manifest's ranges order (DEE4 component order).
    """
    chunks: list[bytes] = []
    component_sha256: dict[str, str] = {}
    source_shards: set[str] = set()
    for component, data_offset, nbytes, _record_offset, tensor in \
            record["ranges"]:
        data = source.fetch(record["shard"], int(data_offset), int(nbytes))
        if len(data) != int(nbytes):
            raise IOError(f"{tensor}: {len(data)} != {nbytes}")
        chunks.append(data)
        component_sha256[component] = hashlib.sha256(data).hexdigest()
        source_shards.add(record["shard"])
    blob = b"".join(chunks)
    if len(blob) != int(record["record_bytes"]):
        raise RuntimeError(
            f"record (bucket {record['bucket']}, expert {record['expert']}) "
            f"assembled {len(blob)} bytes != {record['record_bytes']}"
        )
    return blob, component_sha256, sorted(source_shards)


# ---------------------------------------------------------------------------
# Segmented builder (dee4-v4-segmented)
# ---------------------------------------------------------------------------


def build_segmented(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    source: RangeSource,
    out_dir: Path | str,
    *,
    verify_tail: bool = True,
) -> dict[str, Any]:
    """Build the full store as one segment file per bucket.

    Resumable: ``build.journal.jsonl`` commits one line per FINISHED bucket;
    an interrupted bucket resumes inside its ``*.partial`` segment at the
    last whole-record boundary after re-verifying the last committed record.
    """
    out_dir = Path(out_dir)
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / JOURNAL_FILE
    integrity_path = out_dir / INTEGRITY_FILE
    record_bytes = int(manifest["record_bytes"])
    experts_per_layer = int(manifest["experts_per_layer"])
    n_buckets = int(manifest.get("n_buckets", manifest["n_layers"]))

    journal = _read_lines(journal_path)
    committed_buckets = {int(e["bucket"]) for e in journal}
    integrity_lines = _read_lines(integrity_path)
    by_pair = {r["record_index"]: r for r in records}

    # Reconcile integrity.jsonl with the journal: drop lines beyond the last
    # committed record so a torn tail can never be trusted.  Buckets are
    # built strictly in order, so committed buckets form a contiguous prefix.
    committed_record_count = (
        (max(committed_buckets) + 1) * experts_per_layer
        if committed_buckets else 0)
    if len(integrity_lines) > committed_record_count:
        _truncate_lines(integrity_path, committed_record_count)
        integrity_lines = integrity_lines[:committed_record_count]

    report = {"buckets_built": [], "buckets_skipped": [], "records_written": 0,
              "bytes_written": 0, "errors": []}
    t_start = time.monotonic()

    for bucket in range(n_buckets):
        seg_name = SEGMENT_TEMPLATE.format(bucket=bucket)
        seg_path = seg_dir / seg_name
        partial_path = seg_dir / f"{seg_name}.partial"
        first_index = bucket * experts_per_layer

        if bucket in committed_buckets:
            if not seg_path.is_file():
                raise RuntimeError(
                    f"journal commits bucket {bucket} but {seg_name} missing")
            report["buckets_skipped"].append(bucket)
            continue

        # Resume inside a partial segment: whole records already written are
        # trusted iff their integrity lines exist AND the last one verifies.
        start_expert = 0
        if partial_path.is_file():
            size = partial_path.stat().st_size
            whole = size // record_bytes
            if size % record_bytes:
                with partial_path.open("r+b") as fh:
                    fh.truncate(whole * record_bytes)
                    fh.flush()
                    os.fsync(fh.fileno())
            bucket_integrity = [
                e for e in integrity_lines if int(e["bucket"]) == bucket]
            keep = min(whole, len(bucket_integrity))
            if keep < whole:
                with partial_path.open("r+b") as fh:
                    fh.truncate(keep * record_bytes)
                    fh.flush()
                    os.fsync(fh.fileno())
            if keep < len(bucket_integrity):
                # More integrity lines than bytes: drop this bucket's tail
                # lines (they are the last lines of the file because buckets
                # build strictly in order).
                drop = len(bucket_integrity) - keep
                new_total = len(integrity_lines) - drop
                _truncate_lines(integrity_path, new_total)
                integrity_lines = integrity_lines[:new_total]
            if keep and verify_tail:
                last = [
                    e for e in integrity_lines
                    if int(e["bucket"]) == bucket][-1]
                with partial_path.open("rb") as fh:
                    fh.seek((keep - 1) * record_bytes)
                    tail = fh.read(record_bytes)
                if hashlib.sha256(tail).hexdigest() != last["record_sha256"]:
                    with partial_path.open("r+b") as fh:
                        fh.truncate((keep - 1) * record_bytes)
                        fh.flush()
                        os.fsync(fh.fileno())
                    keep -= 1
                    _truncate_lines(integrity_path, len(integrity_lines) - 1)
                    integrity_lines = integrity_lines[:-1]
            start_expert = keep

        seg_hash = hashlib.sha256()
        # Re-hash the already-written prefix so the segment digest stays
        # honest across a resume (bounded: <= 3.19 GiB, local disk).
        if start_expert:
            with partial_path.open("rb") as fh:
                while True:
                    chunk = fh.read(64 << 20)
                    if not chunk:
                        break
                    seg_hash.update(chunk)

        bucket_t0 = time.monotonic()
        written = 0
        with partial_path.open("ab", buffering=8 << 20) as data_fh:
            for expert in range(start_expert, experts_per_layer):
                rec = by_pair[first_index + expert]
                blob, comp_sha, srcs = assemble_record(source, rec)
                if len(blob) != record_bytes:
                    raise RuntimeError(
                        f"record ({bucket},{expert}) {len(blob)} != "
                        f"{record_bytes}")
                record_sha = hashlib.sha256(blob).hexdigest()
                data_fh.write(blob)
                seg_hash.update(blob)
                data_fh.flush()
                os.fsync(data_fh.fileno())
                _append_line(integrity_path, _integrity_line(
                    bucket, expert, rec["record_index"],
                    rec["record_index"] * record_bytes, record_bytes,
                    record_sha, comp_sha, srcs))
                written += 1
        os.replace(partial_path, seg_path)
        _append_line(journal_path, {
            "bucket": bucket,
            "segment": seg_name,
            "records": experts_per_layer,
            "segment_sha256": seg_hash.hexdigest(),
            "resumed_from_expert": start_expert,
        })
        report["buckets_built"].append({
            "bucket": bucket,
            "records_written": written,
            "resumed_from_expert": start_expert,
            "seconds": round(time.monotonic() - bucket_t0, 3),
        })
        report["records_written"] += written
        report["bytes_written"] += written * record_bytes

    segments = []
    journal = _read_lines(journal_path)
    for bucket in range(n_buckets):
        seg_name = SEGMENT_TEMPLATE.format(bucket=bucket)
        seg_path = seg_dir / seg_name
        entry = next(e for e in journal if int(e["bucket"]) == bucket)
        segments.append({
            "file": f"segments/{seg_name}",
            "bucket": bucket,
            "domain": "main" if bucket < 43 else "mtp",
            "first_record": bucket * experts_per_layer,
            "record_count": experts_per_layer,
            "bytes": experts_per_layer * record_bytes,
            "sha256": entry["segment_sha256"],
        })
        if seg_path.stat().st_size != experts_per_layer * record_bytes:
            raise RuntimeError(f"{seg_name} size mismatch after build")

    metadata = _store_metadata(manifest, "dee4-v4-segmented")
    metadata["segments"] = segments
    metadata["data_file"] = None
    metadata["integrity_file"] = INTEGRITY_FILE
    _write_json_atomic(out_dir / "metadata.json", metadata)
    report["wall_seconds"] = round(time.monotonic() - t_start, 3)
    report["success"] = True
    _write_json_atomic(out_dir / "build_report.json", report)
    return report


# ---------------------------------------------------------------------------
# Single-file builder (dee4-v2, existing reader compatible)
# ---------------------------------------------------------------------------


def build_single(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    source: RangeSource,
    out_dir: Path | str,
    *,
    verify_tail: bool = True,
) -> dict[str, Any]:
    """Build ``experts.dee4`` as ONE fixed-stride file (format dee4-v2).

    Resume: the journal records ``committed_through`` (exclusive record
    index).  On restart the partial data file is truncated to
    ``committed_through * record_bytes`` and the tail record is re-verified,
    so a torn write is always rolled back before appending continues.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / JOURNAL_FILE
    integrity_path = out_dir / INTEGRITY_FILE
    partial_path = out_dir / f"{DATA_FILE}.partial"
    record_bytes = int(manifest["record_bytes"])
    total_experts = int(manifest["total_experts"])

    journal = _read_lines(journal_path)
    committed_through = 0
    data_hash = hashlib.sha256()
    if journal:
        committed_through = max(int(e["committed_through"]) for e in journal)

    integrity_lines = _read_lines(integrity_path)
    if len(integrity_lines) > committed_through:
        _truncate_lines(integrity_path, committed_through)
        integrity_lines = integrity_lines[:committed_through]

    if partial_path.is_file():
        size = partial_path.stat().st_size
        whole = min(size // record_bytes, committed_through)
        if whole != committed_through or size % record_bytes:
            with partial_path.open("r+b") as fh:
                fh.truncate(whole * record_bytes)
                fh.flush()
                os.fsync(fh.fileno())
            committed_through = whole
        if committed_through and verify_tail:
            last = integrity_lines[committed_through - 1]
            with partial_path.open("rb") as fh:
                fh.seek((committed_through - 1) * record_bytes)
                tail = fh.read(record_bytes)
            if hashlib.sha256(tail).hexdigest() != last["record_sha256"]:
                committed_through -= 1
                with partial_path.open("r+b") as fh:
                    fh.truncate(committed_through * record_bytes)
                    fh.flush()
                    os.fsync(fh.fileno())
                _truncate_lines(integrity_path, committed_through)
        # Re-hash the committed prefix for the whole-file digest.
        with partial_path.open("rb") as fh:
            while True:
                chunk = fh.read(64 << 20)
                if not chunk:
                    break
                data_hash.update(chunk)

    report = {"records_written": 0, "bytes_written": 0, "errors": [],
              "resumed_from_record": committed_through}
    t_start = time.monotonic()
    by_index = {r["record_index"]: r for r in records}
    journal_stride = EXPERTS_PER_LAYER  # commit once per finished bucket

    with partial_path.open("ab", buffering=8 << 20) as data_fh:
        for index in range(committed_through, total_experts):
            rec = by_index[index]
            blob, comp_sha, srcs = assemble_record(source, rec)
            if len(blob) != record_bytes:
                raise RuntimeError(
                    f"record index {index}: {len(blob)} != {record_bytes}")
            record_sha = hashlib.sha256(blob).hexdigest()
            data_fh.write(blob)
            data_hash.update(blob)
            written_through = index + 1
            data_fh.flush()
            os.fsync(data_fh.fileno())
            _append_line(integrity_path, _integrity_line(
                rec["bucket"], rec["expert"], index,
                index * record_bytes, record_bytes, record_sha,
                comp_sha, srcs))
            report["records_written"] += 1
            report["bytes_written"] += record_bytes
            if written_through % journal_stride == 0 or \
                    written_through == total_experts:
                _append_line(journal_path, {
                    "committed_through": written_through,
                    "records": written_through,
                })

    expected = total_experts * record_bytes
    if partial_path.stat().st_size != expected:
        raise RuntimeError(
            f"data size {partial_path.stat().st_size} != {expected}")
    os.replace(partial_path, out_dir / DATA_FILE)

    metadata = _store_metadata(manifest, "dee4-v2")
    metadata["data_file"] = DATA_FILE
    metadata["integrity_file"] = INTEGRITY_FILE
    metadata["data_sha256"] = data_hash.hexdigest()
    metadata["total_bytes"] = expected
    _write_json_atomic(out_dir / "metadata.json", metadata)
    report["wall_seconds"] = round(time.monotonic() - t_start, 3)
    report["data_sha256"] = data_hash.hexdigest()
    report["success"] = True
    _write_json_atomic(out_dir / "build_report.json", report)
    return report


def _store_metadata(manifest: dict[str, Any], fmt: str) -> dict[str, Any]:
    """metadata.json fields shared by both shapes.

    Carries every field the existing Dee4ExpertStore parser requires plus
    the Phase-3 universe binding (unknown fields are ignored by the C++
    json_min reader, which looks keys up by name).

    ``num_layers`` is the bucket count so the shipped arithmetic lookup
    ``record_index = layer * experts_per_layer + expert`` covers every
    bucket; for a 46-bucket store, runtime layer indices 43..45 denote the
    mtp draft layers.
    """
    n_buckets = int(manifest.get("n_buckets", manifest["n_layers"]))
    components = manifest["components"]
    weights = [c for c in components if c["component"].endswith(".weight")]
    scales = [c for c in components if c["component"].endswith(".scale")]
    return {
        "format": fmt,
        "codec": "deepseek-fp4-e2m1-e8m0",
        "source_repository": manifest["model"],
        "source_revision": manifest["revision"],
        "start_layer": 0,
        "num_layers": n_buckets,
        "experts_per_layer": manifest["experts_per_layer"],
        "total_experts": manifest["total_experts"],
        "record_bytes": manifest["record_bytes"],
        "group_size": 32,
        "components": [
            {
                "projection": c["component"].split(".")[0],
                "kind": c["component"].split(".")[1],
                "dtype": c["dtype"],
                "shape": c["shape"],
                "offset": c["record_offset"],
                "nbytes": c["nbytes"],
            }
            for c in components
        ],
        "weight_offsets": [c["record_offset"] for c in weights],
        "weight_nbytes": [c["nbytes"] for c in weights],
        "weight_out": [c["shape"][0] for c in weights],
        "weight_stored_in": [c["shape"][1] for c in weights],
        "scale_offsets": [c["record_offset"] for c in scales],
        "scale_nbytes": [c["nbytes"] for c in scales],
        "scale_out": [c["shape"][0] for c in scales],
        "scale_in": [c["shape"][1] for c in scales],
        "universe_sha256": manifest["universe_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "mtp_bucket_offset": N_LAYERS_PLACEHOLDER,
    }


N_LAYERS_PLACEHOLDER = 43  # main-model layers; mtp buckets start at 43


# ---------------------------------------------------------------------------
# Verification / repair
# ---------------------------------------------------------------------------


def verify_store(store_dir: Path | str, *, sample: int | None = None) -> dict[str, Any]:
    """Re-hash records against integrity.jsonl.

    Works on both shapes (single data file or segments dir).  ``sample``
    bounds the check to a deterministic evenly spaced subset for a quick
    audit; None verifies every record.
    """
    store_dir = Path(store_dir)
    metadata = json.loads((store_dir / "metadata.json").read_text("utf-8"))
    record_bytes = int(metadata["record_bytes"])
    integrity = _read_lines(store_dir / INTEGRITY_FILE)
    total = int(metadata["total_experts"])
    if len(integrity) != total:
        raise ValueError(
            f"integrity lines {len(integrity)} != total_experts {total}")

    if metadata["format"] == "dee4-v4-segmented":
        segments = metadata["segments"]
        handles = []
        try:
            for seg in segments:
                handles.append(
                    (int(seg["first_record"]), int(seg["record_count"]),
                     open(store_dir / seg["file"], "rb")))
            indices = _audit_indices(total, sample)
            checked = 0
            for idx in indices:
                entry = integrity[idx]
                first, count, fh = next(
                    (f, c, h) for f, c, h in handles
                    if f <= idx < f + c)
                fh.seek((idx - first) * record_bytes)
                blob = fh.read(record_bytes)
                if len(blob) != record_bytes:
                    raise IOError(f"short record {idx}")
                if hashlib.sha256(blob).hexdigest() != entry["record_sha256"]:
                    raise ValueError(f"record {idx} sha256 mismatch")
                checked += 1
        finally:
            for _f, _c, fh in handles:
                fh.close()
    elif metadata["format"] == "dee4-v2":
        with (store_dir / metadata["data_file"]).open("rb") as fh:
            indices = _audit_indices(total, sample)
            checked = 0
            for idx in indices:
                fh.seek(idx * record_bytes)
                blob = fh.read(record_bytes)
                if len(blob) != record_bytes:
                    raise IOError(f"short record {idx}")
                if hashlib.sha256(blob).hexdigest() != \
                        integrity[idx]["record_sha256"]:
                    raise ValueError(f"record {idx} sha256 mismatch")
                checked += 1
    else:
        raise ValueError(f"unsupported format {metadata['format']}")
    return {"checked_records": checked, "total_records": total,
            "format": metadata["format"], "success": True}


def _audit_indices(total: int, sample: int | None) -> list[int]:
    if sample is None or sample >= total:
        return list(range(total))
    step = total / sample
    return sorted({int(i * step) for i in range(sample)})


def repair_record(
    store_dir: Path | str,
    record_index: int,
    source: RangeSource,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rewrite ONE record in place from the authoritative source.

    Fixed-stride layout makes in-place repair exact; the integrity line is
    rewritten with the new sha256 after fsync only when the line itself was
    wrong (data corruption is repaired silently into place; a wrong
    integrity line is rewritten through the same atomic-replace pattern).
    """
    store_dir = Path(store_dir)
    metadata = json.loads((store_dir / "metadata.json").read_text("utf-8"))
    record_bytes = int(metadata["record_bytes"])
    rec = next(r for r in records if r["record_index"] == record_index)
    blob, comp_sha, srcs = assemble_record(source, rec)
    record_sha = hashlib.sha256(blob).hexdigest()

    if metadata["format"] == "dee4-v4-segmented":
        seg = next(s for s in metadata["segments"]
                   if s["first_record"] <= record_index
                   < s["first_record"] + s["record_count"])
        path = store_dir / seg["file"]
        local = record_index - int(seg["first_record"])
        with path.open("r+b") as fh:
            fh.seek(local * record_bytes)
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
    else:
        with (store_dir / metadata["data_file"]).open("r+b") as fh:
            fh.seek(record_index * record_bytes)
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())

    lines = _read_lines(store_dir / INTEGRITY_FILE)
    entry = lines[record_index]
    repaired = {
        "record_index": record_index,
        "previous_sha256": entry["record_sha256"],
        "new_sha256": record_sha,
        "source_shards": srcs,
    }
    if entry["record_sha256"] != record_sha:
        entry["record_sha256"] = record_sha
        entry["component_sha256"] = comp_sha
        tmp = (store_dir / INTEGRITY_FILE).with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for line in lines:
                fh.write(json.dumps(line, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, store_dir / INTEGRITY_FILE)
        repaired["integrity_rewritten"] = True
    else:
        repaired["integrity_rewritten"] = False
    return repaired
