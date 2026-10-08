"""Dense-only checkpoint format for the segmented-store (dee4_segmented) path.

The runner reads tensors by absolute offset using the committed shard
headers, and the C++ engine mmaps every shard at its original length, so a
dense shard cannot simply drop its expert tensors. Instead each shard is
rebuilt as a sparse file: the original 8-byte prefix + header verbatim, the
original total length, and only the needed (non-routed-expert) tensor bytes
written at their original offsets. Everything else is a hole that is never
read on the segmented path.

S3 stores only the compact blobs plus manifest.json, never the holes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

SCHEMA = "dee-p6-dense-v1"


def prefix_header_len(prefix8: bytes) -> int:
    if len(prefix8) != 8:
        raise ValueError(f"safetensors prefix must be 8 bytes, got {len(prefix8)}")
    return int.from_bytes(prefix8, "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_file_size(header_bytes: bytes, data_ends: list[int]) -> int:
    """Original shard length: prefix + header + the largest tensor end."""
    hlen = prefix_header_len(header_bytes[:8])
    if len(header_bytes) != 8 + hlen:
        raise ValueError("header bytes do not match the declared header length")
    return 8 + hlen + max(data_ends)


def shard_record(*, header_bytes: bytes, file_size: int, blob: str,
                 blob_bytes: int, blob_sha256: str,
                 entries: list[list]) -> dict:
    return {
        "file_size": file_size,
        "header_b64": base64.b64encode(header_bytes).decode("ascii"),
        "blob": blob,
        "blob_bytes": blob_bytes,
        "blob_sha256": blob_sha256,
        "entries": entries,
    }


def manifest(*, model: str, repo: str, revision: str, shards: dict[str, dict]) -> dict:
    tensor_count = sum(len(s["entries"]) for s in shards.values())
    total = sum(s["blob_bytes"] for s in shards.values())
    return {
        "schema": SCHEMA,
        "model": model,
        "repo": repo,
        "revision": revision,
        "tensor_count": tensor_count,
        "total_blob_bytes": total,
        "shards": dict(sorted(shards.items())),
    }


def materialize(manifest_path: Path, blob_dir: Path, out_dir: Path) -> list[Path]:
    """Rebuild sparse dense-only shards under out_dir from compact blobs."""
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"unexpected dense manifest schema: {doc.get('schema')!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for shard, rec in doc["shards"].items():
        blob = blob_dir / rec["blob"]
        if not blob.is_file() or blob.stat().st_size != rec["blob_bytes"]:
            raise ValueError(f"{shard}: blob missing or wrong size: {blob}")
        if sha256_file(blob) != rec["blob_sha256"]:
            raise ValueError(f"{shard}: blob sha256 mismatch: {blob}")
        header = base64.b64decode(rec["header_b64"])
        if len(header) != 8 + prefix_header_len(header[:8]):
            raise ValueError(f"{shard}: header length mismatch")
        file_size = int(rec["file_size"])
        header_end = len(header)
        dst = out_dir / shard
        with dst.open("wb") as fh:
            fh.write(header)
            fh.truncate(file_size)
        with dst.open("r+b") as fh, blob.open("rb") as src:
            for _name, abs_off, nbytes, blob_off in rec["entries"]:
                if abs_off < header_end or abs_off + nbytes > file_size:
                    raise ValueError(f"{shard}: tensor {_name} outside shard bounds")
                src.seek(blob_off)
                fh.seek(abs_off)
                shutil.copyfileobj(_Limit(src, nbytes), fh, length=16 << 20)
                if fh.tell() != abs_off + nbytes:
                    raise ValueError(f"{shard}: short write for tensor {_name}")
        outputs.append(dst)
    return outputs


class _Limit:
    """Read at most `remaining` bytes from a binary file object."""

    def __init__(self, fh, remaining: int) -> None:
        self._fh = fh
        self._remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        want = self._remaining if size is None or size < 0 else min(size, self._remaining)
        data = self._fh.read(want)
        self._remaining -= len(data)
        return data
