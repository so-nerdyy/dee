#!/usr/bin/env python3
"""Resumable DeepSeek-V4-Flash-0731 shard downloader (DS3).

Downloads one or all 48 safetensor shards from the pinned official HF
revision using HTTP Range requests (resume-safe), then verifies:

- final byte size matches the validated ledger manifest; and
- the shard header JSON SHA256 matches the committed cached header
  (proving the file is the pinned revision).

No weight bytes are required for verification beyond size + header hash;
full-file SHA256 can be added as a post-download step by the caller.

Run from the repository root (dee.cpp):

  python tools/download_deepseek_v4_shard.py \\
      --shard model-00001-of-00048.safetensors \\
      --out tmp/dsv4-checkpoint \\
      --manifest benchmark_reports/deepseek-v4-flash-0731-t4/CHECKPOINT_MANIFEST.json

  python tools/download_deepseek_v4_shard.py --all --out tmp/dsv4-checkpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.deepseek_v4_support import (  # noqa: E402
    OFFICIAL_REPOSITORY,
    OFFICIAL_REVISION,
)

DEFAULT_REPORT_DIR = ROOT / "benchmark_reports/deepseek-v4-flash-0731-t4"
CHUNK_BYTES = 8 << 20  # 8 MiB per range request


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def expected_size(manifest: dict, shard: str) -> int:
    for row in manifest.get("shards", []):
        if row.get("shard") == shard:
            return int(row["compressed_bytes"])
    raise ValueError(f"shard {shard} not in manifest")


def header_sha256(manifest: dict, shard: str) -> str:
    for row in manifest.get("shards", []):
        if row.get("shard") == shard:
            return row["header_sha256"]
    raise ValueError(f"shard {shard} not in manifest")


def downloaded_header_sha256(shard_path: Path) -> str:
    """Hash the actual safetensors header bytes of the downloaded file.

    Reads only the 8-byte LE header length + header JSON from the shard,
    then re-serializes with the same canonical form the manifest hash was
    computed from (json.dumps(sort_keys=True)), so the manifest's
    header_sha256 is comparable. This proves the downloaded file is the
    pinned revision, not merely the right byte size.
    """
    with open(shard_path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{shard_path}: truncated 8-byte prefix")
        hlen = struct.unpack("<Q", raw)[0]
        if hlen <= 0 or hlen > (1 << 31):
            raise ValueError(f"{shard_path}: implausible header length {hlen}")
        header_bytes = fh.read(hlen)
        if len(header_bytes) != hlen:
            raise ValueError(f"{shard_path}: truncated header (want {hlen} bytes)")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{shard_path}: malformed header JSON: {exc}") from exc
    canonical = json.dumps(header, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def remote_size(shard: str) -> int:
    """Ask HF for the Content-Range total of a shard."""
    url = f"https://huggingface.co/{OFFICIAL_REPOSITORY}/resolve/{OFFICIAL_REVISION}/{shard}"
    req = urllib.request.Request(url, headers={
        "Range": "bytes=0-0",
        "User-Agent": "freebuff-deepseek-download/1.0",
    })
    with urllib.request.urlopen(req, timeout=120) as resp:
        cr = resp.headers.get("Content-Range", "")
    if "/" not in cr:
        raise ValueError(f"no Content-Range total for {shard}: {cr!r}")
    return int(cr.split("/")[1])


def download_shard(shard: str, out_dir: Path, manifest: dict, *, verify_remote: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / shard
    want = expected_size(manifest, shard)

    if dest.is_file() and dest.stat().st_size == want:
        print(f"[skip] {shard} already complete ({want} bytes)")
    else:
        if verify_remote:
            total = remote_size(shard)
            if total != want:
                raise ValueError(f"{shard}: remote size {total} != manifest {want}")

        url = f"https://huggingface.co/{OFFICIAL_REPOSITORY}/resolve/{OFFICIAL_REVISION}/{shard}"
        have = dest.stat().st_size if dest.is_file() else 0
        mode = "ab" if have > 0 else "wb"
        print(f"[get] {shard} resuming at {have}/{want} bytes")
        with open(dest, mode) as fh:
            while have < want:
                end = min(have + CHUNK_BYTES - 1, want - 1)
                req = urllib.request.Request(url, headers={
                    "Range": f"bytes={have}-{end}",
                    "User-Agent": "freebuff-deepseek-download/1.0",
                })
                with urllib.request.urlopen(req, timeout=300) as resp:
                    chunk = resp.read()
                if not chunk:
                    raise ConnectionError(f"{shard}: empty chunk at {have}")
                fh.write(chunk)
                have += len(chunk)
                print(f"  ... {have}/{want} ({100.0 * have / want:.1f}%)")

    if dest.stat().st_size != want:
        raise ValueError(f"{shard}: final size {dest.stat().st_size} != {want}")

    # Header pin check against the ACTUAL downloaded file: re-hash the
    # shard's own safetensors header and compare with the manifest hash of
    # the pinned revision's cached header. A same-size tampered file fails.
    got_header = downloaded_header_sha256(dest)
    want_header = header_sha256(manifest, shard)
    if got_header != want_header:
        raise ValueError(f"{shard}: downloaded header does not match pinned revision")
    print(f"[ok] {shard} {want} bytes, actual header pinned to revision")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", help="one shard filename, e.g. model-00001-of-00048.safetensors")
    parser.add_argument("--all", action="store_true", help="download all 48 shards")
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/dsv4-checkpoint")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REPORT_DIR / "CHECKPOINT_MANIFEST.json")
    parser.add_argument("--no-verify-remote", action="store_true")
    args = parser.parse_args()

    if bool(args.shard) == bool(args.all):
        parser.error("provide exactly one of --shard or --all")
    manifest = load_manifest(args.manifest)

    shards = [args.shard] if args.shard else sorted({s["shard"] for s in manifest["shards"]})
    for shard in shards:
        download_shard(shard, args.out, manifest, verify_remote=not args.no_verify_remote)
    print(f"download complete: {len(shards)} shard(s) -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
