"""Phase 3: fetch safetensors shard headers over HTTP byte ranges.

Produces the same ``<shard>.json`` header files that DSv4 keeps committed
under ``benchmark_reports/.../shard-headers/`` — one JSON per safetensors
shard, parsed from the shard's own header block (8-byte little-endian
length prefix + JSON).  Header names are enumerated from the repo's
``model.safetensors.index.json`` weight_map when present, else from the
HF file listing API.

Usage::

    python fetch_headers.py --repo XiaomiMiMo/MiMo-V2.6-Flash-RL \
        --revision 5711b268 --out-dir headers/

Honours HF_TOKEN / HUGGING_FACE_HUB_TOKEN for gated repos (never printed).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path


def _auth_headers() -> dict[str, str]:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            return {"Authorization": f"Bearer {tok}"}
    return {}


def _url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def _read_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end}")
    req.headers.update(_auth_headers())
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def list_shards(repo: str, revision: str) -> list[str]:
    """All *.safetensors filenames in the repo at the pinned revision."""
    try:
        idx_url = _url(repo, revision, "model.safetensors.index.json")
        req = urllib.request.Request(idx_url)
        req.headers.update(_auth_headers())
        with urllib.request.urlopen(req, timeout=120) as resp:
            idx = json.loads(resp.read())
        return sorted(set(idx["weight_map"].values()))
    except Exception:
        pass
    api = f"https://huggingface.co/api/models/{repo}?revision={revision}"
    req = urllib.request.Request(api)
    req.headers.update(_auth_headers())
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return sorted(
        f["rfilename"] for f in data.get("siblings", [])
        if f["rfilename"].endswith(".safetensors"))


def fetch_one(repo: str, revision: str, shard: str, out_dir: Path) -> str:
    out_path = out_dir / f"{shard}.json"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return f"cached {shard}"
    url = _url(repo, revision, shard)
    hlen = struct.unpack("<Q", _read_range(url, 0, 7))[0]
    if not (0 < hlen < 100_000_000):
        raise RuntimeError(f"{shard}: implausible header length {hlen}")
    header = _read_range(url, 8, 8 + hlen - 1)
    json.loads(header)  # validate
    tmp = out_path.with_suffix(".json.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(header)
    tmp.replace(out_path)
    return f"fetched {shard} ({hlen} B header)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = list_shards(args.repo, args.revision)
    print(f"{len(shards)} safetensors shards in {args.repo}@{args.revision}",
          flush=True)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, args.repo, args.revision, s,
                            args.out_dir): s for s in shards}
        for fut in concurrent.futures.as_completed(futs):
            print(fut.result(), flush=True)
            done += 1
    print(f"headers complete: {done}/{len(shards)} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
