"""CACHE1f: local weight staging for the Kaggle kernel.

Makes a bounded subset of the 167 GB official checkpoint available as LOCAL
reads inside the kernel, eliminating most per-tensor HTTP Range fetches from
the decode critical path while preserving bit-identical tensors.

Why this exists (measured, not assumed):
  - The DS10 storage probe measured a ~19.5 GiB shared disk for
    /kaggle/working + /kaggle/input: the full checkpoint (166.9 GB) and even
    the 45-shard minimal set (145.3 GiB) can never be pre-staged.
  - The sealed DS10 canonical trace touches 2,365 unique (layer, expert)
    pairs = 14,190 unique routed-expert tensors (34,056 accesses).  At
    TENSOR granularity, ~60-70% of decode-time routed fetches fit in the
    kernel disk when ordered by access frequency.
  - Staging is purely a performance optimization: ``HybridTensorSource``
    falls back to the sealed RemoteTensorSource path for any tensor that is
    not staged, so correctness never depends on the manifest being complete.

Layout: staged tensors are written as PARTIAL safetensors files, one per
official shard (only the staged tensors present), under a staging directory.
``HybridTensorSource`` reads them with safe_open and falls back per tensor.

The staging manifest is generated deterministically from the committed shard
headers + the sealed DS10 token trace and committed to the repo, so the
kernel stages without recomputing anything.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from scripts import deepseek_v4_model as vm
from scripts import deepseek_v4_support as v4support


# ---------------------------------------------------------------------------
# Staging manifest (committed, deterministic).
# ---------------------------------------------------------------------------


def build_staging_manifest(
    headers_dir: Path | str,
    config_path: Path | str,
    evidence_path: Path | str,
    *,
    n_layers: int = 43,
    n_hash_layers: int = 3,
) -> dict[str, Any]:
    """Build the deterministic staging manifest from committed headers +
    the sealed DS10 token trace.

    Priority order (this IS the staging order):
      1. top-level tensors (embed/head/norm/hc_head) -- needed at build;
      2. dense layer tensors (attention/router/norms/compress/index) --
         needed at build, 43 layers;
      3. shared-expert tensors -- fetched every token, every layer;
      4. routed-expert tensors -- sorted by access frequency in the sealed
         trace (most-reused first).

    The manifest records for each tensor: name, shard, byte length, stored
    dtype, shape, priority class, and (for routed) access count.
    """
    headers_dir = Path(headers_dir)
    name_to_info: dict[str, dict[str, Any]] = {}
    shard_bytes: dict[str, int] = {}
    for f in sorted(headers_dir.glob("model-*.json")):
        shard = f.name.replace(".json", "")
        h = json.loads(f.read_text(encoding="utf-8"))
        total = 0
        for name, meta in h.items():
            if name == "__metadata__":
                continue
            d0, d1 = meta["data_offsets"]
            length = int(d1) - int(d0)
            name_to_info[name] = {
                "shard": shard,
                "length": length,
                "dtype": meta["dtype"],
                "shape": [int(d) for d in meta["shape"]],
            }
            total += length
        shard_bytes[shard] = total

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ratios = tuple(int(r) for r in cfg["compress_ratios"][:n_layers])

    # ---- priority 1: top-level -------------------------------------------
    top_level = [
        "embed.weight", "head.weight", "norm.weight",
        "hc_head_fn", "hc_head_base", "hc_head_scale",
    ]

    # ---- priority 2: dense layer tensors ---------------------------------
    dense: list[str] = []
    for layer in range(n_layers):
        dense.extend(v4support.layer_dense_tensor_names(
            layer, hash_layer=layer < n_hash_layers,
            compress_ratio=ratios[layer]))

    # ---- priority 3: shared experts --------------------------------------
    shared: list[str] = []
    for layer in range(n_layers):
        shared.extend(v4support.shared_expert_tensor_names(layer))

    # ---- priority 4: routed experts by access frequency ------------------
    trace = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    token_trace = trace["gates"]["token_trace"]
    freq: Counter[str] = Counter()
    for tok in token_trace.values():
        for ly in tok["layers"]:
            lid = ly["layer"]
            for pos in ly["selected_experts"]:
                for e in pos:
                    for proj in ("w1", "w2", "w3"):
                        for kind in ("weight", "scale"):
                            freq[f"layers.{lid}.ffn.experts.{e}.{proj}.{kind}"] += 1
    routed = sorted(freq, key=lambda n: (-freq[n], n))

    # ---- assemble ---------------------------------------------------------
    tensors: list[dict[str, Any]] = []
    for priority, names in ((1, top_level), (2, dense), (3, shared), (4, routed)):
        for name in names:
            if name not in name_to_info:
                raise KeyError(f"staging manifest: {name} not in committed headers")
            row = dict(name_to_info[name])
            row["name"] = name
            row["priority"] = priority
            row["access_count"] = int(freq.get(name, 0))
            tensors.append(row)

    total_bytes = sum(t["length"] for t in tensors)
    return {
        "schema_version": 1,
        "n_layers": n_layers,
        "n_hash_layers": n_hash_layers,
        "tensor_count": len(tensors),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1 << 30), 3),
        "shard_bytes": {k: v for k, v in sorted(shard_bytes.items())},
        "tensors": tensors,
    }


def select_staged_tensors(manifest: dict[str, Any], budget_bytes: int) -> list[dict[str, Any]]:
    """Greedy budget selection in manifest priority order (must-stage first,
    then routed by access frequency).  Returns the selected tensor rows."""
    selected: list[dict[str, Any]] = []
    used = 0
    for row in manifest["tensors"]:
        if used + row["length"] > budget_bytes:
            continue
        selected.append(row)
        used += row["length"]
    return selected


# ---------------------------------------------------------------------------
# Partial safetensors shard writer.
# ---------------------------------------------------------------------------


def _write_partial_shard(path: Path, tensors: dict[str, tuple[bytes, str, list[int]]]) -> None:
    """Write a partial safetensors file containing only ``tensors``.

    ``tensors`` maps name -> (raw byte blob, safetensors dtype string,
    shape).  Standard safetensors layout: 8-byte little-endian header length,
    JSON header, then the concatenated tensor data.
    """
    header: dict[str, Any] = {}
    data = bytearray()
    for name, (blob, dtype, shape) in tensors.items():
        start = len(data)
        header[name] = {
            "dtype": dtype,
            "shape": [int(d) for d in shape],
            "data_offsets": [start, start + len(blob)],
        }
        data += blob
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_json)))
        fh.write(header_json)
        fh.write(bytes(data))


def stage_partial_shards(
    manifest: dict[str, Any],
    headers_dir: Path | str,
    dest_dir: Path | str,
    *,
    budget_bytes: int,
    revision: str = vm.OFFICIAL_REVISION,
    repository: str = vm.OFFICIAL_REPOSITORY,
    max_workers: int = 32,
    max_attempts: int = 6,
) -> dict[str, Any]:
    """Download the budget-selected tensors and write partial shard files.

    Downloads per tensor through the sealed RemoteTensorSource range-fetch
    path (identical bytes, pinned revision), groups by official shard, and
    writes one partial safetensors file per shard into ``dest_dir``.

    Returns stats: staged tensors, staged bytes, per-shard file counts,
    HTTP requests/bytes/retries used, wall seconds.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    headers_dir = Path(headers_dir)
    dest_dir = Path(dest_dir)
    selected = select_staged_tensors(manifest, budget_bytes)
    selected_by_shard: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        selected_by_shard.setdefault(row["shard"], []).append(row)

    remote = vm.RemoteTensorSource(
        headers_dir, revision=revision, repository=repository,
        max_workers=max_workers, max_attempts=max_attempts)

    t0 = time.monotonic()
    staged_bytes = 0
    files_written = 0
    errors: list[str] = []
    for shard, rows in sorted(selected_by_shard.items()):
        blobs: dict[str, tuple[bytes, str, list[int]]] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(rows))) as ex:
            futures = {
                ex.submit(remote._fetch_bytes, row["name"]): row
                for row in rows
            }
            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    data = fut.result()
                    if len(data) != row["length"]:
                        raise RuntimeError(
                            f"{row['name']}: {len(data)} bytes, expected {row['length']}")
                    blobs[row["name"]] = (data, row["dtype"], row["shape"])
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{row['name']}: {type(exc).__name__}: {exc}")
        if blobs:
            _write_partial_shard(dest_dir / shard, blobs)
            files_written += 1
            staged_bytes += sum(len(b) for b, _, _ in blobs.values())

    dt = time.monotonic() - t0
    return {
        "staged_tensors": sum(len(v) for v in selected_by_shard.values()),
        "staged_bytes": staged_bytes,
        "staged_gib": round(staged_bytes / (1 << 30), 3),
        "partial_shard_files": files_written,
        "shards_covered": len(selected_by_shard),
        "errors": errors,
        "http_requests": remote.stats["requests"],
        "http_bytes": remote.stats["bytes"],
        "http_retries": remote.stats["retries"],
        "wall_seconds": round(dt, 2),
        "budget_bytes": budget_bytes,
        "manifest_tensor_count": manifest["tensor_count"],
        "manifest_total_gib": manifest["total_gib"],
    }


# ---------------------------------------------------------------------------
# Hybrid local-first tensor source.
# ---------------------------------------------------------------------------


class HybridTensorSource(vm.CommittedHeaderSource):
    """Local-first tensor source with per-tensor HTTP fallback.

    Reads tensors from staged partial shard files under ``shards_dir`` when
    present; any tensor not staged (or shard file absent) falls back to the
    sealed ``RemoteTensorSource`` range-fetch path.  Bytes are identical
    either way (same pinned revision), so numerics are unchanged.

    Exposes ``stats`` as a dict with the remote counters plus local
    hit/miss counters, matching the shape the harness reads via
    ``source.stats``.
    """

    def __init__(self, headers_dir: Path | str, shards_dir: Path | str, *,
                 revision: str = vm.OFFICIAL_REVISION,
                 repository: str = vm.OFFICIAL_REPOSITORY,
                 max_workers: int = 16,
                 max_attempts: int = 6):
        super().__init__(headers_dir, revision=revision)
        self.shards_dir = Path(shards_dir)
        self.remote = vm.RemoteTensorSource(
            headers_dir, revision=revision, repository=repository,
            max_workers=max_workers, max_attempts=max_attempts)
        # shard -> (parsed header dict, data bytes) parsed once per shard.
        self._handles: dict[str, tuple[dict[str, Any], bytes]] = {}
        self._local_hits = 0
        self._local_misses = 0

    # -- CommittedHeaderSource overrides -----------------------------------
    def _load_shard(self, shard: str) -> tuple[dict[str, Any], bytes]:
        if shard not in self._handles:
            path = self.shards_dir / shard
            if not path.is_file():
                raise FileNotFoundError(f"staged shard missing: {path}")
            raw = path.read_bytes()
            hlen = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8:8 + hlen].decode("utf-8"))
            self._handles[shard] = (header, raw[8 + hlen:])
        return self._handles[shard]

    def _fetch_bytes(self, name: str) -> bytes:
        row = self.tensor_identity(name)
        shard = row["shard"]
        try:
            header, data = self._load_shard(shard)
            meta = header.get(name)
            if meta is None:
                raise KeyError(f"{name} not staged in {shard}")
            d0, d1 = meta["data_offsets"]
            blob = bytes(data[d0:d1])
            self._local_hits += 1
            return blob
        except (KeyError, FileNotFoundError):
            self._local_misses += 1
            return self.remote._fetch_bytes(name)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "requests": self.remote.stats["requests"],
            "bytes": self.remote.stats["bytes"],
            "retries": self.remote.stats["retries"],
            "local_hits": self._local_hits,
            "local_misses": self._local_misses,
        }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
