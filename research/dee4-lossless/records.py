#!/usr/bin/env python3
"""Exact dee4 trace-bank record reconstruction (research only, no CUDA).

A bank record is the VERBATIM concatenation of 6 source tensors
(w1/w3/w2 weight=i8-packed-FP4 + w1/w3/w2 scale=e8m0), 13369344 bytes,
no header, no padding. Source: deepseek-ai/DeepSeek-V4-Flash-0731
safetensors shards, byte ranges from model.safetensors.index.json.

Two fetch backends:
  hf    : HTTP Range requests against huggingface.co (local pilot)
  mount : local directory of *.safetensors (Kaggle dataset mount)

Reconstruction is verified against dee4-integrity.jsonl record_sha256.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

RECORD_BYTES = 13369344
COMPONENTS = ("w1.weight", "w1.scale", "w3.weight", "w3.scale",
              "w2.weight", "w2.scale")
BASE_URL = ("https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731"
            "/resolve/main")


def tensor_key(layer: int, expert: int, comp: str) -> str:
    return f"layers.{layer}.ffn.experts.{expert}.{comp}"


class HFBackend:
    """Range-fetch tensor bytes from the HF Hub (nothing fully downloaded)."""

    def __init__(self, weight_map: dict, cache: dict | None = None):
        self.weight_map = weight_map
        self._headers: dict[str, dict] = {}
        self.cache = cache if cache is not None else {}

    def _get(self, url: str, start: int, end: int) -> bytes:
        key = (url, start, end)
        if key in self.cache:
            return self.cache[key]
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end}",
                          "User-Agent": "dee-lossless-scan/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            if r.status not in (200, 206):
                raise RuntimeError(f"range fetch {url} [{start}-{end}]: "
                                   f"HTTP {r.status}")
            data = r.read()
        if len(data) != end - start + 1:
            raise RuntimeError(f"short range fetch: {len(data)} != "
                               f"{end - start + 1}")
        self.cache[key] = data
        return data

    def _header(self, shard: str) -> dict:
        if shard not in self._headers:
            url = f"{BASE_URL}/{shard}"
            n = int.from_bytes(self._get(url, 0, 7), "little")
            self._headers[shard] = json.loads(self._get(url, 8, 8 + n - 1))
        return self._headers[shard]

    def read_tensor(self, key: str) -> bytes:
        shard = self.weight_map[key]
        head = self._header(shard)
        off = head[key]["data_offsets"]
        n = int.from_bytes(self._get(f"{BASE_URL}/{shard}", 0, 7), "little")
        base = 8 + n
        return self._get(f"{BASE_URL}/{shard}", base + off[0], base + off[1] - 1)


class MountBackend:
    """Read tensor bytes from a local directory of safetensors shards."""

    def __init__(self, weight_map: dict, root: str | Path):
        self.weight_map = weight_map
        self.root = Path(root)
        self._files: dict[str, object] = {}
        self._headers: dict[str, dict] = {}

    def _header(self, shard: str) -> tuple[dict, int]:
        if shard not in self._headers:
            fh = open(self.root / shard, "rb")
            self._files[shard] = fh
            n = int.from_bytes(fh.read(8), "little")
            self._headers[shard] = (json.loads(fh.read(n)), 8 + n)
        return self._headers[shard]

    def read_tensor(self, key: str) -> bytes:
        shard = self.weight_map[key]
        head, base = self._header(shard)
        off = head[key]["data_offsets"]
        fh = self._files[shard]
        fh.seek(base + off[0])
        data = fh.read(off[1] - off[0])
        if len(data) != off[1] - off[0]:
            raise RuntimeError(f"short read {key}")
        return data


def reconstruct(layer: int, expert: int, backend) -> bytes:
    parts = []
    for comp in ("w1.weight", "w3.weight", "w2.weight",
                 "w1.scale", "w3.scale", "w2.scale"):
        blob = backend.read_tensor(tensor_key(layer, expert, comp))
        parts.append(blob)
    rec = b"".join(parts)
    if len(rec) != RECORD_BYTES:
        raise RuntimeError(f"record {(layer, expert)} size {len(rec)}")
    return rec


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_integrity(path: str | Path) -> dict[tuple[int, int], dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[(r["layer"], r["expert"])] = r
    return out
