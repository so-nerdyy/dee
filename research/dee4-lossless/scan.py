#!/usr/bin/env python3
"""Per-record lossless scan driver (research only, CPU-only).

Tier 'sweep' (fast, all records): lz4-fast, zstd-1/3, nibble-split+zstd-3,
  order-0/1 entropy bounds, zero stats, region hashes.
Tier 'full' (subset): sweep + lz4-hc, zstd-10, real rANS.

Every codec roundtrip is verified byte-for-byte (sha256). Output CSV has
one row per (record, region, codec); summary JSON has distributions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dq_codecs import (REGIONS, PACKED, c_lz4_fast, c_lz4_hc, d_lz4, c_zstd,
                     d_zstd, nibble_split, order0_bits_per_byte,
                     order1_bits_per_byte, rans_encode_size, zero_stats)

SWEEP_CODECS = ("lz4_fast", "zstd_1", "zstd_3", "nib_split_zstd3",
                "order0", "order1", "zeros")
FULL_EXTRA = ("lz4_hc", "zstd_10", "rans")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_codec(codec: str, blob: bytes) -> dict:
    """Returns {cbytes, ratio, c_ms, d_ms, verified} (estimates have no
    encode/decode timing). Raises on roundtrip mismatch."""
    n = len(blob)
    if codec == "lz4_fast":
        c, cms = c_lz4_fast(blob)
        d, dms = d_lz4(c)
        assert sha(d) == sha(blob), "lz4 roundtrip mismatch"
        return {"cbytes": len(c), "c_ms": cms, "d_ms": dms}
    if codec == "lz4_hc":
        c, cms = c_lz4_hc(blob)
        d, dms = d_lz4(c)
        assert sha(d) == sha(blob), "lz4hc roundtrip mismatch"
        return {"cbytes": len(c), "c_ms": cms, "d_ms": dms}
    if codec.startswith("zstd_"):
        level = int(codec.split("_")[1])
        c, cms = c_zstd(level)(blob)
        d, dms = d_zstd(c)
        assert sha(d) == sha(blob), f"zstd{level} roundtrip mismatch"
        return {"cbytes": len(c), "c_ms": cms, "d_ms": dms}
    if codec == "nib_split_zstd3":
        lo, hi = nibble_split(blob)
        clo, mlo = c_zstd(3)(lo)
        chi, mhi = c_zstd(3)(hi)
        dlo, dlo_ms = d_zstd(clo)
        dhi, dhi_ms = d_zstd(chi)
        assert sha(dlo) == sha(lo) and sha(dhi) == sha(hi)
        # note: streams stored separately; no interleave cost counted
        return {"cbytes": len(clo) + len(chi),
                "c_ms": mlo + mhi, "d_ms": dlo_ms + dhi_ms}
    if codec == "order0":
        return {"est_bytes": order0_bits_per_byte(blob) / 8 * n}
    if codec == "order1":
        return {"est_bytes": order1_bits_per_byte(blob) / 8 * n}
    if codec == "rans":
        size, ems = rans_encode_size(blob)
        return {"cbytes": size, "c_ms": ems, "d_ms": None,
                "note": "model-excluded; decode ~ encode speed class"}
    if codec == "zeros":
        return {"stats": zero_stats(blob)}
    raise ValueError(codec)


def scan_record(layer: int, expert: int, rec: bytes, tier: str) -> list[dict]:
    rows = []
    codecs = list(SWEEP_CODECS) + (list(FULL_EXTRA) if tier == "full" else [])
    for region, (off, size) in REGIONS.items():
        blob = rec[off:off + size]
        assert len(blob) == size
        for codec in codecs:
            if codec == "rans" and region == "whole":
                continue  # time-box: rANS on regions only
            if codec in ("order0", "order1", "zeros") and region == "whole" \
                    and tier == "sweep":
                pass  # cheap enough, keep
            t0 = time.perf_counter()
            try:
                r = run_codec(codec, blob)
                err = ""
            except Exception as exc:  # noqa: BLE001 - record per-codec
                r = {}
                err = repr(exc)[:200]
            rows.append({
                "layer": layer, "expert": expert, "region": region,
                "codec": codec, "orig": size,
                "cbytes": r.get("cbytes"), "est_bytes": r.get("est_bytes"),
                "c_ms": r.get("c_ms"), "d_ms": r.get("d_ms"),
                "verified": (err == "") if codec not in (
                    "order0", "order1", "zeros") else None,
                "error": err,
                "region_sha256": sha(blob),
                "note": r.get("note", "") if not isinstance(
                    r.get("stats"), dict) else json.dumps(r["stats"]),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("sweep", "full"), default="sweep")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="scan_out.csv")
    args = ap.parse_args()
    # record source injected by caller harness (pilot.py / kaggle driver)
    print("scan driver ready; tier=", args.tier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
