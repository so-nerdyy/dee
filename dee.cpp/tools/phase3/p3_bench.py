"""Phase 3: honest conversion-throughput measurement.

Measures the per-record pipeline (source fetch -> assemble -> sha256 ->
fsync'd append) on a bounded subset, then reports MiB/s so the full-store
build time can be extrapolated honestly.

Usage:
    python tools/phase3/p3_bench.py --records 8 --source local \
        --shards <dir-of-safetensors-shards>
    python tools/phase3/p3_bench.py --records 2 --source remote   # HF fetch
    python tools/phase3/p3_bench.py --records 16 --source synthetic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_builder  # noqa: E402
import p3_manifest  # noqa: E402


def synth_source(record_bytes: int) -> "p3_builder.RangeSource":
    """Zero-IO source: returns deterministic bytes; isolates hash+write cost."""
    block = hashlib.sha256(b"p3-synth").digest()

    class _Src:
        def fetch(self, shard: str, data_offset: int, nbytes: int) -> bytes:
            return (block * (nbytes // 32 + 1))[:nbytes]

    return _Src()


def run_bench(
    source: "p3_builder.RangeSource",
    records: list[dict[str, Any]],
    out_path: Path,
    *,
    fsync: bool = True,
) -> dict:
    record_bytes = records[0]["record_bytes"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_record_ms = []
    data_hash = hashlib.sha256()
    t0 = time.monotonic()
    total = 0
    with out_path.open("wb", buffering=8 << 20) as fh:
        for rec in records:
            r0 = time.monotonic()
            blob, _comp, _srcs = p3_builder.assemble_record(source, rec)
            data_hash.update(blob)
            fh.write(blob)
            if fsync:
                fh.flush()
                os.fsync(fh.fileno())
            per_record_ms.append((time.monotonic() - r0) * 1000)
            total += len(blob)
    wall = time.monotonic() - t0
    per_record_ms.sort()
    mib = total / (1 << 20)
    universe = int(
        os.environ.get("P3_BENCH_UNIVERSE", "11776"))  # 46 buckets x 256
    return {
        "records": len(records),
        "bytes": total,
        "mib": round(mib, 2),
        "wall_seconds": round(wall, 3),
        "mib_per_s": round(mib / max(wall, 1e-9), 2),
        "per_record_ms_p50": round(per_record_ms[len(per_record_ms) // 2], 1),
        "per_record_ms_max": round(per_record_ms[-1], 1),
        "record_bytes": record_bytes,
        "fsync": fsync,
        "data_sha256": data_hash.hexdigest(),
        # Honest extrapolation to the full universe at this measured rate:
        "est_full_store_seconds": round(
            wall / len(records) * universe, 1),
        "est_full_store_hours": round(
            wall / len(records) * universe / 3600, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=int, default=8)
    ap.add_argument("--source", choices=["local", "remote", "synthetic"],
                    default="synthetic")
    ap.add_argument("--shards", type=Path, default=None)
    ap.add_argument("--headers", type=Path, default=None)
    ap.add_argument("--layers", type=str, default="0",
                    help="comma list of layers to sample records from")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-fsync", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    headers = args.headers or (
        repo / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
             "shard-headers")
    manifest = p3_manifest.build_manifest(headers)
    all_records = manifest["_records"]
    layers = [int(x) for x in args.layers.split(",")]

    picked: list[dict] = []
    per_layer = max(1, args.records // max(1, len(layers)))
    for layer in layers:
        start = layer * manifest["experts_per_layer"]
        picked.extend(all_records[start:start + per_layer])
    picked = picked[: args.records]

    if args.source == "remote":
        source = p3_builder.RemoteRangeSource()
    elif args.source == "local":
        if args.shards is None:
            ap.error("--source local requires --shards")
        source = p3_builder.LocalShardSource(args.shards)
    else:
        source = synth_source(manifest["record_bytes"])

    out = args.out or (Path("tmp") / "p3_bench.bin")
    result = run_bench(source, picked, out, fsync=not args.no_fsync)
    if isinstance(source, p3_builder.RemoteRangeSource):
        result["remote_stats"] = source.stats
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
