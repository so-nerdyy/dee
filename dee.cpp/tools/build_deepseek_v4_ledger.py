#!/usr/bin/env python3
"""Build the authoritative DeepSeek-V4-Flash-0731 tensor/byte ledger.

Fetches only safetensors headers (HTTP range requests) for all 48 pinned
official shards, validates the checkpoint index against every shard header,
and writes:

  benchmark_reports/deepseek-v4-flash-0731-t4/MODEL_LEDGER.json
  benchmark_reports/deepseek-v4-flash-0731-t4/CHECKPOINT_MANIFEST.json

No weight bytes are downloaded. Run from the repository root (dee.cpp).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.deepseek_v4_support import (  # noqa: E402
    EXPECTED_SHARD_COUNT,
    OFFICIAL_REVISION,
    aggregate_by_component,
    build_complete_tensor_ledger,
    fetch_shard_header,
    read_checkpoint_index,
)

DEFAULT_REPORT_DIR = ROOT / "benchmark_reports/deepseek-v4-flash-0731-t4"


def build_ledger(model_dir: Path, report_dir: Path, *, refresh: bool = False) -> dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)
    index = read_checkpoint_index(model_dir)
    shard_names = sorted({shard for shard in index["weight_map"].values()})
    if len(shard_names) != EXPECTED_SHARD_COUNT:
        raise ValueError(f"index references {len(shard_names)} shards, expected {EXPECTED_SHARD_COUNT}")

    headers: dict[str, dict] = {}
    for shard in shard_names:
        if not refresh:
            cached = report_dir / "shard-headers" / f"{shard}.json"
            if cached.is_file():
                headers[shard] = json.loads(cached.read_text(encoding="utf-8"))
                continue
        headers[shard] = fetch_shard_header(OFFICIAL_REVISION, shard)
        (report_dir / "shard-headers").mkdir(parents=True, exist_ok=True)
        (report_dir / "shard-headers" / f"{shard}.json").write_text(
            json.dumps(headers[shard], sort_keys=True), encoding="utf-8"
        )

    rows, summary = build_complete_tensor_ledger(index, headers)
    by_component = aggregate_by_component(rows)

    ledger = {
        "schema_version": 1,
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": OFFICIAL_REVISION,
        "summary": summary,
        "components": by_component,
        "tensors": rows,
    }
    (report_dir / "MODEL_LEDGER.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    header_dir = report_dir / "shard-headers"
    manifest = {
        "schema_version": 1,
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": OFFICIAL_REVISION,
        "shard_count": len(shard_names),
        "tensor_count": summary["tensor_count"],
        "declared_total_size": summary["declared_total_size"],
        "validated_compressed_bytes": summary["compressed_bytes"],
        "shards": [
            {
                "shard": shard,
                "tensor_count": row["tensor_count"],
                "compressed_bytes": row["compressed_bytes"],
                "header_sha256": hashlib.sha256(
                    (header_dir / f"{shard}.json").read_bytes()
                ).hexdigest(),
                "weight_bytes_not_downloaded": True,
            }
            for row, shard in zip(
                sorted(summary["shards"], key=lambda item: item["shard"]),
                sorted(shard_names),
            )
        ],
        "header_hashes_available": True,
    }
    (report_dir / "CHECKPOINT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "summary": summary,
        "components": by_component,
        "manifest_path": str(report_dir / "CHECKPOINT_MANIFEST.json"),
        "ledger_path": str(report_dir / "MODEL_LEDGER.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--refresh", action="store_true", help="refetch shard headers")
    args = parser.parse_args()
    result = build_ledger(args.model_dir.resolve(), args.report_dir.resolve(), refresh=args.refresh)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
