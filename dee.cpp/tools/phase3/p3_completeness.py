"""Phase 3: prove no routing-universe member is missing.

Independent completeness audit between three artifacts:

  1. the committed shard headers (the source of truth for what the pinned
     checkpoint contains);
  2. the Phase-3 manifest + records JSONL (what the store claims to cover);
  3. an optional built store (metadata.json + data/segments) — structural
     audit only (sizes, stride, segment table); content is verified by
     p3_builder.verify_store.

The audit fails closed on ANY of:
  - a ``*.ffn.experts.*`` tensor name in the headers that does not parse
    (covers BOTH ``layers.*`` and ``mtp.*`` namespaces);
  - a (bucket, expert) pair with any of the six components missing or
    duplicated;
  - a record whose ranges disagree with the header data_offsets;
  - coverage != the manifest's declared universe;
  - store structure inconsistent with the manifest.

Run standalone:
    python tools/phase3/p3_completeness.py \
        --headers dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers \
        --manifest <out>/p3_manifest.json --records <out>/p3_records.jsonl \
        [--store <built store dir>]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from p3_manifest import (
    COMPONENTS,
    EXPERTS_PER_LAYER,
    N_BUCKETS,
    N_LAYERS,
    RECORD_BYTES,
    bucket_shard_name,
    load_committed_headers,
    parse_expert_tensor_name,
    tensor_name,
)


def audit_headers_vs_manifest(
    headers_dir: Path | str,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail-closed completeness proof: headers <-> manifest <-> records."""
    headers = load_committed_headers(headers_dir)

    # 1. Manifest geometry.
    n_buckets = int(manifest["n_buckets"])
    if n_buckets not in (N_LAYERS, N_BUCKETS):
        raise ValueError(f"manifest n_buckets {n_buckets} unsupported")
    total_experts = n_buckets * EXPERTS_PER_LAYER
    store_bytes = total_experts * RECORD_BYTES
    if int(manifest["n_layers"]) != N_LAYERS:
        raise ValueError("manifest n_layers mismatch")
    if int(manifest["experts_per_layer"]) != EXPERTS_PER_LAYER:
        raise ValueError("manifest experts_per_layer mismatch")
    if int(manifest["total_experts"]) != total_experts:
        raise ValueError("manifest total_experts mismatch")
    if int(manifest["record_bytes"]) != RECORD_BYTES:
        raise ValueError("manifest record_bytes mismatch")
    if int(manifest["store_bytes"]) != store_bytes:
        raise ValueError("manifest store_bytes mismatch")

    # 2. Every routed-expert header name must parse; collect coverage.
    parsed_names: Counter[tuple[int, int]] = Counter()
    routed_name_count = 0
    out_of_scope = 0
    for shard, header in headers.items():
        for name in header:
            if name == "__metadata__":
                continue
            parsed = parse_expert_tensor_name(name)
            if ".ffn.experts." in name and parsed is None:
                raise ValueError(f"unparsable routed tensor name: {name}")
            if parsed is not None:
                routed_name_count += 1
                if parsed[0] < n_buckets:
                    parsed_names[(parsed[0], parsed[1])] += 1
                else:
                    out_of_scope += 1
    in_scope_expected = total_experts * len(COMPONENTS)
    if sum(parsed_names.values()) != in_scope_expected:
        raise ValueError(
            f"in-scope routed tensors {sum(parsed_names.values())} != "
            f"{in_scope_expected}")
    for pair, count in parsed_names.items():
        if count != len(COMPONENTS):
            raise ValueError(f"pair {pair} has {count} routed tensors")

    # 3. Every universe pair resolves exactly once in the records list.
    if len(records) != total_experts:
        raise ValueError(f"records list {len(records)} != {total_experts}")
    seen_pairs: set[tuple[int, int]] = set()
    for i, rec in enumerate(records):
        if int(rec["record_index"]) != i:
            raise ValueError(f"record {i}: record_index {rec['record_index']}")
        pair = (int(rec["bucket"]), int(rec["expert"]))
        if pair in seen_pairs:
            raise ValueError(f"duplicate record for pair {pair}")
        seen_pairs.add(pair)
        if pair != (i // EXPERTS_PER_LAYER, i % EXPERTS_PER_LAYER):
            raise ValueError(f"record {i} maps to unexpected pair {pair}")
        # Ranges must match the committed header data_offsets exactly.
        expected_names = {
            tensor_name(pair[0], pair[1], p, k) for p, k in COMPONENTS
        }
        header = headers[bucket_shard_name(pair[0])]
        rec_offset = 0
        for component, data_offset, nbytes, record_offset, tensor in \
                rec["ranges"]:
            if tensor not in expected_names:
                raise ValueError(f"record {i}: unexpected tensor {tensor}")
            expected_names.discard(tensor)
            meta = header.get(tensor)
            if meta is None:
                raise ValueError(f"record {i}: tensor absent from header")
            h0, h1 = (int(v) for v in meta["data_offsets"])
            if int(data_offset) != h0 or int(nbytes) != h1 - h0:
                raise ValueError(f"record {i}: range mismatch for {tensor}")
            if int(record_offset) != rec_offset:
                raise ValueError(f"record {i}: record_offset drift")
            rec_offset += int(nbytes)
        if expected_names:
            raise ValueError(f"record {i}: missing components {expected_names}")
        if rec_offset != RECORD_BYTES:
            raise ValueError(f"record {i}: total {rec_offset} != {RECORD_BYTES}")
    missing = {
        (b, e) for b in range(n_buckets) for e in range(EXPERTS_PER_LAYER)
    } - seen_pairs
    if missing:
        raise ValueError(f"universe pairs missing: {sorted(missing)[:8]}")

    return {
        "all_pairs_covered": True,
        "buckets": n_buckets,
        "pairs": total_experts,
        "routed_tensors_in_scope": in_scope_expected,
        "routed_tensors_total": routed_name_count,
        "routed_tensors_out_of_scope": out_of_scope,
        "orphan_tensors": 0,
        "duplicate_pairs": 0,
        "record_bytes": RECORD_BYTES,
        "store_bytes": store_bytes,
    }


def audit_store_structure(
    store_dir: Path | str, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Structural audit of a built store (no content hashing)."""
    store_dir = Path(store_dir)
    metadata = json.loads((store_dir / "metadata.json").read_text("utf-8"))
    fmt = metadata["format"]
    record_bytes = int(metadata["record_bytes"])
    if record_bytes != int(manifest["record_bytes"]):
        raise ValueError("store record_bytes != manifest")
    if int(metadata["total_experts"]) != int(manifest["total_experts"]):
        raise ValueError("store total_experts != manifest")
    if metadata.get("universe_sha256") != manifest["universe_sha256"]:
        raise ValueError("store universe_sha256 != manifest")

    integrity_lines = sum(
        1 for line in (store_dir / "integrity.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip())
    if integrity_lines != int(manifest["total_experts"]):
        raise ValueError(
            f"integrity lines {integrity_lines} != {manifest['total_experts']}")

    if fmt == "dee4-v2":
        size = (store_dir / metadata["data_file"]).stat().st_size
        if size != int(manifest["store_bytes"]):
            raise ValueError(f"data file {size} != {manifest['store_bytes']}")
        return {"format": fmt, "data_bytes": size, "segments": 1}
    if fmt == "dee4-v4-segmented":
        total = 0
        next_first = 0
        for seg in metadata["segments"]:
            size = (store_dir / seg["file"]).stat().st_size
            expected = int(seg["record_count"]) * record_bytes
            if size != expected:
                raise ValueError(
                    f"segment {seg['file']}: {size} != {expected}")
            if int(seg["first_record"]) != next_first:
                raise ValueError("segment table is not contiguous")
            next_first += int(seg["record_count"])
            total += size
        if next_first != int(manifest["total_experts"]):
            raise ValueError("segments do not cover the universe")
        if total != int(manifest["store_bytes"]):
            raise ValueError("segment bytes do not total the store size")
        return {"format": fmt, "data_bytes": total,
                "segments": len(metadata["segments"])}
    raise ValueError(f"unsupported store format {fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headers", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--store", type=Path, default=None)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = audit_headers_vs_manifest(args.headers, manifest, records)
    if args.store is not None:
        report["store"] = audit_store_structure(args.store, manifest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
