"""Phase 3: full-universe routed-expert manifest for DeepSeek-V4-Flash-0731.

Builds a deterministic, completeness-checked index over ALL routed experts
of the pinned checkpoint (``9e165c30...``) using only the committed shard
headers (``benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers/
model-*.json``).  No weight bytes are read; the manifest is a pure function
of the committed headers, so it can be regenerated and diffed anywhere.

Verified universe (this module recomputes and asserts every figure):

  * MAIN model: 43 transformer layers (0..42); layers 0..2 are hash-routed
    (``gate.tid2eid``) but still carry the same 256 packed-FP4 routed
    experts each.  -> 11,008 (layer, expert) pairs, 66,048 tensors.
  * MTP/DSpark draft head: ``mtp.{0,1,2}`` — 3 more expert-bearing modules
    with the IDENTICAL record format (score-routed: gate.weight + bias),
    -> 768 (mtp, expert) pairs, 4,608 tensors.  They are not touched by the
    main-model forward pass, but they ARE members of the checkpoint's
    routed-expert universe and are required if the runtime ever executes
    the draft head for speculative decode.  The manifest therefore assigns
    them dense buckets 43..45 so NO ``*.ffn.experts.*`` tensor in the
    checkpoint is unaccounted.
  * 6 tensors per expert: w1/w3/w2 packed FP4 weights (I8) + matching
    F8_E8M0 scales, in DEE4 record order
    (w1.weight|w3.weight|w2.weight|w1.scale|w3.scale|w2.scale).
  * record_bytes = 13,369,344 (12.75 MiB).
      main-only store   = 147,169,738,752 B = 137.0625 GiB (11,008 records)
      full store (46)   = 157,437,394,944 B = 146.625 GiB (11,776 records)
  * main layer L is entirely contained in shard ``model-{L+2:05d}``;
    mtp.N is entirely contained in shard ``model-{46+N:05d}``.

The emitted manifest is deliberately compact: because the record layout and
the bucket->shard binding are fixed, the only variable data is the six
data-section byte ranges per expert, stored as one JSONL row per record.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
N_LAYERS = 43            # main-model layers 0..42 (0..2 hash-routed)
N_HASH_LAYERS = 3
N_MTP = 3                # mtp.0..2 draft-head layers
N_BUCKETS = N_LAYERS + N_MTP   # 46 expert-bearing modules
EXPERTS_PER_LAYER = 256
MAIN_EXPERTS = N_LAYERS * EXPERTS_PER_LAYER        # 11,008
TOTAL_EXPERTS = N_BUCKETS * EXPERTS_PER_LAYER      # 11,776
SHARD_COUNT = 48
FIRST_LAYER_SHARD = 2    # main layer L -> shard index L+2
FIRST_MTP_SHARD = 46     # mtp.N -> shard index 46+N

# DEE4 record component order (matches kaggle/deepseek-v4-flash-0731/
# repack_to_dee4.py:PROJECTIONS/COMPONENTS): gate/w1, up/w3, down/w2 weights,
# then scales in the same order.
PROJECTIONS = ("w1", "w3", "w2")
COMPONENTS = tuple((p, "weight") for p in PROJECTIONS) + tuple(
    (p, "scale") for p in PROJECTIONS
)

EXPERT_NAME_RE = re.compile(
    r"^(?:layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\."
    r"(weight|scale)$"
)
MAIN_NAME_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$"
)
MTP_NAME_RE = re.compile(
    r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$"
)

EXPECTED_TENSOR_SHAPES = {
    ("w1", "weight"): ("I8", [2048, 2048]),
    ("w3", "weight"): ("I8", [2048, 2048]),
    ("w2", "weight"): ("I8", [4096, 1024]),
    ("w1", "scale"): ("F8_E8M0", [2048, 128]),
    ("w3", "scale"): ("F8_E8M0", [2048, 128]),
    ("w2", "scale"): ("F8_E8M0", [4096, 64]),
}
RECORD_BYTES = 13_369_344
MAIN_STORE_BYTES = MAIN_EXPERTS * RECORD_BYTES      # 147,169,738,752
STORE_BYTES = TOTAL_EXPERTS * RECORD_BYTES          # 157,437,394,944


def tensor_name(bucket: int, expert: int, projection: str, kind: str) -> str:
    """Canonical checkpoint tensor name for one component.

    Buckets 0..42 are main-model ``layers.{L}``; buckets 43..45 are
    ``mtp.{bucket-43}``.
    """
    if bucket < N_LAYERS:
        return f"layers.{bucket}.ffn.experts.{expert}.{projection}.{kind}"
    return f"mtp.{bucket - N_LAYERS}.ffn.experts.{expert}.{projection}.{kind}"


def parse_expert_tensor_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    """Deterministic name -> (bucket, expert, projection, kind) mapper.

    ``layers.L.ffn.experts.E.*`` -> bucket L; ``mtp.N.ffn.experts.E.*`` ->
    bucket 43+N.  Returns None for any non-routed-expert tensor name.  This
    is THE mapping: inverse of :func:`tensor_name`, proved bijective over
    the committed headers by :func:`build_manifest`.
    """
    m = MAIN_NAME_RE.match(name)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    m = MTP_NAME_RE.match(name)
    if m:
        return int(m.group(1)) + N_LAYERS, int(m.group(2)), \
            m.group(3), m.group(4)
    return None


def bucket_shard_name(bucket: int) -> str:
    if bucket < N_LAYERS:
        return (f"model-{bucket + FIRST_LAYER_SHARD:05d}-"
                f"of-{SHARD_COUNT:05d}.safetensors")
    return (f"model-{bucket - N_LAYERS + FIRST_MTP_SHARD:05d}-"
            f"of-{SHARD_COUNT:05d}.safetensors")


def bucket_domain(bucket: int) -> str:
    return "main" if bucket < N_LAYERS else "mtp"


def load_committed_headers(headers_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load all 48 committed shard headers (safetensors header JSON dicts)."""
    headers_dir = Path(headers_dir)
    headers: dict[str, dict[str, Any]] = {}
    for i in range(1, SHARD_COUNT + 1):
        shard = f"model-{i:05d}-of-{SHARD_COUNT:05d}.safetensors"
        path = headers_dir / f"{shard}.json"
        if not path.is_file():
            raise FileNotFoundError(f"committed shard header missing: {path}")
        headers[shard] = json.loads(path.read_text(encoding="utf-8"))
    return headers


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def universe_sha256(n_buckets: int = N_BUCKETS) -> str:
    """SHA-256 over the canonical full-universe pair list.

    Same convention as repack_to_dee4._selection_sha256 but for the complete
    (bucket, expert) universe instead of a trace-selected subset.
    """
    pairs = [
        [bucket, expert]
        for bucket in range(n_buckets)
        for expert in range(EXPERTS_PER_LAYER)
    ]
    return hashlib.sha256(canonical_json_bytes(pairs)).hexdigest()


def iter_record_ranges(
    headers: dict[str, dict[str, Any]], bucket: int, expert: int
) -> Iterator[dict[str, Any]]:
    """Yield the six component range dicts for one record, in record order."""
    shard = bucket_shard_name(bucket)
    header = headers[shard]
    record_offset = 0
    for projection, kind in COMPONENTS:
        name = tensor_name(bucket, expert, projection, kind)
        meta = header.get(name)
        if meta is None:
            raise KeyError(f"routed tensor missing from committed header: {name}")
        expected_dtype, expected_shape = EXPECTED_TENSOR_SHAPES[(projection, kind)]
        dtype = str(meta["dtype"])
        shape = [int(d) for d in meta["shape"]]
        if dtype != expected_dtype or shape != expected_shape:
            raise ValueError(
                f"{name}: layout {(dtype, shape)} != "
                f"{(expected_dtype, expected_shape)}"
            )
        start, end = (int(v) for v in meta["data_offsets"])
        nbytes = end - start
        if nbytes != shape[0] * shape[1]:
            raise ValueError(f"{name}: byte length {nbytes} != shape product")
        yield {
            "tensor": name,
            "shard": shard,
            "data_offset": start,   # offset within the shard data section
            "nbytes": nbytes,
            "record_offset": record_offset,
            "dtype": dtype,
            "shape": shape,
        }
        record_offset += nbytes


def build_manifest(
    headers_dir: Path | str, *, n_buckets: int = N_BUCKETS
) -> dict[str, Any]:
    """Build + self-verify the full-universe manifest.

    Proves, against the committed headers alone:
      * every (bucket, expert) pair in the universe resolves all six tensors;
      * every routed-expert tensor in the headers belongs to exactly one pair
        (bijectivity -- no orphans, no duplicates, no unparsable names);
      * every pair's six tensors live in the bucket's single home shard;
      * per-record and total byte counts match the validated ledger exactly.

    ``n_buckets=43`` builds the main-model-only universe (11,008 records);
    the default 46 additionally covers the three mtp draft layers
    (11,776 records) so every routed tensor in the checkpoint is owned.
    """
    if n_buckets not in (N_LAYERS, N_BUCKETS):
        raise ValueError(f"n_buckets must be {N_LAYERS} or {N_BUCKETS}")
    headers = load_committed_headers(headers_dir)

    # Index every routed-expert tensor name present in the headers.
    seen: dict[tuple[int, int, str, str], str] = {}
    for shard, header in headers.items():
        for name in header:
            if name == "__metadata__":
                continue
            parsed = parse_expert_tensor_name(name)
            if parsed is None:
                if ".ffn.experts." in name:
                    raise ValueError(f"unparsable routed tensor name: {name}")
                continue
            if parsed in seen:
                raise ValueError(f"duplicate routed tensor name: {name}")
            seen[parsed] = shard

    expected_names = n_buckets * EXPERTS_PER_LAYER * len(COMPONENTS)
    if n_buckets == N_BUCKETS and len(seen) != expected_names:
        raise ValueError(
            f"routed tensor count {len(seen)} != {expected_names}")
    # For a main-only manifest, mtp tensors are still seen but excluded from
    # coverage; they are counted for the report.
    in_scope = {k: v for k, v in seen.items() if k[0] < n_buckets}
    if len(in_scope) != expected_names:
        raise ValueError(
            f"in-scope routed tensor count {len(in_scope)} != "
            f"{expected_names}")

    records: list[dict[str, Any]] = []
    total_bytes = 0
    for bucket in range(n_buckets):
        shard = bucket_shard_name(bucket)
        bucket_experts = sorted(
            {e for (b, e, _p, _k) in in_scope if b == bucket}
        )
        if bucket_experts != list(range(EXPERTS_PER_LAYER)):
            missing = sorted(set(range(EXPERTS_PER_LAYER)) -
                             set(bucket_experts))
            raise ValueError(
                f"bucket {bucket} missing expert ids {missing[:8]}")
        for expert in range(EXPERTS_PER_LAYER):
            ranges = list(iter_record_ranges(headers, bucket, expert))
            if sum(r["nbytes"] for r in ranges) != RECORD_BYTES:
                raise ValueError(
                    f"record ({bucket},{expert}) does not total "
                    f"{RECORD_BYTES}")
            for r in ranges:
                if r["shard"] != shard:
                    raise ValueError(
                        f"record ({bucket},{expert}) escapes home shard "
                        f"{shard}")
            records.append({
                "record_index": bucket * EXPERTS_PER_LAYER + expert,
                "bucket": bucket,
                "domain": bucket_domain(bucket),
                "layer": bucket if bucket < N_LAYERS else bucket - N_LAYERS,
                "expert": expert,
                "shard": shard,
                "record_bytes": RECORD_BYTES,
                "ranges": [
                    [r["tensor"].rsplit(".", 2)[-2] + "."
                     + r["tensor"].rsplit(".", 1)[-1],
                     r["data_offset"], r["nbytes"], r["record_offset"],
                     r["tensor"]]
                    for r in ranges
                ],
            })
            total_bytes += RECORD_BYTES
    store_bytes = n_buckets * EXPERTS_PER_LAYER * RECORD_BYTES
    if total_bytes != store_bytes:
        raise ValueError(f"store bytes {total_bytes} != {store_bytes}")

    layer_bytes = EXPERTS_PER_LAYER * RECORD_BYTES
    manifest = {
        "schema": "dee4-p3-full-universe-v1",
        "model": MODEL,
        "revision": REVISION,
        "n_layers": N_LAYERS,
        "n_hash_layers": N_HASH_LAYERS,
        "n_mtp_buckets": n_buckets - N_LAYERS,
        "n_buckets": n_buckets,
        "experts_per_layer": EXPERTS_PER_LAYER,
        "total_experts": n_buckets * EXPERTS_PER_LAYER,
        "record_bytes": RECORD_BYTES,
        "store_bytes": store_bytes,
        "store_gib": round(store_bytes / (1 << 30), 4),
        "layer_bytes": layer_bytes,
        "component_order": [f"{p}.{k}" for p, k in COMPONENTS],
        "components": [
            {
                "component": f"{p}.{k}",
                "dtype": EXPECTED_TENSOR_SHAPES[(p, k)][0],
                "shape": EXPECTED_TENSOR_SHAPES[(p, k)][1],
                "nbytes": EXPECTED_TENSOR_SHAPES[(p, k)][1][0]
                * EXPECTED_TENSOR_SHAPES[(p, k)][1][1],
                "record_offset": off,
            }
            for off, (p, k) in zip(
                _component_record_offsets(), COMPONENTS
            )
        ],
        "universe_sha256": universe_sha256(n_buckets),
        "bucket_shards": {
            str(bucket): {
                "shard": bucket_shard_name(bucket),
                "domain": bucket_domain(bucket),
            }
            for bucket in range(n_buckets)
        },
        "non_expert_shards": {
            "model-00001-of-00048.safetensors": "embed.weight",
            "model-00045-of-00048.safetensors":
                "norm.weight, head.weight, hc_head_{fn,base,scale}",
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    manifest["_records"] = records  # caller splits this out to JSONL
    return manifest


def _component_record_offsets() -> list[int]:
    offsets = []
    off = 0
    for p, k in COMPONENTS:
        offsets.append(off)
        shape = EXPECTED_TENSOR_SHAPES[(p, k)][1]
        off += shape[0] * shape[1]
    return offsets


def write_manifest(manifest: dict[str, Any], out_dir: Path | str) -> dict[str, str]:
    """Write manifest JSON + records JSONL; returns written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = manifest.pop("_records")
    records_path = out_dir / "p3_records.jsonl"
    records_sha = hashlib.sha256()
    with records_path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            line = json.dumps(rec, separators=(",", ":")) + "\n"
            records_sha.update(line.encode("utf-8"))
            fh.write(line)
    manifest["records_file"] = "p3_records.jsonl"
    manifest["records_sha256"] = records_sha.hexdigest()
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        )
    ).hexdigest()
    manifest_path = out_dir / "p3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), "utf-8")
    return {
        "manifest": str(manifest_path),
        "records": str(records_path),
    }


def load_manifest(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_records(path: Path | str) -> list[dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
