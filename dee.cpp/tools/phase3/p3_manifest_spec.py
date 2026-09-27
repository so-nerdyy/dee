"""Phase 3: spec-driven full-universe manifest builder.

``p3_manifest.build_manifest`` is sealed to the pinned DSv4-Flash checkpoint:
its name regexes, shard bijection, and expected dtype table are hardcoded.
This module generalizes the same artifact schema (``dee4-p3-full-universe-v1``
manifest + records JSONL) to any checkpoint described by a JSON spec —
needed for portability targets whose tensor names and shard layouts differ:

  * tensor names come from a ``tensor_pattern`` template (``{layer}``,
    ``{expert}``, ``{proj}``, ``{kind}`` placeholders), e.g. MiMo-V2.6's
    ``model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}`` where
    ``{proj}`` = ``gate_proj``/``up_proj``/``down_proj`` and ``{kind}`` =
    ``weight``/``weight_scale``.
  * no bucket->shard bijection is assumed: EP-sharded checkpoints split a
    layer's experts across files, so every emitted record range carries its
    own ``shard`` (6-element ranges; p3_builder prefers the per-range shard).
  * ``expert_name_re`` is the coverage oracle: every checkpoint tensor name
    matching ``coverage_re`` MUST parse, and every parsed component must be
    owned by exactly one record (bijectivity, same contract as the sealed
    DSv4 manifest).

The emitted record byte layout is identical to DSv4's DEE4 order
(w1.w|w3.w|w2.w|w1.s|w3.s|w2.s); ``projections`` maps the model's native
projection names onto the w1/w3/w2 slots.  The ``codec`` label describes
physical record semantics (e.g. mxfp4 e2m1+e8m0 group-32 is wire-identical
to ``deepseek-fp4-e2m1-e8m0``), NOT the source model name.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import p3_manifest

# DEE4 record component order — fixed by the native ExpertView layout.
# (fp4/mxfp4: three packed weights + three scale tensors.)
COMPONENTS = (
    ("w1", "weight"), ("w3", "weight"), ("w2", "weight"),
    ("w1", "scale"), ("w3", "scale"), ("w2", "scale"),
)
# Scale-free codecs (e.g. raw bf16 expert matrices): the record is just the
# three weight tensors — engine interprets by codec label.
COMPONENTS_NO_SCALE = (
    ("w1", "weight"), ("w3", "weight"), ("w2", "weight"),
)


def load_spec(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_headers_dir(headers_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load every committed/fetched shard header JSON in a directory.

    Unlike ``p3_manifest.load_committed_headers`` (which enumerates DSv4's
    ``model-NNNNN-of-NNNNN`` naming), spec mode accepts arbitrary shard
    file names — every ``*.json`` file whose stem names a safetensors
    shard is one header (``<shard-filename>.json`` as produced by
    ``fetch_headers.py`` or the committed DSv4 header dump).
    """
    headers_dir = Path(headers_dir)
    headers: dict[str, dict[str, Any]] = {}
    for path in sorted(headers_dir.glob("*.json")):
        headers[path.name[: -len(".json")]] = json.loads(
            path.read_text(encoding="utf-8"))
    if not headers:
        raise FileNotFoundError(
            f"no shard header JSONs under {headers_dir}")
    return headers


def _bucket_layer_map(spec: dict[str, Any]) -> list[tuple[int, int]]:
    """Return ordered (bucket, layer) pairs from the spec.

    ``buckets`` is either an explicit list of {bucket, layer, domain} dicts
    or generated from ``n_buckets`` + ``layer_start`` (bucket i maps to
    checkpoint layer ``layer_start + i``, all domain "main").
    """
    if "buckets" in spec:
        out = []
        for entry in spec["buckets"]:
            out.append((int(entry["bucket"]), int(entry["layer"])))
        return sorted(out)
    n = int(spec["n_buckets"])
    start = int(spec.get("layer_start", 0))
    return [(i, start + i) for i in range(n)]


def build_manifest_spec(
    spec: dict[str, Any], headers_dir: Path | str
) -> dict[str, Any]:
    """Build + self-verify a full-universe manifest from a model spec.

    Proves against the headers alone:
      * every routed-expert header name matching ``coverage_re`` parses via
        ``expert_name_re`` (no orphans/unparsable routed tensors);
      * every (bucket, expert) pair resolves all six components;
      * per-record and total byte counts match the spec's ledger;
      * component dtype/shape match the spec's ``expected`` table.
    """
    model = spec["model"]
    revision = spec["revision"]
    experts_per_layer = int(spec["experts_per_layer"])
    record_bytes = int(spec["record_bytes"])
    projections = spec["projections"]          # {"w1": "gate_proj", ...}
    kind_field = spec["kind_field"]            # {"weight": "weight", "scale": "weight_scale"}
    tensor_pattern = spec["tensor_pattern"]    # "...{layer}...{expert}...{proj}.{kind}"
    expected = spec["expected"]                # "w1.weight": {dtype, shape}
    components = COMPONENTS_NO_SCALE if spec.get("no_scales") else COMPONENTS
    expert_re = re.compile(spec["expert_name_re"])
    coverage_re = re.compile(spec["coverage_re"])
    # regex groups: (layer, expert, proj_name, kind_name)
    g_layer, g_expert, g_proj, g_kind = (
        int(spec.get("group_layer", 1)), int(spec.get("group_expert", 2)),
        int(spec.get("group_proj", 3)), int(spec.get("group_kind", 4)))
    layer2bucket = {layer: bucket
                    for bucket, layer in _bucket_layer_map(spec)}
    bucket2layer = {bucket: layer
                    for bucket, layer in _bucket_layer_map(spec)}
    n_buckets = len(layer2bucket)

    headers = load_headers_dir(headers_dir)

    # coverage: every routed-expert name must parse into the universe
    seen: dict[tuple[int, int, str, str], tuple[str, dict[str, Any]]] = {}
    for shard, header in headers.items():
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            m = expert_re.match(name)
            if m is None:
                if coverage_re.search(name):
                    raise ValueError(
                        f"unparsable routed tensor name: {name}")
                continue
            layer = int(m.group(g_layer))
            if layer not in layer2bucket:
                raise ValueError(
                    f"routed tensor on unmapped layer {layer}: {name}")
            key = (layer2bucket[layer], int(m.group(g_expert)),
                   m.group(g_proj), m.group(g_kind))
            if key in seen:
                raise ValueError(f"duplicate routed tensor name: {name}")
            seen[key] = (shard, meta)

    def _tensor_name(bucket: int, expert: int, proj: str, kind: str) -> str:
        return tensor_pattern.format(
            layer=bucket2layer[bucket],
            expert=expert, proj=projections[proj], kind=kind_field[kind])

    records: list[dict[str, Any]] = []
    total_bytes = 0
    for bucket, layer in _bucket_layer_map(spec):
        bucket_shards: set[str] = set()
        expert_ids = sorted(
            {e for (b, e, _p, _k) in seen if b == bucket})
        if expert_ids != list(range(experts_per_layer)):
            missing = sorted(set(range(experts_per_layer)) - set(expert_ids))
            raise ValueError(
                f"bucket {bucket} (layer {layer}) missing expert ids "
                f"{missing[:8]} (have {len(expert_ids)})")
        for expert in range(experts_per_layer):
            ranges = []
            record_offset = 0
            record_shards: set[str] = set()
            for proj, kind in components:
                key = (bucket, expert, projections[proj], kind_field[kind])
                if key not in seen:
                    raise KeyError(
                        f"routed component missing: bucket {bucket} "
                        f"expert {expert} {proj}.{kind}")
                shard, meta = seen[key]
                exp = expected[f"{proj}.{kind}"]
                dtype = str(meta["dtype"])
                shape = [int(d) for d in meta["shape"]]
                if dtype != exp["dtype"] or shape != exp["shape"]:
                    raise ValueError(
                        f"bucket {bucket} expert {expert} {proj}.{kind}: "
                        f"layout {(dtype, shape)} != "
                        f"{(exp['dtype'], exp['shape'])}")
                start, end = (int(v) for v in meta["data_offsets"])
                nbytes = end - start
                if nbytes != exp["nbytes"]:
                    raise ValueError(
                        f"bucket {bucket} expert {expert} {proj}.{kind}: "
                        f"{nbytes} bytes != {exp['nbytes']}")
                ranges.append([f"{proj}.{kind}", start, nbytes,
                               record_offset,
                               _tensor_name(bucket, expert, proj, kind),
                               shard])
                record_offset += nbytes
                record_shards.add(shard)
                bucket_shards.add(shard)
            if record_offset != record_bytes:
                raise ValueError(
                    f"record ({bucket},{expert}) totals {record_offset} "
                    f"!= {record_bytes}")
            records.append({
                "record_index": bucket * experts_per_layer + expert,
                "bucket": bucket,
                "domain": "main",
                "layer": layer,
                "expert": expert,
                "shard": sorted(record_shards)[0] if len(
                    record_shards) == 1 else "multi",
                "source_shards": sorted(record_shards),
                "record_bytes": record_bytes,
                "ranges": ranges,
            })
            total_bytes += record_bytes
        spec.setdefault("_bucket_shards", {})[str(bucket)] = {
            "shards": sorted(bucket_shards), "domain": "main"}

    store_bytes = n_buckets * experts_per_layer * record_bytes
    if total_bytes != store_bytes:
        raise ValueError(f"store bytes {total_bytes} != {store_bytes}")

    manifest = {
        "schema": "dee4-p3-full-universe-v1",
        "model": model,
        "revision": revision,
        "codec": spec.get("codec", "deepseek-fp4-e2m1-e8m0"),
        "n_layers": int(spec.get("n_layers", n_buckets)),
        "n_hash_layers": int(spec.get("n_hash_layers", 0)),
        "n_mtp_buckets": int(spec.get("n_mtp_buckets", 0)),
        "n_buckets": n_buckets,
        "experts_per_layer": experts_per_layer,
        "total_experts": n_buckets * experts_per_layer,
        "record_bytes": record_bytes,
        "store_bytes": store_bytes,
        "store_gib": round(store_bytes / (1 << 30), 4),
        "layer_bytes": experts_per_layer * record_bytes,
        "component_order": [f"{p}.{k}" for p, k in components],
        "components": [
            {
                "component": f"{p}.{k}",
                "dtype": expected[f"{p}.{k}"]["dtype"],
                "shape": expected[f"{p}.{k}"]["shape"],
                "nbytes": expected[f"{p}.{k}"]["nbytes"],
                "record_offset": off,
            }
            for off, (p, k) in zip(
                _component_record_offsets(expected, components), components)
        ],
        "universe_sha256": _universe_sha256(n_buckets, experts_per_layer),
        "bucket_shards": spec.pop("_bucket_shards", {}),
        "spec": {k: v for k, v in spec.items()
                 if not k.startswith("_")},
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        p3_manifest.canonical_json_bytes(manifest)).hexdigest()
    manifest["_records"] = records
    return manifest


def _universe_sha256(n_buckets: int, experts_per_layer: int) -> str:
    pairs = [[b, e] for b in range(n_buckets)
             for e in range(experts_per_layer)]
    return hashlib.sha256(
        p3_manifest.canonical_json_bytes(pairs)).hexdigest()


def _component_record_offsets(expected: dict[str, Any],
                              components=COMPONENTS) -> list[int]:
    offsets = []
    off = 0
    for p, k in components:
        offsets.append(off)
        off += int(expected[f"{p}.{k}"]["nbytes"])
    return offsets
