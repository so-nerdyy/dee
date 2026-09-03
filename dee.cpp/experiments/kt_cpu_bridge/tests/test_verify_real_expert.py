"""Offline loader failure gates. These deliberately use synthetic bytes."""
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from verify_real_expert import COMPONENTS, REPOSITORY, REVISION, load_real_expert, sha


@pytest.fixture
def fixture(tmp_path):
    header, components, chunks = {}, [], []
    for name in COMPONENTS:
        projection, kind = name.split(".")
        shape = ([4096, 1024] if name == "w2.weight" else [2048, 2048] if kind == "weight"
                 else [4096, 64] if projection == "w2" else [2048, 128])
        n = shape[0] * shape[1]
        start = sum(map(len, chunks))
        raw = bytes(n) if kind == "weight" else bytes([127]) * n
        chunks.append(raw)
        dtype = "I8" if kind == "weight" else "F8_E8M0"
        components.append(dict(projection=projection, kind=kind, shape=shape, dtype=dtype,
                               offset=start, nbytes=n))
        header[f"layers.0.ffn.experts.155.{name}"] = dict(shape=shape, dtype=dtype, data_offsets=[start, start+n])
    raw_header = json.dumps(header).encode()
    record = b"".join(chunks)
    shard = tmp_path / "synthetic.safetensors"
    shard.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + record)
    metadata = dict(source_repository=REPOSITORY, source_revision=REVISION,
                    codec="deepseek-fp4-e2m1-e8m0", group_size=32, components=components)
    entry = dict(layer=0, expert=155, record_bytes=len(record), record_sha256=sha(record),
                 component_sha256={n: sha(b) for n, b in zip(COMPONENTS, chunks)})
    artifacts = {"dee4-metadata.json": metadata, "dee4-integrity.jsonl": entry,
                 "routed_experts.jsonl": dict(layer=0, forward_step=0, expert_ids_rank_order=[[155]])}
    seal = tmp_path / "seal.json"

    def save():
        hashes = {}
        for name, data in artifacts.items():
            raw = json.dumps(data).encode()
            (tmp_path / name).write_bytes(raw)
            hashes[name] = sha(raw)
        seal.write_text(json.dumps({"raw_sha256": hashes}))
    save()
    return shard, tmp_path, seal, artifacts, save


def test_canonical_record_reconstructed(fixture):
    shard, bundle, seal, _, _ = fixture
    tensors, record, evidence = load_real_expert(shard, bundle, seal, 0, 155)
    assert len(record) == 13369344
    assert len(tensors) == 6
    assert evidence["sealed_route_occurrences"] == [(0, 0, 0)]


@pytest.mark.parametrize("damage", ["seal", "component", "record", "identity", "route", "shape"])
def test_corruption_fails_closed(fixture, damage):
    shard, bundle, seal, artifacts, save = fixture
    if damage == "seal":
        (bundle / "dee4-metadata.json").write_text("{}")
    elif damage == "component":
        with shard.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"x")
    else:
        if damage == "record":
            artifacts["dee4-integrity.jsonl"]["record_sha256"] = "0" * 64
        elif damage == "identity":
            artifacts["dee4-metadata.json"]["source_revision"] = "not-official"
        elif damage == "route":
            artifacts["routed_experts.jsonl"]["expert_ids_rank_order"] = [[1]]
        elif damage == "shape":
            artifacts["dee4-metadata.json"]["components"][0]["shape"] = [1, 1]
        save()
    with pytest.raises(ValueError):
        load_real_expert(shard, bundle, seal, 0, 155)
