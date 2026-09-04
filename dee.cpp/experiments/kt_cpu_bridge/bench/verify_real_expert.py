"""Offline real-weight correctness, NOT native KT or sealed-activation replay.

Reads exactly one official expert from a local (possibly partial) safetensors
shard. The six tensors AND reconstructed canonical DEE4 record must match a
sealed integrity ledger. No downloads, cache mutation, or scheduler imports.
The route journal proves expert selection only: it contains no activation or
routing-weight values. Input rows and weights are explicitly synthetic probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import torch

BRIDGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE / "python"))
sys.path.insert(0, str(BRIDGE.parents[1]))
from kt_cpu_bridge.reference import expert_forward_reference, kt_emulated_forward
from scripts import deepseek_v4_contract as contract
from scripts import deepseek_v4_expert_reference as trusted

COMPONENTS = ("w1.weight", "w3.weight", "w2.weight", "w1.scale", "w3.scale", "w2.scale")
REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_real_expert(shard: Path, bundle: Path, seal_path: Path, layer: int, expert: int):
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    verified = {}
    for name in ("dee4-metadata.json", "dee4-integrity.jsonl", "routed_experts.jsonl"):
        raw = (bundle / name).read_bytes()
        if sha(raw) != seal["raw_sha256"][name]:
            raise ValueError(f"seal mismatch: {name}")
        verified[name] = raw
    meta = json.loads(verified["dee4-metadata.json"])
    if (meta["source_repository"], meta["source_revision"]) != (REPOSITORY, REVISION):
        raise ValueError("wrong official checkpoint identity")
    if meta["codec"] != "deepseek-fp4-e2m1-e8m0" or meta["group_size"] != 32:
        raise ValueError("unsupported canonical codec")
    entries = [json.loads(line) for line in verified["dee4-integrity.jsonl"].splitlines()]
    selected = [r for r in entries if (r["layer"], r["expert"]) == (layer, expert)]
    if len(selected) != 1:
        raise ValueError("expert missing or duplicated in canonical ledger")
    entry = selected[0]
    routes = [json.loads(line) for line in verified["routed_experts.jsonl"].splitlines()]
    occurrences = [(r["forward_step"], row, rank)
                   for r in routes if r["layer"] == layer
                   for row, ids in enumerate(r["expert_ids_rank_order"])
                   for rank, eid in enumerate(ids) if eid == expert]
    if not occurrences:
        raise ValueError("expert absent from sealed route journal")
    components = {f'{c["projection"]}.{c["kind"]}': c for c in meta["components"]}
    tensors, chunks = {}, []
    with shard.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("missing shard header")
        length, = struct.unpack("<Q", prefix)
        if not 2 <= length <= 16 * 1024 * 1024:
            raise ValueError("unbounded shard header")
        header = json.loads(f.read(length))
        for name in COMPONENTS:
            c = components[name]
            expected_shape = ([4096, 1024] if name == "w2.weight" else
                              [2048, 2048] if name.endswith("weight") else
                              [4096, 64] if name == "w2.scale" else [2048, 128])
            expected_dtype = "I8" if name.endswith("weight") else "F8_E8M0"
            expected_bytes = 4194304 if name.endswith("weight") else 262144
            if (c["shape"], c["dtype"], c["nbytes"], c["offset"]) != (
                    expected_shape, expected_dtype, expected_bytes, sum(map(len, chunks))):
                raise ValueError(f"unsupported canonical component: {name}")
            tensor = header[f"layers.{layer}.ffn.experts.{expert}.{name}"]
            start, end = tensor["data_offsets"]
            if tensor["shape"] != c["shape"] or tensor["dtype"] != c["dtype"]:
                raise ValueError(f"shape/dtype mismatch: {name}")
            if start < 0 or end - start != c["nbytes"]:
                raise ValueError(f"offset mismatch: {name}")
            f.seek(8 + length + start)
            raw = f.read(c["nbytes"])
            if len(raw) != c["nbytes"] or sha(raw) != entry["component_sha256"][name]:
                raise ValueError(f"canonical bytes mismatch: {name}")
            chunks.append(raw)
            tensors[name] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(c["shape"])
    record = b"".join(chunks)
    if len(record) != entry["record_bytes"] or sha(record) != entry["record_sha256"]:
        raise ValueError("canonical record mismatch")
    return tensors, record, {"source_repository": REPOSITORY, "source_revision": REVISION,
        "layer": layer, "expert": expert, "record_sha256": sha(record),
        "record_bytes": len(record), "component_sha256": entry["component_sha256"],
        "seal_sha256": sha(seal_path.read_bytes()),
        "sealed_route_occurrences": occurrences,
        "verified_artifacts": {k: sha(v) for k, v in verified.items()}}


def verify(shard: Path, bundle: Path, seal: Path, executor: Path, layer=0, expert=155):
    tensors, record, evidence = load_real_expert(shard, bundle, seal, layer, expert)
    args = [tensors[name] for name in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")]
    torch.set_num_threads(1)
    x = torch.randn(2, 4096, generator=torch.Generator().manual_seed(20260903))
    x[1] *= 3  # also exercise clamp-active inputs
    weights = (1.0, 0.37)
    evidence.update({"scope": "real official weights; synthetic input rows and routing weights",
        "native_kt_executed": False, "sealed_activation_replay": False,
        "input_sha256": sha(x.numpy().tobytes()), "routing_weights": weights,
        "torch_version": torch.__version__, "executor_sha256": sha(executor.read_bytes()),
        "strict_reference_tolerance": {"atol": 1e-5, "rtol": 1e-4}, "rows": []})
    evidence["observed_scale_ranges"] = {name: [int(t.min()), int(t.max())]
                                         for name, t in tensors.items() if name.endswith("scale")}
    with tempfile.TemporaryDirectory(prefix="dee-kt-real-expert-") as scratch:
        root = Path(scratch)
        (root / "record.bin").write_bytes(record)
        w1, w2, w3 = [trusted.dequantize_expert_weight(tensors[f"{p}.weight"], tensors[f"{p}.scale"])
                      for p in ("w1", "w2", "w3")]
        for i, weight in enumerate(weights):
            row = x[i:i+1]
            gate = torch.clamp(row @ w1.t(), max=10.)
            up = torch.clamp(row @ w3.t(), min=-10., max=10.)
            expected = ((torch.nn.functional.silu(gate) * up) * weight) @ w2.t()
            if weight == 1.0:
                if not torch.equal(expected, trusted.expert_forward(row, *args)):
                    raise AssertionError("trusted weighted reference mismatch")
            python_ref = expert_forward_reference(row, *args, routing_weight=weight)
            emulated = kt_emulated_forward(row, *args, routing_weight=weight)
            (root / "x.bin").write_bytes(row.numpy().tobytes())
            subprocess.run([str(executor), str(root / "record.bin"), str(root / "x.bin"),
                            str(root / "out.bin"), str(weight)], check=True, timeout=120)
            out_bytes = (root / "out.bin").read_bytes()
            if len(out_bytes) != 4096 * 4:
                raise ValueError("wrong executor output size")
            got = torch.frombuffer(bytearray(out_bytes), dtype=torch.float32).reshape(1, 4096)
            metrics = contract.compute_ds8_metrics(expected, got)
            kt_metrics = contract.compute_ds8_metrics(expected, emulated)
            cpp_gate = contract.ds8_gate_report(metrics)
            kt_gate = contract.ds8_gate_report(kt_metrics)
            evidence["rows"].append({"row": i, "python_reference_bitwise": torch.equal(expected, python_ref),
                "cpp_reference_allclose": torch.allclose(expected, got, atol=1e-5, rtol=1e-4),
                "sample_validity": cpp_gate["sample_validity"],
                "cpp_reference_candidate_fidelity": cpp_gate["candidate_fidelity"],
                "kt_emulated_candidate_fidelity": kt_gate["candidate_fidelity"],
                "cpp_reference_ds8_pass": cpp_gate["ds8_gate_passed"], "cpp_reference_metrics": metrics,
                "kt_emulated_ds8_pass": kt_gate["ds8_gate_passed"], "kt_emulated_metrics": kt_metrics})
    evidence["reference_pass"] = all(r["python_reference_bitwise"] and r["cpp_reference_allclose"]
                                      and r["cpp_reference_ds8_pass"] for r in evidence["rows"])
    evidence["unchanged_dee_contract_pass"] = all(r["cpp_reference_ds8_pass"] for r in evidence["rows"])
    evidence["kt_emulated_contract_pass"] = all(r["kt_emulated_ds8_pass"] for r in evidence["rows"])
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shard", "bundle", "seal", "executor", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=155)
    options = vars(parser.parse_args())
    out = options.pop("out")
    report = verify(**options)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"reference_pass": report["reference_pass"], "report": str(out)}))
    raise SystemExit(0 if report["reference_pass"] else 1)
