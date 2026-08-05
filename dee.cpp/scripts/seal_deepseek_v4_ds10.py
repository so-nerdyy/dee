#!/usr/bin/env python3
"""Validate downloaded DS10 evidence and emit a host-side seal.

The Kaggle terminal status and harness verdict are inputs, not proof.  This
validator independently checks every manifest row, the archive and its safe
member names, pinned source identity, copied source hashes, and the exact gate
set required by the recorded DS10 stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
EXPECTED_MODEL_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
EXPECTED_TENSORS = 72_317
EXPECTED_SHARDS = 48
EXPECTED_LAYERS = 43
MODULE_FILES = {
    "cache": "deepseek_v4_module_cache.py",
    "encoding": "deepseek_v4_module_encoding.py",
    "expert_reference": "deepseek_v4_module_expert_reference.py",
    "layer_candidate": "deepseek_v4_module_layer_candidate.py",
    "layer_common": "deepseek_v4_module_layer_common.py",
    "layer_reference": "deepseek_v4_module_layer_reference.py",
    "model": "deepseek_v4_module_model.py",
    "moe_reference": "deepseek_v4_module_moe_reference.py",
    "support": "deepseek_v4_module_support.py",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def discover_evidence(input_dir: Path) -> Path:
    matches = sorted({path.parent for path in input_dir.rglob("ds10-evidence.json")})
    require(len(matches) == 1,
            f"expected exactly one DS10 evidence directory, found {len(matches)}")
    return matches[0]


def validate_manifest(evidence_dir: Path) -> dict[str, Any]:
    path = evidence_dir / "ds10-artifact-manifest.json"
    manifest = read_json(path)
    rows = manifest.get("artifacts")
    require(isinstance(rows, list) and rows, "artifact manifest has no rows")
    seen: set[str] = set()
    for row in rows:
        require(isinstance(row, dict), "invalid artifact manifest row")
        relative = row.get("path")
        require(isinstance(relative, str) and relative not in seen,
                f"invalid or duplicate artifact path {relative!r}")
        pure = PurePosixPath(relative)
        require(not pure.is_absolute() and ".." not in pure.parts,
                f"unsafe artifact path {relative!r}")
        seen.add(relative)
        artifact = evidence_dir / Path(*pure.parts)
        require(artifact.is_file(), f"manifest artifact missing: {relative}")
        require(artifact.stat().st_size == row.get("bytes"),
                f"manifest size mismatch: {relative}")
        require(sha256_file(artifact) == row.get("sha256"),
                f"manifest hash mismatch: {relative}")
    required = manifest.get("required_paths")
    require(isinstance(required, list) and required,
            "manifest required_paths missing")
    for relative in required:
        require(relative in seen, f"required artifact not indexed: {relative}")
    return {"path": path.name, "sha256": sha256_file(path),
            "artifact_count": len(rows), "manifest": manifest}


def validate_identity(evidence_dir: Path, evidence: dict[str, Any],
                      expected_commit: str | None) -> dict[str, Any]:
    identity = read_json(evidence_dir / "harness-identity-ds10.json")
    require(identity == evidence.get("identity"),
            "copied identity differs from evidence identity")
    require(identity.get("model") == EXPECTED_MODEL, "model identity mismatch")
    require(identity.get("model_revision") == EXPECTED_MODEL_REVISION,
            "model revision mismatch")
    commit = identity.get("repository_commit")
    require(isinstance(commit, str) and len(commit) == 40,
            "repository commit is not a full SHA")
    if expected_commit is not None:
        require(commit == expected_commit, "repository commit mismatch")
    harness = evidence_dir / "deepseek_v4_model_runtime.py"
    require(sha256_file(harness) == identity.get("harness_sha256"),
            "copied harness hash mismatch")
    module_hashes = identity.get("module_sha256")
    require(isinstance(module_hashes, dict), "module identity map missing")
    for key, filename in MODULE_FILES.items():
        require(sha256_file(evidence_dir / filename) == module_hashes.get(key),
                f"copied module hash mismatch: {key}")
    return {"repository_commit": commit, "stage": identity.get("stage"),
            "harness_sha256": identity.get("harness_sha256"),
            "module_count": len(MODULE_FILES)}


def validate_archive(input_dir: Path, evidence_dir: Path,
                     manifest_sha256: str) -> dict[str, Any]:
    metadata = read_json(evidence_dir / "archive-metadata.json")
    require(metadata.get("manifest_sha256") == manifest_sha256,
            "archive metadata manifest hash mismatch")
    archive = input_dir / Path(str(metadata.get("archive"))).name
    require(archive.is_file(), f"archive missing: {archive.name}")
    require(sha256_file(archive) == metadata.get("archive_sha256"),
            "archive SHA256 mismatch")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    require(members, "archive is empty")
    prefix = evidence_dir.name
    names: list[str] = []
    for member in members:
        pure = PurePosixPath(member.name)
        require(not pure.is_absolute() and ".." not in pure.parts,
                f"unsafe archive member {member.name!r}")
        require(pure.parts and pure.parts[0] == prefix,
                f"archive member outside evidence root: {member.name}")
        names.append(member.name)
    require(f"{prefix}/ds10-artifact-manifest.json" in names,
            "archive does not contain artifact manifest")
    require(f"{prefix}/archive-metadata.json" not in names,
            "archive unexpectedly contains post-archive metadata")
    return {"path": archive.name, "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive), "member_count": len(members)}


def validate_layer_trace(trace: Any, expected: list[int]) -> bool:
    if not isinstance(trace, list) or len(trace) != len(expected):
        return False
    return all(row.get("layer") == layer_id
               and row.get("order") == order
               and row.get("finite") is True
               and row.get("selected_experts") is not None
               for order, (row, layer_id) in enumerate(zip(trace, expected)))


def validate_runtime(runtime: Any) -> bool:
    if not isinstance(runtime, dict):
        return False
    backends = runtime.get("backends", {})
    caches = [runtime.get("cache0", {}), runtime.get("cache1", {})]
    return (backends.get("cpu_expert_execution") is False
            and backends.get("routed_experts") == "freebuff_ds8_cache_fp16_cuda"
            and all(cache.get("resident_bytes", 1) <= 2 << 30 for cache in caches)
            and all(cache.get("fallbacks") == 0 for cache in caches))


def stage_gates(evidence: dict[str, Any], expected_stage: str) -> dict[str, bool]:
    require(evidence.get("stage") == expected_stage, "evidence stage mismatch")
    gates = evidence.get("gates")
    require(isinstance(gates, dict), "evidence gates missing")
    common = {
        "dual_t4": evidence.get("cuda_available") is True
        and evidence.get("gpu_count") == 2,
        "performance_not_comparable": evidence.get("performance_comparable") is False,
    }
    if expected_stage == "v1":
        coverage = gates.get("coverage", {})
        load = gates.get("model_load", {})
        return {**common,
                "verdict": evidence.get("verdict") == "ACCEPT_MODEL_LOAD",
                "coverage": gates.get("coverage_audit") is True
                and coverage.get("all_resolved") is True
                and coverage.get("tensor_count") == EXPECTED_TENSORS
                and coverage.get("shard_count") == EXPECTED_SHARDS,
                "static_memory": gates.get("memory_ceilings_ok") is True,
                "model_load": gates.get("model_load_ok") is True,
                "state_count": load.get("state_count") == EXPECTED_LAYERS,
                "actual_memory": load.get("actual_memory_ceilings_ok") is True}
    if expected_stage == "v2":
        return {**common,
                "verdict": evidence.get("verdict") == "ACCEPT_PARTITION0",
                "layer_order": validate_layer_trace(
                    gates.get("execution_trace"), list(range(22))),
                "runtime": validate_runtime(gates.get("runtime"))}
    if expected_stage == "v3":
        handoff = gates.get("handoff", {})
        return {**common,
                "verdict": evidence.get("verdict") == "ACCEPT_INTERGPU",
                "layer_order": validate_layer_trace(
                    gates.get("execution_trace"), list(range(23))),
                "handoff": handoff.get("method") == "explicit_pinned_host_staging"
                and handoff.get("checksum_bitwise_equal") is True
                and handoff.get("dtype") == "torch.bfloat16"
                and handoff.get("d2h_bytes") == handoff.get("h2d_bytes"),
                "runtime": validate_runtime(gates.get("runtime"))}
    if expected_stage == "v4":
        logits = gates.get("logits", {})
        return {**common,
                "verdict": evidence.get("verdict") == "ACCEPT_FIRST_LOGITS",
                "layer_order": validate_layer_trace(
                    gates.get("execution_trace"), list(range(EXPECTED_LAYERS))),
                "logits": logits.get("finite") is True
                and logits.get("shape", [0])[-1] == 129_280,
                "runtime": validate_runtime(gates.get("runtime"))}
    minimum = {"v5": 1, "v6": 4, "final": 16}[expected_stage]
    expected_verdict = ("ACCEPT_DUAL_T4_DECODE" if expected_stage == "final"
                        else "ACCEPT_DUAL_T4_FIRST_TOKEN")
    trace = gates.get("token_trace", {})
    tokens = gates.get("tokens", [])
    return {**common,
            "verdict": evidence.get("verdict") == expected_verdict,
            "token_count": isinstance(tokens, list) and len(tokens) >= minimum
            and gates.get("token_count") == len(tokens),
            "token_ids": gates.get("token_ids_in_vocab") is True,
            "token_layers": all(validate_layer_trace(
                trace.get(f"token_{idx}", {}).get("layers"),
                list(range(EXPECTED_LAYERS))) for idx in range(len(tokens))),
            "deterministic": gates.get("deterministic_rerun") is True,
            "cold_warm": gates.get("cold_warm_equal") is True,
            "cache_capacity": gates.get("cache_capacity_variation_equal") is True}


def validate(input_dir: Path, expected_stage: str,
             expected_commit: str | None) -> dict[str, Any]:
    evidence_dir = discover_evidence(input_dir)
    evidence = read_json(evidence_dir / "ds10-evidence.json")
    verdict = read_json(evidence_dir / "ds10-verdict.json")
    require(verdict.get("verdict") == evidence.get("verdict"),
            "verdict file disagrees with evidence")
    require(verdict.get("stage") == evidence.get("stage"),
            "verdict stage disagrees with evidence")
    manifest = validate_manifest(evidence_dir)
    require(manifest["manifest"].get("verdict") == evidence.get("verdict"),
            "manifest verdict disagrees with evidence")
    identity = validate_identity(evidence_dir, evidence, expected_commit)
    require(identity.get("stage") == expected_stage, "identity stage mismatch")
    archive = validate_archive(input_dir, evidence_dir, manifest["sha256"])
    gates = stage_gates(evidence, expected_stage)
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "result": "PASS" if passed else "FAIL",
        "terminal_verdict": evidence.get("verdict"),
        "stage": expected_stage,
        "performance_comparable": False,
        "evidence_dir": evidence_dir.name,
        "identity": identity,
        "manifest": {key: value for key, value in manifest.items()
                     if key != "manifest"},
        "archive": archive,
        "gates": gates,
        "failed_gates": [key for key, value in gates.items() if not value],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-stage", required=True,
                        choices=("v1", "v2", "v3", "v4", "v5", "v6", "final"))
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    seal_path = args.output_dir / "SEAL.json"
    try:
        seal = validate(args.input_dir, args.expected_stage,
                        args.expected_commit)
    except Exception as exc:  # noqa: BLE001
        seal = {"schema_version": 1, "result": "INVALID",
                "stage": args.expected_stage,
                "error": f"{type(exc).__name__}: {exc}"}
    write_json(seal_path, seal)
    write_json(args.output_dir / "seal-manifest.json", {
        "schema_version": 1,
        "artifacts": {"SEAL.json": {
            "bytes": seal_path.stat().st_size,
            "sha256": sha256_file(seal_path),
        }},
    })
    print(json.dumps({"result": seal["result"], "seal": str(seal_path),
                      "failed_gates": seal.get("failed_gates", [])},
                     sort_keys=True))
    return 0 if seal["result"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
