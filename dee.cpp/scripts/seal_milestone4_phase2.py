#!/usr/bin/env python3
"""Revalidate downloaded Milestone 4 Phase 2 evidence and emit SEAL.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_milestone4_seven_variant_cap32_matrix import (  # noqa: E402
    EXPECTED_GENERATED_TOKEN_IDS,
    PRODUCTION_CACHE_CAPACITY,
    SEALED_PHASE1_INTEGRATION,
    SEVEN_VARIANTS,
    VRAM_LIMIT_BYTES_PER_GPU,
    variant_contracts,
)


CRITICAL_MARKERS = (
    "TRACE_ALLOC ABORT",
    "TRACE_ALLOC POST-MORTEM",
    "double free or corruption",
    "Fatal Python error",
    "AddressSanitizer",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return value


def verify_manifest(
    evidence_dir: Path,
    manifest_path: Path,
    *,
    path_key: str,
    size_key: str,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("result") != "PASS":
        raise RuntimeError(f"{manifest_path.name} result is not PASS")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{manifest_path.name} has no artifact rows")
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get(path_key, ""))
        if not relative or relative in seen:
            raise RuntimeError(
                f"{manifest_path.name} has invalid/duplicate path {relative!r}"
            )
        seen.add(relative)
        path = evidence_dir / Path(relative)
        if not path.is_file():
            raise RuntimeError(f"manifest artifact is missing: {relative}")
        actual_size = path.stat().st_size
        if actual_size != int(row.get(size_key, -1)):
            raise RuntimeError(
                f"manifest artifact size changed: {relative} "
                f"{actual_size} != {row.get(size_key)}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != row.get("sha256"):
            raise RuntimeError(f"manifest artifact hash changed: {relative}")
    return {
        "path": manifest_path.name,
        "sha256": sha256_file(manifest_path),
        "artifact_count": len(rows),
        "verified_paths": len(seen),
    }


def validate_final_report(
    report: dict[str, Any], expected_commit: str
) -> dict[str, Any]:
    if report.get("result") != "PASS":
        raise RuntimeError("phase2-final-report result is not PASS")
    if report.get("terminal_reason") != "ALL_SEVEN_UNIQUE_CAP32_MATRIX_ROWS_PASS":
        raise RuntimeError("Phase 2 terminal reason changed")
    if report.get("git_commit") != expected_commit:
        raise RuntimeError(
            f"Phase 2 commit changed: {report.get('git_commit')} != {expected_commit}"
        )
    if report.get("selected_run_ids") != list(SEVEN_VARIANTS):
        raise RuntimeError("Phase 2 run order changed")
    if report.get("configuration_fingerprints_unique") is not True:
        raise RuntimeError("Phase 2 fingerprints are not marked unique")
    if report.get("production_cache_capacity") != PRODUCTION_CACHE_CAPACITY:
        raise RuntimeError("Phase 2 production capacity is not 32")
    override = report.get("explicit_diagnostic_override", {})
    if override != {
        "run_id": "dual-cache-capacity-4",
        "cache_experts_per_layer": 4,
    }:
        raise RuntimeError(f"Phase 2 cap-4 override changed: {override}")

    contracts = {item["run_id"]: item for item in variant_contracts()}
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(SEVEN_VARIANTS):
        raise RuntimeError("Phase 2 does not have exactly seven rows")
    seen: set[str] = set()
    fingerprints: set[str] = set()
    minimum_tps = float("inf")
    maximum_tps = 0.0
    maximum_vram = 0
    for row in rows:
        run_id = row.get("run_id")
        if run_id not in contracts or run_id in seen:
            raise RuntimeError(f"invalid or duplicate Phase 2 row: {run_id}")
        seen.add(run_id)
        if row.get("result") != "PASS" or row.get("git_commit") != expected_commit:
            raise RuntimeError(f"{run_id} is not a PASS at the expected commit")
        fingerprint = row.get("configuration_fingerprint_sha256")
        if fingerprint != contracts[run_id]["configuration_fingerprint_sha256"]:
            raise RuntimeError(f"{run_id} semantic fingerprint changed")
        fingerprints.add(fingerprint)
        correctness = row.get("correctness", {})
        if correctness.get("all_40_layers_executed") is not True:
            raise RuntimeError(f"{run_id} did not execute all 40 layers")
        if correctness.get("expected_tokens_exact") is not True:
            raise RuntimeError(f"{run_id} failed exact expected tokens")
        if correctness.get("generated_token_ids") != EXPECTED_GENERATED_TOKEN_IDS[run_id]:
            raise RuntimeError(f"{run_id} generated-token IDs changed")
        if correctness.get("device_path_share") != 1.0:
            raise RuntimeError(f"{run_id} device path share is not 1.0")
        if correctness.get("host_path_fallback_calls") != 0:
            raise RuntimeError(f"{run_id} used host fallback")
        lifetime = row.get("lifetime", {})
        if (
            lifetime.get("result") != "NO_TRACE_ABORT"
            or lifetime.get("process_exit_code") != 0
            or lifetime.get("critical_markers") != []
        ):
            raise RuntimeError(f"{run_id} lifetime gate failed")
        peak_vram = row.get("peak_process_vram_per_gpu_bytes")
        if not isinstance(peak_vram, dict) or set(peak_vram) != {"cuda:0", "cuda:1"}:
            raise RuntimeError(f"{run_id} peak VRAM identity is incomplete")
        for device, value in peak_vram.items():
            if not isinstance(value, int) or value > VRAM_LIMIT_BYTES_PER_GPU:
                raise RuntimeError(f"{run_id} {device} exceeds the VRAM gate: {value}")
            maximum_vram = max(maximum_vram, value)
        tps = row.get("full_model_single_sequence_decode_tps")
        if not isinstance(tps, (int, float)) or not (float(tps) > 0.0):
            raise RuntimeError(f"{run_id} has invalid full-token TPS: {tps}")
        minimum_tps = min(minimum_tps, float(tps))
        maximum_tps = max(maximum_tps, float(tps))
        for explicit_null in (
            "h2d_bytes_per_token",
            "attention_time",
            "expert_time",
            "synchronization",
        ):
            if row.get(explicit_null, "missing") is not None:
                raise RuntimeError(
                    f"{run_id} projected {explicit_null} instead of recording null"
                )
    if seen != set(SEVEN_VARIANTS) or len(fingerprints) != len(SEVEN_VARIANTS):
        raise RuntimeError("Phase 2 row/fingerprint set is incomplete")
    return {
        "row_count": len(rows),
        "run_ids": list(SEVEN_VARIANTS),
        "unique_fingerprint_count": len(fingerprints),
        "minimum_full_token_tps": minimum_tps,
        "maximum_full_token_tps": maximum_tps,
        "maximum_peak_process_vram_bytes": maximum_vram,
    }


def locate_evidence_dir(download_root: Path) -> Path:
    matches = sorted(download_root.rglob("phase2-final-report.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one phase2-final-report.json under {download_root}, "
            f"found {len(matches)}"
        )
    return matches[0].parent


def validate(download_root: Path, expected_commit: str, expected_run_id: str) -> dict:
    evidence_dir = locate_evidence_dir(download_root)
    required = {
        name: evidence_dir / name
        for name in (
            "environment.json",
            "build-manifest.json",
            "phase2-contract.json",
            "matrix-summary.json",
            "matrix-progress.jsonl",
            "phase2-final-report.json",
            "phase2-artifact-manifest.json",
            "artifact-manifest.json",
            "raw-allocation-trace.log",
        )
    }
    missing = [
        name
        for name, path in required.items()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Phase 2 required artifacts are missing/empty: {missing}")

    environment = read_json(required["environment.json"])
    build = read_json(required["build-manifest.json"])
    contract = read_json(required["phase2-contract.json"])
    summary = read_json(required["matrix-summary.json"])
    report = read_json(required["phase2-final-report.json"])
    notebook_manifest = read_json(required["artifact-manifest.json"])
    for name, value in (
        ("environment run_id", environment.get("run_id")),
        ("notebook manifest run_id", notebook_manifest.get("run_id")),
    ):
        if value != expected_run_id:
            raise RuntimeError(f"{name} changed: {value} != {expected_run_id}")
    for name, value in (
        ("environment commit", environment.get("commit")),
        ("build commit", build.get("commit")),
        ("contract launch commit", contract.get("launch_git_commit")),
        ("notebook manifest commit", notebook_manifest.get("commit")),
    ):
        if value != expected_commit:
            raise RuntimeError(f"{name} changed: {value} != {expected_commit}")
    if contract.get("would_invoke_engine") is not True:
        raise RuntimeError("Phase 2 contract did not invoke the engine")
    if summary.get("selected_run_ids") != list(SEVEN_VARIANTS):
        raise RuntimeError("matrix-summary run order changed")
    experiments = summary.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 7 or any(
        row.get("result") != "PASS" for row in experiments
    ):
        raise RuntimeError("matrix-summary does not contain seven PASS rows")

    phase2_manifest = verify_manifest(
        evidence_dir,
        required["phase2-artifact-manifest.json"],
        path_key="path",
        size_key="size_bytes",
    )
    notebook_manifest_check = verify_manifest(
        evidence_dir,
        required["artifact-manifest.json"],
        path_key="path",
        size_key="bytes",
    )
    report_summary = validate_final_report(report, expected_commit)
    trace_text = required["raw-allocation-trace.log"].read_text(
        encoding="utf-8", errors="replace"
    )
    found_markers = [marker for marker in CRITICAL_MARKERS if marker in trace_text]
    if found_markers:
        raise RuntimeError(f"raw allocation trace contains critical markers: {found_markers}")
    try:
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                SEALED_PHASE1_INTEGRATION,
                expected_commit,
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"expected commit does not descend from Phase 1: {exc.stderr}"
        ) from exc

    return {
        "schema_version": 1,
        "result": "PASS",
        "terminal_reason": "DOWNLOADED_PHASE2_ALL_SEVEN_ROWS_REVALIDATED",
        "expected_run_id": expected_run_id,
        "expected_commit": expected_commit,
        "sealed_phase1_ancestor": SEALED_PHASE1_INTEGRATION,
        "evidence_dir": str(evidence_dir),
        "evidence_dir_name": evidence_dir.name,
        "report_summary": report_summary,
        "source_hashes": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
        },
        "manifest_checks": {
            "phase2": phase2_manifest,
            "notebook": notebook_manifest_check,
        },
        "raw_allocation_trace": {
            "critical_markers": [],
            "result": "NO_TRACE_ABORT",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seal_path = args.output_dir / "SEAL.json"
    try:
        seal = validate(
            args.download_root, args.expected_commit, args.expected_run_id
        )
        write_json(seal_path, seal)
        manifest = {
            "schema_version": 1,
            "result": "PASS",
            "artifacts": {
                "SEAL.json": {
                    "bytes": seal_path.stat().st_size,
                    "sha256": sha256_file(seal_path),
                }
            },
        }
        write_json(args.output_dir / "seal-manifest.json", manifest)
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "seal": str(seal_path),
                    **seal["report_summary"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "result": "FAIL",
            "terminal_reason": f"{type(exc).__name__}: {exc}",
            "expected_run_id": args.expected_run_id,
            "expected_commit": args.expected_commit,
            "download_root": str(args.download_root),
            "traceback": traceback.format_exc(),
        }
        write_json(seal_path, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
