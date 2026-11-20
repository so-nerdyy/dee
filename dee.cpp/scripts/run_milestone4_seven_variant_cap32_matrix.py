#!/usr/bin/env python3
"""Run and seal the exact M3-v17 seven-variant matrix at production cap 32.

Phase 2 is gated on an immutable Phase 1 ``SEAL.json`` with ``result=PASS``.
The proven M3 matrix driver remains the only inference orchestrator. This
wrapper fixes its default expert-cache capacity at 32, preserves the explicit
cap-4 diagnostic override, proves that all seven semantic configurations are
unique before launch, and validates correctness, device path, bounded VRAM,
clean process lifetime, and artifact hashes after the driver returns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MATRIX_DRIVER = SCRIPT_DIR / "run_milestone25_matrix.py"
SEALED_PHASE1_INTEGRATION = "b98696395b51006f71a85c328acacffbf65d29d2"
CANONICAL_PHASE1_SEAL_SHA256 = (
    "edcf26827c3daae509184b368a1fb0f835eafe98ad1b3ab940427feba86b1958"
)
PRODUCTION_CACHE_CAPACITY = 32
VRAM_LIMIT_BYTES_PER_GPU = 8 * 1024**3

# Exact accepted M3-v17 order. Do not reorder: the cold row must run first.
SEVEN_VARIANTS = (
    "dual-cold-primary",
    "dual-warm-profiled",
    "dual-warm-control",
    "dual-warm-reference-present",
    "dual-cache-disabled",
    "dual-cache-capacity-4",
    "dual-long-prompt",
)

# These rows represent the production path or a production-path diagnostic.
# Only the explicit cap-4 diagnostic is permitted to override the cap-32
# default, exactly as it did in M3 v17.
CAP32_VARIANTS = frozenset(SEVEN_VARIANTS) - {"dual-cache-capacity-4"}

EXPECTED_GENERATED_TOKEN_IDS = {
    "dual-cold-primary": [11, 271, 40, 1044],
    "dual-warm-profiled": [11, 271, 40, 1044],
    "dual-warm-control": [11, 271, 40, 1044],
    "dual-warm-reference-present": [11, 271, 40, 1044],
    "dual-cache-disabled": [11, 271],
    "dual-cache-capacity-4": [11, 271],
    "dual-long-prompt": [198, 760],
}

CRITICAL_LIFETIME_PATTERNS = (
    "_abort]",
    "_fail]",
    "trace_alloc post-mortem",
    "double free or corruption",
    "fatal python error",
    "addresssanitizer",
    "corrupted size vs.",
    "munmap_chunk(): invalid pointer",
    "free(): invalid pointer",
    "segmentation fault",
)


def semantic_configurations() -> dict[str, dict[str, Any]]:
    """Return the exact seven M3-v17 semantic configurations with cap 32."""

    common_profiled = {
        "profile_enabled": True,
        "trace_requests": True,
        "profile_timeline": True,
    }
    return {
        "dual-cold-primary": {
            "classification": "cold",
            "prompt": "Hello",
            "max_new_tokens": 4,
            "warmup_generation": False,
            "cache_experts_per_layer": PRODUCTION_CACHE_CAPACITY,
            "cache_disabled": False,
            **common_profiled,
            "torch_profiler_step": 1,
            "reference_parity": True,
            "tensor_map": True,
            "allow_sub_topk_cache": False,
            "require_dual_gpu": True,
        },
        "dual-warm-profiled": {
            "classification": "warm",
            "prompt": "Hello",
            "max_new_tokens": 4,
            "warmup_generation": True,
            "cache_experts_per_layer": PRODUCTION_CACHE_CAPACITY,
            "cache_disabled": False,
            **common_profiled,
            "torch_profiler_step": 1,
            "reference_parity": False,
            "tensor_map": False,
            "allow_sub_topk_cache": False,
            "require_dual_gpu": True,
        },
        "dual-warm-control": {
            "classification": "warm",
            "prompt": "Hello",
            "max_new_tokens": 4,
            "warmup_generation": True,
            "cache_experts_per_layer": PRODUCTION_CACHE_CAPACITY,
            "cache_disabled": False,
            "profile_enabled": False,
            "trace_requests": False,
            "profile_timeline": False,
            "torch_profiler_step": None,
            "reference_parity": False,
            "tensor_map": False,
            "allow_sub_topk_cache": False,
            "require_dual_gpu": True,
        },
        "dual-warm-reference-present": {
            "classification": "warm",
            "prompt": "Hello",
            "max_new_tokens": 4,
            "warmup_generation": True,
            "cache_experts_per_layer": PRODUCTION_CACHE_CAPACITY,
            "cache_disabled": False,
            **common_profiled,
            "torch_profiler_step": None,
            "reference_parity": True,
            "tensor_map": False,
            "allow_sub_topk_cache": False,
            "require_dual_gpu": True,
        },
        "dual-cache-disabled": {
            "classification": "diagnostic",
            "prompt": "Hello",
            "max_new_tokens": 2,
            "warmup_generation": True,
            "cache_experts_per_layer": PRODUCTION_CACHE_CAPACITY,
            "cache_disabled": True,
            **common_profiled,
            "torch_profiler_step": None,
            "reference_parity": False,
            "tensor_map": False,
            "allow_sub_topk_cache": False,
            "require_dual_gpu": True,
        },
        "dual-cache-capacity-4": {
            "classification": "diagnostic",
            "prompt": "Hello",
            "max_new_tokens": 2,
            "warmup_generation": True,
            "cache_experts_per_layer": 4,
            "cache_disabled": False,
            **common_profiled,
            "torch_profiler_step": None,
            "reference_parity": False,
            "tensor_map": False,
            "allow_sub_topk_cache": True,
            "require_dual_gpu": True,
        },
        "dual-long-prompt": {
            "classification": "diagnostic",
            "prompt": "The quick brown fox jumps over the lazy dog.",
            "max_new_tokens": 2,
            "warmup_generation": True,
            "cache_experts_per_layer": PRODUCTION_CACHE_CAPACITY,
            "cache_disabled": False,
            **common_profiled,
            "torch_profiler_step": None,
            "reference_parity": False,
            "tensor_map": False,
            "allow_sub_topk_cache": False,
            "require_dual_gpu": True,
        },
    }


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def configuration_fingerprint(configuration: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(configuration).encode("utf-8")).hexdigest()


def variant_contracts() -> list[dict[str, Any]]:
    configurations = semantic_configurations()
    if tuple(configurations) != SEVEN_VARIANTS:
        raise RuntimeError("semantic configuration order drifted from M3 v17")
    contracts = [
        {
            "run_id": run_id,
            "semantic_configuration": configurations[run_id],
            "configuration_fingerprint_sha256": configuration_fingerprint(
                configurations[run_id]
            ),
        }
        for run_id in SEVEN_VARIANTS
    ]
    fingerprints = {
        item["configuration_fingerprint_sha256"] for item in contracts
    }
    if len(fingerprints) != len(SEVEN_VARIANTS):
        raise RuntimeError("Phase 2 would launch duplicate semantic configurations")
    for item in contracts:
        run_id = item["run_id"]
        capacity = item["semantic_configuration"]["cache_experts_per_layer"]
        expected = 32 if run_id in CAP32_VARIANTS else 4
        if capacity != expected:
            raise RuntimeError(
                f"{run_id} cache capacity drifted: expected {expected}, got {capacity}"
            )
    return contracts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seal-dir",
        required=True,
        type=Path,
        help="Phase 1 seal directory containing SEAL.json",
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--phase1-integration",
        default=SEALED_PHASE1_INTEGRATION,
        help="Required sealed Phase 1 integration ancestor.",
    )
    parser.add_argument(
        "--launch-engine",
        action="store_true",
        help=(
            "Run the seven real-model variants. Without this flag, validate "
            "and print the immutable launch contract without running inference."
        ),
    )
    return parser.parse_args()


def gate(seal_dir: Path) -> dict[str, Any]:
    seal_path = seal_dir / "SEAL.json"
    if not seal_path.is_file():
        raise SystemExit(f"Phase 2 abort: SEAL.json not found at {seal_path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("result") != "PASS":
        reason = seal.get("terminal_reason", "<unspecified>")
        raise SystemExit(f"Phase 2 abort: Phase 1 SEAL is not PASS ({reason})")
    seal_sha256 = sha256_file(seal_path)
    source_sha256 = seal.get("source_seal_sha256")
    if (
        seal_sha256 != CANONICAL_PHASE1_SEAL_SHA256
        and source_sha256 != CANONICAL_PHASE1_SEAL_SHA256
    ):
        raise SystemExit(
            "Phase 2 abort: Phase 1 seal provenance does not match the "
            f"canonical seal {CANONICAL_PHASE1_SEAL_SHA256}"
        )
    seal["validated_seal_file_sha256"] = seal_sha256
    return seal


def verify_phase1_lineage(required_sha: str) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lineage = subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_sha, head],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if lineage.returncode != 0:
        raise SystemExit(
            f"Phase 2 abort: HEAD {head} does not descend from sealed "
            f"Phase 1 integration {required_sha}"
        )
    return head


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lifetime_log(path: Path, run_id: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    matches = [
        pattern for pattern in CRITICAL_LIFETIME_PATTERNS if pattern in lowered
    ]
    if matches:
        raise RuntimeError(f"{run_id} lifetime gate failed: {matches}")
    return {
        "result": "NO_TRACE_ABORT",
        "process_exit_code": 0,
        "critical_markers": [],
        "log": str(path),
        "log_sha256": sha256_file(path),
    }


def validate_runtime_configuration(
    actual: dict[str, Any],
    expected: dict[str, Any],
    run_id: str,
) -> None:
    report_fields = (
        "classification",
        "prompt",
        "max_new_tokens",
        "warmup_generation",
        "cache_experts_per_layer",
        "cache_disabled",
        "profile_enabled",
        "trace_requests",
        "profile_timeline",
        "reference_parity",
    )
    drift = {
        field: {"expected": expected[field], "actual": actual.get(field)}
        for field in report_fields
        if actual.get(field) != expected[field]
    }
    if drift:
        raise RuntimeError(f"{run_id} runtime configuration drifted: {drift}")


def validate_row(
    output_dir: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    run_id = contract["run_id"]
    run_dir = output_dir / "runs" / run_id
    report_path = run_dir / "run-report.json"
    path_proof_path = run_dir / "path-proof.json"
    log_path = output_dir / "logs" / f"{run_id}.log"
    for required in (report_path, path_proof_path, log_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise RuntimeError(f"{run_id} missing required artifact: {required}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("result") != "PASS":
        raise RuntimeError(f"{run_id} run-report result is not PASS")
    validate_runtime_configuration(
        report.get("configuration", {}),
        contract["semantic_configuration"],
        run_id,
    )
    correctness = report.get("correctness", {})
    if not correctness.get("all_40_layers_executed"):
        raise RuntimeError(f"{run_id} did not execute all 40 layers")
    if not correctness.get("baseline_tokens_exact"):
        raise RuntimeError(f"{run_id} failed the expected-token gate")
    generated = report.get("generation", {}).get("generated_token_ids")
    if generated != EXPECTED_GENERATED_TOKEN_IDS[run_id]:
        raise RuntimeError(
            f"{run_id} generated tokens drifted: "
            f"expected {EXPECTED_GENERATED_TOKEN_IDS[run_id]}, got {generated}"
        )

    path_proof = json.loads(path_proof_path.read_text(encoding="utf-8"))
    if path_proof.get("device_path_share") != 1.0:
        raise RuntimeError(f"{run_id} device path share is not 100%")
    if path_proof.get("host_path_fallback_calls") != 0:
        raise RuntimeError(f"{run_id} used the host fallback path")

    generation = report.get("generation", {})
    resources = generation.get("resources", {})
    peak_vram = resources.get("peak_vram_bytes", {})
    if not peak_vram:
        raise RuntimeError(f"{run_id} did not report per-GPU peak process VRAM")
    over_limit = {
        device: value
        for device, value in peak_vram.items()
        if not isinstance(value, int) or value > VRAM_LIMIT_BYTES_PER_GPU
    }
    if over_limit:
        raise RuntimeError(
            f"{run_id} exceeded the 8 GiB/GPU process-VRAM envelope: {over_limit}"
        )

    lifetime = validate_lifetime_log(log_path, run_id)
    runtime_fingerprint = configuration_fingerprint(
        contract["semantic_configuration"]
    )
    if runtime_fingerprint != contract["configuration_fingerprint_sha256"]:
        raise RuntimeError(f"{run_id} configuration fingerprint changed")
    fingerprint_path = run_dir / "configuration-fingerprint.json"
    write_json_atomic(fingerprint_path, {
        **contract,
        "runtime_configuration": report["configuration"],
        "runtime_configuration_validated": True,
    })

    decode_tps = generation.get("single_stream_decode_tokens_per_second")
    native_memory = (
        report.get("layout", {}).get("native_engine_memory", {}).get("aggregate", {})
    )
    return {
        "run_id": run_id,
        "result": "PASS",
        "git_commit": report.get("git_commit"),
        "model": "Ornith-1.0-35B",
        "precision": {
            "model": "BF16 checkpoint",
            "expert_cache": "FP16",
            "expert_transfer": "BF16",
        },
        "gpu": report.get("machine", {}).get("gpus"),
        "batch_size": 1,
        "number_of_sequences": 1,
        "prompt": report.get("configuration", {}).get("prompt"),
        "prompt_token_count": generation.get("prompt_token_count"),
        "cold_or_warm": report.get("configuration", {}).get("classification"),
        "full_model_single_sequence_decode_tps": decode_tps,
        "aggregate_tps": decode_tps,
        "ttft_seconds": generation.get("time_to_first_token_seconds"),
        "tpot_seconds": (1.0 / decode_tps) if decode_tps else None,
        "peak_process_vram_per_gpu_bytes": peak_vram,
        "process_peak_rss_bytes": resources.get("peak_host_rss_bytes"),
        "expert_resident_bytes": native_memory.get("current_vram"),
        "h2d_bytes_per_token": None,
        "cache_hit_rate": native_memory.get("cache_hit_rate"),
        "attention_time": None,
        "expert_time": None,
        "synchronization": None,
        "correctness": {
            "all_40_layers_executed": True,
            "expected_tokens_exact": True,
            "generated_token_ids": generated,
            "device_path_share": 1.0,
            "host_path_fallback_calls": 0,
        },
        "lifetime": lifetime,
        "configuration_fingerprint_sha256": runtime_fingerprint,
        "artifact_path": str(run_dir),
        "artifact_hashes": {
            "run-report.json": sha256_file(report_path),
            "path-proof.json": sha256_file(path_proof_path),
            "configuration-fingerprint.json": sha256_file(fingerprint_path),
        },
        "profiling_fields_note": (
            "H2D/token, attention, expert, and synchronization attribution "
            "are sealed by the subsequent full-token profile/roofline gate; "
            "Phase 2 records explicit nulls rather than projecting values."
        ),
    }


def validate_matrix(
    output_dir: Path,
    contracts: list[dict[str, Any]],
    phase1_seal: dict[str, Any],
    head: str,
) -> dict[str, Any]:
    summary_path = output_dir / "matrix-summary.json"
    if not summary_path.is_file():
        raise RuntimeError("matrix-summary.json is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("selected_run_ids") != list(SEVEN_VARIANTS):
        raise RuntimeError("matrix run order drifted from the accepted M3 v17 order")
    experiments = summary.get("experiments", [])
    if len(experiments) != 7 or any(
        item.get("result") != "PASS" for item in experiments
    ):
        raise RuntimeError("the seven-variant matrix did not finish with seven PASS rows")

    rows = [validate_row(output_dir, contract) for contract in contracts]
    commits = {row["git_commit"] for row in rows}
    if commits != {head}:
        raise RuntimeError(
            f"matrix rows did not execute the launch HEAD {head}: {sorted(commits)}"
        )
    final_report = {
        "schema_version": 1,
        "result": "PASS",
        "terminal_reason": "ALL_SEVEN_UNIQUE_CAP32_MATRIX_ROWS_PASS",
        "phase1": {
            "seal_result": phase1_seal.get("result"),
            "seal_run_id": phase1_seal.get("identity", {}).get("run_id"),
            "sealed_integration_ancestor": SEALED_PHASE1_INTEGRATION,
        },
        "git_commit": head,
        "selected_run_ids": list(SEVEN_VARIANTS),
        "configuration_fingerprints_unique": True,
        "production_cache_capacity": PRODUCTION_CACHE_CAPACITY,
        "explicit_diagnostic_override": {
            "run_id": "dual-cache-capacity-4",
            "cache_experts_per_layer": 4,
        },
        "rows": rows,
    }
    final_path = output_dir / "phase2-final-report.json"
    write_json_atomic(final_path, final_report)

    artifact_paths = [
        output_dir / "phase2-contract.json",
        summary_path,
        output_dir / "matrix-progress.jsonl",
        final_path,
    ]
    for run_id in SEVEN_VARIANTS:
        run_dir = output_dir / "runs" / run_id
        artifact_paths.extend([
            run_dir / "run-report.json",
            run_dir / "path-proof.json",
            run_dir / "configuration-fingerprint.json",
            output_dir / "logs" / f"{run_id}.log",
        ])
    manifest = {
        "schema_version": 1,
        "result": "PASS",
        "git_commit": head,
        "artifacts": [
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    write_json_atomic(output_dir / "phase2-artifact-manifest.json", manifest)
    return final_report


def main() -> int:
    args = parse_args()
    seal = gate(args.seal_dir)
    head = verify_phase1_lineage(args.phase1_integration)
    contracts = variant_contracts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_payload = {
        "schema_version": 1,
        "result": "CONTRACT_VALIDATED",
        "phase1_seal_run_id": seal.get("identity", {}).get("run_id"),
        "phase1_seal_terminal_timestamp": seal.get("terminal_timestamp_utc"),
        "phase1_pass": True,
        "phase1_integration_ancestor": args.phase1_integration,
        "launch_git_commit": head,
        "would_launch": contracts,
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "would_invoke_engine": bool(args.launch_engine),
    }
    write_json_atomic(args.output_dir / "phase2-contract.json", contract_payload)
    if not args.launch_engine:
        print(json.dumps(contract_payload, sort_keys=True), flush=True)
        return 0

    command = [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        str(MATRIX_DRIVER),
        "--model-dir",
        str(args.model_dir),
        "--output-dir",
        str(args.output_dir),
        "--require-dual-gpu",
        "--default-cache-experts",
        str(PRODUCTION_CACHE_CAPACITY),
        "--skip-aggregate",
        "--run-ids",
        *SEVEN_VARIANTS,
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    final_report = validate_matrix(args.output_dir, contracts, seal, head)
    print(
        "MILESTONE4_PHASE2=" + json.dumps(final_report, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
