#!/usr/bin/env python3
"""Revalidate downloaded dual-T4 router Pareto evidence and emit SEAL.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any


MAX_PROCESS_VRAM_BYTES = 8 * 1024**3
EXPECTED_TOKENS = [11, 271, 40, 1044]
EXPECTED_MEASUREMENT_ORDER = [
    "native-host",
    "torch-device",
    "torch-device",
    "native-host",
    "native-host",
    "torch-device",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_root_manifest(evidence_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("result") != "PASS":
        raise RuntimeError("root artifact manifest result is not PASS")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("root artifact manifest has no rows")
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        if not relative or relative in seen:
            raise RuntimeError(f"invalid/duplicate root manifest path: {relative!r}")
        seen.add(relative)
        path = evidence_dir / relative
        if not path.is_file():
            raise RuntimeError(f"root manifest artifact is missing: {relative}")
        if path.stat().st_size != int(row.get("bytes", -1)):
            raise RuntimeError(f"root manifest size changed: {relative}")
        if sha256_file(path) != row.get("sha256"):
            raise RuntimeError(f"root manifest hash changed: {relative}")
    return {
        "sha256": sha256_file(manifest_path),
        "artifact_count": len(rows),
    }


def verify_nested_manifest(
    benchmark_dir: Path, manifest_path: Path, report_path: Path
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("result") != "PASS":
        raise RuntimeError("benchmark artifact manifest result is not PASS")
    row = manifest.get("files", {}).get(report_path.name)
    if not isinstance(row, dict):
        raise RuntimeError("benchmark artifact manifest does not index its report")
    if report_path.stat().st_size != int(row.get("bytes", -1)):
        raise RuntimeError("benchmark report size changed")
    if sha256_file(report_path) != row.get("sha256"):
        raise RuntimeError("benchmark report hash changed")
    return {
        "sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
    }


def validate_report(report: dict[str, Any], expected_commit: str) -> dict[str, Any]:
    if report.get("result") != "PASS":
        raise RuntimeError("router benchmark result is not PASS")
    if report.get("git_commit") != expected_commit:
        raise RuntimeError("router benchmark commit changed")
    config = report.get("configuration", {})
    if config.get("cache_experts_per_layer") != 32:
        raise RuntimeError("router benchmark cache capacity is not 32")
    if config.get("trials_per_backend", 0) < 3:
        raise RuntimeError("router benchmark has fewer than three trials/backend")
    order = config.get("measurement_order")
    if order != EXPECTED_MEASUREMENT_ORDER or any(
        order.count(backend) != config["trials_per_backend"]
        for backend in ("native-host", "torch-device")
    ):
        raise RuntimeError("router benchmark trial order/count is invalid")
    expected_configuration = {
        "cache_experts_per_layer": 32,
        "split_layer": 20,
        "trials_per_backend": 3,
        "prompt": "Hello",
        "greedy": True,
        "load_once": True,
        "warmup_per_backend": 1,
    }
    changed_configuration = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in expected_configuration.items()
        if config.get(key) != expected
    }
    if changed_configuration:
        raise RuntimeError(
            f"router benchmark configuration changed: {changed_configuration}"
        )

    environment = report.get("environment", {})
    gpus = environment.get("gpus")
    if (
        not isinstance(gpus, list)
        or len(gpus) != 2
        or any("T4" not in str(gpu.get("name")) for gpu in gpus)
    ):
        raise RuntimeError(f"router benchmark is not dual T4: {gpus}")

    correctness = report.get("correctness", {})
    if correctness.get("native_tokens") != EXPECTED_TOKENS:
        raise RuntimeError("native-host benchmark tokens changed")
    if correctness.get("optimized_tokens") != EXPECTED_TOKENS:
        raise RuntimeError("torch-device benchmark tokens changed")
    if not all(
        correctness.get(key) is True
        for key in (
            "generated_token_ids_exact",
            "decoded_text_exact",
            "all_trace_categories_passed",
        )
    ):
        raise RuntimeError("router benchmark correctness/trace parity failed")

    summaries = report.get("backend_summaries", {})
    native = summaries.get("native-host", {})
    optimized = summaries.get("torch-device", {})
    for backend, summary in (("native-host", native), ("torch-device", optimized)):
        if summary.get("trial_count") != config["trials_per_backend"]:
            raise RuntimeError(f"{backend} trial count changed")
        if not all(
            summary.get(key) is True
            for key in (
                "all_tokens_exact",
                "all_40_layers_executed",
                "all_expert_calls_device_resident",
            )
        ):
            raise RuntimeError(f"{backend} model/expert correctness gate failed")
        if summary.get("host_fallback_calls") != 0:
            raise RuntimeError(f"{backend} used the host expert fallback")
        peak = summary.get("peak_process_vram_per_gpu_bytes")
        if not isinstance(peak, dict) or set(peak) != {"cuda:0", "cuda:1"}:
            raise RuntimeError(f"{backend} per-GPU VRAM identity is incomplete")
        if any(
            not isinstance(value, int) or value > MAX_PROCESS_VRAM_BYTES
            for value in peak.values()
        ):
            raise RuntimeError(f"{backend} exceeded the 8 GiB/GPU VRAM gate")
        rates = summary.get("tokens_per_second", {})
        if (
            not isinstance(rates.get("median"), (int, float))
            or rates["median"] <= 0
            or len(rates.get("all", [])) != config["trials_per_backend"]
        ):
            raise RuntimeError(f"{backend} throughput evidence is incomplete")
    if native.get("router_native_host_calls", 0) <= 0:
        raise RuntimeError("native control did not exercise the native-host router")
    if native.get("router_torch_device_calls") != 0:
        raise RuntimeError("native control exercised the optimized router")
    if optimized.get("router_torch_device_calls", 0) <= 0:
        raise RuntimeError("optimized run did not exercise torch-device routing")
    for key in (
        "router_native_host_calls",
        "router_hidden_d2h_total_bytes",
        "router_outputs_h2d_total_bytes",
        "router_scalar_sync_calls",
    ):
        if optimized.get(key) != 0:
            raise RuntimeError(f"optimized router host-path counter is nonzero: {key}")

    pareto = report.get("pareto", {})
    baseline_tps = pareto.get("baseline_native_host_median_tps")
    optimized_tps = pareto.get("optimized_torch_device_median_tps")
    speedup = pareto.get("speedup_ratio")
    if not all(isinstance(value, (int, float)) for value in (
        baseline_tps, optimized_tps, speedup
    )):
        raise RuntimeError("router Pareto rates are missing")
    if abs(baseline_tps - native["tokens_per_second"]["median"]) > 1e-12:
        raise RuntimeError("router baseline median does not match trial summary")
    if abs(optimized_tps - optimized["tokens_per_second"]["median"]) > 1e-12:
        raise RuntimeError("router optimized median does not match trial summary")
    if abs(speedup - optimized_tps / baseline_tps) > 1e-9 or speedup < 1.02:
        raise RuntimeError("router speedup is not a >=2% measured improvement")
    gates = pareto.get("gates")
    if not isinstance(gates, dict) or not gates or not all(gates.values()):
        raise RuntimeError(f"router Pareto gates failed: {gates}")
    return {
        "baseline_native_host_median_tps": baseline_tps,
        "optimized_torch_device_median_tps": optimized_tps,
        "speedup_ratio": speedup,
        "speedup_percent": pareto.get("speedup_percent"),
        "generated_token_ids": EXPECTED_TOKENS,
        "trace_categories": sorted(
            key
            for key in correctness.get("comparisons", {})
            if key != "all_categories_passed"
        ),
        "native_host": {
            "trial_tps": native["tokens_per_second"]["all"],
            "median_decode_seconds": native.get("decode_seconds", {}).get(
                "median"
            ),
            "peak_process_vram_per_gpu_bytes": native[
                "peak_process_vram_per_gpu_bytes"
            ],
            "router_native_host_calls": native["router_native_host_calls"],
            "router_hidden_d2h_total_bytes": native[
                "router_hidden_d2h_total_bytes"
            ],
            "router_outputs_h2d_total_bytes": native[
                "router_outputs_h2d_total_bytes"
            ],
            "router_scalar_sync_calls": native["router_scalar_sync_calls"],
        },
        "optimized_torch_device": {
            "trial_tps": optimized["tokens_per_second"]["all"],
            "median_decode_seconds": optimized.get("decode_seconds", {}).get(
                "median"
            ),
            "peak_process_vram_per_gpu_bytes": optimized[
                "peak_process_vram_per_gpu_bytes"
            ],
            "router_torch_device_calls": optimized[
                "router_torch_device_calls"
            ],
            "router_hidden_d2h_total_bytes": optimized[
                "router_hidden_d2h_total_bytes"
            ],
            "router_outputs_h2d_total_bytes": optimized[
                "router_outputs_h2d_total_bytes"
            ],
            "router_scalar_sync_calls": optimized[
                "router_scalar_sync_calls"
            ],
        },
        "optimized_peak_process_vram_per_gpu_bytes": optimized[
            "peak_process_vram_per_gpu_bytes"
        ],
        "trial_count_per_backend": config["trials_per_backend"],
    }


def locate(download_root: Path) -> tuple[Path, Path]:
    matches = sorted(download_root.rglob("router-backend-benchmark.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one router-backend-benchmark.json, found {len(matches)}"
        )
    report_path = matches[0]
    benchmark_dir = report_path.parent
    evidence_dir = benchmark_dir.parent
    return evidence_dir, report_path


def validate(
    download_root: Path, expected_commit: str, expected_run_id: str
) -> dict[str, Any]:
    evidence_dir, report_path = locate(download_root)
    required = {
        "environment.json": evidence_dir / "environment.json",
        "build-manifest.json": evidence_dir / "build-manifest.json",
        "artifact-manifest.json": evidence_dir / "artifact-manifest.json",
        "benchmark/artifact-manifest.json": (
            evidence_dir / "benchmark" / "artifact-manifest.json"
        ),
        "benchmark/router-backend-benchmark.json": report_path,
        "logs/router-backend-benchmark.log": (
            evidence_dir / "logs" / "router-backend-benchmark.log"
        ),
    }
    missing = [
        name
        for name, path in required.items()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"router Pareto artifacts are missing/empty: {missing}")
    environment = read_json(required["environment.json"])
    build = read_json(required["build-manifest.json"])
    root_manifest = read_json(required["artifact-manifest.json"])
    if environment.get("run_id") != expected_run_id:
        raise RuntimeError("router environment run ID changed")
    if root_manifest.get("run_id") != expected_run_id:
        raise RuntimeError("router root manifest run ID changed")
    for label, value in (
        ("environment", environment.get("commit")),
        ("build", build.get("commit")),
        ("root manifest", root_manifest.get("commit")),
    ):
        if value != expected_commit:
            raise RuntimeError(f"router {label} commit changed")
    report_summary = validate_report(read_json(report_path), expected_commit)
    root_check = verify_root_manifest(
        evidence_dir, required["artifact-manifest.json"]
    )
    nested_check = verify_nested_manifest(
        evidence_dir / "benchmark",
        required["benchmark/artifact-manifest.json"],
        report_path,
    )
    return {
        "schema_version": 1,
        "result": "PASS",
        "terminal_reason": "DOWNLOADED_DEVICE_ROUTER_PARETO_REVALIDATED",
        "expected_run_id": expected_run_id,
        "expected_commit": expected_commit,
        "evidence_dir": str(evidence_dir),
        "report_summary": report_summary,
        "manifest_checks": {
            "root": root_check,
            "benchmark": nested_check,
        },
        "source_hashes": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
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
        write_json(
            args.output_dir / "seal-manifest.json",
            {
                "schema_version": 1,
                "result": "PASS",
                "artifacts": {
                    "SEAL.json": {
                        "bytes": seal_path.stat().st_size,
                        "sha256": sha256_file(seal_path),
                    }
                },
            },
        )
        print(json.dumps(seal["report_summary"], sort_keys=True), flush=True)
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
