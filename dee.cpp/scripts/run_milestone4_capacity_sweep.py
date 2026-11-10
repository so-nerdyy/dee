#!/usr/bin/env python3
"""Run the paired, single-variable Milestone 4 cache-capacity sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_milestone25_matrix import (
    append_progress,
    run_tee,
    write_json_atomic,
)


CAPACITIES = (8, 16, 32)
EXPECTED_TOKEN_IDS = [11, 271, 40, 1044]


def capacity_experiments(require_dual_gpu: bool = True) -> list[dict[str, Any]]:
    dual = ["--require-dual-gpu"] if require_dual_gpu else []
    experiments: list[dict[str, Any]] = []
    for capacity in CAPACITIES:
        common = ["--warmup-generation", "--cache-experts", str(capacity), *dual]
        experiments.extend([
            {
                "run_id": f"capacity-{capacity}-control",
                "capacity": capacity,
                "profiled": False,
                "flags": common,
            },
            {
                "run_id": f"capacity-{capacity}-profiled",
                "capacity": capacity,
                "profiled": True,
                "flags": [
                    *common,
                    "--profile",
                    "--trace-requests",
                    "--profile-timeline",
                    "--torch-profiler-step", "1",
                ],
            },
        ])
    return experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-dual-gpu", action="store_true")
    parser.add_argument(
        "--run-ids", nargs="+",
        help="Run only these experiment IDs, preserving declared order.",
    )
    return parser.parse_args()


def _validate_report(report: dict[str, Any], run_id: str, capacity: int) -> None:
    if report.get("result") != "PASS":
        raise RuntimeError(f"{run_id} run-report did not pass")
    configuration = report.get("configuration", {})
    if configuration.get("cache_experts_per_layer") != capacity:
        raise RuntimeError(f"{run_id} did not use cache capacity {capacity}")
    if configuration.get("prompt") != "Hello":
        raise RuntimeError(f"{run_id} prompt drifted from the M4 control")
    if configuration.get("max_new_tokens") != 4:
        raise RuntimeError(f"{run_id} token count drifted from the M4 control")
    correctness = report.get("correctness", {})
    if not correctness.get("all_40_layers_executed"):
        raise RuntimeError(f"{run_id} did not execute all 40 layers")
    if not correctness.get("baseline_tokens_exact"):
        raise RuntimeError(f"{run_id} failed the exact-token gate")
    generated = report.get("generation", {}).get("generated_token_ids")
    if generated != EXPECTED_TOKEN_IDS:
        raise RuntimeError(f"{run_id} generated token drift: {generated}")


def _validate_path_proof(path: Path, run_id: str) -> dict[str, Any]:
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("device_path_share") != 1.0:
        raise RuntimeError(f"{run_id} device path share is not 100%")
    if proof.get("host_path_fallback_calls") != 0:
        raise RuntimeError(f"{run_id} used the host fallback path")
    return proof


def _validate_timing_completeness(path: Path, run_id: str) -> None:
    timing = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for snapshot in timing.get("profile_snapshots", []):
        for layer in snapshot.get("layers", []):
            operations = layer.get("profile", {}).get("operations", {})
            dropped = operations.get("timing_events_dropped")
            if dropped != 0:
                failures.append({
                    "phase": snapshot.get("phase"),
                    "step": snapshot.get("step"),
                    "layer": layer.get("layer"),
                    "timing_events_dropped": dropped,
                })
    if failures:
        raise RuntimeError(
            f"{run_id} CUDA timing instrumentation is incomplete: {failures[:8]}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = capacity_experiments(args.require_dual_gpu)
    if args.run_ids:
        requested = set(args.run_ids)
        known = {item["run_id"] for item in experiments}
        unknown = sorted(requested - known)
        if unknown:
            raise RuntimeError(f"unknown experiment IDs: {unknown}")
        experiments = [item for item in experiments if item["run_id"] in requested]
        if [item["run_id"] for item in experiments] != args.run_ids:
            raise RuntimeError(
                "--run-ids must follow declared order: "
                + ", ".join(item["run_id"] for item in experiments)
            )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "controlled_variables": {
            "model": "Ornith-1.0-35B",
            "prompt": "Hello",
            "prompt_token_ids": [9419],
            "generated_token_ids": EXPECTED_TOKEN_IDS,
            "max_new_tokens": 4,
            "routing_top_k": 8,
            "precision": {"cache": "fp16", "transfer": "bf16"},
            "split_layer": 20,
            "warmup_generation": True,
        },
        "independent_variable": "cache_experts_per_layer",
        "capacities": list(CAPACITIES),
        "selected_run_ids": [item["run_id"] for item in experiments],
        "experiments": [],
    }
    summary_path = args.output_dir / "capacity-sweep-summary.json"
    progress_path = args.output_dir / "capacity-sweep-progress.jsonl"
    write_json_atomic(summary_path, summary)

    base = [
        sys.executable, "-u", "-X", "faulthandler",
        str(REPO_ROOT / "scripts/run_ornith_forensics.py"),
        "--model-dir", str(args.model_dir),
    ]
    for experiment in experiments:
        run_id = experiment["run_id"]
        capacity = experiment["capacity"]
        output = args.output_dir / "runs" / run_id
        command = [
            *base,
            "--output-dir", str(output),
            "--run-id", run_id,
            "--classification", "warm",
            "--prompt", "Hello",
            "--max-new-tokens", "4",
            *experiment["flags"],
        ]
        running = {
            "run_id": run_id,
            "capacity": capacity,
            "profiled": experiment["profiled"],
            "result": "RUNNING",
            "started_unix": time.time(),
        }
        summary["experiments"].append(running)
        write_json_atomic(summary_path, summary)
        append_progress(progress_path, running)
        try:
            elapsed = run_tee(
                command,
                args.output_dir / "logs" / f"{run_id}.log",
                os.environ.copy(),
                args.output_dir / "raw-allocation-trace.log",
            )
            report_path = output / "run-report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            _validate_report(report, run_id, capacity)
            proof = _validate_path_proof(output / "path-proof.json", run_id)

            if experiment["profiled"]:
                _validate_timing_completeness(
                    output / "timing-raw.json", run_id
                )
                trace_path = output / "expert-trace.jsonl"
                warmup_trace_path = output / "warmup-expert-trace.jsonl"
                if not trace_path.is_file() or trace_path.stat().st_size == 0:
                    raise RuntimeError(f"{run_id} expert trace is missing or empty")
                if (
                    not warmup_trace_path.is_file()
                    or warmup_trace_path.stat().st_size == 0
                ):
                    raise RuntimeError(f"{run_id} warmup expert trace is missing or empty")
                run_tee(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts/analyze_milestone25_expert_trace.py"),
                        str(trace_path),
                        "--output-dir", str(output),
                    ],
                    args.output_dir / "logs" / f"{run_id}-analysis.log",
                )
                run_tee(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts/analyze_milestone4_transfer_ledger.py"),
                        "--warmup-trace", str(warmup_trace_path),
                        "--measured-trace", str(trace_path),
                        "--capacity", str(capacity),
                        "--allow-missing-warmup-timing",
                        "--output", str(output / "transfer-ledger.json"),
                    ],
                    args.output_dir / "logs" / f"{run_id}-ledger.log",
                )
                run_tee(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts/analyze_milestone4_capacity_bounds.py"),
                        "--warmup-trace", str(warmup_trace_path),
                        "--measured-trace", str(trace_path),
                        "--output", str(output / "capacity-bounds.json"),
                    ],
                    args.output_dir / "logs" / f"{run_id}-capacity-bounds.log",
                )

            running.clear()
            running.update({
                "run_id": run_id,
                "capacity": capacity,
                "profiled": experiment["profiled"],
                "elapsed_seconds": elapsed,
                "result": "PASS",
                "report": str(report_path.relative_to(args.output_dir)),
                "generated_token_ids": report["generation"]["generated_token_ids"],
                "decode_tokens_per_second": report["generation"][
                    "single_stream_decode_tokens_per_second"
                ],
                "device_path_share": proof["device_path_share"],
                "host_path_fallback_calls": proof["host_path_fallback_calls"],
                "finished_unix": time.time(),
            })
            write_json_atomic(summary_path, summary)
            append_progress(progress_path, running)
        except Exception as exc:
            running.update({
                "result": "FAILED",
                "reason": f"{type(exc).__name__}: {exc}",
                "finished_unix": time.time(),
            })
            write_json_atomic(summary_path, summary)
            append_progress(progress_path, running)
            if isinstance(exc, subprocess.CalledProcessError):
                raise
            raise

    print("MILESTONE4_CAPACITY_SWEEP=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
