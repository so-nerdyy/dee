#!/usr/bin/env python3
"""Orchestrate the controlled Milestone 2.5 diagnostic matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-dual-gpu", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--layer0-regression", type=Path)
    parser.add_argument("--router-parity", type=Path)
    parser.add_argument(
        "--run-ids", nargs="+",
        help="Run only these experiment IDs, preserving the declared order.",
    )
    parser.add_argument(
        "--prior-audit", type=Path,
        default=REPO_ROOT / "benchmark_reports/milestone-2.5/work/prior-30-tps-audit.md",
    )
    parser.add_argument(
        "--kernel-slug", default="nivind/dee-cpp-ornith-milestone-2-5-forensics"
    )
    return parser.parse_args()


def run_tee(command: list[str], log_path: Path,
            environment: dict[str, str] | None = None,
            trace_path: Path | None = None) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trace = None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace = trace_path.open("a", encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if trace is not None and (
                    "[ta_" in line or "[DEE_TA_" in line
                    or "TRACE_ALLOC" in line):
                trace.write(line)
                trace.flush()
                os.fsync(trace.fileno())
        return_code = process.wait()
    if trace is not None:
        trace.close()
    elapsed = time.perf_counter() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return elapsed


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_progress(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = [
        sys.executable, "-u", "-X", "faulthandler",
        str(REPO_ROOT / "scripts/run_ornith_forensics.py"),
        "--model-dir", str(args.model_dir),
    ]
    dual = ["--require-dual-gpu"] if args.require_dual_gpu else []
    experiments = [
        {
            "run_id": "dual-cold-primary", "classification": "cold",
            "prompt": "Hello", "tokens": 4,
            "flags": ["--profile", "--trace-requests", "--profile-timeline",
                      "--torch-profiler-step", "1", "--reference-parity",
                      "--tensor-map", *dual],
        },
        {
            "run_id": "dual-warm-profiled", "classification": "warm",
            "prompt": "Hello", "tokens": 4,
            "flags": ["--warmup-generation", "--profile", "--trace-requests",
                      "--profile-timeline", "--torch-profiler-step", "1", *dual],
        },
        {
            "run_id": "dual-warm-control", "classification": "warm",
            "prompt": "Hello", "tokens": 4,
            "flags": ["--warmup-generation", *dual],
        },
        {
            "run_id": "dual-warm-reference-present", "classification": "warm",
            "prompt": "Hello", "tokens": 4,
            "flags": ["--warmup-generation", "--profile", "--trace-requests",
                      "--profile-timeline", "--reference-parity", *dual],
        },
        {
            "run_id": "dual-cache-disabled", "classification": "diagnostic",
            "prompt": "Hello", "tokens": 2,
            "flags": ["--warmup-generation", "--cache-disabled", "--profile",
                      "--trace-requests", "--profile-timeline", *dual],
        },
        {
            "run_id": "dual-cache-capacity-4", "classification": "diagnostic",
            "prompt": "Hello", "tokens": 2,
            "flags": ["--warmup-generation", "--cache-experts", "4",
                      "--allow-sub-topk-cache", "--profile", "--trace-requests",
                      "--profile-timeline", *dual],
        },
        {
            "run_id": "dual-long-prompt", "classification": "diagnostic",
            "prompt": "The quick brown fox jumps over the lazy dog.", "tokens": 2,
            "flags": ["--warmup-generation", "--profile", "--trace-requests",
                      "--profile-timeline", *dual],
        },
        {
            "run_id": "dual-one-token", "classification": "diagnostic",
            "prompt": "Hello", "tokens": 1,
            "flags": ["--warmup-generation", "--profile", "--trace-requests",
                      "--profile-timeline", *dual],
        },
        {
            "run_id": "single-t4-warm", "classification": "diagnostic",
            "prompt": "Hello", "tokens": 2,
            "flags": ["--warmup-generation", "--profile", "--trace-requests",
                      "--profile-timeline"],
            "single_gpu": True,
        },
    ]
    if args.run_ids:
        requested = set(args.run_ids)
        known = {item["run_id"] for item in experiments}
        unknown = sorted(requested - known)
        if unknown:
            raise RuntimeError(f"unknown experiment IDs: {unknown}")
        experiments = [
            item for item in experiments if item["run_id"] in requested
        ]
        if [item["run_id"] for item in experiments] != args.run_ids:
            raise RuntimeError(
                "--run-ids must follow the declared matrix order: "
                + ", ".join(item["run_id"] for item in experiments)
            )

    summary = {
        "schema_version": 2,
        "selected_run_ids": [item["run_id"] for item in experiments],
        "experiments": [],
    }
    summary_path = args.output_dir / "matrix-summary.json"
    progress_path = args.output_dir / "matrix-progress.jsonl"
    write_json_atomic(summary_path, summary)
    for experiment in experiments:
        run_id = experiment["run_id"]
        output = args.output_dir / "runs" / run_id
        command = [
            *base,
            "--output-dir", str(output),
            "--run-id", run_id,
            "--classification", experiment["classification"],
            "--prompt", experiment["prompt"],
            "--max-new-tokens", str(experiment["tokens"]),
            *experiment["flags"],
        ]
        environment = os.environ.copy()
        if experiment.get("single_gpu"):
            environment["CUDA_VISIBLE_DEVICES"] = "0"
        running = {
            "run_id": run_id,
            "classification": experiment["classification"],
            "result": "RUNNING",
            "started_unix": time.time(),
        }
        summary["experiments"].append(running)
        write_json_atomic(summary_path, summary)
        append_progress(progress_path, running)
        try:
            elapsed = run_tee(
                command, args.output_dir / "logs" / f"{run_id}.log",
                environment,
                args.output_dir / "raw-allocation-trace.log",
            )
        except subprocess.CalledProcessError as exc:
            running.update({
                "result": "FAILED",
                "return_code": exc.returncode,
                "finished_unix": time.time(),
            })
            write_json_atomic(summary_path, summary)
            append_progress(progress_path, running)
            raise
        try:
            report_path = output / "run-report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("result") != "PASS":
                raise RuntimeError(f"{run_id} run-report did not pass")
            trace_path = output / "expert-trace.jsonl"
            trace_expected = run_id != "dual-warm-control"
            if not trace_path.is_file() or (
                    trace_expected and trace_path.stat().st_size == 0):
                raise RuntimeError(f"{run_id} expert trace is missing or empty")
            analysis_command = [
                sys.executable,
                str(REPO_ROOT / "scripts/analyze_milestone25_expert_trace.py"),
                str(trace_path),
                "--output-dir", str(output),
            ]
            if trace_expected:
                run_tee(
                    analysis_command,
                    args.output_dir / "logs" / f"{run_id}-analysis.log",
                )
            required_analysis = [] if not trace_expected else [
                output / "expert-cache-analysis.json",
                output / "transfer-analysis.json",
            ]
            missing_analysis = [
                str(path) for path in required_analysis
                if not path.is_file() or path.stat().st_size == 0
            ]
            if missing_analysis:
                raise RuntimeError(
                    f"{run_id} trace postprocessing omitted: {missing_analysis}"
                )
            running.clear()
            running.update({
                "run_id": run_id,
                "elapsed_seconds": elapsed,
                "result": report["result"],
                "report": str(report_path.relative_to(args.output_dir)),
                "generated_token_ids": report["generation"]["generated_token_ids"],
                "all_40_layers": report["correctness"]["all_40_layers_executed"],
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
            raise
    if not args.skip_aggregate:
        if args.layer0_regression is None:
            raise RuntimeError("--layer0-regression is required unless --skip-aggregate is used")
        aggregate = [
            sys.executable, str(REPO_ROOT / "scripts/analyze_milestone25_matrix.py"),
            "--matrix-dir", str(args.output_dir),
            "--output-dir", str(args.output_dir),
            "--prior-audit", str(args.prior_audit),
            "--layer0-regression", str(args.layer0_regression),
            "--kernel-slug", args.kernel_slug,
        ]
        if args.router_parity is not None:
            aggregate.extend(["--router-parity", str(args.router_parity)])
        run_tee(aggregate, args.output_dir / "logs/matrix-analysis.log")
    print("MILESTONE25_MATRIX=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
