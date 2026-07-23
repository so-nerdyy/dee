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
        "--prior-audit", type=Path,
        default=REPO_ROOT / "benchmark_reports/milestone-2.5/work/prior-30-tps-audit.md",
    )
    parser.add_argument(
        "--kernel-slug", default="nivind/dee-cpp-ornith-milestone-2-5-forensics"
    )
    return parser.parse_args()


def run_tee(command: list[str], log_path: Path,
            environment: dict[str, str] | None = None) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
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
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return elapsed


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

    summary = {"schema_version": 1, "experiments": []}
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
        elapsed = run_tee(command, args.output_dir / "logs" / f"{run_id}.log",
                          environment)
        report_path = output / "run-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("result") != "PASS":
            raise RuntimeError(f"{run_id} did not pass")
        analysis_command = [
            sys.executable, str(REPO_ROOT / "scripts/analyze_milestone25_expert_trace.py"),
            str(output / "expert-trace.jsonl"),
            "--output-dir", str(output),
        ]
        if (output / "expert-trace.jsonl").stat().st_size:
            run_tee(analysis_command, args.output_dir / "logs" / f"{run_id}-analysis.log")
        summary["experiments"].append({
            "run_id": run_id,
            "elapsed_seconds": elapsed,
            "result": report["result"],
            "report": str(report_path.relative_to(args.output_dir)),
            "generated_token_ids": report["generation"]["generated_token_ids"],
            "all_40_layers": report["correctness"]["all_40_layers_executed"],
        })
        (args.output_dir / "matrix-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
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
