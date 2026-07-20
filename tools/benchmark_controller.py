#!/usr/bin/env python3
"""Bounded, evidence-first controller for the Lightning T4 regression workload.

The controller intentionally receives the SSH target as an argument. It never
reads SSH configuration, private keys, or authentication material. Every remote
action is preceded by a clean-check/GPU-check preflight; checkout resets are
confined to the supplied checkout path (normally ~/dee/dee).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def invoke(command: list[str], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_seconds": time.monotonic() - started,
    }


def require_ok(result: dict[str, Any], label: str) -> None:
    if result["exit_code"] == 0:
        return
    raise RuntimeError(f"{label} failed ({result['exit_code']}): {result['stderr'] or result['stdout']}")


def local_clean(repo: Path) -> None:
    result = invoke(["git", "status", "--porcelain"], 30)
    require_ok(result, "local git status")
    if result["stdout"].strip():
        raise RuntimeError("local worktree is not clean; refuse to benchmark an ambiguous candidate")


def ssh(target: str, command: str, timeout: int) -> dict[str, Any]:
    return invoke(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", target, command], timeout)


def preflight(target: str, remote_repo: str, timeout: int, expected_branch: str | None = None,
              expected_commit: str | None = None) -> dict[str, Any]:
    command = (
        "set -e; "
        "printf '__GPU__\\n'; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; "
        f"cd {remote_repo}; "
        "printf '__BRANCH__\\n'; git branch --show-current; "
        "printf '__COMMIT__\\n'; git rev-parse HEAD; "
        "printf '__STATUS__\\n'; git status --porcelain"
    )
    result = ssh(target, command, timeout)
    require_ok(result, "SSH preflight")
    chunks = re.split(r"^__(GPU|BRANCH|COMMIT|STATUS)__\n", result["stdout"], flags=re.MULTILINE)
    if len(chunks) < 9:
        raise RuntimeError(f"unparseable SSH preflight: {result['stdout']}")
    fields = dict(zip(chunks[1::2], chunks[2::2]))
    gpu = fields["GPU"].strip()
    branch = fields["BRANCH"].strip()
    commit = fields["COMMIT"].strip()
    status = fields["STATUS"].strip()
    if "T4" not in gpu:
        raise RuntimeError(f"expected a T4 GPU, got: {gpu}")
    if status:
        raise RuntimeError(f"remote worktree is not clean: {status}")
    if expected_branch and branch != expected_branch:
        raise RuntimeError(f"remote branch {branch} != expected {expected_branch}")
    if expected_commit and commit != expected_commit:
        raise RuntimeError(f"remote commit {commit} != expected {expected_commit}")
    return {"gpu": gpu, "branch": branch, "commit": commit, "raw": result}


def throughput(stdout: str) -> float:
    match = re.search(r"tokens per second\s*:\s*([0-9]+(?:\.[0-9]+)?)", stdout)
    if not match:
        raise RuntimeError("benchmark output did not contain 'tokens per second'")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="noninteractive SSH target")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--stage", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--remote-repo", default="~/dee/dee")
    parser.add_argument("--prepare", help="one-time remote build/setup command")
    parser.add_argument("--benchmark", required=True, help="remote command run exactly three times")
    parser.add_argument("--profile", help="optional one-time remote profiler command")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--reports", type=Path, default=Path("benchmark_reports"))
    args = parser.parse_args()

    local_clean(Path.cwd())
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "candidate": {"branch": args.branch, "commit": args.commit},
        "runs": [],
    }

    # A clean, read-only preflight is required before the only destructive
    # operation, which is explicitly confined to the supplied remote checkout.
    report["before_sync"] = preflight(args.target, args.remote_repo, args.timeout)
    sync = ssh(args.target,
               f"set -e; cd {args.remote_repo}; git fetch origin; git switch {args.branch}; git reset --hard origin/{args.branch}",
               args.timeout)
    require_ok(sync, "remote checkout sync")
    report["sync"] = sync
    report["after_sync"] = preflight(args.target, args.remote_repo, args.timeout, args.branch, args.commit)

    if args.prepare:
        preflight(args.target, args.remote_repo, args.timeout, args.branch, args.commit)
        prepared = ssh(args.target, f"set -e; cd {args.remote_repo}/dee.cpp; {args.prepare}", args.timeout)
        require_ok(prepared, "remote prepare")
        report["prepare"] = prepared

    for index in range(3):
        preflight(args.target, args.remote_repo, args.timeout, args.branch, args.commit)
        run = ssh(args.target, f"set -e; cd {args.remote_repo}/dee.cpp; {args.benchmark}", args.timeout)
        require_ok(run, f"benchmark run {index + 1}")
        run["throughput_tokens_per_second"] = throughput(run["stdout"])
        report["runs"].append(run)

    if args.profile:
        preflight(args.target, args.remote_repo, args.timeout, args.branch, args.commit)
        profile = ssh(args.target, f"set -e; cd {args.remote_repo}/dee.cpp; {args.profile}", args.timeout)
        require_ok(profile, "remote profile")
        report["profile"] = profile

    values = [run["throughput_tokens_per_second"] for run in report["runs"]]
    report["median_tokens_per_second"] = statistics.median(values)
    args.reports.mkdir(parents=True, exist_ok=True)
    path = args.reports / f"{args.stage}-{args.commit[:12]}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    print(f"median_tokens_per_second={report['median_tokens_per_second']:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"benchmark controller: {exc}", file=sys.stderr)
        raise SystemExit(1)
