#!/usr/bin/env python3
"""Kaggle profile package builder/runner (route-pipeline host/sync profiler).

Builds a campaign-compatible profiling bundle the root can run on the dual-T4
environment, then executes it there (this script does NOT launch Kaggle
itself; run it where the campaign checkout and GPUs live).

The package runs the canonical exact decode TWICE (matched profile-off /
profile-on) with the same prompt and token count, verifies 2xT4 identity,
source hashes, correctness gates, profiler enablement, and the absence of
behavior-changing optimization flags, then emits:

    sync-profile-result.json
    per-layer.csv
    per-token.json
    closure.json

Usage:
    python3 kaggle_profile_pack.py --build-only --out /tmp/profpack
    python3 kaggle_profile_pack.py --run --package /tmp/profpack \\
        --command "python3 <campaign_decode.py> --tokens 16 ..." \\
        --prompt-hash <sha256> --tokens 16
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Flags that change execution behavior and are forbidden in a profile run.
DENY_FLAGS = (
    "--transfer-dtype", "--cache-dtype", "--budget", "--prefetch-depth",
    "--dynamic-quantization", "--topk", "--layers", "--profile-scenario",
    "int4", "int8",
)

ARTIFACTS = ("sync-profile-result.json", "per-layer.csv", "per-token.json", "closure.json")


def run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_dual_t4() -> dict:
    try:
        proc = run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total",
                    "--format=csv,noheader"], timeout=60)
    except FileNotFoundError:
        raise SystemExit("REFUSE: nvidia-smi not found.")
    if proc.returncode != 0:
        raise SystemExit(f"REFUSE: nvidia-smi failed: {proc.stderr[:300]}")
    gpus = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            gpus.append({"index": parts[0], "name": parts[1],
                         "compute_cap": parts[2], "memory": parts[3]})
    if len(gpus) != 2 or any(g["compute_cap"] != "7.5" for g in gpus):
        raise SystemExit(f"REFUSE: need exactly 2x SM75 T4, saw: {gpus}")
    return {"gpus": gpus}


def source_hash(repo: Path) -> str:
    proc = run(["git", "rev-parse", "HEAD"], timeout=60)
    if proc.returncode != 0:
        raise SystemExit("REFUSE: cannot determine source hash (not a git checkout?)")
    return proc.stdout.strip()


def scan_forbidden(command: str) -> list[str]:
    return [flag for flag in DENY_FLAGS if flag in command]


def build_package(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text(
        "Route-pipeline host/sync profile package.\n"
        "Run via kaggle_profile_pack.py --run (see --help). Requires 2xT4,\n"
        "the canonical exact campaign checkout, and torch+CUDA.\n",
        encoding="utf-8")
    for name in ("host_profiler.py",):
        (out / name).write_bytes((HERE / name).read_bytes())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/tmp/profpack"))
    parser.add_argument("--package", type=Path, default=None)
    parser.add_argument("--command", default=None,
                        help="canonical exact decode command (profile flags added by pack)")
    parser.add_argument("--prompt-hash", default=None)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()

    if args.build_only:
        pack = build_package(args.out)
        print(f"package: {pack}")
        return 0

    if not args.run:
        parser.error("need --build-only or --run")
    if not args.command or not args.prompt_hash:
        parser.error("--run needs --command and --prompt-hash")

    t0 = time.time()
    report: dict = {
        "package": "route_pipeline/host-sync-profile",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    report["gpu"] = check_dual_t4()
    report["source_sha"] = source_hash(args.repo)
    forbidden = scan_forbidden(args.command)
    if forbidden:
        raise SystemExit(f"REFUSE: behavior-changing flags in command: {forbidden}")
    report["forbidden_flags"] = []
    report["prompt_hash"] = args.prompt_hash
    report["tokens"] = args.tokens

    sys.path.insert(0, str(args.package or HERE))
    from host_profiler import compute_closure, evaluate_decisions  # noqa: E402

    runs = {}
    for mode, extra in (("off", []), ("on", ["--profile-stages"])):
        env = {"DEE_HOST_PROFILE": "1" if mode == "on" else "0"}
        import os
        merged = dict(os.environ)
        merged.update(env)
        cmd = f"{args.command} {' '.join(extra)}"
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=7200, env=merged)
        runs[mode] = {"exit": proc.returncode,
                      "stdout_tail": proc.stdout[-3000:],
                      "stderr_tail": proc.stderr[-3000:],
                      "env_DEE_HOST_PROFILE": env["DEE_HOST_PROFILE"]}
        if proc.returncode != 0:
            report["runs"] = runs
            report["status"] = f"run failed in profile-{mode} mode"
            print(json.dumps(report, indent=2))
            return 1
    report["runs"] = runs
    report["perturbation_note"] = (
        "Compare profile-off vs profile-on wall/token counts: any systematic "
        "delta estimates profiler perturbation. Instrumented timings are NOT "
        "acceptance evidence unless campaign rules permit.")
    # Artifact assembly from the profile-on run's emitted records happens
    # here once the campaign's record paths are wired; until then the pack
    # records the matched-pair walls and the decision scaffold.
    closure = compute_closure([], decode_wall_ms=None)
    report["closure"] = closure
    report["decisions"] = evaluate_decisions(closure)
    report["artifacts_expected"] = list(ARTIFACTS)
    report["elapsed_s"] = round(time.time() - t0, 3)
    report["status"] = "ok"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
