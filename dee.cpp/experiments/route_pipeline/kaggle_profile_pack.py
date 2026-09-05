#!/usr/bin/env python3
"""Kaggle profile package runner (route-pipeline host/sync profiler).

Runs the canonical exact decode TWICE (matched profile-off / profile-on),
validates the pair, ingests the profile-on records through the real
evidence pipeline (validate -> merge -> CSV/token rollups -> closure), and
emits machine-readable artifacts. No stdout scraping: walls come from
result.json metrics; spans come from host-profile.jsonl + stage-profile.json.

Canonical command contract (the command MUST accept these):
    <command> --out <dir> [--profile-stages] [--extra ...]
and write profile-run/{result.json, host-profile.jsonl, stage-profile.json,
correctness.json} under --out. See MOCK_CAMPAIGN_PROTOCOL.md; the mock
campaign implements exactly this contract.

Usage:
    python3 kaggle_profile_pack.py --run --out /tmp/profpack \\
        --command "python3 <campaign_decode.py> --tokens 16 ..." \\
        --prompt-hash <sha256> --tokens 16 [--extra-cmd-args "..."]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evidence import (  # noqa: E402
    EvidenceError,
    PairError,
    load_profile_run,
    merge_records,
    records_to_csv,
    rollup_per_token,
    validate_pair,
)
from host_profiler import compute_closure  # noqa: E402

# Flags that change execution behavior and are forbidden in a profile run.
DENY_FLAGS = (
    "--transfer-dtype", "--cache-dtype", "--budget", "--prefetch-depth",
    "--dynamic-quantization", "--topk", "--layers", "--profile-scenario",
    "int4", "int8",
)

ARTIFACTS = ("sync-profile-result.json", "per-layer.csv", "per-token.json",
             "closure.json", "host-sync-attribution.json")


def run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_dual_t4() -> dict:
    try:
        proc = run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total",
                    "--format=csv,noheader"])
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
    proc = run(["git", "rev-parse", "HEAD"])
    if proc.returncode != 0:
        raise SystemExit("REFUSE: cannot determine source hash (not a git checkout?)")
    return proc.stdout.strip()


def scan_forbidden(command: str) -> list[str]:
    return [flag for flag in DENY_FLAGS if flag in command]


def execute_pair(command: str, out: Path, extra: str, skip_gpu_check: bool) -> dict:
    """Run the canonical command in off/on modes. Returns run metadata.
    Walls are NEVER taken from stdout; only exit status gates progress."""
    runs = {}
    for mode, env_flag, extra_flags in (
            ("off", "0", ""), ("on", "1", "--profile-stages")):
        modedir = out / mode
        modedir.mkdir(parents=True, exist_ok=True)
        merged = dict(os.environ)
        merged["DEE_HOST_PROFILE"] = env_flag
        cmd = f"{command} --out {modedir} {extra_flags} {extra}".strip()
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=7200, env=merged)
        runs[mode] = {"exit": proc.returncode, "dir": str(modedir),
                      "env_DEE_HOST_PROFILE": env_flag}
        if proc.returncode != 0:
            runs[mode]["stderr_tail"] = proc.stderr[-2000:]
            raise RuntimeError(f"canonical command failed in {mode} mode")
    return runs


def assemble(out: Path, prompt_hash: str, tokens: int | None) -> dict:
    """Validate pair -> ingest on-run -> merge -> artifacts + closure."""
    off = load_profile_run(out / "off")
    on = load_profile_run(out / "on")
    pair = validate_pair(off, on)
    if prompt_hash != off["result"].get("prompt_hash"):
        raise PairError("INVALID_PROFILE_PAIR: prompt hash mismatch vs --prompt-hash")
    if tokens is not None and off["result"].get("tokens") != tokens:
        raise PairError("INVALID_PROFILE_PAIR: token count mismatch vs --tokens")
    cpp_records = on["stage"].get("host_layer_records", {}).get("records", [])
    if not isinstance(cpp_records, list):
        raise EvidenceError("stage-profile.json: host_layer_records.records must be a list")
    records, merge_notes = merge_records(cpp_records, on["py_rows"])
    if not records:
        raise EvidenceError("INVALID_PROFILE_EVIDENCE: zero merged records")
    wall_s = pair["on_wall_s"]
    closure = compute_closure(records, decode_wall_ms=wall_s * 1000.0)
    attribution = {"records": records, "merge_notes": merge_notes,
                   "provenance": "merged C++ HostSpan + Python DEE_HOST_PROFILE rows"}
    (out / "host-sync-attribution.json").write_text(
        json.dumps(attribution, indent=2), encoding="utf-8")
    (out / "per-layer.csv").write_text(records_to_csv(records), encoding="utf-8")
    (out / "per-token.json").write_text(
        json.dumps({"tokens": rollup_per_token(records)}, indent=2), encoding="utf-8")
    (out / "closure.json").write_text(json.dumps(closure, indent=2), encoding="utf-8")
    return {"pair": pair, "records": len(records), "closure": closure,
            "merge_notes": merge_notes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/tmp/profpack"))
    parser.add_argument("--command", default=None)
    parser.add_argument("--prompt-hash", default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--extra-cmd-args", default="")
    parser.add_argument("--skip-gpu-check", action="store_true",
                        help="for mock/local validation only; never for T4 evidence")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("need --run")
    if not args.command or not args.prompt_hash:
        parser.error("--run needs --command and --prompt-hash")

    t0 = time.time()
    report: dict = {
        "package": "route_pipeline/host-sync-profile",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        report["gpu"] = ("mocked-off" if args.skip_gpu_check
                         else check_dual_t4())
        report["source_sha"] = source_hash(args.repo)
        forbidden = scan_forbidden(args.command + " " + args.extra_cmd_args)
        if forbidden:
            raise SystemExit(
                f"REFUSE: behavior-changing flags in command: {forbidden}")
        report["forbidden_flags"] = []
        report["prompt_hash"] = args.prompt_hash
        args.out.mkdir(parents=True, exist_ok=True)
        report["runs"] = execute_pair(args.command, args.out,
                                      args.extra_cmd_args, args.skip_gpu_check)
        built = assemble(args.out, args.prompt_hash, args.tokens)
        report.update({k: v for k, v in built.items() if k != "records"})
        report["record_count"] = built["records"]
        report["artifacts"] = list(ARTIFACTS)
        report["status"] = "ok"
        code = 0
    except (EvidenceError, PairError, RuntimeError) as exc:
        code_name = getattr(exc, "code", type(exc).__name__)
        report["status"] = f"{code_name}: {exc}"
        code = 1
    report["elapsed_s"] = round(time.time() - t0, 3)
    (args.out / "sync-profile-result.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
