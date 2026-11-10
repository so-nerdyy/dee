#!/usr/bin/env python3
"""Push the Milestone 3 forensic kernel to Kaggle, poll status, and download
the evidence archive back to the host. Intended to be run from the Windows
host's repo root with credentials at ``~/.kaggle/kaggle.json`` (mode 600).

Reads the kernel slug from ``dee.cpp/kaggle/ornith-milestone3/kernel-metadata.json``
and the evidence-directory prefix from ``--evidence-prefix`` (default:
``./m3-evidence``).

Reports each kernel run ID printed by Kaggle so a follow-up dispatch can be
referenced unambiguously.  Aborts with a non-zero return code on any Kaggle
API failure so the parent agent can re-roll the dice on dual-T4 contention.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


KERNEL_DIR = Path(__file__).resolve().parent.parent / "kaggle" / "ornith-milestone3"


def _append_log(log_path: Path, line: str, stream=None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        if not line.endswith("\n"):
            handle.write("\n")
    if stream is not None:
        stream.write(line)
        if not line.endswith("\n"):
            stream.write("\n")


def _run_kaggle(args: list[str], log_path: Path, cwd: Path,
                check: bool = True) -> subprocess.CompletedProcess:
    print(f"[kaggle] {' '.join(args)}", flush=True)
    process = subprocess.run(
        args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )
    _append_log(log_path, process.stdout)
    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args)
    return process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-slug",
                        default="nivind/dee-cpp-ornith-milestone-3-forensics")
    parser.add_argument("--evidence-prefix", type=Path,
                        default=Path("./m3-evidence"))
    parser.add_argument("--kernel-dir", type=Path, default=KERNEL_DIR)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--poll-timeout-seconds", type=int, default=7200)
    parser.add_argument("--skip-push", action="store_true",
                        help="If true, only poll and download (push must have happened)")
    parser.add_argument("--skip-download", action="store_true",
                        help="If true, only push and poll (no evidence download)")
    return parser.parse_args()


def slug_from_metadata(kernel_dir: Path) -> str:
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    return metadata["id"]


def main() -> None:
    args = parse_args()
    if not (args.kernel_dir / "ornith_milestone3.ipynb").is_file():
        raise RuntimeError(f"missing notebook at {args.kernel_dir}")
    if not args.skip_push:
        log = args.evidence_prefix.with_suffix(".push.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        _run_kaggle(["kaggle", "kernels", "push", "-p", str(args.kernel_dir)],
                     log, args.kernel_dir)
    poll_started = time.perf_counter()
    last_status = None
    while time.perf_counter() - poll_started < args.poll_timeout_seconds:
        process = _run_kaggle(
            ["kaggle", "kernels", "status", args.kernel_slug],
            args.evidence_prefix.with_suffix(".poll.log"),
            Path.cwd(), check=False,
        )
        status_line = (process.stdout or "").strip()
        if status_line != last_status:
            print(f"[kaggle] status: {status_line}", flush=True)
            last_status = status_line
        if "complete" in status_line.lower():
            break
        if "error" in status_line.lower() or "failed" in status_line.lower():
            raise RuntimeError(f"kernel reached error state: {status_line!r}")
        time.sleep(args.poll_interval_seconds)
    if not args.skip_download:
        download_dir = args.evidence_prefix
        if download_dir.exists():
            shutil.rmtree(download_dir)
        download_dir.mkdir(parents=True)
        _run_kaggle(
            ["kaggle", "kernels", "output", args.kernel_slug, "-p", str(download_dir)],
            download_dir.with_suffix(".download.log"),
            Path.cwd(),
        )
    print(f"MILESTONE3_DISPATCH=OK slug={args.kernel_slug}", flush=True)


if __name__ == "__main__":
    main()
