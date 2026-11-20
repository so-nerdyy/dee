#!/usr/bin/env python3
"""Minimal bisect probe for the silent single-GPU generate-kernel deaths.

Runs exactly the same early phases as deepseek_v4_native_generate.py
(clone -> RESOURCES log -> cmake configure -> build dee_core -j2) and stops.
If this dies like the full script, the killer is in build/setup; if it
completes, the killer is in download/model-load/decode.

Every step writes a heartbeat file into /kaggle/working so a hard kill
still leaves a breadcrumb trail.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
ROOT = Path("/kaggle/working/dee")
DEE = ROOT / "dee.cpp"
BUILD = DEE / "build"
WORK = Path("/kaggle/working")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(WORK / "progress.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def heartbeat(stage: str) -> None:
    try:
        (WORK / "heartbeat.txt").write_text(
            f"{stage} @ {time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
            encoding="utf-8")
    except OSError:
        pass


def log_resources(stage: str) -> None:
    info = {}
    for mnt in ("/", "/tmp", "/kaggle/working"):
        try:
            t, u, f = shutil.disk_usage(mnt)
            info[mnt] = {"free_gb": round(f / 2**30, 1)}
        except OSError:
            pass
    try:
        mi = {}
        for ln in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = ln.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                mi[k] = round(int(v.split()[0]) / 2**20, 1)
        info["ram_gb"] = mi
    except OSError:
        pass
    log(f"RESOURCES[{stage}] {json.dumps(info)}")


def run(cmd: list[str]) -> None:
    t0 = time.monotonic()
    log(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout or "")[-800:] + (r.stderr or "")[-800:]
    if r.returncode != 0:
        log(f"FAILED rc={r.returncode}\n{tail}")
        raise SystemExit(1)
    log(f"ok ({time.monotonic() - t0:.0f}s)")


def main() -> int:
    gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    log(f"GPUS: {gpus.replace(chr(10), ' | ')}")
    log_resources("startup")

    heartbeat("clone-start")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    run(["git", "clone", "--depth", "1", "--branch", BRANCH,
         "--single-branch", REPO, str(ROOT)])
    head = subprocess.check_output(
        ["git", "-C", str(DEE), "rev-parse", "HEAD"], text=True).strip()
    log(f"HEAD {head}")

    heartbeat("configure-start")
    log_resources("pre-configure")
    run(["cmake", "-S", str(DEE), "-B", str(BUILD),
         "-DCMAKE_CUDA_ARCHITECTURES=60;75", "-DDEE_CUDA=ON",
         "-DDEE_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=Release"])

    heartbeat("build-start")
    jobs = max(1, min(2, os.cpu_count() or 2))
    log(f"building dee_core -j{jobs}")
    t0 = time.monotonic()
    r = subprocess.run(["cmake", "--build", str(BUILD),
                        "--target", "dee_core", "--", f"-j{jobs}"],
                       capture_output=True, text=True)
    dt = time.monotonic() - t0
    if r.returncode != 0:
        log(f"BUILD FAILED rc={r.returncode} after {dt:.0f}s\n"
            + (r.stderr or "")[-3000:])
        return 1
    log(f"dee_core built ok in {dt:.0f}s")

    heartbeat("build-done")
    log_resources("post-build")
    log("PROBE COMPLETE — early phases are NOT the killer; "
        "investigate download/model-load phases next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
