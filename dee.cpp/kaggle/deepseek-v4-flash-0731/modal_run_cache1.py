"""CACHE1 campaign run on Modal dual-T4 (Kaggle weekly GPU quota exhausted).

Launches the sealed DS10 harness (identity stage=cache1, pinned commit
cf7cf09d8...) inside a Modal container with 2x NVIDIA T4, mirroring the
Kaggle kernel contract:

  - the container runs as root, so it can create /kaggle/temp + /kaggle/working
  - the harness self-clones the pinned repo commit into /kaggle/temp/dsv4-source
    and verifies harness + module SHA-256s against harness-identity-ds10.json
  - model tensors stream over HF range requests (Modal has outbound internet
    by default; the checkpoint is never staged on local disk)
  - evidence, logs and the evidence tarball land on a persistent Modal Volume

Usage:
    modal run dee.cpp/kaggle/deepseek-v4-flash-0731/modal_run_cache1.py
    modal run --detach dee.cpp/kaggle/deepseek-v4-flash-0731/modal_run_cache1.py

Expected cost: 2x T4 ($0.59/h each) for ~1.5-4h -> roughly $2-5 total.
Expected verdict: ACCEPT_CACHE_PARTIAL (the >=70% target is structurally
unreachable on the canonical 16-token trace, per CACHE1_ANALYSIS.md).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

PINNED_COMMIT = "710e82d26b5d2c5c15bfba6c8ce0ee64c626944b"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
REPOSITORY = "https://github.com/so-nerdyy/dee.git"
RUNNER_DIR = "/kaggle/temp/dsv4-runner"
HARNESS_REL = "dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_model_runtime.py"
LOG_PATH = "/kaggle/working/cache1-run.log"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "safetensors", "requests",
                 "huggingface_hub")
)

app = modal.App("dee-cpp-dsv4-cache1")
evidence_volume = modal.Volume.from_name(
    "dee-dsv4-cache1-evidence", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4:2",
    timeout=8 * 3600,  # 8h cap; v13 took ~4.4h (build + cold/warm + alternate)
    volumes={"/kaggle/working": evidence_volume},
    retries=0,
)
def run_cache1() -> dict:
    import shutil

    os.makedirs("/kaggle/temp", exist_ok=True)
    os.makedirs("/kaggle/working", exist_ok=True)
    log_path = Path(LOG_PATH)

    def log(line: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {line}\n")
        print(line, flush=True)

    summary: dict = {"pinned_commit": PINNED_COMMIT, "stage": "cache1"}

    try:
        if os.path.exists(RUNNER_DIR):
            shutil.rmtree(RUNNER_DIR)
        log(f"cloning {REPOSITORY} @ branch {BRANCH} ...")
        subprocess.run(
            ["git", "clone", "--branch", BRANCH, "--single-branch",
             REPOSITORY, RUNNER_DIR],
            check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", RUNNER_DIR, "checkout", "--quiet", PINNED_COMMIT],
            check=True, capture_output=True, text=True)
        head = subprocess.check_output(
            ["git", "-C", RUNNER_DIR, "rev-parse", "HEAD"], text=True).strip()
        if head != PINNED_COMMIT:
            raise RuntimeError(f"checkout mismatch: {head} != {PINNED_COMMIT}")
        log(f"checked out {head}")

        harness = Path(RUNNER_DIR) / HARNESS_REL
        if not harness.is_file():
            raise FileNotFoundError(harness)

        env = dict(os.environ)
        env["DS10_STAGE"] = "cache1"
        log("running harness (DS10_STAGE=cache1, T4x2) ...")
        t0 = time.monotonic()
        proc = subprocess.Popen(
            [sys.executable, str(harness)], env=env, cwd="/kaggle/working",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1)
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[harness] {line}\n")
            tail.append(line)
            if len(tail) > 60:
                tail.pop(0)
        code = proc.wait()
        wall_s = round(time.monotonic() - t0, 1)
        log(f"harness exit code {code} after {wall_s}s")
        summary["exit_code"] = code
        summary["wall_s"] = wall_s
        summary["log_tail"] = tail[-40:]

        verdict = "UNKNOWN"
        for line in tail:
            if line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip()
            if line.startswith("FATAL:"):
                verdict = f"FATAL: {line.split(':', 1)[1].strip()}"
        summary["verdict"] = verdict
    except Exception as exc:  # noqa: BLE001
        summary["verdict"] = f"{type(exc).__name__}: {exc}"
        log(f"runner error: {exc}")
        raise
    finally:
        evidence_volume.commit()
        ls = evidence_volume.listdir("/")
        summary["volume_root"] = [e.path for e in ls]
    return summary


if __name__ == "__main__":
    run_cache1.remote()  # dispatched to Modal by `modal run`
