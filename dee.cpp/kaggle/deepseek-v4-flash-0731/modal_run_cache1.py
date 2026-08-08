"""CACHE1 campaign run on Modal dual-T4 (Kaggle weekly GPU quota exhausted).

Launches the sealed DS10 harness (identity stage=cache1, pinned commit
710e82d...) inside a Modal container with 2x NVIDIA T4, mirroring the
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

History:
  - v1 (cf7cf09): 8h timeout was too tight. CACHE1b workload = 2 full model
    builds + 3x 16-token decodes at ~240-494s/token (HTTP-bound expert
    fetches) needs ~7-10h; the function was killed at 8h with ZERO evidence
    (the harness only serializes evidence in its finally block).
  - v2 (34d3d1d): timeout=16h, heartbeat thread every 10 min that logs
    elapsed time + harness process liveness + per-GPU util/mem (nvidia-smi)
    + harness RSS, and commits the volume so progress is observable even
    while the harness prints nothing.
  - v3 (this file): CACHE1c ended INVALID_EXPERIMENT because the heartbeat
    thread died at 01:05Z on an unbounded evidence_volume.commit() and the
    harness then ran silently to the 16h timeout (zero gates).  Fixes:
      * volume.commit() now runs under a 120s hard timeout in a dedicated
        executor so a hung commit can never kill the heartbeat thread;
      * heartbeat loop is fully exception-guarded (keeps logging forever);
      * FUNCTION_TIMEOUT_SECONDS raised to 20h (CACHE1c was still in
        decode at 16h);
      * the harness itself now prints [cache1] progress lines per decode
        step (primary/warm/alternate) which the runner tails as
        [harness] lines, so progress is visible even without the heartbeat.

Expected cost: 2x T4 ($0.59/h each) for ~8-14h -> roughly $9-17.
Expected verdict: ACCEPT_CACHE_PARTIAL (the >=70% target is structurally
unreachable on the canonical 16-token trace, per CACHE1_ANALYSIS.md).
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

PINNED_COMMIT = "710e82d26b5d2c5c15bfba6c8ce0ee64c626944b"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
REPOSITORY = "https://github.com/so-nerdyy/dee.git"
RUNNER_DIR = "/kaggle/temp/dsv4-runner"
HARNESS_REL = "dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_model_runtime.py"
LOG_PATH = "/kaggle/working/cache1-run.log"
HEARTBEAT_SECONDS = 600  # 10 min
COMMIT_TIMEOUT_SECONDS = 120  # volume commit must never hang the heartbeat
FUNCTION_TIMEOUT_SECONDS = 20 * 3600  # v3: 20h (CACHE1c hit 16h in decode)

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
    timeout=FUNCTION_TIMEOUT_SECONDS,
    volumes={"/kaggle/working": evidence_volume},
    retries=0,
)
def run_cache1() -> dict:
    import shutil

    os.makedirs("/kaggle/temp", exist_ok=True)
    os.makedirs("/kaggle/working", exist_ok=True)
    log_path = Path(LOG_PATH)
    lock = threading.Lock()

    def log(line: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with lock:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[{ts}] {line}\n")
        print(line, flush=True)

    def _gpu_snapshot() -> str:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
                 "memory.total,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            return " ; ".join(lines) if lines else "nvidia-smi: no GPUs"
        except Exception as exc:  # noqa: BLE001
            return f"nvidia-smi unavailable: {type(exc).__name__}"

    def _rss_mb(pid: int) -> str:
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return line.split()[1]
        except Exception:  # noqa: BLE001
            pass
        return "?"

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

        # Heartbeat thread: proves the container is alive and working even
        # though the harness prints nothing for hours during build/decode.
        # v3: volume.commit() runs under a hard timeout via ThreadPoolExecutor
        # so a hung commit can NEVER kill the heartbeat (CACHE1c: heartbeat
        # died at 01:05Z on an unbounded commit() and the run went dark).
        stop_heartbeat = threading.Event()
        commit_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _commit_volume_bounded() -> str:
            """Commit with a hard timeout; returns status string."""
            try:
                fut = commit_pool.submit(evidence_volume.commit)
                try:
                    fut.result(timeout=COMMIT_TIMEOUT_SECONDS)
                    return "ok"
                except concurrent.futures.TimeoutError:
                    log("WARN volume.commit() exceeded "
                        f"{COMMIT_TIMEOUT_SECONDS}s; continuing without commit")
                    return "timeout"
            except Exception as exc:  # noqa: BLE001
                log(f"WARN volume.commit() failed: {type(exc).__name__}: {exc}")
                return f"error:{type(exc).__name__}"

        last_commit_ok = [time.monotonic()]

        def _heartbeat() -> None:
            while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
                try:
                    elapsed_h = round((time.monotonic() - t0) / 3600.0, 2)
                    alive = proc.poll() is None
                    snap = _gpu_snapshot()
                    rss = _rss_mb(proc.pid) if alive else "exited"
                    log(f"heartbeat: elapsed={elapsed_h}h harness_alive={alive} "
                        f"harness_rss_kB={rss} gpu=[{snap}]")
                    commit_status = _commit_volume_bounded()
                    if commit_status == "ok":
                        last_commit_ok[0] = time.monotonic()
                    # watchdog: if a commit keeps failing, still keep logging
                    # every heartbeat so the run never goes dark again.
                except Exception as exc:  # noqa: BLE001
                    log(f"heartbeat error (continuing): "
                        f"{type(exc).__name__}: {exc}")

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()

        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            log(f"[harness] {line}")
            tail.append(line)
            if len(tail) > 60:
                tail.pop(0)
        code = proc.wait()
        stop_heartbeat.set()
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
