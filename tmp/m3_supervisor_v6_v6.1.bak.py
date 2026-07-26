#!/usr/bin/env python3
"""M3 v6.1 supervisor: push kernel, poll, download, run diagnostic analyzer.

v6.1 additions over v6:
- FIX-10c: stdout / stderr reconfigured to utf-8 at module top, so the
  print path itself cannot blow up with `charmap` on Windows hosts.
- FIX-10a: per-launch RUN_ID (timestamp + 8-hex) is generated at module
  import and threaded into:
    * `kaggle kernels push --env-vars RUN_ID=...` so the notebook picks
      it up from os.environ['RUN_ID'].
    * Every old-evidence child gets renamed `RUN_ID__<original>` when
      archived to <evidence>_prev/ before the new launch starts.
- FIX-10b: stale evidence_dir is moved into <evidence>_prev/ at startup
  so this run cannot be confused with prior runs.
- FIX-10d: while the kernel is RUNNING or COMPLETE, we attempt a
  best-effort `kaggle kernels output -p <evidence>/_live` pull every poll
  so partial logs are captured even if the kernel is later cancelled.
- FIX-10e: exit codes reflect terminal outcome (COMPLETE -> 0,
  ERROR/FAILED/CANCELLED -> 2, budget exhausted -> 3).

Forbidden changes outside this file + the notebook under
dee.cpp/kaggle/ornith-milestone3/: dee.cpp/src/*.cpp, dee.cpp/include/dee/*.h.
"""
from __future__ import annotations

# v6.1 / FIX-10c TIGHTEST: bind `sys`, immediately reconfigure stdout/stderr
# BEFORE any other stdlib import that might emit a DeprecationWarning or
# platform handshake message that would otherwise bypass the UTF-8 sink.
import sys
try:
    sys.stdout.reconfigure(
        encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr.reconfigure(
        encoding="utf-8", errors="replace", line_buffering=True
    )
except (AttributeError, OSError):
    pass

import argparse
import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLUG_DEFAULT = "nivind/dee-cpp-ornith-milestone-3-forensics"
KERNEL_DIR_DEFAULT = ROOT.parent / "dee.cpp" / "kaggle" / "ornith-milestone3"
EVIDENCE_DIR_DEFAULT = ROOT / "m3_v6_output"
ANALYZER_SCRIPT = ROOT / "m3_analyze_trace.py"
LOG_FILE = ROOT / "m3_supervisor_v6.log"
STATE_FILE = ROOT / "m3_supervisor_v6_state.json"
# FIX-10d-pivot: heartbeat JSONL tracks kernel liveness while polling;
# written OUTSIDE evidence_dir so it survives the terminal-state
# `download_artifacts` rmtree wipe.
HEARTBEAT_FILE = ROOT / "m3_v6_heartbeat.jsonl"

#  v6.1 / FIX-10a: unique run id.  Generated once at module import; the
#  same RUN_ID propagates to the Kaggle kernel via env-vars and to the
#  archived evidence_dir children via the move-into-_prev loop.
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


POLL_INTERVAL_SEC = 30.0
HARD_BUDGET_SEC_DEFAULT = 75 * 60

TERMINAL_STATES = {"COMPLETE", "ERROR", "FAILED", "CANCELLED"}
OK_STATES = {"COMPLETE"}
RETRY_STATES = {"ERROR", "FAILED", "CANCELLED"}


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str, tag: str = "SUP") -> None:
    line = "[" + stamp() + "] [" + tag + "] " + msg + "\n"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line)
    sys.stdout.write(line)
    sys.stdout.flush()


def save_state(payload: dict[str, object]) -> None:
    tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, STATE_FILE)
    except OSError:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def run(cmd, timeout=120, env=None):
    """Run shell command; return (rc, combined stdout+stderr text)."""
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout, env=env,
        )
        return cp.returncode, (cp.stdout or "") + (cp.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except FileNotFoundError as exc:
        return 127, "NOT_FOUND " + str(exc)
    except Exception as exc:
        return 1, "EXC " + type(exc).__name__ + ": " + str(exc)


def kaggle_invoke(args_list, timeout=180, env=None):
    """Invoke kaggle CLI; falls back to python -m kaggle if PATH is bare."""
    rc, out = run(["kaggle", *args_list], timeout=timeout)
    if rc == 0:
        return rc, out
    if rc != 127:
        return rc, out
    log("kaggle CLI missing on PATH; trying python -m kaggle fallback", "SUP")
    py_guess = shutil.which("python") or shutil.which("python3") or "python"
    return run([py_guess, "-m", "kaggle", *args_list], timeout=timeout, env=env)


def parse_status(raw):
    for line in raw.splitlines():
        if "has status" in line:
            return line.split("has status")[-1].strip().strip('"').upper()
    return None


def push_kernel(kernel_dir, env=None):
    rid = (env or {}).get("RUN_ID", "NONE")
    log("pushing kernel at " + str(kernel_dir) + " RUN_ID=" + rid, "PUSH")
    rc, out = kaggle_invoke(
        ["kernels", "push", "-p", str(kernel_dir)], timeout=300, env=env
    )
    log("kaggle kernels push rc=" + str(rc) + " (first 600 chars)\n" + out[:600], "PUSH")
    if rc != 0:
        log("kernel push FAILED rc=" + str(rc), "PUSH")
        return False
    new_version = None
    for line in out.splitlines():
        if "Version" in line or "version " in line:
            try:
                new_version = int(line.strip().split()[-1].rstrip("."))
            except (ValueError, IndexError):
                pass
    log("kernel push OK; new version: " + str(new_version), "PUSH")
    return True


def get_status(slug):
    rc, out = kaggle_invoke(["kernels", "status", slug], timeout=120)
    return parse_status(out), out


def download_artifacts(slug, evidence_dir):
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir, ignore_errors=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rc, out = kaggle_invoke(
        ["kernels", "output", slug, "-p", str(evidence_dir)], timeout=600
    )
    log("kaggle kernels output rc=" + str(rc) + " (first 500 chars)\n" + out[:500], "DL")
    if rc != 0:
        log("download FAILED", "DL")
        return False
    size = sum(p.stat().st_size for p in evidence_dir.rglob("*") if p.is_file())
    log("download OK; size=" + str(size), "DL")
    return True


def run_analyzer(evidence_dir, output_dir):
    log("running analyzer evidence=" + str(evidence_dir) + " -> " + str(output_dir), "ANL")
    rc, out = run(
        [sys.executable, str(ANALYZER_SCRIPT),
         "--evidence-dir", str(evidence_dir),
         "--output-dir", str(output_dir)],
        timeout=120,
    )
    log("analyzer rc=" + str(rc) + " (first 800 chars)\n" + out[:800], "ANL")
    return rc


def live_tail(slug, evidence_dir):
    """FIX-10d-pivot: best-effort partial download + heartbeat JSONL append.
    The heartbeat file lives at ROOT (not under evidence_dir) so
    terminal-state `download_artifacts` cannot wipe it.
    """
    target = evidence_dir / "_live"
    target.mkdir(parents=True, exist_ok=True)
    rc, _ = kaggle_invoke(
        ["kernels", "output", slug, "-p", str(target)], timeout=120
    )
    if rc != 0:
        return
    try:
        size = sum(
            p.stat().st_size
            for p in target.rglob("*")
            if p.is_file()
        )
    except Exception:
        size = -1
    try:
        with HEARTBEAT_FILE.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"poll_tick_utc": stamp(), "log_size_bytes": size},
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception as exc:
        log("heartbeat write failed: " + str(exc), "DL")
    log(
        "LIVE-TAIL tick ok log_size=" + str(size) + "B at " + str(target),
        "DL",
    )


def archive_stale(evidence_dir):
    """FIX-10b-pivot: rename active evidence_dir to
    `<evidence>_prev_at_<RUN_ID>` exactly once per launch, so this run
    cannot be confused with prior runs. The HEARTBEAT_FILE sibling
    (defined at module scope) survives because it lives at ROOT, not
    under evidence_dir.
    """
    if not evidence_dir.exists():
        evidence_dir.mkdir(parents=True, exist_ok=True)
        return
    if not any(evidence_dir.iterdir()):
        return
    dest = evidence_dir.with_name(
        evidence_dir.name + "_prev_at_" + RUN_ID
    )
    if dest.exists():
        # two consecutive pushes within the same RUN_ID: blow away the
        # duplicate so we never silently overwrite prior evidence.
        shutil.rmtree(dest, ignore_errors=True)
    try:
        shutil.move(str(evidence_dir), str(dest))
        log("archived evidence_dir -> " + str(dest), "SUP")
    except Exception as exc:
        log("could not archive " + str(evidence_dir) + ": " + str(exc), "SUP")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-push", action="store_true",
                        help="Skip the kernel push step.")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloading artifacts.")
    parser.add_argument("--no-archive", action="store_true",
                        help="Skip evidence-dir archive (use only on first run).")
    parser.add_argument("--no-live-tail", action="store_true",
                        help="Disable live-tail partial capture.")
    parser.add_argument("--slug", default=SLUG_DEFAULT)
    parser.add_argument("--kernel-dir", type=Path, default=KERNEL_DIR_DEFAULT)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR_DEFAULT)
    parser.add_argument("--budget-min", type=int, default=75)
    parser.add_argument("--commit-expected", default="4d8ccf2")
    args = parser.parse_args(argv)

    HARD_BUDGET_SEC = args.budget_min * 60
    log("=== m3_supervisor_v6.1 starting ===", "SUP")
    log("RUN_ID=" + RUN_ID + " slug=" + args.slug
        + " budget=" + str(args.budget_min) + "min"
        + " commit_expected=" + args.commit_expected, "SUP")
    log("kaggle on PATH: "
        + ("present at " + shutil.which("kaggle") if shutil.which("kaggle") else "MISSING"),
        "SUP")
    log("python: " + sys.executable + " version=" + sys.version.split()[0], "SUP")

    if not args.no_archive:
        archive_stale(args.evidence_dir)

    start = time.time()
    save_state({
        "schema_version": 2, "started_at": stamp(), "slug": args.slug,
        "run_id": RUN_ID, "evidence_dir": str(args.evidence_dir),
        "no_push": args.no_push, "no_download": args.no_download,
        "commit_expected": args.commit_expected,
    })

    if not args.no_push:
        env_for_kernel = {
            "RUN_ID": RUN_ID,
            "COMMIT_EXPECTED": args.commit_expected,
        }
        if not push_kernel(args.kernel_dir, env=env_for_kernel):
            log("push failed; will still poll status in case someone else submitted", "SUP")
    prev_state = None

    while True:
        elapsed = time.time() - start
        if elapsed > HARD_BUDGET_SEC:
            log("BUDGET EXCEEDED (" + str(round(elapsed/60, 1)) + " min > "
                + str(args.budget_min) + "); attempting best-effort download+analyzer",
                "SUP")
            if not args.no_download:
                download_artifacts(args.slug, args.evidence_dir)
            run_analyzer(args.evidence_dir, args.evidence_dir / "analysis")
            log("supervisor exits 3 (budget exhausted)", "SUP")
            return 3

        state, raw = get_status(args.slug)
        if state == "RUNNING" and not args.no_live_tail:
            live_tail(args.slug, args.evidence_dir)
        if state is None:
            log("no state parsed; raw=" + repr((raw or "")[:200]), "STATE")
        elif state != prev_state:
            log("STATE " + repr(prev_state) + " -> " + repr(state)
                + " (elapsed=" + str(int(elapsed)) + "s)", "STATE")
            prev_state = state
            save_state({
                "schema_version": 2, "elapsed_sec": elapsed,
                "last_kaggle_state": state, "raw_status": raw[:600],
                "started_at_iso": stamp(),
            })
            if state in TERMINAL_STATES:
                log("terminal state " + state + " reached; downloading + analyzing", "SUP")
                if not args.no_download:
                    download_artifacts(args.slug, args.evidence_dir)
                else:
                    log("download skipped (--no-download)", "DL")
                run_analyzer(args.evidence_dir, args.evidence_dir / "analysis")
                save_state({
                    "schema_version": 2, "elapsed_sec": elapsed,
                    "last_kaggle_state": state, "complete": True,
                    "completed_at": stamp(),
                })
                #  v6.1 / FIX-10e: distinguish recovered-completion from
                #  unsuccessful terminal states via exit code.
                if state in OK_STATES:
                    log("supervisor done after terminal COMPLETE; exits 0", "SUP")
                    return 0
                else:
                    log("supervisor done after non-COMPLETE terminal " + state
                        + "; exits 2", "SUP")
                    return 2
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
