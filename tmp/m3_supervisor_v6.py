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
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def _m3v62_resolve_kaggle_dir():
    """Return the user's ~/.kaggle directory on any platform.

    Windows `pathlib.Path.home()` ERRORS when only ``HOME`` (Cygwin form
    ``C:\\Users\\carth``) is set and ``USERPROFILE`` is missing -- which
    is exactly what `setsid + nohup </dev/null` produces for v10-style
    supervised launches. Try ``USERPROFILE`` first (Windows Python's
    real home), fall back to ``HOME`` (Cygwin/Linux MSYS form), and
    finally ``HOMEPATH``+``HOMEDRIVE`` as a last resort. Returning the
    attempted path even if it does not exist is fine because callers
    guard with ``is_file()``.
    """
    up = os.environ.get("USERPROFILE")
    if up:
        return Path(up) / ".kaggle"
    home = os.environ.get("HOME")
    if home:
        return Path(home) / ".kaggle"
    drive = os.environ.get("HOMEDRIVE", "")
    pth = os.environ.get("HOMEPATH", "")
    if drive and pth:
        return Path(drive + pth) / ".kaggle"
    return Path.home() / ".kaggle"  # last resort; raises if undetermined


#  v6.2 / FIX-13: propagate Kaggle auth into every subprocess lineage.
#  The Kaggle CLI honours KAGGLE_USERNAME + KAGGLE_KEY (legacy) and
#  KAGGLE_API_TOKEN (post-OAuth `kaggle auth login`).  When invoked via
#  a detached MSYS/MINGW supervisor process tree, the in-process subprocess
#  does NOT always inherit the OAuth credentials.json automatically; we
#  read it explicitly here so every `subprocess.run(["kaggle", ...])` works.
try:
    _kdir = _m3v62_resolve_kaggle_dir()
    _leg = _kdir / "kaggle.json"
    if _leg.is_file():
        _j = json.loads(_leg.read_text(encoding="utf-8"))
        for _src, _env in (("username", "KAGGLE_USERNAME"), ("key", "KAGGLE_KEY")):
            if _src in _j and not os.environ.get(_env):
                os.environ[_env] = _j[_src]
    _cred = _kdir / "credentials.json"
    if _cred.is_file():
        _c = json.loads(_cred.read_text(encoding="utf-8"))
        # credentials.json OAuth format, post-2026 wins over legacy
        for _src, _env in (
            ("kaggle_username", "KAGGLE_USERNAME"),
            ("kaggle_key", "KAGGLE_KEY"),
            ("kaggle_token", "KAGGLE_API_TOKEN"),
        ):
            if _src in _c and _c[_src]:
                os.environ[_env] = _c[_src]
except Exception:
    pass

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
LOCK_FILE = ROOT / "m3_supervisor_v6.lock"

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


def _pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_supervisor_lock():
    payload = {"pid": os.getpid(), "run_id": RUN_ID, "created_at": stamp()}
    for _ in range(2):
        try:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(payload) + "\n")
            return payload
        except FileExistsError:
            try:
                existing = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
            if _pid_alive(existing.get("pid")):
                raise RuntimeError(
                    "another M3 supervisor owns " + str(LOCK_FILE)
                    + ": " + json.dumps(existing)
                )
            try:
                LOCK_FILE.unlink()
            except FileNotFoundError:
                pass
    raise RuntimeError("could not acquire supervisor lock " + str(LOCK_FILE))


def release_supervisor_lock(payload):
    try:
        current = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        if current.get("pid") == payload.get("pid") and (
                current.get("run_id") == payload.get("run_id")):
            LOCK_FILE.unlink()
    except (FileNotFoundError, OSError, ValueError):
        pass


def save_state(payload: dict[str, object], *, merge: bool = True) -> None:
    tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
    try:
        current = {}
        if merge and STATE_FILE.is_file():
            try:
                current = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                current = {}
        current.update(payload)
        tmp.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, STATE_FILE)
    except OSError:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def run(cmd, timeout=120, env=None):
    """Run shell command; return (rc, combined stdout+stderr text).

    v6.2 / FIX-12: merge PYTHONUTF8 + PYTHONIOENCODING into every
    child env so subprocess Python (including the Kaggle CLI) never
    defaults to cp1252 charmap and crashes mid-write on non-ASCII
    bytes captured from the kernel log.
    """
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUTF8", "1")
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout, env=merged_env,
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
    rc, out = run(["kaggle", *args_list], timeout=timeout, env=env)
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
            value = line.split("has status")[-1].strip().strip('"').upper()
            value = value.rsplit(".", 1)[-1]
            if value == "FINISHED":
                return "COMPLETE"
            if value == "CANCELED":
                return "CANCELLED"
            return value
    return None


def _stage_kernel(kernel_dir, run_id, commit_expected):
    """Create the exact, isolated kernel source tree that Kaggle will execute.

    Kaggle's current CLI has no --env-vars option. Injecting identity into a
    temporary notebook copy is therefore the only local-to-remote path that is
    both explicit and inspectable. The tracked notebook is never modified.
    """
    stage = ROOT / ("m3_kernel_stage_" + run_id)
    resolved_root = ROOT.resolve()
    if stage.exists():
        resolved_stage = stage.resolve()
        if resolved_stage.parent != resolved_root or not stage.name.startswith(
                "m3_kernel_stage_"):
            raise RuntimeError("refusing to replace unsafe kernel stage " + str(stage))
        shutil.rmtree(stage)
    shutil.copytree(kernel_dir, stage)
    metadata_path = stage / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    notebook_path = stage / metadata["code_file"]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    found_run_id = 0
    found_commit = 0
    found_nonce = 0
    harness_nonce = uuid.uuid4().hex
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        for index, line in enumerate(source):
            if line.startswith("RUN_ID = "):
                source[index] = "RUN_ID = " + repr(run_id) + "\n"
                found_run_id += 1
            elif line.startswith("COMMIT_EXPECTED = "):
                source[index] = (
                    "COMMIT_EXPECTED = " + repr(commit_expected) + "\n"
                )
                found_commit += 1
            elif line.startswith("HARNESS_NONCE = "):
                source[index] = "HARNESS_NONCE = " + repr(harness_nonce) + "\n"
                found_nonce += 1
    if found_run_id != 1 or found_commit != 1 or found_nonce != 1:
        raise RuntimeError(
            "kernel identity injection failed: RUN_ID sites="
            + str(found_run_id) + " COMMIT_EXPECTED sites=" + str(found_commit)
            + " HARNESS_NONCE sites=" + str(found_nonce)
        )
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    notebook_sha256 = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    identity = {
        "schema_version": 1,
        "run_id": run_id,
        "commit_expected": commit_expected,
        "notebook": notebook_path.name,
        "notebook_sha256": notebook_sha256,
        "harness_nonce": harness_nonce,
        "staged_at": stamp(),
    }
    (stage / "run-identity.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    return stage, identity


def push_kernel(kernel_dir, env=None):
    rid = (env or {}).get("RUN_ID", "NONE")
    commit_expected = (env or {}).get("COMMIT_EXPECTED", "")
    stage, identity = _stage_kernel(kernel_dir, rid, commit_expected)
    log(
        "pushing isolated kernel stage " + str(stage)
        + " RUN_ID=" + rid
        + " notebook_sha256=" + str(identity["notebook_sha256"]),
        "PUSH",
    )
    try:
        rc, out = kaggle_invoke(
            ["kernels", "push", "-p", str(stage)], timeout=300, env=env
        )
    finally:
        resolved_stage = stage.resolve()
        if (resolved_stage.parent == ROOT.resolve()
                and stage.name.startswith("m3_kernel_stage_")):
            shutil.rmtree(stage, ignore_errors=True)
    log("kaggle kernels push rc=" + str(rc) + " (first 600 chars)\n" + out[:600], "PUSH")
    if rc != 0:
        log("kernel push FAILED rc=" + str(rc), "PUSH")
        return False
    new_version = None
    for line in out.splitlines():
        match = re.search(r"\bversion\s+(\d+)\b", line, flags=re.IGNORECASE)
        if match:
            new_version = int(match.group(1))
    log("kernel push OK; new version: " + str(new_version), "PUSH")
    save_state({
        "kaggle_version": new_version,
        "notebook_sha256": identity["notebook_sha256"],
        "harness_nonce": identity["harness_nonce"],
    })
    return True


def get_status(slug):
    rc, out = kaggle_invoke(["kernels", "status", slug], timeout=120)
    return parse_status(out), out


def download_artifacts(slug, evidence_dir):
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


EXPECTED_RUN_IDS = [
    "dual-cold-primary",
    "dual-warm-profiled",
    "dual-warm-control",
    "dual-warm-reference-present",
    "dual-cache-disabled",
    "dual-cache-capacity-4",
    "dual-long-prompt",
]


def find_evidence_root(evidence_dir, run_id):
    candidates = [
        p for p in evidence_dir.rglob("ornith-milestone3-evidence-" + run_id)
        if p.is_dir() and "_live" not in p.parts
    ]
    return candidates[0] if len(candidates) == 1 else None


def validate_artifacts(
    evidence_dir, run_id, commit_expected=None, harness_nonce=None
):
    """Validate the downloaded experimental identity and required evidence."""
    candidates = [
        p for p in evidence_dir.rglob("ornith-milestone3-evidence-" + run_id)
        if p.is_dir() and "_live" not in p.parts
    ]
    if len(candidates) != 1:
        result = {
            "result": "FAIL",
            "run_id": run_id,
            "reason": "expected exactly one run-specific evidence root",
            "candidates": [str(p) for p in candidates],
        }
    else:
        root = candidates[0]
        if harness_nonce is None:
            try:
                harness_nonce = json.loads(
                    STATE_FILE.read_text(encoding="utf-8")
                ).get("harness_nonce")
            except (OSError, TypeError, ValueError):
                harness_nonce = None
        required = [
            root / "matrix-summary.json",
            root / "final_report.json",
            root / "summary_metrics.csv",
            root / "analysis" / "milestone-3-report.json",
            root / "analysis" / "correctness-report.json",
            root / "analysis" / "layer-timing.json",
            root / "analysis" / "transfer-analysis.json",
            root / "analysis" / "multi-gpu-timeline.json",
            root / "analysis" / "overlap-analysis.json",
            root / "analysis" / "expert-cache-analysis.json",
            root / "analysis" / "synchronization-analysis.json",
            root / "analysis" / "host-memory-breakdown.json",
            root / "analysis" / "gpu-memory-breakdown.json",
            root / "analysis" / "path-proof.json",
            root / "analysis" / "expert-trace.jsonl.gz",
            root / "analysis" / "profiler-summary.md",
            root / "analysis" / "bottleneck-ranking.json",
            root / "ornith-router-parity.json",
            root / "ornith-layer0-regression.json",
            root / "build-manifest.json",
            root / "commit-binary-identity.json",
            root / "artifact-manifest.json",
            root / "lifetime-analysis" / "lifetime_report.json",
            root / "lifetime-analysis" / "abort_summary.txt",
            root / "lifetime-analysis" / "markers_index.json",
            root / "post-download-manifest.json",
        ]
        missing = [
            str(path.relative_to(root))
            for path in required
            if not path.is_file() or path.stat().st_size == 0
        ]
        host_generated = {
            "lifetime-analysis/lifetime_report.json",
            "lifetime-analysis/abort_summary.txt",
            "lifetime-analysis/markers_index.json",
            "post-download-manifest.json",
        }
        kernel_required = {
            path.relative_to(root).as_posix()
            for path in required
            if path != root / "artifact-manifest.json"
            and path.relative_to(root).as_posix() not in host_generated
        }
        summary_ok = False
        manifest_ok = False
        identity_ok = False
        hash_errors = []
        lifetime_ok = False
        trace_gzip_ok = False
        summary_path = root / "matrix-summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                experiments = summary.get("experiments", [])
                summary_ok = (
                    summary.get("selected_run_ids") == EXPECTED_RUN_IDS
                    and
                    len(experiments) == 7
                    and all(item.get("result") == "PASS" for item in experiments)
                )
            except (OSError, ValueError):
                summary_ok = False
        try:
            manifest = json.loads(
                (root / "artifact-manifest.json").read_text(encoding="utf-8")
            )
            entry_rows = manifest.get("artifacts", [])
            entries = {}
            for item in entry_rows:
                relative = item["path"]
                posix_path = PurePosixPath(relative)
                if (
                    posix_path.is_absolute()
                    or ".." in posix_path.parts
                    or relative in entries
                    or relative != posix_path.as_posix()
                ):
                    hash_errors.append(relative + ": unsafe or duplicate path")
                    continue
                entries[relative] = item
            for relative, entry in entries.items():
                path = root / relative
                if not path.is_file():
                    hash_errors.append(relative + ": missing")
                    continue
                if path.stat().st_size != int(entry.get("bytes", -1)):
                    hash_errors.append(relative + ": size mismatch")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != entry.get("sha256"):
                    hash_errors.append(relative + ": sha256 mismatch")
            required_manifest_paths = set(manifest.get("required_paths", []))
            manifest_ok = (
                manifest.get("result") == "PASS"
                and manifest.get("run_id") == run_id
                and manifest.get("commit") == commit_expected
                and manifest.get("harness_nonce") == harness_nonce
                and manifest.get("seven_pass_matrix") is True
                and required_manifest_paths.issubset(entries)
                and kernel_required.issubset(required_manifest_paths)
                and kernel_required.issubset(entries)
                and not hash_errors
                and bool(entries)
            )
            build = json.loads(
                (root / "build-manifest.json").read_text(encoding="utf-8")
            )
            binary = json.loads(
                (root / "commit-binary-identity.json").read_text(encoding="utf-8")
            )
            identity_ok = all(
                item.get("run_id") == run_id
                and item.get("commit") == commit_expected
                and item.get("harness_nonce") == harness_nonce
                for item in (build, binary)
            ) and binary.get("marker_strings_present") == [
                "[DEE_TA_SELFTEST_BEGIN]",
                "[DEE_TA_ALLOC]",
                "[DEE_TA_FREE]",
                "[DEE_TA_SELFTEST_PASS]",
            ]
            lifetime = json.loads(
                (root / "lifetime-analysis" / "lifetime_report.json").read_text(
                    encoding="utf-8"
                )
            )
            marker_counts = lifetime.get("per_file_marker_counts", [])
            lifetime_ok = (
                lifetime.get("result") == "NO_TRACE_ABORT"
                and not lifetime.get("glibc_double_free_anywhere")
                and not lifetime.get("glibc_generic_corruption_anywhere")
                and not lifetime.get("address_sanitizer_anywhere")
                and sum(int(row.get("alloc", 0)) for row in marker_counts) > 0
                and sum(int(row.get("free", 0)) for row in marker_counts) > 0
            )
            post_manifest = json.loads(
                (root / "post-download-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            post_entries = {
                item["path"]: item
                for item in post_manifest.get("artifacts", [])
            }
            post_required = {
                "lifetime-analysis/lifetime_report.json",
                "lifetime-analysis/abort_summary.txt",
                "lifetime-analysis/markers_index.json",
            }
            post_ok = (
                post_manifest.get("result") == "PASS"
                and post_required.issubset(post_entries)
            )
            for relative, entry in post_entries.items():
                posix_path = PurePosixPath(relative)
                if posix_path.is_absolute() or ".." in posix_path.parts:
                    post_ok = False
                    continue
                path = root / relative
                if (
                    not path.is_file()
                    or path.stat().st_size != int(entry.get("bytes", -1))
                    or hashlib.sha256(path.read_bytes()).hexdigest()
                    != entry.get("sha256")
                ):
                    post_ok = False
            lifetime_ok = lifetime_ok and post_ok
            import gzip
            with gzip.open(
                    root / "analysis" / "expert-trace.jsonl.gz", "rb") as stream:
                decompressed = stream.read(4096)
            trace_gzip_ok = (
                bool(decompressed.strip())
                and b'"not_measured": true' not in decompressed
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            hash_errors.append(type(exc).__name__ + ": " + str(exc))
        result = {
            "result": "PASS" if (
                not missing
                and summary_ok
                and manifest_ok
                and identity_ok
                and lifetime_ok
                and trace_gzip_ok
            ) else "FAIL",
            "run_id": run_id,
            "evidence_root": str(root),
            "missing": missing,
            "seven_pass_matrix": summary_ok,
            "manifest_hashes": manifest_ok,
            "identity": identity_ok,
            "lifetime": lifetime_ok,
            "expert_trace_gzip": trace_gzip_ok,
            "hash_errors": hash_errors,
        }
    report = evidence_dir / "artifact-validation.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log("artifact validation " + result["result"] + ": " + json.dumps(result), "MANIFEST")
    return result["result"] == "PASS"


def run_analyzer(evidence_dir, output_dir):
    log("running analyzer evidence=" + str(evidence_dir) + " -> " + str(output_dir), "ANL")
    rc, out = run(
        [sys.executable, str(ANALYZER_SCRIPT),
         "--evidence-dir", str(evidence_dir),
         "--output-dir", str(output_dir)],
        timeout=600,
    )
    log("analyzer rc=" + str(rc) + " (first 800 chars)\n" + out[:800], "ANL")
    if rc == 0:
        try:
            report = json.loads(
                (output_dir / "lifetime_report.json").read_text(encoding="utf-8")
            )
            marker_counts = report.get("per_file_marker_counts", [])
            safe = (
                report.get("result") == "NO_TRACE_ABORT"
                and not report.get("glibc_double_free_anywhere")
                and not report.get("glibc_generic_corruption_anywhere")
                and not report.get("address_sanitizer_anywhere")
                and sum(int(row.get("alloc", 0)) for row in marker_counts) > 0
                and sum(int(row.get("free", 0)) for row in marker_counts) > 0
            )
            post_files = [
                output_dir / "lifetime_report.json",
                output_dir / "abort_summary.txt",
                output_dir / "markers_index.json",
                output_dir / "timeline.txt",
            ]
            post_manifest = {
                "schema_version": 1,
                "result": "PASS" if safe else "FAIL",
                "artifacts": [
                    {
                        "path": str(path.relative_to(evidence_dir)),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in post_files if path.is_file()
                ],
            }
            (evidence_dir / "post-download-manifest.json").write_text(
                json.dumps(post_manifest, indent=2) + "\n", encoding="utf-8"
            )
            if not safe:
                log("lifetime semantic gate FAILED", "ANL")
                return 6
        except (OSError, TypeError, ValueError) as exc:
            log("lifetime semantic gate exception: " + repr(exc), "ANL")
            return 6
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


def _main_unlocked(argv):
    global RUN_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-push", action="store_true",
                        help="Skip the kernel push step.")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloading artifacts.")
    parser.add_argument("--no-archive", action="store_true",
                        help="Skip evidence-dir archive (use only on first run).")
    parser.add_argument("--no-live-tail", action="store_true",
                        help="Disable live-tail partial capture.")
    parser.add_argument(
        "--run-id",
        help="Exact remote RUN_ID for --no-push resume. Defaults to state file.",
    )
    parser.add_argument("--slug", default=SLUG_DEFAULT)
    parser.add_argument("--kernel-dir", type=Path, default=KERNEL_DIR_DEFAULT)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR_DEFAULT)
    parser.add_argument("--budget-min", type=int, default=75)
    parser.add_argument(
        "--commit-expected",
        help="Exact remote commit SHA. Defaults to the current local HEAD.",
    )
    args = parser.parse_args(argv)
    if args.run_id:
        RUN_ID = args.run_id
    elif args.no_push:
        try:
            RUN_ID = json.loads(
                STATE_FILE.read_text(encoding="utf-8")
            )["run_id"]
        except (KeyError, OSError, TypeError, ValueError):
            raise RuntimeError(
                "--no-push resume requires --run-id or a valid supervisor state file"
            )
    if not args.commit_expected:
        args.commit_expected = subprocess.check_output(
            ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
            text=True, encoding="utf-8",
        ).strip()

    HARD_BUDGET_SEC = args.budget_min * 60
    log("=== m3_supervisor_v6.1 starting ===", "SUP")
    log("RUN_ID=" + RUN_ID + " slug=" + args.slug
        + " budget=" + str(args.budget_min) + "min"
        + " commit_expected=" + args.commit_expected, "SUP")
    log("kaggle on PATH: "
        + ("present at " + shutil.which("kaggle") if shutil.which("kaggle") else "MISSING"),
        "SUP")
    log("python: " + sys.executable + " version=" + sys.version.split()[0], "SUP")

    if not args.no_archive and not args.no_push:
        archive_stale(args.evidence_dir)

    start = time.time()
    save_state({
        "schema_version": 2, "started_at": stamp(), "slug": args.slug,
        "run_id": RUN_ID, "evidence_dir": str(args.evidence_dir),
        "no_push": args.no_push, "no_download": args.no_download,
        "commit_expected": args.commit_expected,
    }, merge=False)

    if not args.no_push:
        env_for_kernel = {
            "RUN_ID": RUN_ID,
            "COMMIT_EXPECTED": args.commit_expected,
        }
        if not push_kernel(args.kernel_dir, env=env_for_kernel):
            log("push failed; refusing to poll or attribute a stale kernel", "SUP")
            return 4
    prev_state = None
    seen_running = False

    while True:
        elapsed = time.time() - start
        if elapsed > HARD_BUDGET_SEC:
            log("BUDGET EXCEEDED (" + str(round(elapsed/60, 1)) + " min > "
                + str(args.budget_min) + "); attempting best-effort download+analyzer",
                "SUP")
            if not args.no_download:
                download_artifacts(args.slug, args.evidence_dir)
            root = find_evidence_root(args.evidence_dir, RUN_ID)
            analyzer_source = root or args.evidence_dir
            run_analyzer(
                analyzer_source, analyzer_source / "lifetime-analysis"
            )
            log("supervisor exits 3 (budget exhausted)", "SUP")
            return 3

        state, raw = get_status(args.slug)
        if state == "RUNNING":
            seen_running = True
            if not args.no_live_tail:
                live_tail(args.slug, args.evidence_dir)
        if (state in TERMINAL_STATES and not args.no_push and not seen_running
                and elapsed < 60):
            log(
                "ignoring possible stale terminal state " + state
                + " during the first 60s after push",
                "STATE",
            )
            time.sleep(POLL_INTERVAL_SEC)
            continue
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
                download_ok = True
                if not args.no_download:
                    download_ok = download_artifacts(args.slug, args.evidence_dir)
                else:
                    log("download skipped (--no-download)", "DL")
                root = find_evidence_root(args.evidence_dir, RUN_ID)
                analyzer_source = root or args.evidence_dir
                analyzer_rc = run_analyzer(
                    analyzer_source, analyzer_source / "lifetime-analysis"
                )
                artifacts_ok = False
                if state in OK_STATES and download_ok and not args.no_download:
                    artifacts_ok = validate_artifacts(
                        args.evidence_dir, RUN_ID, args.commit_expected
                    )
                save_state({
                    "schema_version": 2, "elapsed_sec": elapsed,
                    "last_kaggle_state": state, "complete": True,
                    "completed_at": stamp(), "download_ok": download_ok,
                    "analyzer_rc": analyzer_rc,
                    "artifact_manifest_ok": artifacts_ok,
                })
                #  v6.1 / FIX-10e: distinguish recovered-completion from
                #  unsuccessful terminal states via exit code.
                if state in OK_STATES and download_ok and analyzer_rc == 0 and artifacts_ok:
                    log("supervisor done after terminal COMPLETE; exits 0", "SUP")
                    return 0
                elif state in OK_STATES:
                    log(
                        "Kaggle COMPLETE but download/analyzer/manifest gate failed; exits 5",
                        "SUP",
                    )
                    return 5
                else:
                    log("supervisor done after non-COMPLETE terminal " + state
                        + "; exits 2", "SUP")
                    return 2
        time.sleep(POLL_INTERVAL_SEC)


def main(argv):
    lock = acquire_supervisor_lock()
    try:
        return _main_unlocked(argv)
    finally:
        release_supervisor_lock(lock)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
