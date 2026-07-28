#!/usr/bin/env python3
"""Phase 1 Milestone-4 supervisor: push the integration commit, poll,
download evidence, and run the local seal harness against the downloaded
artifacts. Modeled on tmp/m3_supervisor_v6.py; the same `kaggle_invoke()`
fallback chain, lock + heartbeat + state machine, and `_stage_kernel()`-style
literal-override pattern are reused verbatim so the M3-proven execution
path carries over.

Forbidden changes outside these files: kernel source stays at
dee.cpp/kaggle/ornith-milestone4/ UNTIL this supervisor stages it.
"""
from __future__ import annotations

#  v6.1 / FIX-10c (carried over from M3 supervisor): bind `sys` first, then
#  reconfigure stdout/stderr to UTF-8 BEFORE any stdlib import that might
#  emit a deprecation warning or platform handshake that would otherwise
#  bypass the UTF-8 sink on Windows hosts.
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


def _resolve_kaggle_dir() -> Path:
    """Resolve ~/.kaggle on any platform (Windows / WSL / linux)."""
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
    return Path.home() / ".kaggle"


# v6.2 / FIX-13 carry-over: explicit propagation of Kaggle creds into the
# subprocess env. Mirrors tmp/m3_supervisor_v6.py lines 80-108 but is
# reduced to the parts the M4 launch path actually needs.
try:
    _kdir = _resolve_kaggle_dir()
    for legacy in (_kdir / "kaggle.json", _kdir / "credentials.json"):
        if not legacy.is_file():
            continue
        try:
            credentials = json.loads(legacy.read_text(encoding="utf-8"))
        except Exception:
            continue
        # legacy kaggle.json schema
        for src, env in (("username", "KAGGLE_USERNAME"), ("key", "KAGGLE_KEY")):
            if src in credentials and not os.environ.get(env):
                os.environ[env] = credentials[src]
        # post-OAuth credentials.json schema
        for src, env in (
            ("kaggle_username", "KAGGLE_USERNAME"),
            ("kaggle_key", "KAGGLE_KEY"),
            ("kaggle_token", "KAGGLE_API_TOKEN"),
        ):
            if src in credentials and credentials[src]:
                os.environ[env] = credentials[src]
except Exception:
    pass


ROOT = Path(__file__).resolve().parent.parent  # dynamic_expert_eviction/
SLUG_DEFAULT = "nivind/dee-cpp-ornith-milestone-4-capacity-sweep"
KERNEL_DIR_DEFAULT = ROOT / "dee.cpp" / "kaggle" / "ornith-milestone4"
EVIDENCE_DIR_DEFAULT = ROOT / "tmp" / "m4_ledger_seal_redo"
SEAL_HARNESS_PATH = (
    ROOT / "dee.cpp" / "scripts" / "run_milestone4_ledger_seal.py"
)
LOG_FILE = ROOT / "tmp" / "m4_supervisor.log"
STATE_FILE = ROOT / "tmp" / "m4_supervisor_state.json"
HEARTBEAT_FILE = ROOT / "tmp" / "m4_supervisor_heartbeat.jsonl"
LOCK_FILE = ROOT / "tmp" / "m4_supervisor.lock"
INTEGRATION_BRANCH_PREFIX = "freeze/phase1-integration-"

#  Module-import run id. The same RUN_ID propagates into the integration
#  branch name, the staged kernel literals, and the downloaded evidence
#  directory. Generated exactly once.
RUN_ID = (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + "-"
    + uuid.uuid4().hex[:8]
)

POLL_INTERVAL_SEC = 30.0
HARD_BUDGET_SEC_DEFAULT = 75 * 60
TERMINAL_STATES = {"COMPLETE", "ERROR", "FAILED", "CANCELLED"}
OK_STATES = {"COMPLETE"}


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str, tag: str = "SUP") -> None:
    line = "[%s] [%s] %s\n" % (stamp(), tag, msg)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line)
    sys.stdout.write(line)
    sys.stdout.flush()


def run(cmd, timeout=120, env=None):
    """Run a shell command; return (rc, combined stdout+stderr text)."""
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUTF8", "1")
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, env=merged_env,
        )
        return cp.returncode, (cp.stdout or "") + (cp.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except FileNotFoundError as exc:
        return 127, "NOT_FOUND " + str(exc)
    except Exception as exc:
        return 1, "EXC %s: %s" % (type(exc).__name__, exc)


def kaggle_invoke(args_list, timeout=180, env=None):
    """Invoke kaggle CLI; fall back to `python -m kaggle` if PATH is bare."""
    rc, out = run(["kaggle", *args_list], timeout=timeout, env=env)
    if rc == 0:
        return rc, out
    if rc != 127:
        return rc, out
    log("kaggle CLI missing on PATH; trying python -m kaggle fallback", "SUP")
    py_guess = shutil.which("python") or shutil.which("python3") or "python"
    return run([py_guess, "-m", "kaggle", *args_list], timeout=timeout, env=env)


def parse_status(raw: str) -> str | None:
    """Parse `kaggle kernels status` output into a normalised state token."""
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
        tmp.write_text(
            json.dumps(current, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp, STATE_FILE)
    except OSError:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def load_state() -> dict[str, object]:
    if not STATE_FILE.is_file():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _pid_alive(pid) -> bool:
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


def acquire_lock() -> dict[str, object]:
    for _ in range(2):
        try:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "pid": os.getpid(), "run_id": RUN_ID, "created_at": stamp(),
                }) + "\n")
            return {"pid": os.getpid(), "run_id": RUN_ID}
        except FileExistsError:
            try:
                existing = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
            if _pid_alive(existing.get("pid")):
                raise RuntimeError(
                    "another M4 supervisor owns %s: %s"
                    % (LOCK_FILE, json.dumps(existing))
                )
            try:
                LOCK_FILE.unlink()
            except FileNotFoundError:
                pass
    raise RuntimeError("could not acquire supervisor lock %s" % LOCK_FILE)


def release_lock(payload: dict[str, object]) -> None:
    try:
        current = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        if (
            current.get("pid") == payload.get("pid")
            and current.get("run_id") == payload.get("run_id")
        ):
            LOCK_FILE.unlink()
    except (FileNotFoundError, OSError, ValueError):
        pass


def _stage_kernel(
    kernel_dir: Path,
    run_id: str,
    integration_sha: str,
    integration_branch: str,
    harness_nonce: str,
) -> tuple[Path, dict[str, object]]:
    """Copy `kernel_dir` into an isolated stage and override FIVE literals
    deterministically via exact-string replacement (no regex for the
    highly-contextual lines):

      1. `RUN_ID = "<old>"` → `RUN_ID = "<new run_id>"`
      2. `EXPECTED_COMMIT = "<old>"` → `EXPECTED_COMMIT = "<new sha>"`
      3. `git clone ... "--branch", "opt/real-model-t1", "--single-branch", ...`
         → `--branch "<new branch>"`
      4. `["git", "checkout", "origin/opt/real-model-t1"]`
         → `["git", "checkout", "<new sha>"]`
      5. Inject `HARNESS_NONCE = "<new nonce>"` on the line immediately
         after `EXPECTED_COMMIT`.

    Each replacement uses `str.replace(old, new, 1)` and verifies that the
    expected substitution occurred exactly once. Any misscount != 1 leaves
    the staged tree on disk but raises before the caller can `.push()` to
    Kaggle. The stage directory is ALWAYS removed on a per-site misscount
    failure so subsequent runs do not inherit a half-staged tree.

    Returns (stage_dir, identity_record) where identity_record captures the
    exact overrides + the post-staged notebook SHA so downstream validators
    can correlate.
    """
    stage = ROOT / ("m4_kernel_stage_" + run_id)
    if stage.exists():
        if stage.resolve().parent != ROOT.resolve():
            raise RuntimeError(
                "refusing to replace unsafe kernel stage %s" % stage
            )
        shutil.rmtree(stage)
    shutil.copytree(kernel_dir, stage)
    kernel_script = stage / "ornith_milestone4.py"
    text = kernel_script.read_text(encoding="utf-8")

    def _literal_replace_once(label: str, old_token: str, new_token: str) -> None:
        nonlocal text
        if old_token not in text:
            raise RuntimeError(
                "staging literal miss for %s: token %r not in kernel script"
                % (label, old_token)
            )
        occurrences = text.count(old_token)
        if occurrences != 1:
            raise RuntimeError(
                "staging literal %s appeared %d times (expected 1); "
                "kernel script may have been refactored"
                % (label, occurrences)
            )
        text = text.replace(old_token, new_token, 1)

    try:
        _literal_replace_once(
            "RUN_ID",
            old_token='RUN_ID = "20260727T024309Z-capacity-sweep"',
            new_token='RUN_ID = "' + run_id + '"',
        )
        _literal_replace_once(
            "EXPECTED_COMMIT",
            old_token='EXPECTED_COMMIT = "726fbfce22d676e9c86e1af1e1cb197d21ea8612"',
            new_token='EXPECTED_COMMIT = "' + integration_sha + '"',
        )
        _literal_replace_once(
            "git_clone_branch",
            old_token='"--branch", "opt/real-model-t1"',
            new_token='"--branch", "' + integration_branch + '"',
        )
        _literal_replace_once(
            "git_checkout_target",
            old_token='["git", "checkout", "origin/opt/real-model-t1"]',
            new_token='["git", "checkout", "' + integration_sha + '"]',
        )
        # Inject HARNESS_NONCE on the line immediately after EXPECTED_COMMIT.
        # We anchor on the post-replacement EXPECTED_COMMIT literal so the
        # exact line context is enforced.
        commit_anchor = (
            'EXPECTED_COMMIT = "' + integration_sha + '"\n'
        )
        if commit_anchor not in text:
            raise RuntimeError(
                "staging literal miss for HARNESS_NONCE: post-replacement "
                "EXPECTED_COMMIT anchor not found"
            )
        commit_count = text.count(commit_anchor)
        if commit_count != 1:
            raise RuntimeError(
                "staging literal HARNESS_NONCE: EXPECTED_COMMIT anchor "
                "appeared %d times (expected 1)" % commit_count
            )
        nonce_line = 'HARNESS_NONCE = "' + harness_nonce + '"\n'
        if nonce_line in text:
            raise RuntimeError(
                "staging literal HARNESS_NONCE: nonce line already present"
            )
        text = text.replace(commit_anchor, commit_anchor + nonce_line, 1)
    except RuntimeError:
        # Stage-cleanup so the partial tree does not leak into the next run.
        try:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
        except Exception:
            pass
        raise

    kernel_script.write_text(text, encoding="utf-8")
    kernel_sha = hashlib.sha256(kernel_script.read_bytes()).hexdigest()
    identity = {
        "schema_version": 1,
        "run_id": run_id,
        "integration_sha": integration_sha,
        "integration_branch": integration_branch,
        "kernel_script": kernel_script.name,
        "kernel_script_sha256": kernel_sha,
        "harness_nonce": harness_nonce,
        "staged_at": stamp(),
    }
    (stage / "run-identity.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    return stage, identity


def resolve_integration_sha(
    integration_sha_arg: str | None,
    integration_branch: str,
) -> str:
    """Resolve the integration SHA. If a SHA was passed in, use it; else
    resolve `git rev-parse origin/<integration_branch>` from the local
    git remote.
    """
    if integration_sha_arg and re.fullmatch(r"[0-9a-f]{7,64}", integration_sha_arg):
        return integration_sha_arg
    if not integration_sha_arg:
        rc, out = run([
            "git", "-C", str(ROOT),
            "rev-parse", "origin/" + integration_branch,
        ], timeout=30)
        if rc == 0 and out.strip():
            return out.strip()
    raise RuntimeError(
        "could not resolve integration SHA for branch %s (got=%s)"
        % (integration_branch, integration_sha_arg)
    )


def integrate_branch(integration_sha: str) -> str:
    """Create (if absent) + push freeze/phase1-integration-<sha[:12]> off
    the integration SHA. FAIL-CLOSED on a pre-existing branch whose HEAD
    does NOT match the requested SHA: a stale Phase-1 orphan MUST NOT be
    silently reused. We either refuse (preferred) or force-update via
    `git branch -f`.

    Returns the integration branch name.
    """
    integration_branch = INTEGRATION_BRANCH_PREFIX + integration_sha[:12]
    rc, show = run([
        "git", "-C", str(ROOT),
        "show-ref", "--verify", "--quiet",
        "refs/heads/" + integration_branch,
    ], timeout=15)
    if rc == 0:
        # Local branch exists; verify HEAD == requested SHA. If not, refuse.
        rc_p, out_p = run([
            "git", "-C", str(ROOT),
            "rev-parse", "refs/heads/" + integration_branch,
        ], timeout=15)
        if rc_p == 0 and out_p.strip() == integration_sha:
            log(
                "integration branch %s already at requested SHA %s"
                % (integration_branch, integration_sha),
                "INT",
            )
        else:
            existing_sha = out_p.strip() if rc_p == 0 else "<unknown>"
            raise RuntimeError(
                "integration branch %s already exists at SHA %s but caller "
                "requested SHA %s. A stale Phase-1 orphan MUST NOT be "
                "silently reused. Either pick the existing SHA via "
                "--integration-sha %s or remove the orphan branch with "
                "`git branch -D %s` before re-launching."
                % (
                    integration_branch, existing_sha, integration_sha,
                    existing_sha, integration_branch,
                )
            )
    else:
        rc_b, out_b = run([
            "git", "-C", str(ROOT),
            "branch", integration_branch, integration_sha,
        ], timeout=30)
        if rc_b != 0:
            raise RuntimeError(
                "git branch %s %s failed: %s"
                % (integration_branch, integration_sha, out_b[:400])
            )
        log(
            "created local integration branch %s -> %s"
            % (integration_branch, integration_sha),
            "INT",
        )
    rc_p, out_p = run([
        "git", "-C", str(ROOT),
        "push", "-u", "origin", integration_branch,
    ], timeout=120)
    if rc_p != 0:
        raise RuntimeError(
            "git push origin %s failed: %s"
            % (integration_branch, out_p[:600])
        )
    log("pushed integration branch to origin: %s" % integration_branch, "INT")
    return integration_branch


def push_kernel(
    kernel_dir: Path,
    run_id: str,
    integration_sha: str,
    integration_branch: str,
    env=None,
) -> dict[str, object]:
    """Stage the kernel locally with the run_id + integration SHA injected,
    then push the staged tree to Kaggle.

    The staged script's contents are SHA-256-fingerprinted before push so
    validators downstream can verify the same engine source ran.
    Returns the staged-identity dict.
    """
    harness_nonce = uuid.uuid4().hex
    stage, identity = _stage_kernel(
        kernel_dir, run_id, integration_sha, integration_branch, harness_nonce
    )
    log(
        "pushing isolated kernel stage %s RUN_ID=%s SHA=%s branch=%s"
        " notebook_sha256=%s nonce=%s"
        % (
            stage, run_id, integration_sha, integration_branch,
            identity["kernel_script_sha256"], harness_nonce[:12],
        ),
        "PUSH",
    )
    try:
        rc, out = kaggle_invoke(
            ["kernels", "push", "-p", str(stage)], timeout=300, env=env
        )
    finally:
        if (
            stage.resolve().parent == ROOT.resolve()
            and stage.name.startswith("m4_kernel_stage_")
        ):
            shutil.rmtree(stage, ignore_errors=True)
    log(
        "kaggle kernels push rc=%s (first 600 chars)\n%s"
        % (rc, out[:600]),
        "PUSH",
    )
    if rc != 0:
        raise RuntimeError(
            "kernel push FAILED rc=%s out=%s" % (rc, out[:600])
        )
    new_version = None
    for line in out.splitlines():
        match = re.search(r"\bversion\s+(\d+)\b", line, flags=re.IGNORECASE)
        if match:
            new_version = int(match.group(1))
    log("kernel push OK; new version: %s" % new_version, "PUSH")
    identity["kaggle_version"] = new_version
    return identity


def get_status(slug: str) -> tuple[str | None, str]:
    rc, out = kaggle_invoke(["kernels", "status", slug], timeout=120)
    return parse_status(out), out


def download_artifacts(slug: str, evidence_dir: Path) -> bool:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rc, out = kaggle_invoke(
        ["kernels", "output", slug, "-p", str(evidence_dir)],
        timeout=600,
    )
    log(
        "kaggle kernels output rc=%s (first 500 chars)\n%s"
        % (rc, out[:500]),
        "DL",
    )
    if rc != 0:
        return False
    try:
        size = sum(
            p.stat().st_size
            for p in evidence_dir.rglob("*")
            if p.is_file()
        )
    except Exception:
        size = -1
    log("download OK; size=%s bytes written to %s" % (size, evidence_dir), "DL")
    return True


def archive_stale(evidence_dir: Path) -> None:
    """Move a non-empty existing `evidence_dir` to <evidence>_prev_at_<RUN_ID>.
    Mirrors the M3 supervisor pattern.
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
        shutil.rmtree(dest, ignore_errors=True)
    try:
        shutil.move(str(evidence_dir), str(dest))
        log("archived evidence_dir -> %s" % dest, "SUP")
    except Exception as exc:
        log("could not archive %s: %s" % (evidence_dir, exc), "SUP")


def live_tail(slug: str, evidence_dir: Path) -> None:
    """Best-effort partial-download tick + heartbeat row append."""
    target = evidence_dir / "_live"
    target.mkdir(parents=True, exist_ok=True)
    rc, _ = kaggle_invoke(
        ["kernels", "output", slug, "-p", str(target)], timeout=120
    )
    if rc != 0:
        return
    try:
        size = sum(
            p.stat().st_size for p in target.rglob("*") if p.is_file()
        )
    except Exception:
        size = -1
    try:
        with HEARTBEAT_FILE.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({
                    "poll_tick_utc": stamp(), "log_size_bytes": size,
                }, ensure_ascii=False) + "\n"
            )
    except Exception as exc:
        log("heartbeat write failed: %s" % exc, "DL")


def find_evidence_root(evidence_dir: Path, run_id: str) -> Path | None:
    candidates = [
        p for p in evidence_dir.rglob("ornith-milestone4-evidence-" + run_id)
        if p.is_dir() and "_live" not in p.parts
    ]
    return candidates[0] if len(candidates) == 1 else None


def run_local_seal_harness(
    downloaded_evidence_dir: Path,
    model_dir: Path,
    run_id: str,
    identity: dict[str, object],
    timeout_sec: int,
) -> int:
    """After Kaggle terminal-state COMPLETE, invoke the local seal harness
    against the downloaded evidence. Fail-closed: harness returns non-zero
    on validation failure, OR the supervisor returns 5.
    """
    seal_output_dir = downloaded_evidence_dir.parent / (
        "seal_validation_" + run_id
    )
    seal_logs_dir = seal_output_dir / "logs"
    seal_logs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-X", "faulthandler",
        str(SEAL_HARNESS_PATH),
        "--model-dir", str(model_dir),
        "--output-dir", str(seal_output_dir),
        "--label", "phase1-redo",
        "--timeout-seconds", str(timeout_sec),
    ]
    log_path = seal_logs_dir / "seal_harness.log"
    env = os.environ.copy()
    env["M4_IDENTITY_JSON"] = json.dumps(identity)
    log(
        "invoking local seal harness model_dir=%s evidence=%s"
        % (model_dir, downloaded_evidence_dir),
        "SEAL",
    )
    rc, out = run(cmd, timeout=timeout_sec + 600, env=env)
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n" + (out or ""),
        encoding="utf-8",
        errors="replace",
    )
    log(
        "seal harness rc=%s (first 800 chars)\n%s"
        % (rc, out[:800]),
        "SEAL",
    )
    return rc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slug", default=SLUG_DEFAULT)
    p.add_argument("--kernel-dir", type=Path, default=KERNEL_DIR_DEFAULT)
    p.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR_DEFAULT)
    p.add_argument(
        "--integration-sha", default=None,
        help="Exact 7-64 hex SHA to push to Kaggle. Defaults to "
             "git rev-parse of the integration branch.",
    )
    p.add_argument(
        "--integration-branch", default=None,
        help="Local+remote branch name. Defaults to "
             "freeze/phase1-integration-<sha[:12]>.",
    )
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--budget-min", type=int, default=75)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--no-archive", action="store_true")
    p.add_argument("--no-live-tail", action="store_true")
    p.add_argument("--no-seal", action="store_true",
                   help="Skip running the local seal harness after download.")
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def _main_unlocked(argv: list[str]) -> int:
    args = parse_args()
    global RUN_ID
    if args.run_id:
        RUN_ID = args.run_id
    elif args.no_push:
        state = load_state()
        if not state.get("run_id"):
            raise RuntimeError(
                "--no-push resume requires --run-id or valid supervisor state"
            )
        RUN_ID = state["run_id"]

    integration_sha = resolve_integration_sha(
        args.integration_sha, args.integration_branch or "UNRESOLVED"
    )
    integration_branch = args.integration_branch
    if not integration_branch:
        if not args.no_push:
            integration_branch = integrate_branch(integration_sha)
        else:
            integration_branch = INTEGRATION_BRANCH_PREFIX + integration_sha[:12]

    HARD_BUDGET_SEC = args.budget_min * 60
    log("=== m4_supervisor starting ===", "SUP")
    log("RUN_ID=%s slug=%s integration_sha=%s branch=%s budget=%smin"
        % (
            RUN_ID, args.slug, integration_sha, integration_branch,
            args.budget_min,
        ), "SUP")
    log("kaggle on PATH: %s"
        % (shutil.which("kaggle") or "MISSING"), "SUP")

    if not args.no_archive and not args.no_push:
        archive_stale(args.evidence_dir)

    save_state({
        "schema_version": 2,
        "started_at": stamp(),
        "slug": args.slug,
        "run_id": RUN_ID,
        "evidence_dir": str(args.evidence_dir),
        "integration_sha": integration_sha,
        "integration_branch": integration_branch,
        "no_push": args.no_push,
        "no_download": args.no_download,
        "no_seal": args.no_seal,
    }, merge=False)

    identity: dict[str, object] = {}
    if not args.no_push:
        identity = push_kernel(
            kernel_dir=args.kernel_dir,
            run_id=RUN_ID,
            integration_sha=integration_sha,
            integration_branch=integration_branch,
        )
        save_state({
            "kaggle_version": identity.get("kaggle_version"),
            "kernel_script_sha256": identity["kernel_script_sha256"],
            "harness_nonce": identity["harness_nonce"],
        })
    else:
        state = load_state()
        identity = {
            "run_id": RUN_ID,
            "integration_sha": state.get("integration_sha", integration_sha),
            "integration_branch": state.get(
                "integration_branch", integration_branch
            ),
            "kernel_script_sha256": state.get("kernel_script_sha256"),
            "harness_nonce": state.get("harness_nonce"),
            "kaggle_version": state.get("kaggle_version"),
        }

    start = time.time()
    prev_state = None
    seen_running = False
    while True:
        elapsed = time.time() - start
        if elapsed > HARD_BUDGET_SEC:
            log(
                "BUDGET EXCEEDED (%s min > %s); attempting best-effort "
                "download + local seal"
                % (round(elapsed / 60, 1), args.budget_min),
                "SUP",
            )
            if not args.no_download:
                download_artifacts(args.slug, args.evidence_dir)
            break
        state_v, raw = get_status(args.slug)
        if state_v == "RUNNING":
            seen_running = True
            if not args.no_live_tail:
                live_tail(args.slug, args.evidence_dir)
        if (
            state_v in TERMINAL_STATES
            and not args.no_push
            and not seen_running
            and elapsed < 60
        ):
            log(
                "ignoring possible stale terminal state %s during the first "
                "60s after push" % state_v,
                "STATE",
            )
            time.sleep(POLL_INTERVAL_SEC)
            continue
        if state_v is None:
            log("no state parsed; raw=%s" % repr((raw or "")[:200]), "STATE")
        elif state_v != prev_state:
            log(
                "STATE %s -> %s (elapsed=%ss)"
                % (repr(prev_state), repr(state_v), int(elapsed)),
                "STATE",
            )
            prev_state = state_v
            save_state({
                "elapsed_sec": elapsed,
                "last_kaggle_state": state_v,
                "raw_status": raw[:600],
                "started_at_iso": stamp(),
            })
            if state_v in TERMINAL_STATES:
                log("terminal state %s reached" % state_v, "SUP")
                download_ok = True
                if not args.no_download:
                    download_ok = download_artifacts(
                        args.slug, args.evidence_dir
                    )
                if state_v in OK_STATES and download_ok:
                    root = find_evidence_root(args.evidence_dir, RUN_ID)
                    if root is not None and not args.no_seal:
                        seal_rc = run_local_seal_harness(
                            downloaded_evidence_dir=root,
                            model_dir=args.model_dir,
                            run_id=RUN_ID,
                            identity=identity,
                            timeout_sec=HARD_BUDGET_SEC,
                        )
                        save_state({
                            "elapsed_sec": elapsed,
                            "last_kaggle_state": state_v,
                            "complete": True,
                            "completed_at": stamp(),
                            "download_ok": download_ok,
                            "seal_rc": seal_rc,
                        })
                        if seal_rc == 0:
                            log("Phase 1 COMPLETE; SEAL harness PASS; exits 0",
                                "SUP")
                            return 0
                        log(
                            "Kaggle COMPLETE but local SEAL harness gates "
                            "failed (rc=%s); exits 5" % seal_rc,
                            "SUP",
                        )
                        return 5
                    log(
                        "evidence root not found in download; "
                        "skipping seal harness; exits 5",
                        "SUP",
                    )
                    return 5
                log(
                    "terminal state %s but download ok=%s; exits 2"
                    % (state_v, download_ok),
                    "SUP",
                )
                return 2
        time.sleep(POLL_INTERVAL_SEC)


def main(argv: list[str]) -> int:
    lock = acquire_lock()
    try:
        return _main_unlocked(argv)
    finally:
        release_lock(lock)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
