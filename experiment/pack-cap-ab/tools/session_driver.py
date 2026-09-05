#!/usr/bin/env python3
"""Session driver for the pack-cap A/B (experiment/pack-cap-ab).

Runs BOTH arms of ONE counterbalanced session in fresh OS processes:
  - arm A: the byte-original seal-era harness (LRU_TOTAL_CAP_GIB = 17.0)
  - arm B: the same harness with EXACTLY ONE constant changed (cap 20.0)

Both arm scripts are embedded below (base64) and verified by sha256 before
execution — the driver trusts nothing on disk. Each arm gets a fresh
process and fresh /tmp workspace (the harness refuses a stale
/tmp/dsv4-native-src and validates-in-place the shared trace bank), while
the DEE4 trace bank directory is SHARED: created+validated by the first
arm, validated in place by the second (byte-identical store both arms).

The driver touches NO engine behavior: it stages files, spawns the
harness, moves evidence into per-arm directories, applies the
pre-registered memory-abort thresholds from experiment-contract.json
(min checkpoint MemAvailable >= 1.5 GiB; VmHWM <= 30.0 GiB), verifies the
effective lru_cap from the harness log (a mem_avail-3.5 clamp silently
shrinking arm B is a contract violation), and records provenance.

NOTE: the harness ALWAYS exits 0 (so Kaggle snapshots its output), so
arm validity is judged from evidence classification
(native-generate-result.json -> ACCEPT_CORRECTNESS), never exit codes.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- embedded --
# ARM_A_B64 / ARM_B_B64 and their sha256s are substituted by make_harness.py
# at package build time. PREAD_B64 (optional) is appended to the LAST arm's
# workspace only when PREAD_ENABLE = "1" and only AFTER that arm is terminal.
ARM_A_B64 = "@@ARM_A_B64@@"
ARM_A_SHA256 = "@@ARM_A_SHA256@@"
ARM_B_B64 = "@@ARM_B_B64@@"
ARM_B_SHA256 = "@@ARM_B_SHA256@@"
SESSION_ID = "@@SESSION_ID@@"          # "session1" or "session2"
ARM_ORDER = "@@ARM_ORDER@@"            # "A,B" or "B,A" (pre-registered, fixed)
PREAD_B64 = "@@PREAD_B64@@"            # bench_expert_pread.py (rider)
PREAD_SHA256 = "@@PREAD_SHA256@@"
PREAD_ENABLE = "@@PREAD_ENABLE@@"      # "1" only on session 2

# Fixed /tmp paths the harness hardcodes; everything EXCEPT the trace bank is
# wiped between arms so each arm starts fresh and the "refusing stale
# preparation" guard cannot fire.
HARNESS_TMP_PATHS = [
    "/tmp/dsv4-native-src",
    "/tmp/dsv4-checkpoint",
    "/tmp/dsv4-dee4-v2",
    "/tmp/dsv4-dee4-v2-component",
    "/tmp/host-reuse-candidate-from-baseline.bundle",
]
TRACE_BANK = "/tmp/dsv4-dee4-v3-trace"   # SHARED by design (validated in place)
WORK = Path("/kaggle/working")

# Pre-registered memory gates (experiment-contract.json decision_rule.memory).
MIN_AVAIL_ABORT_GIB = 1.5
VMHWM_ABORT_GIB = 30.0
# Contract expectations as the harness LOGS them: lru_cap= is the TOTAL cap
# constant; host_pack= a/b are the effective per-GPU budgets after any
# mem_avail clamp. Arm B shrinking to 8.5/8.5 means the clamp engaged ->
# contract violation (candidate arm not actually running cap 20.0).
EXPECTED_CAP_GIB = {"A": {"lru_cap_total": 17.0, "per_gpu": 8.5},
                    "B": {"lru_cap_total": 20.0, "per_gpu": 10.0}}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [driver] {msg}", flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify_embedded() -> dict:
    arms = {}
    for arm, b64, want in (("A", ARM_A_B64, ARM_A_SHA256),
                           ("B", ARM_B_B64, ARM_B_SHA256)):
        raw = base64.b64decode(b64)
        got = sha256_bytes(raw)
        if got != want:
            raise SystemExit(f"embedded arm {arm} sha256 mismatch: {got} != {want}")
        arms[arm] = raw
    log(f"embedded harnesses verified: A={ARM_A_SHA256[:12]} B={ARM_B_SHA256[:12]}")
    return arms


def meminfo_gib(path: Path, prefix: str) -> float:
    try:
        for line in path.read_text().splitlines():
            if line.startswith(prefix):
                return int(line.split()[1]) / (1 << 20)  # kB -> GiB
    except OSError:
        pass
    return -1.0


def clean_between_arms() -> dict:
    """Remove every harness-owned /tmp path EXCEPT the shared trace bank."""
    removed = []
    for p in HARNESS_TMP_PATHS:
        pp = Path(p)
        if pp.exists() or pp.is_symlink():
            if pp.is_dir() and not pp.is_symlink():
                shutil.rmtree(pp, ignore_errors=True)
            else:
                try:
                    pp.unlink()
                except OSError:
                    pass
            removed.append(p)
    # Any stray trace-bank *lock/temp* is fine to leave; the bank itself and
    # its metadata.json must survive (validated in place by the next arm).
    if not (Path(TRACE_BANK) / "metadata.json").is_file():
        log(f"note: trace bank not yet built at {TRACE_BANK} (arm 1 builds it)")
    log(f"between-arms cleanup removed {len(removed)} /tmp paths (kept {TRACE_BANK})")
    return {"removed": removed, "kept": TRACE_BANK}


def run_arm(arm: str, script: bytes, out_dir: Path) -> dict:
    """Run one arm in a fresh process; collect evidence; keep logs."""
    work_root = Path(f"/tmp/pack-cap-{SESSION_ID}-{arm}")
    if work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True)
    script_path = work_root / f"harness_arm_{arm}.py"
    script_path.write_bytes(script)

    # NOTHING NATIVE_*: the harness preflight fails closed on them. Config
    # flows only through the harness's own constants (arm A vs arm B file).
    env = dict(os.environ)
    env.pop("NATIVE_NTFY", None)
    for k in list(env):
        if k.startswith("NATIVE_"):
            log(f"contract violation: refusing to pass {k} to arm (preflight would fail)")
            del env[k]
    env.setdefault("PYTHONUNBUFFERED", "1")

    t0 = time.time()
    log(f"arm {arm}: launching fresh process (workspace {work_root})")
    # Stream stdout to a growing log so progress is visible in kernel logs.
    stdout_path = work_root / "arm-stdout.live.log"
    with stdout_path.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen([sys.executable, str(script_path)],
                                cwd=str(work_root), env=env, stdout=out,
                                stderr=subprocess.STDOUT)
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            raise
    wall = time.time() - t0
    log(f"arm {arm}: exit={rc} wall={wall/60:.1f} min")

    out_dir.mkdir(parents=True, exist_ok=True)
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    (out_dir / "arm-stdout.log").write_text(text[-4_000_000:], encoding="utf-8")

    # Effective cap check: parse the harness budget line
    # ("... lru_cap=<x>GiB ...") and the actual per-GPU budgets.
    lru_cap_vals = re.findall(r"lru_cap=([0-9.]+)GiB", text)
    host_pack_vals = re.findall(
        r"host_pack=([0-9.]+)/([0-9.]+)GiB", text)
    effective = {
        "lru_cap_total_gib_logged": [float(v) for v in lru_cap_vals],
        "host_pack_budget_gib_logged": [[float(a), float(b)] for a, b in host_pack_vals],
        "expected": EXPECTED_CAP_GIB[arm],
    }
    cap_ok = True
    if not lru_cap_vals or not host_pack_vals:
        cap_ok = False
    else:
        want = EXPECTED_CAP_GIB[arm]
        cap_ok = all(abs(float(v) - want["lru_cap_total"]) < 0.26
                     for v in lru_cap_vals) and all(
            abs(float(a) - want["per_gpu"]) < 0.26
            and abs(float(b) - want["per_gpu"]) < 0.26
            for a, b in host_pack_vals)
    if not cap_ok:
        log(f"CAP-CHECK: arm {arm} effective cap {effective} != "
            f"{EXPECTED_CAP_GIB[arm]} (mem_avail clamp may have engaged; "
            f"contract violation)")

    # Move every evidence artifact the harness published to /kaggle/working.
    copied = []
    for p in sorted(WORK.iterdir()):
        if p.is_file():
            shutil.copy2(p, out_dir / p.name)
            copied.append(p.name)
            try:
                p.unlink()          # reset WORK for the next arm
            except OSError:
                pass

    driver_mem = {
        "mem_total_gib": round(meminfo_gib(Path("/proc/meminfo"), "MemTotal:"), 3),
        "mem_available_gib_after_arm": round(
            meminfo_gib(Path("/proc/meminfo"), "MemAvailable:"), 3),
        "note": "driver snapshot post-arm; memory.json (harness) is authoritative",
    }
    return {
        "arm": arm, "exit_code": rc, "wall_s": round(wall, 1),
        "evidence_files": copied, "workspace": str(work_root),
        "driver_mem": driver_mem, "cap_check": effective,
        "cap_check_ok": cap_ok,
    }


def parse_arm_evidence(arm_dir: Path) -> dict:
    """Parse the harness evidence with the REAL field names."""
    out: dict = {"classification": None, "correctness_ok": False,
                 "decode_wall_s": None, "host_pack": None,
                 "engine_cache": None, "h2d_bytes": None,
                 "memory_gate_inputs": None}
    res_p = arm_dir / "native-generate-result.json"
    if res_p.is_file():
        try:
            r = json.loads(res_p.read_text(encoding="utf-8"))
            out["classification"] = r.get("classification")
            out["run_id"] = r.get("run_id")
            out["commit"] = r.get("commit")
            out["model_revision"] = r.get("model_revision")
            out["decode_wall_s"] = r.get("decode_wall_s")
            out["decode_tok_s"] = r.get("decode_tok_s")
            out["prefill_ms"] = r.get("prefill_ms")
            out["total_wall_seconds"] = r.get("total_wall_seconds")
            out["host_pack_budget_gib"] = r.get("host_pack_budget_gib")
            out["host_pack"] = r.get("host_pack")
            out["engine_stats"] = r.get("engine_stats")
            hp = r.get("host_pack") or {}
            out["h2d_bytes"] = {
                g: (r.get("engine_stats", {}).get(g, {}) or {}).get("h2d_bytes")
                for g in ("cuda0", "cuda1")}
            out["byte_accounting"] = r.get("byte_accounting")
            out["correctness"] = r.get("correctness")
            out["performance_eligible"] = r.get("performance_eligible")
            out["hardware_classification"] = r.get("hardware_classification")
        except Exception as exc:
            out["parse_error"] = repr(exc)
    mem_p = arm_dir / "memory.json"
    if mem_p.is_file():
        try:
            m = json.loads(mem_p.read_text(encoding="utf-8"))
            peaks = m.get("process_final_and_peak_gib") or {}
            out["memory_gate_inputs"] = {
                "vmhwm_gib": peaks.get("VmHWM"),
                "vmrss_gib": peaks.get("VmRSS"),
                "vmdata_gib": peaks.get("VmData"),
                "mem_total_gib": (m.get("system_final_gib") or {}).get("MemTotal"),
                "min_checkpoint_mem_available_gib":
                    m.get("minimum_checkpoint_host_mem_available_gib"),
                "host_pack_budget_bytes": m.get("host_pack_budget_bytes"),
                "checkpoint_records": m.get("checkpoint_records"),
            }
        except Exception as exc:
            out["memory_parse_error"] = repr(exc)
    return out


def memory_gate(evidence: dict) -> dict:
    """Pre-registered thresholds applied to the harness's own measurements."""
    gates = {"triggered": False, "reason": None, "thresholds": {
        "min_checkpoint_mem_available_gib_below": MIN_AVAIL_ABORT_GIB,
        "process_vmhwm_gib_above": VMHWM_ABORT_GIB}}
    mi = evidence.get("memory_gate_inputs") or {}
    if not mi:
        gates["reason"] = "memory.json absent/unparseable (arm failed early?)"
        gates["triggered"] = True
        return gates
    av = mi.get("min_checkpoint_mem_available_gib")
    hwm = mi.get("vmhwm_gib")
    if isinstance(av, (int, float)) and av < MIN_AVAIL_ABORT_GIB:
        gates["triggered"] = True
        gates["reason"] = (f"min checkpoint MemAvailable {av} GiB < "
                           f"{MIN_AVAIL_ABORT_GIB} GiB (pre-registered)")
    if isinstance(hwm, (int, float)) and hwm > VMHWM_ABORT_GIB:
        gates["triggered"] = True
        gates["reason"] = (f"VmHWM {hwm} GiB > {VMHWM_ABORT_GIB} GiB "
                           f"(pre-registered)")
    return gates


def maybe_pread_rider(arm_dir: Path) -> dict:
    """Rider: ONLY after the last arm of the LAST session is terminal."""
    if PREAD_ENABLE != "1" or not last_arm_of_experiment():
        return {"ran": False, "why": "not enabled / not the final session tail"}
    if not PREAD_B64 or "@@" in PREAD_B64:
        return {"ran": False, "why": "rider not embedded"}
    raw = base64.b64decode(PREAD_B64)
    if sha256_bytes(raw) != PREAD_SHA256:
        return {"ran": False, "why": "rider sha256 mismatch"}
    store = Path(TRACE_BANK)
    if not (store / "metadata.json").is_file():
        return {"ran": False, "why": "trace bank absent"}
    bench = Path("/tmp/pread-rider/bench_expert_pread.py")
    bench.parent.mkdir(parents=True, exist_ok=True)
    bench.write_bytes(raw)
    cmd = [sys.executable, str(bench), "--store", str(store / "metadata.json"),
           "--records", "96", "--out", str(WORK / "pread-rider.json")]
    log(f"pread rider: {' '.join(cmd)}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        out = {"ran": True, "exit_code": r.returncode,
               "wall_s": round(time.time() - t0, 1),
               "stdout_tail": (r.stdout or "")[-4000:],
               "stderr_tail": (r.stderr or "")[-2000:]}
    except Exception as exc:
        out = {"ran": True, "error": repr(exc)}
    (arm_dir / "pread-rider-launch.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


_LAST_ARM = {"flag": False}


def last_arm_of_experiment() -> bool:
    return _LAST_ARM["flag"]


def main() -> int:
    log(f"session {SESSION_ID} order {ARM_ORDER} (pre-registered; fixed)")
    arms = verify_embedded()
    order = [a.strip() for a in ARM_ORDER.split(",")]
    results = {}
    for i, arm in enumerate(order):
        if i > 0:
            clean_between_arms()
        out_dir = WORK / f"{SESSION_ID}-{arm}"
        r = run_arm(arm, arms[arm], out_dir)
        evidence = parse_arm_evidence(out_dir)
        r["evidence"] = evidence
        r["memory_gate"] = memory_gate(evidence)
        log(f"arm {arm}: classification={evidence.get('classification')} "
            f"decode_wall_s={evidence.get('decode_wall_s')} "
            f"memory_gate={r['memory_gate']}")
        is_last = (i == len(order) - 1)
        if is_last and SESSION_ID == "session2":
            _LAST_ARM["flag"] = True
            r["pread_rider"] = maybe_pread_rider(out_dir)
        results[arm] = r
        (out_dir / "arm-summary.json").write_text(
            json.dumps(r, indent=2) + "\n", encoding="utf-8")
        if r["memory_gate"]["triggered"]:
            log("PRE-REGISTERED MEMORY ABORT TRIGGERED; stopping session")
            break
    session_summary = {
        "schema": "pack-cap-ab/session-summary-v1",
        "session": SESSION_ID, "arm_order": order,
        "arm_A_sha256": ARM_A_SHA256, "arm_B_sha256": ARM_B_SHA256,
        "results": results,
        "trace_bank_shared": TRACE_BANK,
        "note": "trace bank shared by design: created/validated by first arm, "
                "validated in place by second (byte-identical store; harness "
                "refuses to overwrite). Harness always exits 0; validity is "
                "judged from evidence classification.",
    }
    (WORK / f"{SESSION_ID}-summary.json").write_text(
        json.dumps(session_summary, indent=2) + "\n", encoding="utf-8")
    valid = all(
        (results.get(a, {}).get("evidence") or {}).get("classification")
        == "ACCEPT_CORRECTNESS"
        for a in order if a in results)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
