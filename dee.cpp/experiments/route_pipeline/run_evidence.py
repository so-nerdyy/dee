#!/usr/bin/env python3
"""One-command end-to-end evidence pipeline (route-pipeline track).

Live dual-T4 sequence (all steps real, same code path as mocks):
  1. profile pack (matched off/on, pair validation, ingestion, closure)
  2. ABC mechanics (parsed, normalized)
  3. ranking from parsed evidence + next-A/B derivation
  4. evidence bundle with manifest (hash-validated, resumable)

Mock/local modes (same ingestion/aggregation code, fixture data only):
  --mock-abc PATH      ingest ABC JSON from file instead of executing CUDA
  --dry-run-live       validate readiness without executing the model
  --resume             reuse hash-verified terminal stage outputs

Without a live GPU (and without --dry-run-live) it writes a BLOCKED bundle
(environment proof, empty schemas, zero timings) and exits 3. It never
invents timings: every numeric timing field is measured on the executing
host, parsed from explicit fixture files, or null/UNKNOWN.

Usage (T4 host, campaign checkout):
    python3 run_evidence.py --out evidence/ --command "python3 <decode.py> ..." \\
        --prompt-hash <sha256> --tokens 16 --reps 5
Usage (mock end-to-end, any host):
    python3 mock_campaign.py --out /tmp/m0 --scenario 1
    python3 run_evidence.py --out evidence/ --command "python3 mock_campaign.py ..." \\
        --prompt-hash mock-prompt-hash --mock-abc /tmp/m0/abc.json [--extra-cmd-args ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evidence import (  # noqa: E402
    build_next_ab,
    ingest_abc,
    rank_from_evidence,
)

CLASSIFICATION_BLOCKED = "BLOCKED_LIVE_GPU"
EXIT_BLOCKED = 3
MANIFEST_NAME = "manifest.json"
STAGES = ("profile", "abc", "rank")


def probe_environment() -> dict:
    """Honest capability probe. No timing, no fabrication."""
    env: dict = {"nvidia_smi": False, "gpus": [], "cuda": False,
                 "torch": None, "sm75_only": False}
    if shutil.which("nvidia-smi") is not None:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,compute_cap",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=60)
            if proc.returncode == 0:
                env["nvidia_smi"] = True
                for line in proc.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        env["gpus"].append(
                            {"index": parts[0], "name": parts[1],
                             "compute_cap": parts[2]})
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import torch  # noqa: PLC0415
        env["torch"] = torch.__version__
        env["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    env["sm75_only"] = (len(env["gpus"]) == 2
                        and all(g["compute_cap"] == "7.5" for g in env["gpus"]))
    env["live_dual_t4"] = bool(env["cuda"] and env["sm75_only"])
    return env


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(out: Path) -> dict:
    try:
        data = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_manifest(out: Path, manifest: dict) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out / MANIFEST_NAME, manifest)


def stage_verified(out: Path, manifest: dict, stage: str,
                   artifacts: list[str]) -> bool:
    """A stage is reusable only if the manifest marks it terminal AND every
    listed artifact exists with a matching sha256. Partial output is never
    reused."""
    entry = manifest.get("stages", {}).get(stage)
    if not entry or entry.get("status") != "terminal":
        return False
    recorded = entry.get("artifacts", {})
    for name in artifacts:
        blob = out / name
        if not blob.is_file():
            return False
        if recorded.get(name) != sha256_file(blob):
            return False
    return True


def mark_stage(out: Path, manifest: dict, stage: str,
               artifacts: list[str]) -> None:
    manifest.setdefault("stages", {})[stage] = {
        "status": "terminal",
        "artifacts": {name: sha256_file(out / name) for name in artifacts},
    }
    save_manifest(out, manifest)


def blocked_bundle(out: Path, env: dict, reason: str) -> dict:
    """BLOCKED evidence: environment proof + empty schemas, zero timings."""
    bundle = {
        "classification": CLASSIFICATION_BLOCKED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "environment": env,
        "timings_invented": 0,
    }
    write_json(out / "environment.json", {"environment": env, "reason": reason})
    empty_attribution = {
        "classification": CLASSIFICATION_BLOCKED,
        "records": [],
        "note": "no live GPU on the executing host; no timings measured or inferred",
    }
    write_json(out / "host-sync-attribution.json", empty_attribution)
    (out / "per-layer.csv").write_text(
        "token,layer,device,route_d2h_host_wait_ms,native_call_wall_ms,"
        "source_wait_ms,fill_wait_ms,h2d_ms,decode_ms,expert_compute_ms,"
        "native_output_sync_wait_ms,shared_expert_ms,combine_ms,"
        "orchestration_ms,handoff_ms,provenance\n", encoding="utf-8")
    write_json(out / "per-token.json", {"classification": CLASSIFICATION_BLOCKED,
                                        "tokens": []})
    write_json(out / "closure.json", {"classification": CLASSIFICATION_BLOCKED,
                                      "accounted_ms": None, "unknown_ms": None,
                                      "closure_fraction": None,
                                      "note": "no wall to reconcile without execution"})
    write_json(out / "abc-mechanics.json", {"classification": CLASSIFICATION_BLOCKED,
                                            "cases": {}})
    write_json(out / "candidate-ranking.json",
               {"classification": CLASSIFICATION_BLOCKED,
                "candidates": {name: {"rank": "UNMEASURED", "reason": "no live data"}
                               for name in ("HASH_EARLY_STAGING", "SHARED_OVERLAP",
                                            "EVENT_HANDOFF", "ROUTE_D2H_NARROW", "C1_C2")}})
    write_json(out / "next-ab.json",
               {"classification": CLASSIFICATION_BLOCKED,
                "next": "run this same entry point on a dual-T4 host; "
                        "no A/B is specified until live attribution exists"})
    for mode in ("profile-off", "profile-on"):
        (out / mode).mkdir(parents=True, exist_ok=True)
        write_json(out / mode / "status.json",
                   {"classification": CLASSIFICATION_BLOCKED,
                    "note": "matched validation not executed without a live GPU"})
    bundle["artifacts"] = sorted(p.name for p in out.iterdir())
    write_json(out / "bundle.json", bundle)
    return bundle


def run_profile_stage(out: Path, args) -> dict:
    """Matched off/on via the pack. Returns the pack's sync-profile-result."""
    cmd = [sys.executable, str(HERE / "kaggle_profile_pack.py"), "--run",
           "--out", str(out / "profile-pack"),
           "--command", args.command, "--prompt-hash", args.prompt_hash]
    if args.tokens is not None:
        cmd += ["--tokens", str(args.tokens)]
    if args.extra_cmd_args:
        cmd += ["--extra-cmd-args", args.extra_cmd_args]
    if args.no_gpu_check:
        cmd += ["--skip-gpu-check"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        detail = proc.stderr[-2000:]
        try:
            pack_report = json.loads(
                (out / "profile-pack" / "sync-profile-result.json").read_text(encoding="utf-8"))
            detail = f"{pack_report.get('status')}; {detail}"
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        raise RuntimeError(f"profile pack failed: {detail}")
    return json.loads((out / "profile-pack" / "sync-profile-result.json").read_text(
        encoding="utf-8"))


def run_abc_stage(out: Path, args) -> dict:
    """ABC mechanics: execute the CUDA runner, or ingest --mock-abc file."""
    if args.mock_abc is not None:
        try:
            payload = json.loads(Path(args.mock_abc).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"mock ABC unreadable: {exc}")
        return ingest_abc(payload)
    cmd = [sys.executable, str(HERE / "kaggle_runner_abc.py"),
           "--workdir", str(out / "abc"), "--reps", str(args.reps),
           "--out-json", str(out / "abc_raw.json")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"ABC runner failed: {proc.stderr[-2000:]}")
    payload = json.loads((out / "abc_raw.json").read_text(encoding="utf-8"))
    return ingest_abc(payload)


def publish_pack_outputs(out: Path) -> None:
    """Copy pack artifacts into the evidence bundle layout."""
    pack = out / "profile-pack"
    (out / "profile-off").mkdir(parents=True, exist_ok=True)
    (out / "profile-on").mkdir(parents=True, exist_ok=True)
    for mode in ("off", "on"):
        src = pack / mode
        dst = out / ("profile-off" if mode == "off" else "profile-on")
        if src.is_dir():
            for name in ("result.json", "host-profile.jsonl",
                         "stage-profile.json", "correctness.json"):
                blob = src / name
                if blob.is_file():
                    (dst / name).write_bytes(blob.read_bytes())
    for name in ("host-sync-attribution.json", "per-layer.csv",
                 "per-token.json", "closure.json"):
        blob = pack / name
        if blob.is_file():
            (out / name).write_bytes(blob.read_bytes())
        else:
            raise RuntimeError(f"pack did not produce {name}")


PROFILE_ARTIFACTS = ["host-sync-attribution.json", "per-layer.csv",
                     "per-token.json", "closure.json"]
ABC_ARTIFACTS = ["abc-mechanics.json"]
RANK_ARTIFACTS = ["candidate-ranking.json", "next-ab.json"]


def _repo_root() -> Path:
    for candidate in (HERE.parents[2], Path.cwd()):
        if (candidate / "dee.cpp" / "include" / "dee" / "profiling.h").is_file():
            return candidate
    return Path.cwd()


def _profiler_source_markers() -> dict:
    """Verify the default-off instrumentation is present in source."""
    root = _repo_root()
    checks = {
        "profiling.h HostSpanGuard":
            root / "dee.cpp" / "include" / "dee" / "profiling.h",
        "engine.cpp NativeOutputSync marker":
            root / "dee.cpp" / "src" / "engine.cpp",
        "candidate DEE_HOST_PROFILE gate":
            root / "dee.cpp" / "scripts" / "deepseek_v4_layer_candidate.py",
    }
    needles = {
        "profiling.h HostSpanGuard": "HostSpanGuard",
        "engine.cpp NativeOutputSync marker": "NativeOutputSync",
        "candidate DEE_HOST_PROFILE gate": 'os.environ.get("DEE_HOST_PROFILE")',
    }
    return {name: (path.is_file() and needles[name] in path.read_text(encoding="utf-8"))
            for name, path in checks.items()}


def dry_run_live(out: Path, args, env: dict) -> dict:
    """Readiness validation WITHOUT executing the model. Every check is
    explicit; any failure names the missing piece."""
    checks: dict = {}
    checks["dual_t4"] = {"pass": bool(env["live_dual_t4"]), "detail": env["gpus"]}
    first = (args.command or "").split()[0] if args.command else ""
    cmd_path = Path(first) if first else None
    checks["command_exists"] = {
        "pass": bool(cmd_path and (cmd_path.is_file() or shutil.which(first))),
        "detail": first or "(no --command given)"}
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".writetest"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError as exc:
        writable = str(exc)
    checks["out_writable"] = {"pass": writable is True, "detail": writable}
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=60)
        checks["source_sha"] = {"pass": proc.returncode == 0,
                                "detail": proc.stdout.strip()[:12]}
    except OSError as exc:
        checks["source_sha"] = {"pass": False, "detail": str(exc)}
    markers = _profiler_source_markers()
    try:
        import host_profiler  # noqa: PLC0415
        modules_ok: object = True
    except ImportError as exc:
        modules_ok = str(exc)
    checks["profiler_source"] = {"pass": all(markers.values()), "detail": markers}
    checks["python_modules"] = {"pass": modules_ok is True, "detail": modules_ok}
    checks["abc_runner"] = {"pass": (HERE / "kaggle_runner_abc.py").is_file()}
    try:
        from host_profiler import compute_closure  # noqa: PLC0415
        compute_closure([], decode_wall_ms=1.0)
        schema_ok: object = True
    except Exception as exc:  # noqa: BLE001
        schema_ok = str(exc)
    checks["schemas"] = {"pass": schema_ok is True, "detail": schema_ok}
    if args.fixture_shard is not None:
        checks["checkpoint_path"] = {"pass": Path(args.fixture_shard).exists(),
                                     "detail": str(args.fixture_shard)}
    report = {"dry_run": True, "checks": checks,
              "ready": all(c.get("pass") for c in checks.values())}
    write_json(out / "dry-run.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("research/route-pipeline/evidence"))
    parser.add_argument("--command", default=None)
    parser.add_argument("--prompt-hash", default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--fixture-shard", type=Path, default=None)
    parser.add_argument("--extra-cmd-args", default="")
    parser.add_argument("--mock-abc", type=Path, default=None)
    parser.add_argument("--no-gpu-check", action="store_true",
                        help="mock/local validation only; never for T4 evidence")
    parser.add_argument("--dry-run-live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    env = probe_environment()
    if args.dry_run_live:
        report = dry_run_live(args.out, args, env)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 2

    if not env["live_dual_t4"] and not args.no_gpu_check:
        reasons = []
        if not env["nvidia_smi"]:
            reasons.append("nvidia-smi absent")
        if not env["cuda"]:
            reasons.append("torch.cuda unavailable")
        if env["nvidia_smi"] and not env["sm75_only"]:
            reasons.append(f"GPUs are not dual SM75: {env['gpus']}")
        bundle = blocked_bundle(args.out, env, "; ".join(reasons) or "unknown")
        print(json.dumps(bundle, indent=2))
        return EXIT_BLOCKED

    if not args.command or not args.prompt_hash:
        print(json.dumps({"classification": "LIVE_HOST_MISSING_ARGS",
                          "note": "--command and --prompt-hash are required "
                                  "for a live run"}))
        return 2
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.out) if args.resume else {}
    reused: list[str] = []
    try:
        if stage_verified(args.out, manifest, "profile", PROFILE_ARTIFACTS):
            reused.append("profile")
        else:
            pack_result = run_profile_stage(args.out, args)
            publish_pack_outputs(args.out)
            mark_stage(args.out, manifest, "profile", PROFILE_ARTIFACTS)
        if stage_verified(args.out, manifest, "abc", ABC_ARTIFACTS):
            reused.append("abc")
            mechanics = ingest_abc(json.loads(
                (args.out / "abc-mechanics.json").read_text(encoding="utf-8")))
        else:
            mechanics = run_abc_stage(args.out, args)
            write_json(args.out / "abc-mechanics.json", mechanics)
            mark_stage(args.out, manifest, "abc", ABC_ARTIFACTS)
        closure = json.loads((args.out / "closure.json").read_text(encoding="utf-8"))
        ranking = rank_from_evidence(closure, mechanics)
        write_json(args.out / "candidate-ranking.json", ranking)
        next_ab = build_next_ab(ranking)
        write_json(args.out / "next-ab.json", next_ab)
        mark_stage(args.out, manifest, "rank", RANK_ARTIFACTS)
        bundle = {"classification": "LIVE_RUN_OK",
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "reused_stages": reused,
                  "ranking": {k: v.get("rank") for k, v in
                              ranking.get("candidates", {}).items()},
                  "next": next_ab.get("classification"),
                  "closure_status": ranking.get("closure_status"),
                  "elapsed_s": round(time.time() - t0, 3)}
        write_json(args.out / "bundle.json", bundle)
        print(json.dumps(bundle, indent=2))
        return 0
    except (RuntimeError, ValueError) as exc:
        failure = {"classification": "EVIDENCE_FAILURE", "error": str(exc),
                   "reused_stages": reused}
        write_json(args.out / "bundle.json", failure)
        print(json.dumps(failure, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
