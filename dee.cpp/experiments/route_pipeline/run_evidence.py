#!/usr/bin/env python3
"""One-command evidence entry point (route-pipeline host/sync profile track).

On a dual-T4 host with the campaign checkout, this runs the full evidence
sequence and writes the bundle:

    profile-off/ + profile-on/ matched validation  (Part A)
    host-sync-attribution.json / per-layer.csv / per-token.json  (Part B)
    closure.json  (Part C, non-double-counting)
    route-D2H / final-sync / shared-expert attribution  (Parts D-F)
    abc-mechanics.json  (Part G, kaggle_runner_abc.py)
    candidate-ranking.json / next-ab.json  (Part H/J)

Without a live GPU it writes a BLOCKED bundle (environment proof, empty
schemas, zero timings) and exits 3. It never invents timings: every numeric
timing field is either measured on the executing host or null/UNKNOWN.

Usage (T4 host, campaign checkout):
    python3 run_evidence.py --out evidence/ --command "python3 <decode.py> ..." \\
        --prompt-hash <sha256> --tokens 16 --reps 5
Usage (any host, always safe):
    python3 run_evidence.py --out evidence/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSIFICATION_BLOCKED = "BLOCKED_LIVE_GPU"
EXIT_BLOCKED = 3


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


def run_live(out: Path, env: dict, args) -> dict:
    """Live sequence. Each step shells to the existing pack/runners so this
    file owns orchestration only, never measurement logic."""
    steps: dict = {"classification": "LIVE_RUN", "steps": {}}

    def sh(cmd: list[str], label: str) -> dict:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        rec = {"exit": proc.returncode,
               "stdout_tail": proc.stdout[-2000:],
               "stderr_tail": proc.stderr[-2000:]}
        steps["steps"][label] = rec
        if proc.returncode not in (0,):
            raise RuntimeError(f"live step failed: {label}")
        return rec

    pack = [sys.executable, str(HERE / "kaggle_profile_pack.py"), "--run",
            "--out", str(out / "profile-pack"),
            "--command", args.command, "--prompt-hash", args.prompt_hash,
            "--tokens", str(args.tokens)]
    sh(pack, "profile_pack_off_on")
    abc = [sys.executable, str(HERE / "kaggle_runner_abc.py"),
           "--workdir", str(out / "abc"), "--reps", str(args.reps)]
    if args.fixture_shard:
        abc += ["--shard", str(args.fixture_shard)]
    sh(abc, "abc_mechanics")
    steps["note"] = ("closure/ranking aggregation reads the pack + ABC "
                     "artifacts via host_profiler; see evidence bundle.")
    return steps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("research/route-pipeline/evidence"))
    parser.add_argument("--command", default=None)
    parser.add_argument("--prompt-hash", default=None)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--fixture-shard", type=Path, default=None)
    args = parser.parse_args(argv)

    env = probe_environment()
    if not env["live_dual_t4"]:
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
    steps = run_live(args.out, env, args)
    steps["elapsed_s"] = round(time.time() - t0, 3)
    write_json(args.out / "bundle.json", steps)
    print(json.dumps(steps, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
