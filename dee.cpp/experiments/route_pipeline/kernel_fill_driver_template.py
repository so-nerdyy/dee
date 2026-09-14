#!/usr/bin/env python3
"""Fill-measurement session driver template (route-pipeline fill track).

Single decode arm (OFF-equivalent, 17 GiB cap, sealed prompt/16 tokens) to
build the trace bank + fresh wall datum, then the fill measurement matrix
against the REAL bank with the REAL store/cache objects:

  rider:   sequential vs random record reads, lanes 1..8, cold/warm passes
  replay:  journal-order production pattern, lanes {1,3}, qdepth {1,6},
           cold-ish pass 1 vs warm pass 2, mincore deltas

Built by build_kernel.py (fills @@FILL_PATCH_B64@@ / _SHA256 plus the shared
VARIANT_* blobs). Always exits 0 (validity from evidence, pack-cap pattern).
No optimization, no cache-policy change, no tolerance change.
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
import urllib.request
from pathlib import Path

FILL_PATCH_B64 = "@@FILL_PATCH_B64@@"
FILL_PATCH_SHA256 = "@@FILL_PATCH_SHA256@@"
VARIANT_RULES_B64 = "@@VARIANT_RULES_B64@@"
VARIANT_RULES_SHA256 = "@@VARIANT_RULES_SHA256@@"

SEALED_ARM_URL = ("https://raw.githubusercontent.com/so-nerdyy/dee/"
                  "45b2a1659d0226c15ceb8821f104b1684798be4f/"
                  "experiment/pack-cap-ab/kernels/session1/session-driver.py")
SEALED_ARM_SHA256 = "5d1f862631e403433f88478137e8321ef46664bb80c63a7fb5d703282a1617df"
ENGINE_COMMIT = "217a33359b06a0453444a698ec52e4078b77e388"
MODEL_REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
TRACE_BANK = "/tmp/dsv4-dee4-v3-trace"
WORK = Path("/kaggle/working")
OUTBASE = Path("/tmp/host-sync-kernel/out")
BANK_META = Path(TRACE_BANK) / "metadata.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [filldriver] {msg}", flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dry_run_gate() -> dict:
    report: dict = {"status": "CHECKING"}
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader"], text=True, timeout=60)
        rows = [r.strip() for r in smi.strip().splitlines() if r.strip()]
        report["nvidia_smi"] = rows
        if len(rows) != 2 or not all(", 7.5," in r and "T4" in r for r in rows):
            raise RuntimeError(f"need exactly 2x SM75 T4, saw: {rows}")
        shard_dir = Path("/kaggle/input/datasets/nivind/deepseek-v4-flash-0731-shards")
        shards = sorted(shard_dir.glob("model-*-of-00048.safetensors")) \
            if shard_dir.is_dir() else []
        if len(shards) != 48:
            for m in Path("/kaggle/input").iterdir():
                shards = sorted(m.rglob("model-*-of-00048.safetensors"))
                if len(shards) == 48:
                    break
        if len(shards) != 48:
            raise RuntimeError("48-shard dataset mount absent")
        report["dataset"] = str(shards[0].parent)
        if shutil.disk_usage("/tmp").free < 45 * (1 << 30):
            raise RuntimeError("/tmp free < 45 GiB")
        for tool in ("git", "cmake", "gcc"):
            if shutil.which(tool) is None:
                raise RuntimeError(f"missing tool: {tool}")
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            raise RuntimeError("torch CUDA prerequisite failed")
        report["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        report.update(status="FAIL_CLOSED", error=repr(exc))
    log("DRY-RUN gate: " + json.dumps(report)[:600])
    return report


def fetch_sealed_arm(workdir: Path) -> bytes:
    req = urllib.request.Request(SEALED_ARM_URL, headers={"User-Agent": "dee-fill/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        text = resp.read().decode("utf-8")
    m = re.search("ARM_A_B64 = \"([A-Za-z0-9+/=]+)\"", text)
    if not m:
        raise RuntimeError("ARM_A_B64 not found in fetched driver")
    raw = base64.b64decode(m.group(1))
    if sha256_bytes(raw) != SEALED_ARM_SHA256:
        raise RuntimeError("sealed arm-A sha mismatch")
    log(f"sealed arm-A verified: {SEALED_ARM_SHA256[:12]}")
    return raw


def build_variant(arm_bytes: bytes, workdir: Path) -> tuple[Path, dict]:
    rules_raw = base64.b64decode(VARIANT_RULES_B64)
    if sha256_bytes(rules_raw) != VARIANT_RULES_SHA256:
        raise RuntimeError("embedded variant-rules sha mismatch")
    rules = json.loads(rules_raw.decode("utf-8"))
    text = arm_bytes.decode("utf-8").replace("\r\n", "\n")
    applied = []
    for rule in rules:
        old, new = rule["old"], rule["new"]
        if text.count(old) != 1:
            raise RuntimeError(f"variant rule {rule['id']}: anchor != 1x")
        text = text.replace(old, new)
        applied.append(rule["id"])
    path = workdir / "harness_fill_variant.py"
    path.write_text(text, encoding="utf-8")
    proof = {"rules_applied": applied,
             "variant_sha256": sha256_bytes(path.read_bytes())}
    (workdir / "variant-proof.json").write_text(json.dumps(proof, indent=2))
    return path, proof


def write_patch_file(workdir: Path) -> Path:
    raw = base64.b64decode(FILL_PATCH_B64)
    if sha256_bytes(raw) != FILL_PATCH_SHA256:
        raise RuntimeError("embedded fill patch sha mismatch")
    path = workdir / "fill_instrumentation.patch"
    path.write_bytes(raw)
    return path


def run_variant(variant: Path, patch_path: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    for stale in ("dsv4-native-src", "dsv4-checkpoint", "dsv4-dee4-v2",
                  "dsv4-dee4-v2-component",
                  "host-reuse-candidate-from-baseline.bundle"):
        pp = Path("/tmp") / stale
        if pp.exists() or pp.is_symlink():
            if pp.is_dir() and not pp.is_symlink():
                shutil.rmtree(pp, ignore_errors=True)
            else:
                try:
                    pp.unlink()
                except OSError:
                    pass
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("NATIVE_") or key in (
                "DEE_HOST_PROFILE", "HOST_SYNC_EMIT", "HOST_SYNC_PROFILE_ARM",
                "HOST_SYNC_PATCH_PATH", "HOST_SYNC_PATCH_SHA256",
                "HOST_SYNC_PATCH_FILES"):
            del env[key]
    env.update({"DEE_HOST_PROFILE": "1", "HOST_SYNC_EMIT": "1",
                "HOST_SYNC_PROFILE_ARM": "1",
                "HOST_SYNC_PATCH_PATH": str(patch_path),
                "HOST_SYNC_PATCH_SHA256": FILL_PATCH_SHA256,
                "HOST_SYNC_PATCH_FILES": ",".join([
                    "dee.cpp/CMakeLists.txt",
                    "dee.cpp/include/dee/engine.h",
                    "dee.cpp/include/dee/expert_store.h",
                    "dee.cpp/include/dee/host_pack_cache.h",
                    "dee.cpp/include/dee/profiling.h",
                    "dee.cpp/pydee/pydee.cpp",
                    "dee.cpp/scripts/deepseek_v4_layer_candidate.py",
                    "dee.cpp/src/engine.cpp",
                    "dee.cpp/src/expert_store.cpp",
                    "dee.cpp/src/host_pack_cache.cpp",
                    "dee.cpp/src/profiling.cpp",
                    "dee.cpp/tools/fill_replay.cpp",
                ]),
                "PYTHONUNBUFFERED": "1"})
    stdout_path = out_dir / "arm-stdout.log"
    t0 = time.time()
    with stdout_path.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen([sys.executable, str(variant)],
                                cwd=str(out_dir), env=env, stdout=out,
                                stderr=subprocess.STDOUT)
        rc = proc.wait()
    wall = time.time() - t0
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    (out_dir / "arm-stdout.tail.log").write_text(text[-4_000_000:], encoding="utf-8")
    res_p = out_dir / "native-generate-result.json"
    result = json.loads(res_p.read_text(encoding="utf-8")) if res_p.is_file() else {}
    lru = re.findall(r"lru_cap=([0-9.]+)GiB", text)
    hp = re.findall(r"host_pack=([0-9.]+)/([0-9.]+)GiB", text)
    cap_ok = (bool(lru) and bool(hp)
              and all(abs(float(v) - 17.0) < 0.26 for v in lru)
              and all(abs(float(a) - 8.5) < 0.26 and abs(float(b) - 8.5) < 0.26
                      for a, b in hp))
    return {"exit_code": rc, "wall_s": round(wall, 1),
            "classification": result.get("classification"),
            "decode_wall_s": result.get("decode_wall_s"),
            "cap_check_ok": cap_ok,
            "run_id": result.get("run_id")}


def journal_to_replay_input(route_journal: Path, out_path: Path) -> dict:
    """Flatten routed_experts.jsonl rows to 'layer expert token' lines."""
    n_rows = n_uses = 0
    with open(route_journal, encoding="utf-8") as src, \
            open(out_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            token = row.get("forward_step", n_rows)
            for rank in row.get("expert_ids_rank_order", []):
                for expert in rank:
                    dst.write(f"{row['layer']} {expert} {token}\n")
                    n_uses += 1
            n_rows += 1
    return {"rows": n_rows, "uses": n_uses}


def run_fill_matrix(binary: Path, bank_meta: Path, journal_txt: Path,
                    out_dir: Path) -> dict:
    if not binary.is_file():
        return {"status": "skipped", "reason": "fill_replay not built"}
    if not bank_meta.is_file():
        return {"status": "skipped", "reason": "trace bank metadata absent"}
    if not journal_txt.is_file():
        return {"status": "skipped", "reason": "replay journal absent"}
    report: dict = {"bank": str(bank_meta), "cases": {}}
    rider = subprocess.run(
        [str(binary), "--mode", "rider", "--bank", str(bank_meta),
         "--lanes", "8", "--pattern", "seq",
         "--out", str(out_dir / "rider_seq.json")],
        capture_output=True, text=True, timeout=3600)
    report["cases"]["rider_seq"] = {"exit": rider.returncode,
                                    "output": rider.stdout[-2000:]}
    rider_rand = subprocess.run(
        [str(binary), "--mode", "rider", "--bank", str(bank_meta),
         "--lanes", "8", "--pattern", "rand",
         "--out", str(out_dir / "rider_rand.json")],
        capture_output=True, text=True, timeout=3600)
    report["cases"]["rider_rand"] = {"exit": rider_rand.returncode,
                                     "output": rider_rand.stdout[-2000:]}
    for lanes, qdepth, tag in ((3, 6, "production"), (1, 6, "lanes1"),
                               (3, 1, "qdepth1")):
        rep = subprocess.run(
            [str(binary), "--mode", "replay", "--bank", str(bank_meta),
             "--journal", str(journal), "--lanes", str(lanes),
             "--qdepth", str(qdepth),
             "--out", str(out_dir / f"replay_{tag}.timeline.json")],
            capture_output=True, text=True, timeout=3600)
        report["cases"][f"replay_{tag}"] = {
            "exit": rep.returncode, "output": rep.stdout[-2000:],
            "stderr_tail": rep.stderr[-1000:]}
    report["status"] = "ok"
    return report


def main() -> int:
    workdir = Path("/tmp/host-sync-kernel")
    workdir.mkdir(parents=True, exist_ok=True)
    log(f"fill driver build: @@DRIVER_BUILD_ID@@")
    session: dict = {"session": "host-sync-fill", "arms": {}}
    gate = dry_run_gate()
    session["dry_run_gate"] = gate
    if gate["status"] != "PASS":
        (WORK / "session-summary.json").write_text(json.dumps(session, indent=2))
        log("DRY-RUN gate failed; refusing expensive decode")
        return 0
    arm_bytes = fetch_sealed_arm(workdir)
    variant, proof = build_variant(arm_bytes, workdir)
    session["variant_proof"] = proof
    patch_path = write_patch_file(workdir)
    session["fill_patch_sha256"] = FILL_PATCH_SHA256
    out_dir = OUTBASE / "profile-fill"
    rec = run_variant(variant, patch_path, out_dir)
    session["arms"]["FILL"] = rec
    log(f"arm FILL: classification={rec.get('classification')} "
        f"decode_wall_s={rec.get('decode_wall_s')}")
    journal = out_dir / "routed_experts.jsonl"
    replay_journal = OUTBASE / "journal.txt"
    if journal.is_file():
        session["journal"] = journal_to_replay_input(journal, replay_journal)
    else:
        session["journal"] = {"error": "routed_experts.jsonl absent"}
    build_dir = OUTBASE / "fill-build"
    build_dir.mkdir(parents=True, exist_ok=True)
    # Configure a SEPARATE host-only build dir (never touches the arm's
    # CUDA build): DEE_CUDA=OFF + DEE_BUILD_FILL_REPLAY=ON + tests off.
    # Source root is the variant's verified clone (/tmp/dsv4-native-src).
    try:
        dee_src = Path("/tmp/dsv4-native-src/dee.cpp")
        if not (dee_src / "tools" / "fill_replay.cpp").is_file():
            raise RuntimeError("patched fill_replay.cpp absent from clone")
        replay_build = dee_src / "build-fill"
        cfg = subprocess.run(
            ["cmake", "-S", str(dee_src), "-B", str(replay_build),
             "-DDEE_CUDA=OFF", "-DDEE_BUILD_TESTS=OFF",
             "-DDEE_BUILD_FILL_REPLAY=ON", "-DCMAKE_BUILD_TYPE=Release"],
            capture_output=True, text=True, timeout=1800)
        if cfg.returncode != 0:
            raise RuntimeError("fill_replay configure failed: " + cfg.stderr[-2000:])
        bld = subprocess.run(
            ["cmake", "--build", str(replay_build), "--target", "fill_replay",
             "-j", str(max(1, min(4, os.cpu_count() or 2)))],
            capture_output=True, text=True, timeout=3600)
        if bld.returncode != 0:
            raise RuntimeError("fill_replay build failed: " + bld.stderr[-2000:])
        binary = replay_build / "fill_replay"
        session["fill_build"] = {"ok": True, "binary": str(binary)}
    except Exception as exc:  # noqa: BLE001
        session["fill_build"] = {"ok": False, "error": repr(exc)}
        binary = None
    if binary is not None and replay_journal.is_file():
        session["fill_matrix"] = run_fill_matrix(
            binary, Path(TRACE_BANK) / "metadata.json", replay_journal, out_dir)
    else:
        session["fill_matrix"] = {"status": "skipped",
                                  "reason": "no binary or no journal"}
    (WORK / "session-summary.json").write_text(json.dumps(session, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
