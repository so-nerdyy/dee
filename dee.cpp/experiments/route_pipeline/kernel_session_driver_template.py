#!/usr/bin/env python3
"""Kernel session driver template for the host-sync profile run.

Built by build_kernel.py, which fills @@PROFILER_PATCH_B64@@,
@@PROFILER_PATCH_SHA256@@ and @@VARIANT_RULES_JSON@@ (rules audited in
make_profile_variant.py). The driver itself is generic orchestration.

Sequence: dry-run gate -> fetch sealed arm-A (sha-pinned) -> build variant
-> for each arm (OFF1, ON, OFF2): clean workspace, run variant subprocess
  with arm env, collect evidence -> pair validation -> ABC mechanics ->
  profile-run layout -> bundle. Always exits 0 (validity from evidence).
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

PROFILER_PATCH_B64 = "@@PROFILER_PATCH_B64@@"
PROFILER_PATCH_SHA256 = "@@PROFILER_PATCH_SHA256@@"
VARIANT_RULES_B64 = "@@VARIANT_RULES_B64@@"
VARIANT_RULES_SHA256 = "@@VARIANT_RULES_SHA256@@"

SEALED_ARM_URL = ("https://raw.githubusercontent.com/so-nerdyy/dee/"
                  "45b2a1659d0226c15ceb8821f104b1684798be4f/"
                  "experiment/pack-cap-ab/kernels/session1/session-driver.py")
SEALED_ARM_SHA256 = "5d1f862631e403433f88478137e8321ef46664bb80c63a7fb5d703282a1617df"
SEALED_ARM_RE = "ARM_A_B64 = \"([A-Za-z0-9+/=]+)\""
ENGINE_COMMIT = "217a33359b06a0453444a698ec52e4078b77e388"
MODEL_REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
TRACE_BANK = "/tmp/dsv4-dee4-v3-trace"
WORK = Path("/kaggle/working")
# Per-arm collection lives OUTSIDE WORK (the variant writes evidence into
# WORK; collecting inside WORK would trip the empty-at-start guard).
OUTBASE = Path("/tmp/host-sync-kernel/out")
EXPERT_BYTES = 13369344
HASH_READS = 18

ARMS = (
    {"name": "OFF1", "env": {}},
    {"name": "ON", "env": {"NATIVE_PROFILE": "1", "DEE_HOST_PROFILE": "1",
                           "HOST_SYNC_EMIT": "1", "HOST_SYNC_PROFILE_ARM": "1",
                           "HOST_SYNC_PATCH_PATH": "__PATCH_PATH__"}},
    {"name": "OFF2", "env": {}},
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [profdriver] {msg}", flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dry_run_gate() -> dict:
    """Fail fast before anything expensive. Returns the gate report."""
    report: dict = {"status": "CHECKING"}
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader"], text=True, timeout=60)
        rows = [r.strip() for r in smi.strip().splitlines() if r.strip()]
        report["nvidia_smi"] = rows
        if len(rows) != 2 or not all(", 7.5," in r and "T4" in r for r in rows):
            raise RuntimeError(f"need exactly 2x SM75 T4, saw: {rows}")
        mounts = [p for p in Path("/kaggle/input").iterdir()] if \
            Path("/kaggle/input").is_dir() else []
        # Canonical mount (mirrors the sealed harness DATASET_DIR).
        shard_dir = Path("/kaggle/input/datasets/nivind/deepseek-v4-flash-0731-shards")
        shards = sorted(shard_dir.glob("model-*-of-00048.safetensors")) \
            if shard_dir.is_dir() else []
        if len(shards) != 48:
            # Fallback: recursive search under every input mount.
            for m in mounts:
                shards = sorted(m.rglob("model-*-of-00048.safetensors"))
                if len(shards) == 48:
                    shard_dir = shards[0].parent
                    break
        if len(shards) != 48:
            raise RuntimeError("48-shard dataset mount absent")
        report["dataset"] = str(shard_dir)
        free_tmp = shutil.disk_usage("/tmp").free
        if free_tmp < 45 * (1 << 30):
            raise RuntimeError(f"/tmp free {free_tmp} < 45 GiB")
        for tool in ("git", "cmake", "nvcc", "gcc"):
            if shutil.which(tool) is None:
                raise RuntimeError(f"missing tool: {tool}")
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("torch CUDA 2-GPU prerequisite failed")
        report["torch"] = torch.__version__
        report["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        report.update(status="FAIL_CLOSED", error=repr(exc))
    log("DRY-RUN gate: " + json.dumps(report)[:800])
    return report


def fetch_sealed_arm(workdir: Path) -> bytes:
    log("fetching sealed arm-A bytes (sha-pinned)")
    req = urllib.request.Request(SEALED_ARM_URL, headers={"User-Agent": "dee-profiler/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        text = resp.read().decode("utf-8")
    m = re.search(SEALED_ARM_RE, text)
    if not m:
        raise RuntimeError("ARM_A_B64 not found in fetched driver")
    raw = base64.b64decode(m.group(1))
    if sha256_bytes(raw) != SEALED_ARM_SHA256:
        raise RuntimeError("sealed arm-A sha mismatch")
    (workdir / "armA_sealed.py").write_bytes(raw)
    log(f"sealed arm-A verified: {SEALED_ARM_SHA256[:12]} ({len(raw)} bytes)")
    return raw


def build_variant(arm_bytes: bytes, workdir: Path) -> tuple[Path, dict]:
    rules_raw = base64.b64decode(VARIANT_RULES_B64)
    if sha256_bytes(rules_raw) != VARIANT_RULES_SHA256:
        raise RuntimeError("embedded variant-rules sha mismatch")
    rules = json.loads(rules_raw.decode("utf-8"))
    text = arm_bytes.decode("utf-8")
    applied = []
    for rule in rules:
        old, new = rule["old"], rule["new"]
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"variant rule {rule['id']}: anchor {count}x (need 1)")
        text = text.replace(old, new)
        applied.append(rule["id"])
    variant_path = workdir / "harness_profile_variant.py"
    variant_path.write_text(text, encoding="utf-8")
    proof = {"rules_applied": applied,
             "variant_sha256": sha256_bytes(variant_path.read_bytes()),
             "rule_ids": [r["id"] for r in rules],
             "rule_whys": {r["id"]: r["why"] for r in rules}}
    (workdir / "variant-proof.json").write_text(json.dumps(proof, indent=2))
    log(f"variant built: {proof['variant_sha256'][:12]} rules={applied}")
    return variant_path, proof


def write_patch_file(workdir: Path) -> Path:
    raw = base64.b64decode(PROFILER_PATCH_B64)
    if sha256_bytes(raw) != PROFILER_PATCH_SHA256:
        raise RuntimeError("embedded profiler patch sha mismatch")
    path = workdir / "host_sync_profiler.patch"
    path.write_bytes(raw)
    log(f"profiler patch staged: {PROFILER_PATCH_SHA256[:12]}")
    return path


def run_arm(name: str, env_extra: dict, variant: Path, patch_path: Path,
            out_dir: Path) -> dict:
    leftovers = [p.name for p in WORK.iterdir()] if WORK.is_dir() else []
    if leftovers:
        raise RuntimeError(f"WORK not empty at arm {name} start: {leftovers}")
    work_root = Path(f"/tmp/host-sync-prof-{name}")
    if work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("NATIVE_") or key in (
                "DEE_HOST_PROFILE", "HOST_SYNC_EMIT", "HOST_SYNC_PROFILE_ARM",
                "HOST_SYNC_PATCH_PATH"):
            del env[key]
    for key, value in env_extra.items():
        env[key] = str(patch_path) if value == "__PATCH_PATH__" else value
    env.setdefault("PYTHONUNBUFFERED", "1")
    mem_before = _mem_snapshot()
    stdout_path = work_root / "arm-stdout.live.log"
    t0 = time.time()
    with stdout_path.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen([sys.executable, str(variant)],
                                cwd=str(work_root), env=env, stdout=out,
                                stderr=subprocess.STDOUT)
        rc = proc.wait()
    wall = time.time() - t0
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arm-stdout.log").write_text(text[-4_000_000:], encoding="utf-8")
    copied = []
    for p in sorted(WORK.iterdir()):
        if p.is_file():
            shutil.copy2(p, out_dir / p.name)
            copied.append(p.name)
            try:
                p.unlink()
            except OSError:
                pass
    lru = re.findall(r"lru_cap=([0-9.]+)GiB", text)
    hp = re.findall(r"host_pack=([0-9.]+)/([0-9.]+)GiB", text)
    cap_ok = (bool(lru) and bool(hp)
              and all(abs(float(v) - 17.0) < 0.26 for v in lru)
              and all(abs(float(a) - 8.5) < 0.26 and abs(float(b) - 8.5) < 0.26
                      for a, b in hp))
    return {"arm": name, "exit_code": rc, "wall_s": round(wall, 1),
            "evidence_files": copied, "workspace": str(work_root),
            "lru_cap_logged": [float(v) for v in lru],
            "host_pack_logged": [[float(a), float(b)] for a, b in hp],
            "cap_check_ok": cap_ok,
            "mem_before": mem_before, "mem_after": _mem_snapshot()}


def _mem_snapshot() -> dict:
    info: dict = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            for key in ("MemTotal:", "MemAvailable:", "Cached:"):
                if line.startswith(key):
                    info[key.strip(":")] = int(line.split()[1]) // 1024
    except OSError:
        pass
    return info


def clean_between_arms() -> None:
    for name in ("dsv4-native-src", "dsv4-checkpoint", "dsv4-dee4-v2",
                 "dsv4-dee4-v2-component",
                 "host-reuse-candidate-from-baseline.bundle"):
        pp = Path("/tmp") / name
        if pp.exists() or pp.is_symlink():
            if pp.is_dir() and not pp.is_symlink():
                shutil.rmtree(pp, ignore_errors=True)
            else:
                try:
                    pp.unlink()
                except OSError:
                    pass
    log("between-arms cleanup done (kept trace bank)")


def parse_result(arm_dir: Path) -> dict:
    out: dict = {"classification": None}
    res_p = arm_dir / "native-generate-result.json"
    if res_p.is_file():
        try:
            r = json.loads(res_p.read_text(encoding="utf-8"))
            for key in ("classification", "run_id", "commit",
                        "model_revision", "decode_wall_s", "decode_tok_s",
                        "prefill_ms", "generated_token_ids", "decoded_text",
                        "host_pack_budget_gib", "correctness",
                        "performance_eligible", "host_sync_emission"):
                out[key] = r.get(key)
        except Exception as exc:  # noqa: BLE001
            out["parse_error"] = repr(exc)
    return out


def abc_mechanics(workdir: Path) -> dict:
    """Torch-only A/B/C mechanics (same keys as kaggle_runner_abc.py)."""
    import torch  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    rep: dict = {"gpu": torch.cuda.get_device_name(0), "cases": {}}
    f = workdir / "hash18.bin"
    if not f.is_file() or f.stat().st_size != HASH_READS * EXPERT_BYTES:
        with open(f, "wb") as fh:
            fh.write(os.urandom(1 << 20))
            fh.seek(HASH_READS * EXPERT_BYTES - 1)
            fh.write(b"\x00")
    s = torch.cuda.Stream()
    serial, batched = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        with open(f, "rb") as fh:
            for _ in range(HASH_READS):
                buf = fh.read(EXPERT_BYTES)
                d = torch.empty(EXPERT_BYTES, dtype=torch.uint8, device="cuda")
                d.copy_(torch.from_numpy(np.frombuffer(buf, dtype="uint8")),
                        non_blocking=False)
        torch.cuda.synchronize()
        serial.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        with open(f, "rb") as fh:
            blobs = [fh.read(EXPERT_BYTES) for _ in range(HASH_READS)]
        with torch.cuda.stream(s):
            devs = []
            for b in blobs:
                d = torch.empty(EXPERT_BYTES, dtype=torch.uint8, device="cuda")
                d.copy_(torch.from_numpy(np.frombuffer(b, dtype="uint8")),
                        non_blocking=True)
                devs.append(d)
        torch.cuda.synchronize()
        batched.append((time.perf_counter() - t0) * 1000.0)
        del devs
    import statistics as st
    rep["cases"]["A_hash_staging"] = {
        "status": "ok", "fixture": "synthetic", "reads": HASH_READS,
        "bytes": HASH_READS * EXPERT_BYTES,
        "route_table_realism": "ROUTE_TABLE_REALISM_NOT_MEASURED",
        "serial_ms": {"p50": st.median(serial), "mean": st.fmean(serial),
                      "min": min(serial), "n": len(serial)},
        "batched_ms": {"p50": st.median(batched), "mean": st.fmean(batched),
                       "min": min(batched), "n": len(batched)}}
    h, ic = 4096, 2048
    w = torch.randn(ic, h, dtype=torch.float16, device="cuda")
    x = torch.randn(h, dtype=torch.float16, device="cuda")
    import torch.nn.functional as Fn
    alone, loaded = [], []
    big = torch.empty(1 << 28, dtype=torch.uint8, device="cuda")
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            y = Fn.linear(x, w)
        torch.cuda.synchronize()
        alone.append((time.perf_counter() - t0) * 1000.0 / 20)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(s):
            for _ in range(20):
                big.copy_(torch.zeros_like(big))
        for _ in range(20):
            y = Fn.linear(x, w)
        torch.cuda.synchronize()
        loaded.append((time.perf_counter() - t0) * 1000.0 / 20)
    a, b = st.median(alone), st.median(loaded)
    rep["cases"]["B_contention"] = {
        "status": "ok", "fixture": "synthetic",
        "shape_note": "ASSUMED 4096x2048 fp16",
        "alone_ms": {"p50": a, "mean": st.fmean(alone), "min": min(alone), "n": 5},
        "loaded_ms": {"p50": b, "mean": st.fmean(loaded), "min": min(loaded), "n": 5},
        "efficiency_c": round(a / b, 4) if b else None}
    syncs, events = [], []
    cs = torch.cuda.current_stream()
    for _ in range(20):
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        syncs.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        e = torch.cuda.Event()
        e.record(cs)
        e.synchronize()
        events.append((time.perf_counter() - t0) * 1000.0)
    rep["cases"]["C_barrier"] = {
        "status": "ok",
        "sync_ms": {"p50": st.median(syncs), "mean": st.fmean(syncs),
                    "min": min(syncs), "n": len(syncs)},
        "event_ms": {"p50": st.median(events), "mean": st.fmean(events),
                     "min": min(events), "n": len(events)}}
    rep["status"] = "ok"
    return rep


def to_profile_run(arm_dir: Path, out_dir: Path, prompt_hash: str,
                   profiling: bool) -> dict:
    """Convert arm evidence to the canonical profile-run/ layout."""
    res = parse_result(arm_dir)
    hs = arm_dir / "host-sync-records.json"
    rows = sorted(arm_dir.glob("host-profile-rows-l*.json"))
    py_rows = []
    for p in rows:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            py_rows.extend(data if isinstance(data, list) else [data])
        except Exception as exc:  # noqa: BLE001
            return {"error": f"python rows unparseable in {p.name}: {exc}"}
    cpp_records = []
    if hs.is_file():
        try:
            cpp_records = json.loads(hs.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {"error": f"host-sync-records.json unparseable: {exc}"}
    if isinstance(cpp_records, dict):
        merged_cpp = []
        for gpu_recs in cpp_records.values():
            if isinstance(gpu_recs, dict) and isinstance(
                    gpu_recs.get("records"), list):
                merged_cpp.extend(gpu_recs["records"])
            elif isinstance(gpu_recs, list):
                merged_cpp.extend(gpu_recs)
        cpp_records = merged_cpp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps({
        "status": res.get("classification") == "ACCEPT_CORRECTNESS"
        and "ok" or "arm-invalid",
        "metrics": {"decode_wall_s": res.get("decode_wall_s"), "tokens": 16},
        "generated_ids": res.get("generated_token_ids"),
        "decoded_text": res.get("decoded_text"),
        "source_sha": res.get("commit"),
        "prompt_hash": prompt_hash,
        "config": {"cache_dtype": "fp4", "expert_store": "dee4_trace",
                   "lanes": 3, "qdepth": 6, "cap_gib_total": 17.0,
                   "profile_stages": profiling,
                   "DEE_HOST_PROFILE": profiling},
        "hardware": {"count": 2, "names": ["Tesla T4", "Tesla T4"]},
    }, indent=2))
    (out_dir / "host-profile.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in py_rows), encoding="utf-8")
    (out_dir / "stage-profile.json").write_text(json.dumps(
        {"host_layer_records": {"records": cpp_records}}, indent=2),
        encoding="utf-8")
    (out_dir / "correctness.json").write_text(json.dumps(
        {"classification": res.get("classification"),
         "gates": {"accept_correctness":
                   res.get("classification") == "ACCEPT_CORRECTNESS"}},
        indent=2))
    return {"records_cpp": len(cpp_records), "py_rows": len(py_rows),
            "classification": res.get("classification")}


PROMPT_TEXT = "Who is Alan Turing?"


def main() -> int:
    workdir = Path("/tmp/host-sync-kernel")
    workdir.mkdir(parents=True, exist_ok=True)
    session = {"session": "host-sync-profile", "arms": {}}
    gate = dry_run_gate()
    session["dry_run_gate"] = gate
    if gate["status"] != "PASS":
        (WORK / "session-summary.json").write_text(json.dumps(session, indent=2))
        log("DRY-RUN gate failed; refusing expensive decode")
        return 0
    arm_bytes = fetch_sealed_arm(workdir)
    variant, proof = build_variant(arm_bytes, workdir)
    patch_path = write_patch_file(workdir)
    session["variant_proof"] = proof
    session["profiler_patch_sha256"] = PROFILER_PATCH_SHA256
    prompt_hash = hashlib.sha256(PROMPT_TEXT.encode()).hexdigest()
    order = ["OFF1", "ON", "OFF2"]
    arm_envs = {"OFF1": ARMS[0]["env"], "ON": ARMS[1]["env"], "OFF2": ARMS[2]["env"]}
    for i, name in enumerate(order):
        if i > 0:
            clean_between_arms()
        out_dir = OUTBASE / f"profile-{name.lower()}"
        rec = run_arm(name, arm_envs[name], variant, patch_path, out_dir)
        rec["evidence"] = parse_result(out_dir)
        session["arms"][name] = rec
        log(f"arm {name}: classification={rec['evidence'].get('classification')} "
            f"decode_wall_s={rec['evidence'].get('decode_wall_s')}")
    conv = {}
    for name in order:
        key = {"OFF1": "off", "ON": "on", "OFF2": "off2"}[name]
        conv[key] = to_profile_run(OUTBASE / f"profile-{name.lower()}",
                                   WORK / f"profile-run-{key}", prompt_hash,
                                   profiling=(name == "ON"))
    session["conversion"] = conv
    try:
        session["abc"] = abc_mechanics(workdir)
        (WORK / "abc-mechanics.json").write_text(
            json.dumps(session["abc"], indent=2))
    except Exception as exc:  # noqa: BLE001
        session["abc"] = {"status": f"error: {exc}"}
    (WORK / "session-summary.json").write_text(json.dumps(session, indent=2))
    # Publish per-arm evidence into WORK (Kaggle snapshots /kaggle/working).
    for name in order:
        src = OUTBASE / f"profile-{name.lower()}"
        dst = WORK / f"profile-{name.lower()}"
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    for sub in ("profile-run-off", "profile-run-on", "profile-run-off2"):
        src = WORK / sub
        if not src.is_dir():
            log(f"note: {sub} absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
