"""GPU BATCH #1 driver — Phase-2 mechanism/causal campaign (GPU 1/2).

Single Kaggle kernel that runs the PHASE2_T4_CAMPAIGN.md section-2 design:
one submission executes the mechanism segment, the pinned-direct microbench,
the bank prep, and all four scored arms (A0 baseline / A1 VRAM-only /
A2 host-only / A3 combined) plus the A0 drift-bracket rep, then seals the
evidence package.

Arm-rep execution model: every rep is a FRESH subprocess of the proven
single-run harness (deepseek_v4_native_generate.py) with
NATIVE_REUSE_TREE=1 — a fresh process is the strongest possible regime-A
cold start (no in-process teardown can leave stale CUDA/tier state), and a
rep that crashes or is OOM-killed cannot take the other arms with it
(section 4.9: "a broken arm must not starve the others").  Shared phases
(P1 build, P4 bank repack + full validation) run exactly once in this
driver; the inner runs reuse them through the reuse fast-path.

Phase order is hard-gated per section 2.4:
  P0  env + hardware probe        (2xT4 SM75 gate)
  P1  clone pinned commit + build (dee_core + 8 CUDA ctests + pydee + bench)
  P2  mechanism segment           (HARD ABORT -> MECHANISM_FAIL on any fail)
  P3  microbench segment          (pinned_direct_bench; never aborts)
  --  hardware != 2xT4 exits here as INVALID_HARDWARE_LOTTERY
  P4  bank prepare + full validate
  P5  arm loop: A0 -> A1 -> A2 -> A3 (smoke + 3 measured reps each),
      then the A0 rep-4 drift bracket
  P6  evidence pack + seal (manifest.sha256 + registry record)

The kernel always exits 0 so Kaggle snapshots /kaggle/working (the v47
lesson: error-exit kernels drop their output tarball).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Pinned environment (mirrors the single-run harness; the kernel clones this
# exact commit and the inner runs verify HEAD against it every rep).
# ---------------------------------------------------------------------------
REPO = "https://github.com/so-nerdyy/dee.git"
BRANCH = os.environ.get("NATIVE_BRANCH", "integration/phase2-campaign")
COMMIT = os.environ.get("NATIVE_COMMIT", "")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
ROOT = Path("/tmp/dsv4-native-src")
DEE = ROOT / "dee.cpp"
BUILD = DEE / "build-kaggle"
KAGGLE_DIR = DEE / "kaggle" / "deepseek-v4-flash-0731"
INNER_RUNNER = KAGGLE_DIR / "deepseek_v4_native_generate.py"
RUN_CONFIG_SRC = KAGGLE_DIR / "run_config.json"
CKPT = Path("/tmp/dsv4-checkpoint")
DATASET_DIR = Path("/kaggle/input/deepseek-v4-flash-0731-shards")
WORK = Path("/kaggle/working")
PROGRESS = WORK / "progress.log"
ARMS_DIR = WORK / "arms"
BANK_DIR = Path(os.environ.get("NATIVE_DEE4_TRACE_PATH",
                               "/tmp/dsv4-dee4-v3-trace"))

# ---------------------------------------------------------------------------
# Sealed contract (v50/v60 evidence; PHASE2_T4_CAMPAIGN sections 4.5/4.6).
# The route journal hashes run_id into every record, so the sealed terminal
# chain only reproduces under the sealed v60 run id — every measured rep
# MUST run with SEALED_RUN_ID (rep identity lives in the artifact names).
# ---------------------------------------------------------------------------
SEALED_TOKEN_IDS = [
    666, 95140, 96807, 343, 4470, 20, 1127, 3298,
    22, 22604, 515, 411, 3947, 85349, 14, 6341,
]
SEALED_DECODED_TEXT = (
    "**Alan Turing (1912\u20131954)** was an English mathematician, computer")
# Same fail-closed guard as the single-run harness (v47 cp1252 incident):
# refuse to judge exactness if this file was ever re-transcoded.
assert SEALED_DECODED_TEXT.encode("utf-8") == (
    b"**Alan Turing (1912\xe2\x80\x931954)**"
    b" was an English mathematician, computer"), (
    "SEALED_DECODED_TEXT corrupted in transit; refusing to judge exactness")
SEALED_RUN_ID = "p2.3-dee4-trace-fillpool-ab-q8l4-v60"
SEALED_CHAIN_SHA256 = (
    "d8539b6e7c61d18820ccdcc17e492168a4132186f021ba30ca182581c5fcc75e")
SEALED_RECORD_COUNT = 688            # 43 layers x 16 forwards
SEALED_ITL_P95_MS = 6843.79          # v60 result.json
SEALED_DECODE_WALL_S = 72.607        # v60 result.json
SEALED_TOTAL_WALL_S = 168.53         # v60 result.json (gen wall)
# v60 measured counters per engine (the A0 reproduction target, section 4.6
# "sealed" column and the drift gate in section 4.9).
SEALED_COUNTERS = {
    "cuda0": {"resident_hits": 328, "cold_loads": 2285,
              "h2d_copies": 2285, "h2d_bytes": 30548951040,
              "evictions": 2004},
    "cuda1": {"resident_hits": 327, "cold_loads": 2159,
              "h2d_copies": 2159, "h2d_bytes": 28864413696,
              "evictions": 1878},
}
# Section 4.6 predeclared VRAM-repair contract at 281 slots (sim-validated).
VRAM_EXPECTED = {
    "cuda0": {"resident_hits": 680, "cold_loads": 1933},
    "cuda1": {"resident_hits": 1057, "cold_loads": 1429},
}
COUNTER_BAND = 12                    # section 4.6 reject band
# v60 host_pack pooled hit/miss surface (host-arm sanity, section 4.7).
SEALED_HOST_PACK = {"hits": 2618, "misses": 2481}

# V50 bank provenance (binds the dee4_trace bank to the sealed journal).
V50_TRACE_JOURNAL_RELATIVE = Path(
    "benchmark_reports/deepseek-v4-flash-0731-t4/"
    "v50-evidence-20260829T195940Z/routed_experts.jsonl")
V50_TRACE_JOURNAL_SHA256 = (
    "665aac3e8db570237c6dc6acaf08dc39f2af890e8a04e400ce7154f1a858dae1")
V50_TRACE_FINAL_CHAIN_SHA256 = (
    "086f8ca83b6a3c467cdf950096141fa9bc3e55a285d7d1fed8a0ad9913e3eb3d")

# ---------------------------------------------------------------------------
# Mechanism segment (section 3.1): all 8 CUDA-gated ctests.
# ---------------------------------------------------------------------------
CUDA_MECHANISM_TESTS = [
    "test_swiglu_cuda",
    "test_rmsnorm_cuda",
    "test_router_cuda",
    "test_expert_batch_cuda",
    "test_profiling_cuda",
    "test_deepseek_v4_fp4_cuda",
    "test_deepseek_v4_fp4_expert",
    "test_phase2_cuda",
]

# ---------------------------------------------------------------------------
# Arm table (section 2.2).  Only the phase2 switches differ between arms;
# every other knob is pinned identically for every rep.
# ---------------------------------------------------------------------------
PHASE2_HOST_ENV = {
    "NATIVE_PHASE2_HOST_BYTES": "9126805504",   # v60 host_pack envelope
    "NATIVE_PHASE2_HOST_SLOTS": "682",          # floor(envelope / record)
    "NATIVE_PHASE2_HOST_POLICY_SLOTS": "0",     # plain LRU only
    "NATIVE_PHASE2_HOST_SLOT_BYTES": "0",       # engine derives 13,369,344
    "NATIVE_PHASE2_HOST_ALIGNMENT": "4096",
    "NATIVE_PHASE2_TRY_PIN": "1",
    "NATIVE_PHASE2_MODEL_IDENTITY":
        f"deepseek-v4-flash-0731@{REV}",
}
ARMS = [
    {"id": "A0", "name": "baseline",
     "description": "legacy host_pack + prefetch + priority-LRU VRAM "
                    "(sealed v60 semantics)",
     "env": {"NATIVE_PHASE2": "off"}},
    {"id": "A1", "name": "vram_only",
     "description": "identical data path to A0; only the eviction score "
                    "changes (experimental_plain_lru)",
     "env": {"NATIVE_PHASE2": "vram"}},
    {"id": "A2", "name": "host_only",
     "description": "tier path: stage_expert -> DeviceExpertTier::stage -> "
                    "HostExpertTier acquire -> prefetch_host_lease -> "
                    "direct H2D",
     "env": {"NATIVE_PHASE2": "host", **PHASE2_HOST_ENV}},
    {"id": "A3", "name": "combined",
     "description": "tier path + plain-LRU VRAM",
     "env": {"NATIVE_PHASE2": "both", **PHASE2_HOST_ENV}},
]
REPS = ["smoke", "r1", "r2", "r3"]   # smoke = n_tokens 2 (unscored)
REP_N_TOKENS = {"smoke": 2, "r1": 16, "r2": 16, "r3": 16, "r4": 16}
REP_TIMEOUT_S = {"smoke": 1500, "measured": 2700}
WALL_WATCHDOG_S = int(os.environ.get("CAMPAIGN_WALL_WATCHDOG_S",
                                     str(5 * 3600)))       # section 4.9
ITL_WATCHDOG_MS = 3.0 * SEALED_ITL_P95_MS                  # 3x sealed p95
ITL_WATCHDOG_CONSECUTIVE = 3

# Artifacts the inner single-run harness publishes per rep, renamed into the
# arm dir with the section-4.10 <arm>.<rep> suffix scheme.
REP_ARTIFACTS = {
    "generated_checkpoint.jsonl": "generated_checkpoint.{a}.{r}.jsonl",
    "routed_experts.jsonl": "routed_experts.{a}.{r}.jsonl",
    "result.json": "result.{a}.{r}.json",
    "run_config.json": "run_config.{a}.{r}.json",
    "integrity.json": "integrity.{a}.{r}.json",
    "profile.json": "profile.{a}.{r}.json",
    "memory.json": "memory.{a}.{r}.json",
    "environment.json": "environment.{a}.{r}.json",
    "native-generate-result.json": "native-generate-result.{a}.{r}.json",
    "error.txt": "error.{a}.{r}.txt",
}

CAMPAIGN_ID = "PHASE2-T4"
LAUNCHED_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] [campaign] {msg}"
    print(line, flush=True)
    try:
        with open(PROGRESS, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_evidence(name: str, payload: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), "utf-8")
    temporary.replace(path)


def run(cmd, **kw):
    log("+ " + (" ".join(str(c) for c in cmd)
                if isinstance(cmd, list) else str(cmd)))
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        log(f"FAILED (exit {r.returncode})")
        raise RuntimeError("command failed: " + (
            " ".join(str(c) for c in cmd) if isinstance(cmd, list)
            else str(cmd)))
    return r


def terminal_verdict(verdict: str, detail: dict) -> int:
    """Seal whatever evidence exists and exit 0 (Kaggle output snapshot)."""
    completed = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result = {
        "campaign": CAMPAIGN_ID,
        "run_number": "GPU 1/2",
        "status": "COMPLETE",
        "verdict": verdict,
        "launched_utc": LAUNCHED_UTC,
        "completed_at_utc": completed,
        "repository": REPO,
        "branch": BRANCH,
        "repository_commit": COMMIT,
        "detail": detail,
    }
    write_evidence("native-generate-result.json", result)
    write_manifest(extra={"verdict": verdict})
    log(f"=== CAMPAIGN VERDICT: {verdict} ===")
    return 0


def write_manifest(extra: dict | None = None) -> None:
    """manifest.sha256 over every artifact present under WORK (section 4.10)."""
    entries = {}
    for path in sorted(WORK.rglob("*")):
        if path.is_file() and path.name not in {
                "manifest.sha256", "manifest.sha256.tmp"}:
            try:
                entries[str(path.relative_to(WORK))] = sha256_file(path)
            except OSError:
                pass
    lines = [f"{digest}  {name}" for name, digest in entries.items()]
    payload = "\n".join(lines) + "\n"
    manifest = {"file_count": len(entries), "files": entries,
                "sealed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime())}
    if extra:
        manifest.update(extra)
    (WORK / "manifest.json").write_text(
        json.dumps(manifest, indent=2), "utf-8")
    (WORK / "manifest.sha256").write_text(payload, "utf-8")


# ---------------------------------------------------------------------------
# P0 — environment + hardware probe
# ---------------------------------------------------------------------------
def p0_probe() -> dict:
    env: dict = {
        "campaign": CAMPAIGN_ID,
        "run_number": "GPU 1/2",
        "recorded_at_utc": LAUNCHED_UTC,
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "storage_mount": str(DATASET_DIR),
        "dataset_mount_present": DATASET_DIR.is_dir(),
    }
    try:
        env["nvidia_smi_L"] = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True,
            stderr=subprocess.STDOUT).strip().splitlines()
    except Exception as exc:
        env["nvidia_smi_L"] = []
        env["nvidia_smi_error"] = repr(exc)
    try:
        env["nvidia_smi_query"] = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
             "--format=csv,noheader"], text=True,
            stderr=subprocess.STDOUT).strip().splitlines()
    except Exception as exc:
        env["nvidia_smi_query"] = []
        env["nvidia_smi_query_error"] = repr(exc)
    try:
        info = {}
        for mnt in ("/", "/tmp", str(WORK)):
            try:
                total, _used, free = shutil.disk_usage(mnt)
                info[mnt] = {"total_gb": round(total / 2**30, 1),
                             "free_gb": round(free / 2**30, 1)}
            except OSError:
                pass
        meminfo = {}
        for ln in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = ln.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                meminfo[key] = round(int(value.split()[0]) / 2**20, 1)
        info["ram_gb"] = meminfo
        env["resources"] = info
    except Exception as exc:  # noqa: BLE001
        env["resources_error"] = repr(exc)
    names = env.get("nvidia_smi_L") or []
    env["gpu_count"] = len(names)
    env["hardware_ok_2xt4"] = (
        len(names) == 2 and all("Tesla T4" in n for n in names))
    env["tmp_free_ok_45gib"] = (
        env.get("resources", {}).get("/tmp", {}).get("free_gb", 0) >= 45)
    write_evidence("environment.json", env)
    log(f"P0 hardware: {names} -> 2xT4={env['hardware_ok_2xt4']} "
        f"resources={env.get('resources')}")
    return env


# ---------------------------------------------------------------------------
# P1 — clone pinned commit + build (dee_core + mechanism targets + pydee)
# ---------------------------------------------------------------------------
def p1_build() -> str:
    log("=== P1 clone + checkout ===")
    if ROOT.exists():
        run(["rm", "-rf", str(ROOT)])
    run(["git", "clone", "--branch", BRANCH, "--single-branch",
         REPO, str(ROOT)])
    if COMMIT:
        run(["git", "-C", str(ROOT), "checkout", "--quiet", COMMIT])
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    log(f"pinned commit {head}")

    build_jobs = max(1, min(2, os.cpu_count() or 2))
    log("=== P1 configure (sm_60;sm_75, DEE_CUDA=ON) ===")
    run(["cmake", "-S", str(DEE), "-B", str(BUILD),
         "-DCMAKE_CUDA_ARCHITECTURES=60;75", "-DDEE_CUDA=ON",
         "-DDEE_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=Release"])
    log("=== P1 build dee_core + mechanism tests + bench (-j2) ===")
    run(["cmake", "--build", str(BUILD), "--target", "dee_core",
         "-j", str(build_jobs)])
    run(["cmake", "--build", str(BUILD), "--target", "dee_test_assets",
         "-j", str(build_jobs)])
    for target in [*CUDA_MECHANISM_TESTS, "pinned_direct_bench"]:
        run(["cmake", "--build", str(BUILD), "--target", target,
             "-j", str(build_jobs)])
    log("=== P1 build pydee ===")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--user",
         "pybind11"])
    run([sys.executable, "pydee/setup.py", "build_ext", "--inplace"],
        env={**os.environ, "DEE_BUILD_DIR": str(BUILD)}, cwd=str(DEE))
    return head


# ---------------------------------------------------------------------------
# P2 — mechanism segment (hard abort on any failure)
# ---------------------------------------------------------------------------
def p2_mechanism() -> dict:
    """Run every CUDA-gated ctest under ctest; SKIP(77) counts as FAIL here
    because a GPU is present (section 3.1 hard-abort rule)."""
    segment: dict = {"tests": {}, "all_pass": True,
                     "audit_probes": []}
    for name in CUDA_MECHANISM_TESTS:
        t0 = time.monotonic()
        proc = subprocess.run(
            ["ctest", "--test-dir", str(BUILD), "-R", f"^{name}$",
             "--output-on-failure"],
            capture_output=True, text=True, timeout=600)
        elapsed = round(time.monotonic() - t0, 1)
        out = (proc.stdout or "") + (proc.stderr or "")
        status = "FAIL"
        if proc.returncode == 0:
            # ctest exits 0 for both Passed and Skipped (SKIP_RETURN_CODE).
            status = "SKIP" if "Skipped" in out else "PASS"
        if status != "PASS":
            segment["all_pass"] = False
        segment["tests"][name] = {
            "status": status, "exit_code": proc.returncode,
            "seconds": elapsed, "output_tail": out[-2000:],
        }
        log(f"P2 {name}: {status} ({elapsed}s)")
    # Section 3.2 audit-probe bookkeeping.  Probe (b) is exercised by
    # test_phase2_cuda's run(false) half (backend.pin -> false gives
    # pin_failures>0 and the pageable fallback path).  Probe (a) needs the
    # fault-injection hook specced in docs/CAMPAIGN_FAULT_INJECTION.md — not
    # landed at this commit, so it degrades to an observational note per the
    # campaign doc ("verdict note, not a gate").
    phase2 = segment["tests"].get("test_phase2_cuda", {})
    segment["audit_probes"] = [
        {"probe": "a_submit_failure_event_cleanup",
         "status": "NOT_RUN",
         "reason": "fault-injection hook not landed at this commit; spec at "
                   "docs/CAMPAIGN_FAULT_INJECTION.md (post-merge work). "
                   "Degrades to observational note per campaign section 3.2."},
        {"probe": "b_forced_pageable_fallback",
         "status": "COVERED" if phase2.get("status") == "PASS"
                   else "NOT_RUN",
         "evidence": "test_phase2_cuda run(false) half: backend.pin->false "
                     "yields pin_failures>0 + pageable fallback copy path"},
        {"probe": "c_inflight_synchronize_reset",
         "status": "PARTIAL",
         "evidence": "test_phase2_cuda pinned half holds a DMA behind a "
                     "cudaLaunchHostFunc gate; proves wait_on_stream cannot "
                     "release the host source early.  Full synchronize_all/"
                     "reset-under-inflight variant awaits the same hook spec."},
    ]
    write_evidence("segment_mechanism.json", segment)
    return segment


# ---------------------------------------------------------------------------
# P3 — pinned-direct microbench (informational; never aborts the campaign)
# ---------------------------------------------------------------------------
def p3_bench() -> dict:
    out_path = WORK / "pinned_direct_bench.json"
    bench: dict = {"status": "NOT_RUN"}
    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            [str(BUILD / "pinned_direct_bench"),
             "--out", str(out_path)],
            capture_output=True, text=True, timeout=1800)
        bench = {"status": "RAN", "exit_code": proc.returncode,
                 "seconds": round(time.monotonic() - t0, 1),
                 "stderr_tail": (proc.stderr or "")[-2000:]}
        if out_path.is_file():
            try:
                bench["result"] = json.loads(out_path.read_text("utf-8"))
            except json.JSONDecodeError:
                bench["result_parse"] = "failed"
    except Exception as exc:  # noqa: BLE001 - informational segment
        bench = {"status": "ERROR", "error": repr(exc)}
    log(f"P3 pinned_direct_bench: {bench.get('status')} "
        f"exit={bench.get('exit_code')}")
    return bench


# ---------------------------------------------------------------------------
# P4 — bank prepare + full validation (once; reps reuse it cheaply)
# ---------------------------------------------------------------------------
def p4_bank() -> dict:
    sys.path.insert(0, str(KAGGLE_DIR))
    import deepseek_v4_native_generate as native  # cloned harness helpers
    from repack_to_dee4 import (
        repack_trace,
        validate_dee4_trace_store,
    )

    log("=== P4 shard presence ===")
    shard_paths = native.download_all_shards()
    source_dir = Path(shard_paths[0]).parent
    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        import urllib.request
        url = ("https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
               f"resolve/{REV}/model.safetensors.index.json")
        index_path = WORK / "model.safetensors.index.json"
        log(f"P4: downloading index from HF: {url}")
        index_path.write_bytes(
            urllib.request.urlopen(url, timeout=300).read())

    metadata_path = (BANK_DIR if BANK_DIR.name == "metadata.json"
                     else BANK_DIR / "metadata.json")
    trace_journal = DEE / V50_TRACE_JOURNAL_RELATIVE
    created = False
    if metadata_path.is_file():
        log(f"P4: bank exists; validating {metadata_path} in place "
            "(full 29.4 GiB record hash walk)")
    else:
        if BANK_DIR.exists() and any(BANK_DIR.iterdir()):
            raise RuntimeError(
                "P4: bank dir non-empty without metadata.json; refusing to "
                f"overwrite {BANK_DIR}")
        log(f"P4: repacking sealed v50 journal into {BANK_DIR}")
        repack_trace(source_dir, BANK_DIR, trace_journal,
                     index_path=index_path,
                     expected_journal_sha256=V50_TRACE_JOURNAL_SHA256,
                     expected_final_chain_sha256=
                     V50_TRACE_FINAL_CHAIN_SHA256)
        created = True
    t0 = time.monotonic()
    validation = validate_dee4_trace_store(
        metadata_path, trace_journal,
        expected_journal_sha256=V50_TRACE_JOURNAL_SHA256,
        expected_final_chain_sha256=V50_TRACE_FINAL_CHAIN_SHA256)
    validation["validated_seconds"] = round(time.monotonic() - t0, 1)
    validation["created_this_run"] = created
    validation["configured_metadata_path"] = str(metadata_path)
    write_evidence("dee4-trace-validation.json", validation)
    log(f"P4 bank ready: data_sha256={validation['data_sha256'][:16]} "
        f"records={validation['total_experts']} created={created}")
    return validation


# ---------------------------------------------------------------------------
# P5 — arm loop
# ---------------------------------------------------------------------------
def rep_env(arm: dict, rep: str) -> dict:
    env = dict(os.environ)
    env.update({
        "NATIVE_REUSE_TREE": "1",
        "NATIVE_CACHE_DTYPE": "fp4",
        "NATIVE_EXPERT_STORE": "dee4_trace",
        "NATIVE_DEE4_TRACE_PATH": str(BANK_DIR),
        "NATIVE_DEE4_VALIDATE_SAMPLES": "12",
        "NATIVE_SOURCE_READ_LANES": "4",
        "NATIVE_SOURCE_READ_QUEUE_DEPTH": "8",
        "NATIVE_PROFILE": "1",
        "NATIVE_BUDGET_BYTES": "3758096384",   # 281 slots x 12.75 MiB
        "NATIVE_N_TOKENS": str(REP_N_TOKENS[rep]),
        # The sealed terminal route chain only reproduces under the v60 run
        # id (run_id is hashed into every journal record).
        "NATIVE_RUN_ID": SEALED_RUN_ID,
    })
    env.update(arm["env"])
    return env


def move_rep_artifacts(arm_dir: Path, arm: str, rep: str) -> dict:
    moved = {}
    for src_name, dst_pattern in REP_ARTIFACTS.items():
        src = WORK / src_name
        if not src.is_file():
            continue
        dst = arm_dir / dst_pattern.format(a=arm, r=rep)
        shutil.move(str(src), str(dst))
        moved[dst.name] = sha256_file(dst)
    return moved


def rep_exactness(arm: dict, rep: str, result: dict,
                  journal_path: Path) -> dict:
    """Section-4.5 gates.  Smoke reps only prove boot + the first two
    tokens of the sealed prefix; measured reps must satisfy every gate."""
    toks = [int(t) for t in result.get("generated_token_ids", [])]
    record = {"arm": arm["id"], "rep": rep,
              "n_tokens": len(toks), "tokens": toks,
              "status": result.get("status"),
              "classification": result.get("classification")}
    if rep == "smoke":
        record["exact"] = (result.get("status") == "COMPLETE"
                           and toks[:2] == SEALED_TOKEN_IDS[:2])
        record["scored"] = False
        return record
    gates = result.get("correctness", {}).get(
        "sealed_contract_gates", {})
    chain = None
    if journal_path.is_file():
        try:
            tail = journal_path.read_text("utf-8").strip().splitlines()
            if tail:
                chain = json.loads(tail[-1]).get("chain_sha256")
        except Exception:  # noqa: BLE001
            pass
    record["journal_tail_chain_sha256"] = chain
    record["sealed_chain_match"] = (chain == SEALED_CHAIN_SHA256)
    record["gates"] = gates
    record["exact"] = bool(
        result.get("status") == "COMPLETE"
        and toks == SEALED_TOKEN_IDS
        and result.get("decoded_text") == SEALED_DECODED_TEXT
        and record["sealed_chain_match"]
        and all(v for k, v in gates.items()
                if k != "required_performance_hardware"))
    record["scored"] = True
    return record


def rep_counters(result: dict) -> dict:
    out = {}
    for key, stats in result.get("engine_stats", {}).items():
        out[key] = {k: stats.get(k) for k in (
            "resident_hits", "inflight_hits", "cold_loads", "evictions",
            "duplicate_requests", "h2d_bytes", "h2d_copies",
            "resident_experts")}
    out["host_pack"] = {
        k: {kk: v.get(kk) for kk in ("hits", "misses", "evictions", "bytes")}
        for k, v in result.get("host_pack", {}).items()}
    out["phase2"] = result.get("phase2", {}).get("metrics", {})
    out["expert_store"] = {
        k: {kk: v.get(kk) for kk in (
            "source_reads", "bytes_requested", "read_bandwidth_mib_s",
            "materialization_mode", "lookup_failures")}
        for k, v in result.get("expert_store", {}).items()}
    out["wall_s"] = result.get("total_wall_seconds")
    out["decode_wall_s"] = result.get("decode_wall_s")
    out["itl"] = result.get("inter_token_latency_ms")
    return out


def baseline_drift_ok(counters: dict) -> tuple[bool, dict]:
    """A0 measured reps must reproduce the sealed v60 counters within the
    section-4.6 band; drift means the platform/trace is not the sealed one."""
    detail = {}
    ok = True
    for eng, sealed in SEALED_COUNTERS.items():
        got = counters.get(eng, {})
        row = {}
        for field, want in sealed.items():
            actual = got.get(field)
            within = (isinstance(actual, (int, float))
                      and abs(int(actual) - want) <= COUNTER_BAND)
            row[field] = {"sealed": want, "actual": actual,
                          "within_band": within}
            if not within:
                ok = False
        detail[eng] = row
    return ok, detail


def vram_band_verdict(counters: dict) -> dict:
    """Section 4.6: counters ~= sealed => switch inert (REJECT_MECHANISM);
    inside band => contract reproduced; between => PROVISIONAL."""
    per_engine = {}
    for eng, want in VRAM_EXPECTED.items():
        got = counters.get(eng, {})
        sealed = SEALED_COUNTERS[eng]
        row = {}
        for field in ("resident_hits", "cold_loads"):
            actual = got.get(field)
            if not isinstance(actual, (int, float)):
                row[field] = "MISSING"
            elif abs(int(actual) - sealed[field]) <= COUNTER_BAND:
                row[field] = "AT_SEALED"      # switch did not take
            elif abs(int(actual) - want[field]) <= COUNTER_BAND:
                row[field] = "IN_BAND"
            else:
                row[field] = "DRIFTED"
        per_engine[eng] = row
    flat = [v for row in per_engine.values() for v in row.values()]
    if all(v == "IN_BAND" for v in flat):
        verdict = "CONTRACT_REPRODUCED"
    elif any(v == "AT_SEALED" for v in flat):
        verdict = "REJECT_MECHANISM"   # counters at sealed values
    else:
        verdict = "PROVISIONAL"
    return {"per_engine": per_engine, "verdict": verdict}


def host_sanity(result: dict) -> dict:
    """Section 4.8 ACCEPT_HOST_TIER mechanism sanity (A2/A3)."""
    metrics = result.get("phase2", {}).get("metrics", {})
    host_budget = int(PHASE2_HOST_ENV["NATIVE_PHASE2_HOST_BYTES"])
    sane = {}
    all_ok = True
    for eng, m in metrics.items():
        host = m.get("host", {})
        checks = {
            "pin_failures_zero": int(host.get("pin_failures", -1)) == 0,
            "failures_zero": int(host.get("failures", -1)) == 0,
            "device_failures_zero":
                int(m.get("device_failures", -1)) == 0,
            "resident_within_budget":
                0 <= int(host.get("resident_bytes", -1)) <= host_budget,
            "peak_within_budget":
                0 <= int(host.get("peak_resident_bytes", -1)) <= host_budget,
            "tokens_denominator": int(m.get("tokens", -1)) > 0,
        }
        sane[eng] = checks
        all_ok = all_ok and all(checks.values())
    return {"per_engine": sane, "all_ok": all_ok}


def run_rep(arm: dict, rep: str, arm_dir: Path,
            campaign_start: float) -> dict:
    n_tokens = REP_N_TOKENS[rep]
    timeout = (REP_TIMEOUT_S["smoke"] if rep == "smoke"
               else REP_TIMEOUT_S["measured"])
    log(f"P5 {arm['id']}.{rep}: {n_tokens} tokens "
        f"(phase2={arm['env']['NATIVE_PHASE2']}, timeout={timeout}s)")
    t0 = time.monotonic()
    record = {"arm": arm["id"], "rep": rep, "n_tokens_requested": n_tokens,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime())}
    try:
        proc = subprocess.run(
            [sys.executable, str(INNER_RUNNER)],
            env=rep_env(arm, rep), cwd=str(WORK), timeout=timeout)
        record["exit_code"] = proc.returncode
    except subprocess.TimeoutExpired:
        record["exit_code"] = None
        record["terminal"] = "timeout"
        log(f"P5 {arm['id']}.{rep}: TIMEOUT after {timeout}s")
    record["wall_s"] = round(time.monotonic() - t0, 1)
    record["artifacts"] = move_rep_artifacts(arm_dir, arm["id"], rep)

    result_path = arm_dir / f"result.{arm['id']}.{rep}.json"
    journal_path = arm_dir / f"routed_experts.{arm['id']}.{rep}.jsonl"
    result = {}
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text("utf-8"))
        except json.JSONDecodeError:
            record["terminal"] = "result_unparseable"
    exactness = rep_exactness(arm, rep, result, journal_path)
    record.update({k: exactness[k] for k in (
        "exact", "scored", "classification", "status")})
    if "sealed_chain_match" in exactness:
        record["sealed_chain_match"] = exactness["sealed_chain_match"]
    if "gates" in exactness:
        record["gates"] = exactness["gates"]
    if result:
        record["counters"] = rep_counters(result)
        # ITL watchdog evidence: 3 consecutive tokens beyond 3x sealed p95.
        timings = result.get("decode_timings_ms", [])
        record["itl_watchdog_tripped"] = any(
            all(t > ITL_WATCHDOG_MS for t in timings[i:i + 3])
            for i in range(max(0, len(timings) - 2)))
    if record.get("exact") is not True and rep != "smoke":
        record.setdefault("terminal", "exactness_break")
    elif record.get("exact") is not True and rep == "smoke":
        record.setdefault("terminal", "smoke_failed")
    return record


def p5_arms(campaign_start: float) -> dict:
    arms_out: dict = {"arms": {}, "rep_order": [],
                      "drift_bracket": None, "aborted": None,
                      "frozen": False}
    invalidated = False

    def write_arm_summary(arm: dict, reps: list[dict]) -> dict:
        measured = [r for r in reps if r.get("scored")]
        summary = {
            "arm_id": arm["id"], "name": arm["name"],
            "description": arm["description"],
            "phase2_env": arm["env"],
            "reps": reps,
            "measured_reps": len(measured),
            "exact_reps": sum(1 for r in measured if r.get("exact")),
            "verdict": ("ALL_EXACT" if measured and all(
                r.get("exact") for r in measured) else "INCOMPLETE_OR_BROKEN"),
            "counters_last_measured": (
                measured[-1].get("counters") if measured else None),
        }
        if arm["id"] in ("A1", "A3") and measured:
            summary["vram_band"] = vram_band_verdict(
                measured[-1].get("counters", {}))
        if arm["id"] in ("A2", "A3") and measured:
            last = {}
            result_path = (ARMS_DIR / f"{arm['id']}_{arm['name']}"
                           / f"result.{arm['id']}."
                             f"{measured[-1]['rep']}.json")
            if result_path.is_file():
                try:
                    last = json.loads(result_path.read_text("utf-8"))
                except json.JSONDecodeError:
                    pass
            summary["host_sanity"] = host_sanity(last)
        write_evidence(f"result.{arm['id']}.summary.json", summary)
        return summary

    for arm in ARMS:
        if invalidated or arms_out["frozen"]:
            break
        arm_dir = ARMS_DIR / f"{arm['id']}_{arm['name']}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "run_config.json").write_text(json.dumps({
            "arm_id": arm["id"], "name": arm["name"],
            "description": arm["description"],
            "campaign": CAMPAIGN_ID, "run_number": "GPU 1/2",
            "resolved_env": rep_env(arm, "r1"),
            "sealed_contract": {
                "run_id": SEALED_RUN_ID,
                "token_ids": SEALED_TOKEN_IDS,
                "route_chain_sha256": SEALED_CHAIN_SHA256,
                "itl_p95_ms": SEALED_ITL_P95_MS,
            },
        }, indent=2), "utf-8")
        reps: list[dict] = []
        for rep in REPS:
            elapsed = time.monotonic() - campaign_start
            if elapsed > WALL_WATCHDOG_S:
                arms_out["frozen"] = "TIMEOUT_PARTIAL"
                log(f"P5 watchdog: cumulative wall {elapsed:.0f}s > "
                    f"{WALL_WATCHDOG_S}s — freezing remaining reps")
                break
            record = run_rep(arm, rep, arm_dir, campaign_start)
            reps.append(record)
            arms_out["rep_order"].append(f"{arm['id']}.{rep}")
            # Section 4.9: an exactness break skips the rest of THIS arm.
            if record.get("terminal") in (
                    "exactness_break", "smoke_failed", "timeout"):
                log(f"P5 {arm['id']}.{rep} terminal={record['terminal']} — "
                    f"skipping remaining reps of {arm['id']}")
                break
            # A0 rep-1 drift gate: baseline must reproduce sealed v60
            # counters or the experiment is invalid (section 4.9).
            if arm["id"] == "A0" and rep == "r1":
                ok, detail = baseline_drift_ok(record.get("counters", {}))
                record["baseline_drift_gate"] = detail
                if not ok:
                    invalidated = True
                    arms_out["aborted"] = "INVALID_EXPERIMENT"
                    log("P5 A0.r1 baseline counter drift outside sealed "
                        "band — INVALID_EXPERIMENT")
                    write_arm_summary(arm, reps)
                    break
            if record.get("itl_watchdog_tripped"):
                arms_out["frozen"] = "TIMEOUT_PARTIAL"
                log(f"P5 {arm['id']}.{rep}: ITL watchdog tripped "
                    f"(3 consecutive tokens > {ITL_WATCHDOG_MS:.0f} ms)")
                break
        write_arm_summary(arm, reps)
        arms_out["arms"][arm["id"]] = {
            "name": arm["name"], "reps_completed": len(reps),
            "exact_reps": sum(1 for r in reps
                              if r.get("scored") and r.get("exact")),
        }

    # A0 rep-4 drift bracket (section 2.3): detects slow environmental
    # drift inside the session instead of silently attributing it to an arm.
    if not invalidated and not arms_out["frozen"]:
        elapsed = time.monotonic() - campaign_start
        if elapsed <= WALL_WATCHDOG_S:
            arm = ARMS[0]
            arm_dir = ARMS_DIR / f"{arm['id']}_{arm['name']}"
            record = run_rep(arm, "r4", arm_dir, campaign_start)
            ok, detail = baseline_drift_ok(record.get("counters", {}))
            record["baseline_drift_gate"] = detail
            arms_out["drift_bracket"] = {
                "rep": "A0.r4", "exact": record.get("exact"),
                "counters_within_sealed_band": ok,
                "record": record,
            }
            log(f"P5 A0.r4 drift bracket: exact={record.get('exact')} "
                f"within_band={ok}")
    return arms_out


# ---------------------------------------------------------------------------
# Verdict + seal
# ---------------------------------------------------------------------------
def final_verdict(arms_out: dict) -> str:
    if arms_out.get("aborted"):
        return arms_out["aborted"]
    if arms_out.get("frozen"):
        return "TIMEOUT_PARTIAL"
    arms = arms_out.get("arms", {})
    a0 = arms.get("A0", {})
    if a0.get("exact_reps", 0) < 3:
        return "INVALID_EXPERIMENT"
    return "COMPLETE"


def main() -> int:
    campaign_start = time.monotonic()
    WORK.mkdir(parents=True, exist_ok=True)
    log(f"=== PHASE2-T4 GPU Batch #1 (GPU 1/2) launched {LAUNCHED_UTC} ===")

    environment = p0_probe()                                    # P0
    head = p1_build()                                           # P1
    run_config_payload = {
        "campaign": CAMPAIGN_ID, "run_number": "GPU 1/2",
        "recorded_at_utc": LAUNCHED_UTC,
        "repository": REPO, "branch": BRANCH,
        "repository_commit": COMMIT, "resolved_head": head,
        "arms": [{"id": a["id"], "name": a["name"],
                  "env": a["env"]} for a in ARMS],
        "reps_per_arm": REPS, "rep_n_tokens": REP_N_TOKENS,
        "drift_bracket": "A0.r4",
        "watchdogs": {"wall_s": WALL_WATCHDOG_S,
                      "itl_ms": ITL_WATCHDOG_MS,
                      "itl_consecutive": ITL_WATCHDOG_CONSECUTIVE},
        "shared_config": {
            "cache_dtype": "fp4", "expert_store": "dee4_trace",
            "dee4_trace_path": str(BANK_DIR),
            "source_read_lanes": 4, "source_read_queue_depth": 8,
            "budget_bytes_per_gpu": 3758096384,
            "n_tokens_measured": 16, "n_tokens_smoke": 2,
            "profile_stages": True, "run_id": SEALED_RUN_ID,
        },
        "source_run_config_sha256": sha256_file(RUN_CONFIG_SRC),
        "source_run_config_path": str(RUN_CONFIG_SRC),
    }
    write_evidence("run_config.json", run_config_payload)

    segment = p2_mechanism()                                    # P2
    if not segment["all_pass"]:
        return terminal_verdict("MECHANISM_FAIL", {
            "segment": "P2", "environment": environment,
            "segment_mechanism": segment})
    bench = p3_bench()                                          # P3
    if not environment["hardware_ok_2xt4"]:
        return terminal_verdict("INVALID_HARDWARE_LOTTERY", {
            "environment": environment,
            "note": "mechanism + bench segments banked; arms never run "
                    "on non-2xT4 hardware (section 4.9)",
            "segment_mechanism": segment, "pinned_direct_bench": bench})
    validation = p4_bank()                                      # P4
    arms_out = p5_arms(campaign_start)                          # P5
    verdict = final_verdict(arms_out)                           # P6
    # Re-publish the campaign-level records: each rep's inner environment.json
    # / run_config.json was moved into its arm dir, so the top-level files
    # must be rewritten before the manifest hashes the package.
    write_evidence("environment.json", environment)
    write_evidence("run_config.json", run_config_payload)

    registry = {
        "run_id": f"{CAMPAIGN_ID}-b1",
        "campaign": CAMPAIGN_ID,
        "platform": "kaggle",
        "gpu": "T4:2" if environment["hardware_ok_2xt4"] else "other",
        "kernel": "nivind/dee-cpp-dsv4-native-generate",
        "kernel_version": os.environ.get("KERNEL_VERSION", ""),
        "repository_commit": COMMIT, "resolved_head": head,
        "branch_head_at_launch": BRANCH,
        "hardware": environment.get("nvidia_smi_L"),
        "launched_utc": LAUNCHED_UTC,
        "policy": {"arms": run_config_payload["arms"],
                   "reps": REPS, "shared": run_config_payload["shared_config"]},
        "verdict": verdict,
        "gates_summary": {
            "mechanism_all_pass": segment["all_pass"],
            "arms": arms_out.get("arms", {}),
            "drift_bracket": (arms_out.get("drift_bracket") or {}).get(
                "counters_within_sealed_band"),
        },
        "performance_comparable": environment["hardware_ok_2xt4"],
        "evidence_dir": str(WORK),
    }
    write_evidence("run_registry_record.json", registry)
    return terminal_verdict(verdict, {
        "environment": environment,
        "segment_mechanism_all_pass": segment["all_pass"],
        "pinned_direct_bench_status": bench.get("status"),
        "dee4_trace_validation": {
            "data_sha256": validation.get("data_sha256"),
            "total_experts": validation.get("total_experts")},
        "arms": arms_out,
    })


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        log("FATAL " + tb)
        try:
            (WORK / "error.txt").write_text(tb)
        except OSError:
            pass
        try:
            terminal_verdict("INVALID_EXPERIMENT", {
                "fatal": str(exc), "traceback_tail": tb[-2000:]})
        except Exception:  # noqa: BLE001
            pass
        sys.exit(0)
