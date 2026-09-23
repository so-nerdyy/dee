"""Phase-5b mechanism test driver (dual-T4, Kaggle).

Question answered: WHERE does the warm-process numerical divergence come
from?  The P5 campaign proved: inference unit >=2 in a warm process
diverges from unit 0 on byte-identical inputs, deterministically within
each forward (pad rows mutually agree), injected during layers 0-2
compute.  This session bisects the source with three arms sharing one
process-per-arm discipline:

  v3 result (kernel v3): decisive localization — at (step0, layer0)
  router_scores/routing_weights/expert_ids/shared_out all match unit 0
  while moe_out diverges => the engine's routed-expert `raw` output
  differs on bit-identical inputs.  mD crashed on reference-cache
  capacity (shared pins > 1 GiB budget); mF was a null arm
  (NATIVE_BATCHED never consulted on the device path).
  v2 result: divergence survives BOTH use_deterministic_algorithms AND
  CUBLAS_WORKSPACE_CONFIG — documented cuBLAS-workspace fix is dead.
  v1 result: sequential path itself diverges (not cohort-specific).

  mG  full-model probe: NATIVE_PROBE_L0=1 dumps layer-0 prefill engine
      I/O tensors (hidden_fp16/raw/ids) + a same-unit replay call into
      a second buffer — gives divergence MAGNITUDE (ULP vs corruption),
      position pattern, and miss-vs-hit asymmetry inside a unit.
  mI  engine-only micro suite (NATIVE_MICRO_PROBE=1): no model; drives
      moe_forward_batch_device directly across coldreset/resident/churn/
      postchurn configs — convicts/exonerates dee_core standalone.
  mJ  same micro suite under CUDA_LAUNCH_BLOCKING=1 — serializes every
      kernel launch; clean => stream-timing race; dirty => state.

Instruments on model arms: NATIVE_ROUTE_WEIGHT_JOURNAL (per-step,layer
sha of routing weights + expert ids) and NATIVE_CAPTURE_JOURNAL (sha of
moe_out/shared_out/router_scores captures) — the capture shas split
layer-0 internals: moe_out diverging => engine FFN; shared_out alone =>
shared expert; weights-only => attention/residual side.

Accept/reject is per-arm in p5b_report.json:
  engine_standalone_convicted = mI any config distinct>1
  hit_path_implicated         = mI resident diverges but coldreset stable
  timing_race                 = mI dirty but mJ clean under launch-blocking
  model_level_trigger         = mI clean but mG still diverges

Phases identical to phase5_session_driver.py: P0 gate, P1 clone+build,
P2 store assembly + seal, P3 one runner subprocess per arm, P4 report
(always exits 0).

--selfcheck validates ARMS/env/groups without launching anything.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "https://github.com/so-nerdyy/dee.git"
BRANCH = "research/phase4-cache-hierarchy"
COMMIT = os.environ.get("P5_PINNED_COMMIT", "")
ROOT = Path(os.environ.get("P5_SOURCE_ROOT", "/tmp/dee-p5b-src"))
DEE = ROOT / "dee.cpp"
BUILD = DEE / "build-kaggle"
WORK = Path("/kaggle/working")
OUT = WORK / "p5b-out"
RUNNER = (DEE / "kaggle" / "deepseek-v4-flash-0731"
          / "deepseek_v4_native_generate.py")

INDEX_DS = "nivind/dee4-p3-full-u2081ada5e37e-index"
SEG_PREFIX = "dee4-p3-full-u2081ada5e37e-b"
N_BUCKETS = 46
STORE_ROOT = Path("/tmp/dee4-full")
N_TOKENS = int(os.environ.get("P5B_N_TOKENS", "8"))


def _env_num(cast, name, default):
    try:
        return cast(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return cast(default)


ARM_TIMEOUT_S = _env_num(int, "P5B_ARM_TIMEOUT_S", 2400)
HEARTBEAT_S = 60
# run_id is inside the route-journal canonical hash payload; constant
# across arms so journal structures compare cleanly.
RUN_ID = "p5b-mechanism"
CP_WORKERS = 4

PACK_8_5_GIB = int(8.5 * (1 << 30))      # Phase-4 committed pack/GPU
VRAM_3_5_GIB = 3584 << 20                # Phase-4 committed VRAM arena

# The single prompt under test: q0 (the mRNA prompt -- the campaign's
# known-divergent pair c1-c0 vs c1-c7 both used it).
_B = ("<" + chr(0xFF5C) + "begin" + chr(0x2581) + "of" + chr(0x2581)
      + "sentence" + chr(0xFF5C) + ">")
_A = "<" + chr(0xFF5C) + "Assistant" + chr(0xFF5C) + ">"
Q0 = _B + ("Explain how mRNA vaccines work, from injection to immune "
           "memory.") + _A

_BASE = {
    "expert_store": "dee4_segmented",
    "cache_dtype": "fp4",
    "eviction_policy": "lru",
    "host_cache_mode": "lru",
    "budget_bytes": VRAM_3_5_GIB,
    "lanes": 4,
    "queue_depth": 6,
    "n_tokens": N_TOKENS,
    "cache_reset": "cold",
    "trace_requests": "1",
    "host_pack_gpu0_bytes": PACK_8_5_GIB,
    "host_pack_gpu1_bytes": PACK_8_5_GIB,
}

ARMS = [
    {
        # mK: byte-fingerprint bisect — the engine-only micro suite with
        # NATIVE_MICRO_FP=1.  After every forward the runner calls
        # debug_expert_fingerprint(layer, expert) for all 64 experts:
        # FNV-1a/64 of (a) the store's mmap record (ground truth), (b) the
        # host pack entry, (c) the resident VRAM block after D2H readback,
        # plus block pointer/generation/pins/all-zero and the FP4 decode
        # scratch hash.  store!=pack => host fill lane; pack!=dev =>
        # staging/H2D/arena addressing; dev==store but output wrong =>
        # decode/GEMM.  Churn config now uses bounded allocations (v4's
        # 16 GiB matmul OOM fixed) and micro_probe.json flushes per
        # iteration so a mid-suite failure keeps prior evidence.
        **_BASE, "arm_id": "mK",
        "prompts_json": [Q0],
        "cohort": None,
        "micro_probe": "1",
        "micro_fp": "1",
    },
    {
        # mL: same fingerprinted suite under CUDA_LAUNCH_BLOCKING=1 —
        # byte-level confirmation that the corruption is not a stream-
        # timing race (v4 showed output-level dirtiness under blocking;
        # this shows whether the BYTES on device are still wrong).
        **_BASE, "arm_id": "mL",
        "prompts_json": [Q0],
        "cohort": None,
        "micro_probe": "1",
        "micro_fp": "1",
        "extra_env": {"CUDA_LAUNCH_BLOCKING": "1"},
    },
    {
        # mM: same fingerprinted suite with cache_dtype=fp16 — the FP4-
        # specificity test.  If fp16 stays byte-stable across resets, the
        # packed-FP4 fill/stage/decode machinery is convicted and the
        # generic arena/pack path is exonerated.  (dev_sha isn't
        # comparable to store_sha here — fp16 blobs are the dequantized
        # record — only cross-iteration stability + output correctness
        # apply.)
        **_BASE, "arm_id": "mM",
        "cache_dtype": "fp16",
        "prompts_json": [Q0],
        "cohort": None,
        "micro_probe": "1",
        "micro_fp": "1",
        # fp16 experts are ~50 MiB: 64 = ~3.2 GiB, close to the 3.5 GiB
        # arena — drop to 48 experts so the suite stays eviction-free and
        # its byte-stability signal stays clean.
        "extra_env": {"NATIVE_MICRO_EXPERTS": "48"},
    },
]

STALE_PATTERNS = (
    "native-generate-result*.json", "native-generate-all.json",
    "routed_experts*.jsonl", "route_weights*.jsonl", "captures*.jsonl",
    "generated_checkpoint*.jsonl", "cache_events*.jsonl",
    "arm_config*.json", "cohort-c*.json", "result-c*.json",
    "SHA256SUMS*.json", "error.txt", "result*.json", "integrity*.json",
    "environment*.json", "run_config*.json", "profile*.json",
    "memory*.json", "progress.log", "dee4-*.json", "dee4-*.jsonl",
    "probe_*.npy", "micro_*.npy", "micro_probe.json",
    "p2.2-dee4-evidence.json")

report = {"job": "phase-5b warm-process divergence mechanism test",
          "branch": BRANCH, "commit": COMMIT or "(branch head)",
          "run_id": RUN_ID, "n_tokens": N_TOKENS,
          "checks": [], "runs": {}, "arms": {},
          "analysis": {}}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [p5b] {msg}", flush=True)


def write_report():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p5b_report.json").write_text(json.dumps(report, indent=1))


def check(name, ok, detail=""):
    report["checks"].append({"check": name, "ok": bool(ok),
                             "detail": detail})
    log(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    write_report()


def bail(verdict):
    report["verdict"] = verdict
    write_report()


def sh(cmd, timeout=600, env=None):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       env=env)
    return {"rc": p.returncode, "secs": round(time.time() - t0, 1),
            "out": (p.stdout or "")[-200000:], "err": (p.stderr or "")[-8000:]}


def sha256_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(64 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tail_lines(path, n=3):
    try:
        data = Path(path).read_text(errors="replace")
        lines = [ln for ln in data.splitlines() if ln.strip()]
        return " | ".join(lines[-n:])[:500]
    except OSError:
        return "(no log yet)"


def arm_env(arm):
    env = dict(os.environ)
    env.update({
        "NATIVE_RUN_ID": RUN_ID,
        "NATIVE_ARM_ID": arm["arm_id"],
        "NATIVE_N_TOKENS": str(arm["n_tokens"]),
        "NATIVE_EXPERT_STORE": arm["expert_store"],
        "NATIVE_CACHE_DTYPE": arm["cache_dtype"],
        "NATIVE_EVICTION_POLICY": arm["eviction_policy"],
        "NATIVE_HOST_CACHE_MODE": arm["host_cache_mode"],
        "NATIVE_HOST_PACK_GPU0_BYTES": str(arm["host_pack_gpu0_bytes"]),
        "NATIVE_HOST_PACK_GPU1_BYTES": str(arm["host_pack_gpu1_bytes"]),
        "NATIVE_BUDGET_BYTES": str(arm["budget_bytes"]),
        "NATIVE_SOURCE_READ_LANES": str(arm["lanes"]),
        "NATIVE_SOURCE_READ_QUEUE_DEPTH": str(arm["queue_depth"]),
        "NATIVE_CACHE_RESET": arm["cache_reset"],
        "NATIVE_TRACE_REQUESTS": arm["trace_requests"],
        "NATIVE_IGNORE_EOS": "1",
        "NATIVE_PROFILE": "1",
        "NATIVE_DIAGNOSTICS": "1",
        "NATIVE_PROMPTS_JSON": json.dumps(arm["prompts_json"]),
        "NATIVE_SOURCE_TREE": str(ROOT),
        "PYTHONPATH": str(DEE),
    })
    if arm.get("cohort"):
        env["NATIVE_COHORT_JSON"] = json.dumps(arm["cohort"])
    if arm.get("route_weight_journal"):
        env["NATIVE_ROUTE_WEIGHT_JOURNAL"] = arm["route_weight_journal"]
    if arm.get("capture_journal"):
        env["NATIVE_CAPTURE_JOURNAL"] = arm["capture_journal"]
    if arm.get("ffn_backend"):
        env["NATIVE_FFN_BACKEND"] = arm["ffn_backend"]
    if arm.get("batched"):
        env["NATIVE_BATCHED"] = arm["batched"]
    if arm.get("torch_deterministic"):
        env["NATIVE_TORCH_DETERMINISTIC"] = arm["torch_deterministic"]
    if arm.get("probe_l0"):
        env["NATIVE_PROBE_L0"] = arm["probe_l0"]
    if arm.get("micro_probe"):
        env["NATIVE_MICRO_PROBE"] = arm["micro_probe"]
    if arm.get("micro_fp"):
        env["NATIVE_MICRO_FP"] = arm["micro_fp"]
    for k, v in (arm.get("extra_env") or {}).items():
        env[k] = v
    if COMMIT:
        env["NATIVE_COMMIT"] = COMMIT
    env["NATIVE_DEE4_SEGMENTED_STORE"] = str(STORE_ROOT)
    env["DEE4_STORE_SKIP_SEAL"] = "1"
    return env


def run_arm(arm):
    """One runner process per arm (one engine build + one store open)."""
    arm_id = arm["arm_id"]
    groups = (arm.get("cohort") or {}).get("groups")
    n_units = len(groups) if groups else len(arm["prompts_json"])
    arm_out = OUT / arm_id
    arm_out.mkdir(parents=True, exist_ok=True)
    env = arm_env(arm)
    report["arms"][arm_id] = {
        "env": {k: v for k, v in env.items()
                if k.startswith(("NATIVE_", "DEE4_", "CUBLAS_"))},
        "resolved": dict(arm, prompts_json=f"{len(arm['prompts_json'])} prompts"),
    }
    for pat in STALE_PATTERNS:
        for stale in WORK.glob(pat):
            stale.unlink()
    log_path = arm_out / f"log-{arm_id}.txt"
    t0 = time.time()
    deadline = t0 + ARM_TIMEOUT_S
    timed_out = False
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(RUNNER)], env=env, cwd=str(WORK),
            stdout=lf, stderr=subprocess.STDOUT, text=True)
        last_hb = 0.0
        while True:
            rc = proc.poll()
            now = time.time()
            if now - last_hb >= HEARTBEAT_S:
                log(f"{arm_id} heartbeat elapsed={now - t0:.0f}s "
                    f"rc={rc} tail: {tail_lines(log_path)}")
                last_hb = now
            if rc is not None:
                break
            if now > deadline:
                timed_out = True
                log(f"{arm_id} TIMEOUT at {ARM_TIMEOUT_S}s; killing")
                proc.kill()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                break
            time.sleep(5)
    wall = round(time.time() - t0, 1)
    log(f"{arm_id} {'TIMEOUT' if timed_out else f'exit={proc.returncode}'} "
        f"wall={wall}s")
    rc_val = proc.returncode
    default_cls = "TIMEOUT" if timed_out else "NO_RESULT"
    recs = {}
    for ui in range(n_units):
        tag = f"{arm_id}-{'c' if groups else 'q'}{ui}"
        rec = {"arm": arm_id, "unit": ui, "rc": rc_val, "wall_s": wall}
        stem = f"c{ui}" if groups else f"q{ui}"
        res_path = WORK / f"result-{stem}.json"
        journal_path = WORK / f"routed_experts-{stem}.jsonl"
        weights_path = WORK / f"route_weights-{stem}.jsonl"
        for base in ("result", "integrity", "memory", "profile",
                     "run_config", "environment", "arm_config", "cohort"):
            src = WORK / f"{base}-{stem}.json"
            if src.is_file():
                (arm_out / src.name).write_bytes(src.read_bytes())
        for src in sorted(WORK.glob(f"generated_checkpoint-{stem}*.jsonl")):
            (arm_out / src.name).write_bytes(src.read_bytes())
        for src in sorted(WORK.glob(f"result-{stem}-r*.json")):
            (arm_out / src.name).write_bytes(src.read_bytes())
        for src in (WORK / f"cache_events-{stem}.jsonl",
                    WORK / f"native-generate-result-{stem}.json"):
            if src.is_file():
                (arm_out / src.name).write_bytes(src.read_bytes())
        result = None
        if res_path.is_file():
            raw = res_path.read_bytes()
            (arm_out / res_path.name).write_bytes(raw)
            try:
                result = json.loads(raw)
            except ValueError:
                rec["classification"] = "UNPARSEABLE_RESULT"
        else:
            err_path = WORK / f"native-generate-result-{stem}.json"
            if err_path.is_file():
                raw = err_path.read_bytes()
                (arm_out / err_path.name).write_bytes(raw)
                try:
                    rec["classification"] = json.loads(raw).get(
                        "classification", "ERROR")
                    rec["error"] = json.loads(raw).get("error")
                except ValueError:
                    rec["classification"] = "ERROR"
        if journal_path.is_file():
            dst = arm_out / journal_path.name
            dst.write_bytes(journal_path.read_bytes())
            rec["journal_sha256"] = sha256_path(dst)
        if weights_path.is_file():
            dst = arm_out / weights_path.name
            dst.write_bytes(weights_path.read_bytes())
            rec["weights_sha256"] = sha256_path(dst)
        captures_path = WORK / f"captures-{stem}.jsonl"
        if captures_path.is_file():
            dst = arm_out / captures_path.name
            dst.write_bytes(captures_path.read_bytes())
            rec["captures_sha256"] = sha256_path(dst)
        if result is not None:
            rec["classification"] = result.get("classification")
            rec["k"] = result.get("cohort_k", 1)
            rec["n_forward_steps"] = result.get("n_forward_steps")
            rec["emitted_tokens"] = result.get("emitted_tokens")
            rec["row_shas"] = {
                r.get("prompt_index"): r.get("token_ids_sha256")
                for r in (result.get("rows") or [])}
            if not groups:
                toks = result.get("generated_token_ids") or []
                rec["row_shas"] = {ui: hashlib.sha256(json.dumps(
                    [int(t) for t in toks]).encode()).hexdigest()}
                rec["n_tokens"] = len(toks)
            rec["total_wall_seconds"] = result.get("total_wall_seconds")
        elif "classification" not in rec:
            rec["classification"] = default_cls
        recs[tag] = rec
    for name in ("native-generate-all.json", "integrity.json",
                 "environment.json", "run_config.json", "profile.json",
                 "memory.json", "progress.log", "error.txt",
                 "dee4-segmented-store.json", "result.json",
                 "micro_probe.json",
                 "native-generate-result.json"):
        src = WORK / name
        if src.is_file():
            (arm_out / src.name).write_bytes(src.read_bytes())
    # v4 probes: layer-0 engine I/O dumps (mG) and micro-suite raws (mI/mJ)
    for src in sorted(WORK.glob("probe_*.npy")):
        (arm_out / src.name).write_bytes(src.read_bytes())
    for src in sorted(WORK.glob("micro_raw-*.npy")):
        (arm_out / src.name).write_bytes(src.read_bytes())
    return recs


def _first_divergence(arm_id, n_units, kind="weights_sha256",
                    fname="route_weights"):
    """Scan {fname}-c{u}.jsonl across units; return the first
    (step, layer) whose per-record sha differs between unit 0 and a
    later unit.  kind selects the fingerprint field."""
    per_unit = []
    for ui in range(n_units):
        p = OUT / arm_id / f"{fname}-c{ui}.jsonl"
        if not p.is_file():
            p = OUT / arm_id / f"{fname}-q{ui}.jsonl"
        if not p.is_file():
            return None
        rows = [json.loads(ln) for ln in p.read_text().splitlines()
                if ln.strip()]
        per_unit.append({(r["step"], r["layer"]): r for r in rows})
    if len(per_unit) < 2:
        return None
    keys = sorted(per_unit[0])
    for key in keys:
        ref = per_unit[0][key].get(kind)
        for ui in range(1, len(per_unit)):
            got = per_unit[ui].get(key, {})
            if key not in per_unit[ui]:
                return {"step": key[0], "layer": key[1],
                        "unit": ui, "missing": True}
            if got.get(kind) != ref:
                return {"step": key[0], "layer": key[1],
                        "unit": ui, "kind": kind,
                        "ref": str(ref)[:16], "got": str(got.get(kind))[:16]}
    return None


def selfcheck():
    problems = []
    if len(ARMS) != 3:
        problems.append(f"expected 3 arms, got {len(ARMS)}")
    for arm in ARMS:
        aid = arm["arm_id"]
        co = arm.get("cohort")
        if co is not None:
            groups = co.get("groups")
            if not groups or not all(groups):
                problems.append(f"{aid}: empty cohort groups")
            if max(max(g) for g in groups) >= len(arm["prompts_json"]):
                problems.append(f"{aid}: group index out of prompt range")
    envs = {}
    for arm in ARMS:
        try:
            envs[arm["arm_id"]] = arm_env(arm)
        except Exception as exc:
            problems.append(f"{arm['arm_id']}: arm_env raised {exc!r}")
    print("[selfcheck] resolved per-arm env diff (vs first arm):")
    ref_env = envs.get(ARMS[0]["arm_id"], {})
    for aid, env in envs.items():
        knob_keys = sorted(k for k in env
                           if k.startswith(("NATIVE_", "CUBLAS_",
                                            "CUDA_", "PYTORCH_")))
        diff = {k: env[k] for k in knob_keys
                if env.get(k) != ref_env.get(k)
                and k not in ("NATIVE_ARM_ID", "NATIVE_SOURCE_TREE",
                              "NATIVE_COMMIT")}
        print(f"  {aid}: {json.dumps(diff, sort_keys=True)[:400]}")
    print(f"[selfcheck] {'PROBLEMS: ' + '; '.join(problems) if problems else 'OK'}")
    return not problems


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---------------- P0 gate ----------------
    r = sh(["nvidia-smi", "-L"], timeout=60)
    names = [l for l in r["out"].splitlines() if "Tesla T4" in l]
    check("2x T4", len(names) == 2, "; ".join(names))
    if len(names) != 2:
        bail("FAIL_CLOSED"); return
    free_tmp = os.statvfs("/tmp").f_bavail * os.statvfs("/tmp").f_frsize
    log(f"[p0] /tmp free: {free_tmp / (1 << 30):.0f} GiB")

    # ---------------- P1 clone + build ----------------
    if os.environ.get("P5_SOURCE_ROOT"):
        check("driver-provided tree", (DEE / "CMakeLists.txt").is_file(),
              str(ROOT))
        if not (DEE / "CMakeLists.txt").is_file():
            bail("FAIL_CLOSED"); return
    else:
        r = sh(["git", "clone", "--branch", BRANCH, "--single-branch",
                REPO, str(ROOT)], timeout=900)
        check("clone", r["rc"] == 0, r["err"][-200:])
        if r["rc"] != 0:
            bail("FAIL_CLOSED"); return
    if COMMIT:
        r = sh(["git", "-C", str(ROOT), "checkout", "--quiet", COMMIT],
               timeout=120)
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True).strip()
        check("pinned commit", r["rc"] == 0 and head == COMMIT, head[:12])
        if r["rc"] != 0 or head != COMMIT:
            bail("FAIL_CLOSED"); return
    else:
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True).strip()
        check("branch head (unpinned)", True, head[:12])
    r = sh(["cmake", "-S", str(DEE), "-B", str(BUILD),
            "-DCMAKE_CUDA_ARCHITECTURES=60;75", "-DDEE_CUDA=ON",
            "-DDEE_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=Release"],
           timeout=900)
    check("cmake configure", r["rc"] == 0, r["err"][-200:])
    r = sh(["cmake", "--build", str(BUILD), "--target", "dee_core", "-j", "2"],
           timeout=3600)
    check("dee_core build", r["rc"] == 0, r["err"][-200:])
    if r["rc"] != 0:
        (OUT / "build-dee_core.log").write_text(
            r["out"] + "\n---STDERR---\n" + r["err"])
        bail("FAIL_BUILD"); return
    r2 = sh(["cmake", "--build", str(BUILD), "--target", "test_dee4_segmented",
             "-j", "2"], timeout=1200)
    check("test_dee4_segmented build", r2["rc"] == 0, r2["err"][-200:])
    if r2["rc"] != 0:
        bail("FAIL_BUILD"); return
    r = sh(["ctest", "--test-dir", str(BUILD), "-R", "^test_dee4_segmented$",
            "--output-on-failure"], timeout=600)
    check("test_dee4_segmented", r["rc"] == 0, r["out"][-300:])
    if r["rc"] != 0:
        bail("FAIL_MECHANISM"); return
    r = sh([sys.executable, "-m", "pip", "install", "--quiet", "--user",
            "pybind11"], timeout=600)
    r = subprocess.run(
        [sys.executable, "pydee/setup.py", "build_ext", "--inplace"],
        capture_output=True, text=True, timeout=1800, cwd=str(DEE),
        env={**os.environ, "DEE_BUILD_DIR": str(BUILD)})
    r = {"rc": r.returncode, "out": (r.stdout or "")[-200000:],
         "err": (r.stderr or "")[-8000:]}
    check("pydee build", r["rc"] == 0, r["err"][-300:])
    if r["rc"] != 0:
        bail("FAIL_CLOSED"); return

    # ---------------- P2 store assembly (symlinks; skip re-seal — the
    # sealed evidence came from the same mounted datasets in the P5
    # campaign, and the runner's own store integrity gate still runs) ----
    def ds_path(slug, name):
        for base in (Path("/kaggle/input/datasets/nivind"),
                     Path("/kaggle/input")):
            p = base / slug
            if (p / name).is_file():
                return p
        return None

    mount_deadline = time.time() + 1500
    while True:
        missing = []
        if not any((base / "deepseek-v4-flash-0731-shards").is_dir()
                   for base in (Path("/kaggle/input/datasets/nivind"),
                                Path("/kaggle/input"))):
            missing.append("shards")
        idx_dir = ds_path(INDEX_DS.split("/", 1)[-1], "metadata.json")
        if idx_dir is None:
            missing.append("index")
        for b in range(N_BUCKETS):
            p = ds_path(f"{SEG_PREFIX}{b:02d}",
                        f"experts-bucket-{b:02d}.dee4")
            if p is None:
                missing.append(f"b{b:02d}")
        if not missing:
            break
        if time.time() > mount_deadline:
            break
        log(f"waiting for {len(missing)} mounts ({missing[:4]}...)")
        time.sleep(30)
    check("index dataset mounted",
          idx_dir is not None and (idx_dir / "metadata.json").is_file(),
          f"missing: {missing[:8]}")
    if not (idx_dir and (idx_dir / "metadata.json").is_file()):
        bail("FAIL_CLOSED"); return
    if "shards" in missing:
        check("shard dataset mounted (post-wait)", False,
              "shards dataset never mounted")
        bail("FAIL_CLOSED"); return
    meta = json.loads((idx_dir / "metadata.json").read_text())
    segs = meta["segments"]
    check("segment table 46", len(segs) == N_BUCKETS, f"{len(segs)}")
    if len(segs) != N_BUCKETS:
        bail("FAIL_CLOSED"); return
    (STORE_ROOT / "segments").mkdir(parents=True, exist_ok=True)
    (STORE_ROOT / "metadata.json").write_bytes(
        (idx_dir / "metadata.json").read_bytes())

    def fetch_bucket(entry):
        b = int(entry["bucket"])
        dsrc = ds_path(f"{SEG_PREFIX}{b:02d}", Path(entry["file"]).name)
        src = dsrc / Path(entry["file"]).name if dsrc else None
        dst = STORE_ROOT / entry["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src is None or not src.is_file():
            return (b, f"mount missing {SEG_PREFIX}{b:02d}")
        if src.stat().st_size != int(entry["bytes"]):
            return (b, f"mount size {src.stat().st_size}")
        try:
            dst.unlink(missing_ok=True)
            dst.symlink_to(src)
            if dst.stat().st_size != int(entry["bytes"]):
                return (b, f"linked size {dst.stat().st_size}")
        except Exception as exc:
            return (b, repr(exc)[:120])
        return (b, None)

    t0 = time.time()
    errors = []
    with ThreadPoolExecutor(max_workers=CP_WORKERS) as ex:
        for b, err in ex.map(fetch_bucket, segs):
            if err:
                errors.append(f"b{b}:{err}")
    check("46 segments assembled", not errors,
          "; ".join(errors[:5]) or f"{time.time() - t0:.0f}s")
    if errors:
        bail("FAIL_CLOSED"); return

    # ---------------- P3 arm loop ----------------
    for arm in ARMS:
        report["runs"].update(run_arm(arm))
        write_report()

    # ---------------- P4 analysis + verdict ----------------
    # Per-arm: do all units produce bit-identical token streams?  Micro
    # arms (mI/mJ) produce micro_probe.json configs instead of journals.
    analysis = {}
    for arm in ARMS:
        aid = arm["arm_id"]
        groups = (arm.get("cohort") or {}).get("groups")
        n_units = len(groups) if groups else len(arm["prompts_json"])
        if arm.get("micro_probe"):
            mp_path = OUT / aid / "micro_probe.json"
            mp = (json.loads(mp_path.read_text())
                  if mp_path.is_file() else None)
            cfg_rows = (mp or {}).get("results") or {}
            micro = {
                "micro": True,
                "units": n_units,
                # Completion = the probe file exists AND the suite ran at
                # least one config iteration — incremental flush means a
                # mid-suite failure still yields partial evidence.
                "accepted": (n_units if mp is not None and
                             any((v.get("iters") or v.get("raw_shas"))
                                 for v in cfg_rows.values()) else 0),
                "configs": {k: v.get("distinct") for k, v in
                            cfg_rows.items()},
                "configs_detail": {
                    k: {kk: vv for kk, vv in v.items() if kk != "iters"}
                    for k, v in cfg_rows.items()},
            }
            # v5: per-iteration boundary bisect — for each config, count
            # experts crossing each corruption boundary:
            #   pack_sha != store_sha  => host fill lane wrong bytes
            #   dev_sha  != pack_sha   => staging/H2D/arena addressing
            #   dev_all_zero           => device block is all zeros
            # and per-expert output state from the raw .npy dumps
            # (zero/wrong (t,s) block -> expert via ids[t,s]).
            fp_rows = {}
            try:
                import numpy as _np
            except Exception:
                _np = None
            mpc = (mp or {}).get("config") or {}
            _mn = int(mpc.get("n") or 18)
            _mtopk = int(mpc.get("topk") or 6)
            _mexp = int(mpc.get("experts") or 64)
            for cfg_name, cfg in cfg_rows.items():
                iters = cfg.get("iters") or []
                if not any(it.get("fp") for it in iters):
                    continue
                # Rebuild the ids mapping the probe used.
                ids = [[(t * _mtopk + s) % _mexp
                        for s in range(_mtopk)] for t in range(_mn)]
                raws = []
                for i in range(len(iters)):
                    p = OUT / aid / f"micro_raw-{cfg_name}-i{i}.npy"
                    try:
                        raws.append(
                            _np.load(str(p))
                            if (p.is_file() and _np is not None)
                            else None)
                    except Exception:
                        raws.append(None)
                iter_rows = []
                # dev-vs-pack byte equality is only meaningful when the
                # device holds the same packed record bytes — under
                # cache_dtype=fp16 the block is the dequantized blob, so
                # dev_sha != pack_sha is expected and not a defect signal.
                _fp4_cache = arm.get("cache_dtype", "fp4") == "fp4"
                for i, it in enumerate(iters):
                    fps = it.get("fp") or []
                    row = {
                        "raw_sha": str(it.get("raw_sha"))[:16],
                        "n_fp": len(fps),
                        "store_ok": sum(1 for f in fps
                                        if f.get("store_ok")),
                        "pack_ready": sum(1 for f in fps
                                          if f.get("pack_ready")),
                        "pack_ne_store": sum(
                            1 for f in fps
                            if f.get("store_ok") and f.get("pack_ready")
                            and f.get("pack_sha") != f.get("store_sha")),
                        "dev_resident": sum(1 for f in fps
                                            if f.get("dev_resident")),
                        "dev_ne_pack": sum(
                            1 for f in fps
                            if _fp4_cache
                            and f.get("dev_resident") and f.get("pack_ready")
                            and f.get("dev_sha") != f.get("pack_sha")),
                        "dev_eq_pack": sum(
                            1 for f in fps
                            if _fp4_cache
                            and f.get("dev_resident") and f.get("pack_ready")
                            and f.get("dev_sha") == f.get("pack_sha")),
                        "dev_zero": sum(1 for f in fps
                                        if f.get("dev_all_zero")),
                        # v5 root-cause tripwire: a live pack entry whose
                        # stamped fp4_regions sum to 0 bytes means the
                        # pinned-slot gather copied nothing and the H2D
                        # shipped stale slot bytes (clear_host_cache
                        # region-nbytes poison, fixed in 0dd39ef+).
                        "staging_zero_regions": sum(
                            1 for f in fps
                            if f.get("pack_ready") and f.get("dev_resident")
                            and f.get("staging_present")
                            and not f.get("staging_region_bytes")),
                    }
                    scr = it.get("scratch") or {}
                    row["scratch_zero"] = scr.get("all_zero")
                    # Per-expert chain: join fingerprints with the raw
                    # output's zero/diff blocks for this iteration.
                    experts = {}
                    if raws[i] is not None:
                        ref = raws[0]
                        for t in range(min(_mn, raws[i].shape[0])):
                            for s in range(min(_mtopk, raws[i].shape[1])):
                                blk = raws[i][t, s]
                                e = ids[t][s]
                                st = experts.setdefault(e, {"pos": 0})
                                st["pos"] += 1
                                if not _np.any(blk):
                                    st["out_zero"] = st.get(
                                        "out_zero", 0) + 1
                                if i > 0 and ref is not None and \
                                        not _np.array_equal(
                                            blk, ref[t, s]):
                                    st["out_diff"] = st.get(
                                        "out_diff", 0) + 1
                    row["expert_states"] = experts
                    # Third boundary: experts whose DEVICE bytes verified
                    # identical to the store record yet whose output block
                    # is zero/differs — the decode/GEMM conviction.
                    row["dev_clean_but_out_bad"] = sum(
                        1 for e, st in experts.items()
                        if (st.get("out_zero") or st.get("out_diff"))
                        and e < len(fps)
                        and fps[e].get("dev_resident")
                        and fps[e].get("store_ok")
                        and fps[e].get("dev_sha") == fps[e].get(
                            "store_sha"))
                    iter_rows.append(row)
                fp_rows[cfg_name] = iter_rows
            if fp_rows:
                micro["fp_iters"] = fp_rows
            analysis[aid] = micro
            continue
        shas = []
        ok_units = 0
        for ui in range(n_units):
            tag = f"{aid}-{'c' if groups else 'q'}{ui}"
            rec = report["runs"].get(tag, {})
            # Completion = produced tokens + journals.  Mechanism arms may
            # not pass the native integrity classifier — journals carry
            # the evidence.
            produced = (rec.get("rc") == 0
                        and (rec.get("classification")
                             == "ACCEPT_CORRECTNESS"
                             or rec.get("n_tokens") or rec.get("row_shas")
                             or rec.get("journal_sha256")))
            if produced:
                ok_units += 1
            sha_map = rec.get("row_shas") or {}
            # cohort rows: one row per prompt_index (all index 0 here);
            # sequential: keyed by unit index.
            sha = next(iter(sha_map.values()), None)
            shas.append(sha)
        identical = (len({s for s in shas if s}) == 1
                     and all(s for s in shas))
        first_div = _first_divergence(aid, n_units)
        cap_divs = {}
        for key in ("moe_out_sha256", "shared_out_sha256",
                    "router_scores_sha256", "expert_ids_sha256"):
            d = _first_divergence(aid, n_units, kind=key,
                                  fname="captures")
            if d is not None:
                cap_divs[key] = d
        rec_a = {"units": n_units, "accepted": ok_units,
                 "token_shas": [str(s)[:16] for s in shas],
                 "units_bit_identical": identical,
                 "first_divergent_weight": first_div,
                 "first_divergent_ids": _first_divergence(
                     aid, n_units, kind="ids_sha256"),
                 "first_divergent_captures": cap_divs}
        # v4 mG probe: layer-0 engine I/O dumps — cross-unit magnitude +
        # same-unit replay (miss vs hit path) comparison.
        if arm.get("probe_l0"):
            try:
                import numpy as _np
                dumps = {}
                for ui in range(n_units):
                    stem = f"q{ui}"
                    entry = {}
                    for k in ("hidden", "raw", "raw_replay", "ids"):
                        p = OUT / aid / f"probe_{stem}_{k}.npy"
                        if p.is_file():
                            entry[k] = str(p)
                    dumps[stem] = entry
                if "q0" in dumps and "raw" in dumps["q0"]:
                    ref = _np.load(dumps["q0"]["raw"])
                    for stem, entry in dumps.items():
                        if "raw" not in entry:
                            continue
                        cur = _np.load(entry["raw"])
                        d = _np.abs(ref - cur)
                        entry["vs_q0_max_abs"] = float(d.max())
                        entry["vs_q0_ndiff"] = int((d > 0).sum())
                    for stem, entry in dumps.items():
                        if "raw" in entry and "raw_replay" in entry:
                            d = _np.abs(_np.load(entry["raw"])
                                        - _np.load(entry["raw_replay"]))
                            entry["replay_max_abs"] = float(d.max())
                            entry["replay_ndiff"] = int((d > 0).sum())
                # keep only the comparison metrics, not the path strings
                rec_a["probe_l0_dumps"] = {
                    s: {k: v for k, v in e.items()
                        if k.startswith(("vs_q0", "replay"))}
                    for s, e in dumps.items()}
            except Exception as exc:
                rec_a["probe_l0_error"] = repr(exc)[:200]
        analysis[aid] = rec_a
    report["analysis"] = analysis

    mK = analysis.get("mK", {})
    mL = analysis.get("mL", {})
    mM = analysis.get("mM", {})

    def _fp_tallies(arm_analysis):
        """Sum the per-iteration boundary counters across a micro arm's
        configs -> {pack_ne_store, dev_ne_pack, dev_zero, dev_eq_pack}."""
        tot = {"pack_ne_store": 0, "dev_ne_pack": 0, "dev_zero": 0,
               "dev_eq_pack": 0, "dev_clean_but_out_bad": 0,
               "staging_zero_regions": 0, "iters_with_fp": 0}
        for cfg_name, rows in (arm_analysis.get("fp_iters") or {}).items():
            for row in rows:
                if not row.get("n_fp"):
                    continue
                tot["iters_with_fp"] += 1
                for k in ("pack_ne_store", "dev_ne_pack", "dev_zero",
                          "dev_eq_pack", "dev_clean_but_out_bad",
                          "staging_zero_regions"):
                    tot[k] += int(row.get(k) or 0)
        return tot

    mk_t = _fp_tallies(mK)
    ml_t = _fp_tallies(mL)
    mm_t = _fp_tallies(mM)
    mk_cfgs = mK.get("configs") or {}
    mm_cfgs = mM.get("configs") or {}
    mk_repro = any((v or 0) > 1 for v in mk_cfgs.values())
    mm_repro = any((v or 0) > 1 for v in mm_cfgs.values())
    interp = {
        # Byte-boundary verdicts (the decisive fields):
        "host_fill_convicted": mk_t["pack_ne_store"] > 0,
        "device_fill_convicted": (mk_t["pack_ne_store"] == 0
                                  and mk_t["dev_ne_pack"] > 0),
        "decode_or_gemm_convicted": (mk_t["pack_ne_store"] == 0
                                     and mk_t["dev_ne_pack"] == 0
                                     and mk_t["dev_clean_but_out_bad"] > 0),
        "device_blocks_zero": mk_t["dev_zero"],
        "fp4_path_specific": mk_repro and not mm_repro,
        "bytes_stable_under_blocking": (
            ml_t["iters_with_fp"] > 0 and ml_t["dev_ne_pack"] == 0
            and ml_t["dev_zero"] == 0 and ml_t["pack_ne_store"] == 0),
        "fp16_also_divergent": mm_repro,
        "micro_reproduced": mk_repro,
        "mk_boundaries": mk_t,
        "ml_boundaries": ml_t,
        "mm_boundaries": mm_t,
        "reading": (
            "fp_iters[cfg][iter] counts experts crossing each byte "
            "boundary: pack_ne_store = host fill lane delivered wrong "
            "bytes; dev_ne_pack = staging/H2D/arena wrote wrong bytes; "
            "dev_zero = resident block is all zeros; dev_eq_pack = "
            "device bytes verified == pack bytes.  expert_states maps "
            "expert -> {pos, out_zero, out_diff} from the raw output so "
            "each expert's full chain store->pack->dev->output is "
            "visible per iteration.  fp4_path_specific = defect repros "
            "under fp4 (mK) but not fp16 (mM)."),
    }
    report["interpretation"] = interp
    # The mechanism test's verdict is informational, not a gate:
    # PASS = all arms produced their units' data.
    complete = all(a.get("accepted") == a.get("units")
                   for a in analysis.values())
    report["verdict"] = "PASS" if complete else "FAIL"
    write_report()
    print("P5B VERDICT:", report["verdict"], flush=True)
    print("P5B ANALYSIS:", json.dumps(analysis, indent=1)[:4000],
          flush=True)


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        sys.exit(0 if selfcheck() else 1)
    try:
        main()
    except Exception as exc:
        report["verdict"] = f"ERROR: {exc!r}"[:300]
        try:
            (OUT / "p5b_report.json").write_text(
                json.dumps(report, indent=1))
        except Exception:
            pass
    sys.exit(0)
