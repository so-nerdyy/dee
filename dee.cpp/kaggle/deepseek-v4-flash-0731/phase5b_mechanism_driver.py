"""Phase-5b mechanism test driver (dual-T4, Kaggle).

Question answered: WHERE does the warm-process numerical divergence come
from?  The P5 campaign proved: inference unit >=2 in a warm process
diverges from unit 0 on byte-identical inputs, deterministically within
each forward (pad rows mutually agree), injected during layers 0-2
compute.  This session bisects the source with three arms sharing one
process-per-arm discipline:

  v2 result (kernel v2, all-sequential arms): divergence survives BOTH
  torch.use_deterministic_algorithms AND CUBLAS_WORKSPACE_CONFIG — the
  documented cuBLAS-workspace fix is dead.  Warm trajectory is itself
  low-entropy (mB-q1 == mC-q1 divergent sha) — consistent with
  allocator-layout/alignment-dependent kernel dispatch, not randomness.
  v1 result: sequential path itself diverges (not cohort-specific);
  injection sits inside layer-0's compute output (layer-1 router
  weights differ at step 0 while layer-0's match).

  mD  reference torch expert path (ffn_backend=cache_fp16): routed
      experts via DeepseekV4CacheFfn torch GEMMs over FP16 payloads —
      bypasses moe_forward_batch_device entirely.  Clean => dee_core
      native FFN path convicted.  Dirty => mechanism lives above the
      engine (torch/dense/attention side).
  mE  PYTORCH_NO_CUDA_MEMORY_CACHING=1: disables the caching allocator —
      every tensor alloc goes through cudaMalloc (uniform driver
      alignment).  Clean => allocator-layout/alignment-dependent
      dispatch convicted.
  mF  NATIVE_BATCHED=1: engine uses cublasGemmBatchedEx pointer-batch
      instead of per-expert cublasGemmEx.  Clean => per-expert GEMM
      dispatch specifically implicated; batched = candidate fix.

Instruments on every arm: NATIVE_ROUTE_WEIGHT_JOURNAL (per-step,layer
sha of routing weights + expert ids) and NATIVE_CAPTURE_JOURNAL (sha of
moe_out/shared_out/router_scores captures) — the capture shas split
layer-0 internals: moe_out diverging => engine FFN; shared_out alone =>
shared expert; weights-only => attention/residual side.

Accept/reject is per-arm in p5b_report.json:
  engine_convicted     = mD bit-identical (dee_core path necessary)
  allocator_convicted  = mE bit-identical (layout/algo dispatch)
  batched_stable       = mF bit-identical (per-expert GEMM implicated)
  unresolved           = all three still diverge

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
        # mD: reference torch expert path (cache_fp16) — routed experts
        # computed by DeepseekV4CacheFfn torch GEMMs over FP16 payloads,
        # bypassing moe_forward_batch_device entirely.  The python cache
        # persists across units, so warm units run the SAME payload
        # tensors — clean result convicts dee_core; dirty => the
        # mechanism lives above the engine (torch-side/dense path).
        # NOTE: engine-integrity gates in classify_full_generation are
        # meaningless for this arm (engine unused) — the journals carry
        # the evidence; produced-tokens is the completion criterion.
        **_BASE, "arm_id": "mD",
        "prompts_json": [Q0, Q0, Q0],
        "cohort": None,
        "route_weight_journal": "1",
        "capture_journal": "1",
        "ffn_backend": "cache_fp16",
    },
    {
        # mE: allocator-layout probe — PYTORCH_NO_CUDA_MEMORY_CACHING=1
        # disables the caching allocator: every tensor alloc goes to
        # cudaMalloc (uniform >=256B driver alignment).  The leading
        # surviving theory is address/alignment-dependent kernel/algo
        # dispatch (warm units' tensors land at different sub-offsets of
        # recycled pool blocks).  Clean => allocator layout convicted.
        **_BASE, "arm_id": "mE",
        "prompts_json": [Q0, Q0, Q0],
        "cohort": None,
        "route_weight_journal": "1",
        "capture_journal": "1",
        "extra_env": {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"},
    },
    {
        # mF: pointer-batched GEMM path — NATIVE_BATCHED=1 switches the
        # engine to cublasGemmBatchedEx (pointer-table batch) instead of
        # per-expert cublasGemmEx.  Divergent here too => mechanism deeper
        # than per-expert GEMM dispatch; clean => the per-expert path is
        # specifically implicated (and batched mode is a candidate fix).
        **_BASE, "arm_id": "mF",
        "prompts_json": [Q0, Q0, Q0],
        "cohort": None,
        "route_weight_journal": "1",
        "capture_journal": "1",
        "batched": "1",
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
                 "native-generate-result.json"):
        src = WORK / name
        if src.is_file():
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
                                            "PYTORCH_")))
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
    # Per-arm: do all units produce bit-identical token streams?
    analysis = {}
    for arm in ARMS:
        aid = arm["arm_id"]
        groups = (arm.get("cohort") or {}).get("groups")
        n_units = len(groups) if groups else len(arm["prompts_json"])
        shas = []
        ok_units = 0
        for ui in range(n_units):
            tag = f"{aid}-{'c' if groups else 'q'}{ui}"
            rec = report["runs"].get(tag, {})
            # Completion = produced tokens + journals.  The reference-FFN
            # arm (mD) never exercises the engine, so the native-path
            # integrity classifier REJECTs it by design — the journals
            # are the evidence, not the classification.
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
        analysis[aid] = {
            "units": n_units, "accepted": ok_units,
            "token_shas": [str(s)[:16] for s in shas],
            "units_bit_identical": identical,
            "first_divergent_weight": first_div,
            "first_divergent_ids": _first_divergence(
                aid, n_units, kind="ids_sha256"),
            "first_divergent_captures": cap_divs,
        }
    report["analysis"] = analysis

    mD = analysis.get("mD", {})
    mE = analysis.get("mE", {})
    mF = analysis.get("mF", {})
    interp = {
        "engine_convicted": (mD.get("units_bit_identical") is True
                             and mD.get("accepted") == 3),
        "allocator_convicted": (mE.get("units_bit_identical") is True
                                and mE.get("accepted") == 3),
        "batched_stable": (mF.get("units_bit_identical") is True
                           and mF.get("accepted") == 3),
        "engine_exonerated": mD.get("units_bit_identical") is False,
        "unresolved": all(a.get("units_bit_identical") is False
                          for a in (mD, mE, mF)),
        "reading": (
            "mD clean => dee_core native path necessary for the defect.  "
            "mE clean => allocator-layout/alignment-dependent dispatch "
            "convicted (fix: allocator discipline).  mF clean => "
            "per-expert cublasGemmEx implicated; batched path is a "
            "candidate fix.  All dirty => driver/context-level or "
            "attention-side mechanism remains.  "
            "first_divergent_captures splits layer-0 internals: "
            "moe_out diverging => engine FFN output; shared_out alone "
            "=> shared expert; only downstream weights => attention/residual."),
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
