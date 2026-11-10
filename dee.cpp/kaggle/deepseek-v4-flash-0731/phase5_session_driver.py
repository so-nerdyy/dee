"""Phase-5 cohort-serving campaign driver (dual-T4, Kaggle).

Question answered: does lockstep cohort execution preserve per-request
exactness (every row's token stream identical to its sequential run over
the same padded input), and how do shared-cache hit rates / store reads /
wall time scale with cohort size K?  The W0 replay sim predicts a VRAM
LRU cliff at K=4 on 3.5 GiB (48% -> 38% -> ~0% residency) and host-RAM
rescue as the real serving tier; this run validates both directions.

Arms (same kernel, same padded workload, same RUN_ID, same device split):
  c0_anchor  sequential b=1 over UNPADDED prompt q0 only -- the
             cross-campaign anchor: its token sha must equal the
             committed Phase-4 a2_fp4 q0 sha (reproves determinism of the
             whole rebuilt stack against sealed evidence)
  c1         cohort mode, groups=[[0]..[7]], pad_to="max" -- eight
             sequential singleton cohorts over the PADDED inputs; the
             in-session exactness reference and the K=1 residency point
  c2         groups=[[0,1],[2,3],[4,5],[6,7]] -- K=2 cohorts
  c4         groups=[[0,1,2,3],[4,5,6,7]] -- K=4 cohorts (sim: VRAM cliff)
  c8h        groups=[[0..7]] + enlarged host pack (largest the session
             can carry) -- K=8 host-rescue direction probe

Accept: per cohort, classification=="ACCEPT_CORRECTNESS" AND every row
has n_tokens==128 AND store lookup_failures==0; every cohort row's
token_ids_sha256 equals the c1 row sha for the same prompt_index
(non-null both sides); c0's token sha equals the committed Phase-4 a2
q0 sha.  Reject: any divergence, fail-closed miss, seal failure.

Phases (identical discipline to phase4_session_driver.py):
  P0  gate: 2x T4 SM75, /tmp disk, dataset mount probe
  P1  clone pinned branch + CUDA build + ctest + pydee build_ext
  P2  store assembly: symlink 46 mounted segments, driver-side sha seal
  P3  arm loop: one runner process per arm, watcher heartbeat, per-arm
      artifact harvest into p5-out/{arm_id}/
  P4  verdict + p5_report.json (always exits 0)

--selfcheck validates ARMS/env/groups and prints the resolved per-arm
env diff WITHOUT launching anything.
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
ROOT = Path(os.environ.get("P5_SOURCE_ROOT", "/tmp/dee-p5-src"))
DEE = ROOT / "dee.cpp"
BUILD = DEE / "build-kaggle"
WORK = Path("/kaggle/working")
OUT = WORK / "p5-out"
RUNNER = (DEE / "kaggle" / "deepseek-v4-flash-0731"
          / "deepseek_v4_native_generate.py")

INDEX_DS = "nivind/dee4-p3-full-u2081ada5e37e-index"
SEG_PREFIX = "dee4-p3-full-u2081ada5e37e-b"
N_BUCKETS = 46
STORE_ROOT = Path("/tmp/dee4-full")
N_TOKENS = 128
def _env_num(cast, name, default):
    try:
        return cast(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return cast(default)


ARM_TIMEOUT_S = _env_num(int, "P5_ARM_TIMEOUT_S", 3 * 3600)
HEARTBEAT_S = 60
# run_id is inside the route-journal canonical hash payload; constant
# across arms so journal structures compare cleanly.
RUN_ID = "p5-cohort-campaign"
CP_WORKERS = 4

ONE_RECORD_BYTES = 13_369_344            # packed dee4 expert record
PACK_8_5_GIB = int(8.5 * (1 << 30))      # Phase-4 committed pack/GPU
VRAM_3_5_GIB = 3584 << 20                # Phase-4 committed VRAM arena
# c8h's enlarged host pack: sized UNDER the measured ~21 GiB total-LRU
# death point (runner comment: v12/v14 OOM'd at ~21 GiB LRU + 5-8 GiB
# baseline) -- 9.75 GiB/GPU = 19.5 GiB total leaves ~1.5 GiB margin.
# Needs NATIVE_LRU_TOTAL_CAP_GIB > 17 or the runner scales it back to
# the 8.5 GiB baseline (the arm would measure nothing distinct).
PACK_C8H_GIB = _env_num(float, "P5_C8H_HOST_PACK_GIB", 9.75)
PACK_C8H = int(PACK_C8H_GIB * (1 << 30))
LRU_CAP_C8H_GIB = _env_num(float, "P5_C8H_LRU_CAP_GIB", 19.5)

# Committed Phase-4 reference (inside the cloned tree -- sealed evidence).
P4_TOKEN_MANIFEST = (DEE / "benchmark_reports/deepseek-v4-flash-0731-t4"
                     / "gpu2-phase4-cache-hierarchy/token-sha-manifest.json")

# ---- 8-prompt workload (identical to Phase-4) --------------------------
_B = ("<" + chr(0xFF5C) + "begin" + chr(0x2581) + "of" + chr(0x2581)
      + "sentence" + chr(0xFF5C) + ">")
_A = "<" + chr(0xFF5C) + "Assistant" + chr(0xFF5C) + ">"
_TEXTS = [
    "Explain how mRNA vaccines work, from injection to immune memory.",
    "Write a Python function that finds the longest common subsequence "
    "of two strings, and explain its time and space complexity.",
    "A fair coin is flipped until two consecutive heads appear. What is "
    "the expected number of flips? Show your reasoning step by step.",
    "Write the opening paragraph of a hard science fiction novel about "
    "a generation ship whose crew discovers the laws of physics are "
    "slightly different two light-years from Earth.",
    "Explain in one sentence why the sky is blue.",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three causes of the French Revolution.",
]
PROMPTS = [_B + t + _A for t in _TEXTS]
PROMPTS.append(PROMPTS[0])  # q7 = q0 verbatim (determinism probe)

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
}

ARMS = [
    {
        # c0: one unpadded sequential prompt (q0) through the legacy path
        # -- the cross-campaign anchor vs the committed Phase-4 a2 sha.
        # Warm reset + fp4/lru matches the a2 config it anchors to.
        **_BASE, "arm_id": "c0_anchor", "cache_reset": "warm",
        "host_pack_gpu0_bytes": PACK_8_5_GIB,
        "host_pack_gpu1_bytes": PACK_8_5_GIB,
        "prompts_json": [PROMPTS[0]],
        "cohort": None,
    },
    {
        # c1: singleton cohorts, pad_to="max" -- every prompt runs the
        # SAME padded input the cohort arms see.  This is the in-session
        # exactness reference AND the K=1 residency datapoint.
        **_BASE, "arm_id": "c1",
        "host_pack_gpu0_bytes": PACK_8_5_GIB,
        "host_pack_gpu1_bytes": PACK_8_5_GIB,
        "prompts_json": PROMPTS,
        "cohort": {"groups": [[i] for i in range(8)], "pad_to": "max"},
    },
    {
        **_BASE, "arm_id": "c2",
        "host_pack_gpu0_bytes": PACK_8_5_GIB,
        "host_pack_gpu1_bytes": PACK_8_5_GIB,
        "prompts_json": PROMPTS,
        "cohort": {"groups": [[0, 1], [2, 3], [4, 5], [6, 7]],
                   "pad_to": "max"},
    },
    {
        **_BASE, "arm_id": "c4",
        "host_pack_gpu0_bytes": PACK_8_5_GIB,
        "host_pack_gpu1_bytes": PACK_8_5_GIB,
        "prompts_json": PROMPTS,
        "cohort": {"groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
                   "pad_to": "max"},
    },
    {
        # c8h: one 8-row cohort + the largest host pack the session can
        # carry -- the host-rescue direction probe.
        **_BASE, "arm_id": "c8h",
        "host_pack_gpu0_bytes": PACK_C8H,
        "host_pack_gpu1_bytes": PACK_C8H,
        "lru_total_cap_gib": LRU_CAP_C8H_GIB,
        "prompts_json": PROMPTS,
        "cohort": {"groups": [list(range(8))], "pad_to": "max"},
    },
]

STALE_PATTERNS = (
    "native-generate-result*.json", "native-generate-all.json",
    "routed_experts*.jsonl", "generated_checkpoint*.jsonl",
    "cache_events*.jsonl", "arm_config*.json", "cohort-c*.json",
    "result-c*.json", "SHA256SUMS*.json",
    "error.txt", "result*.json", "integrity*.json",
    "environment*.json", "run_config*.json", "profile*.json",
    "memory*.json", "progress.log", "dee4-*.json",
    "dee4-*.jsonl", "p2.2-dee4-evidence.json")

report = {"job": "phase-5 cohort-serving campaign (dee-serve v0)",
          "branch": BRANCH, "commit": COMMIT or "(branch head)",
          "run_id": RUN_ID, "checks": [], "runs": {}, "arms": {}}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [p5] {msg}", flush=True)


def write_report():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p5_report.json").write_text(json.dumps(report, indent=1))


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
        "NATIVE_PROMPTS_JSON": json.dumps(arm["prompts_json"]),
        "NATIVE_SOURCE_TREE": str(ROOT),
        "PYTHONPATH": str(DEE),
    })
    if arm.get("cohort"):
        env["NATIVE_COHORT_JSON"] = json.dumps(arm["cohort"])
    if arm.get("lru_total_cap_gib"):
        env["NATIVE_LRU_TOTAL_CAP_GIB"] = str(arm["lru_total_cap_gib"])
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
                if k.startswith(("NATIVE_", "DEE4_"))},
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
        # Per-cohort artifacts: result-c{i}.json is the cohort aggregate;
        # result-c{i}-r{r}.json the per-row files; routed_experts-c{i}.jsonl
        # the K-row journal; generated_checkpoint-c{i}[-r{r}].jsonl the
        # checkpoint streams.
        stem = f"c{ui}" if groups else f"q{ui}"
        res_path = WORK / f"result-{stem}.json"
        journal_path = WORK / f"routed_experts-{stem}.jsonl"
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
            # Runner error path writes native-generate-result-{stem}.json
            # (classification=ERROR); harvest it so the unit records the
            # real classification instead of NO_RESULT.
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
        # Loud clamp detection: the runner scales the requested host pack
        # down under the total-LRU/mem-avail caps -- surface it so an arm
        # can't silently run a different capacity than configured.
        cfg_path = arm_out / f"arm_config-{stem}.json"
        if cfg_path.is_file():
            try:
                acfg = json.loads(cfg_path.read_bytes())
                _res = acfg.get("resolved") or {}
                _req = _res.get("host_pack_cache_bytes_requested") or []
                _eff = _res.get("host_pack_cache_bytes_effective") or []
                if _req and _eff and list(_req) != list(_eff):
                    rec["host_pack_clamped"] = {
                        "requested_gib": [round(v / (1 << 30), 2)
                                          for v in _req],
                        "effective_gib": [round(v / (1 << 30), 2)
                                          for v in _eff]}
            except Exception:
                pass
        if result is not None:
            rec["classification"] = result.get("classification")
            rec["k"] = result.get("cohort_k", 1)
            rec["n_forward_steps"] = result.get("n_forward_steps")
            rec["emitted_tokens"] = result.get("emitted_tokens")
            rec["row_shas"] = {
                r.get("prompt_index"): r.get("token_ids_sha256")
                for r in (result.get("rows") or [])}
            if not groups:
                # sequential unit: single prompt stream -- sha convention
                # is default json.dumps separators, matching the committed
                # Phase-4 token-sha-manifest.json values.
                toks = result.get("generated_token_ids") or []
                rec["row_shas"] = {ui: hashlib.sha256(json.dumps(
                    [int(t) for t in toks]).encode()).hexdigest()}
                rec["n_tokens"] = len(toks)
            rec["dedup"] = result.get("dedup")
            rec["cohort_scoped_counters"] = result.get(
                "cohort_scoped_counters") or result.get(
                "prompt_scoped_counters")
            rec["byte_accounting"] = result.get("byte_accounting")
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


def selfcheck():
    problems = []
    if len(PROMPTS) != 8:
        problems.append(f"expected 8 prompts, got {len(PROMPTS)}")
    if PROMPTS[7] != PROMPTS[0]:
        problems.append("q7 must be a verbatim repeat of q0")

    required = {"arm_id", "expert_store", "cache_dtype", "eviction_policy",
                "host_cache_mode", "host_pack_gpu0_bytes",
                "host_pack_gpu1_bytes", "budget_bytes", "lanes",
                "queue_depth", "n_tokens", "cache_reset", "trace_requests",
                "prompts_json", "cohort"}
    seen_ids = set()
    for arm in ARMS:
        aid = arm.get("arm_id", "?")
        missing = required - set(arm)
        if missing:
            problems.append(f"{aid}: missing keys {sorted(missing)}")
            continue
        if aid in seen_ids:
            problems.append(f"duplicate arm_id {aid}")
        seen_ids.add(aid)
        if arm["cache_dtype"] != "fp4":
            problems.append(f"{aid}: all arms must be fp4 (a2 config)")
        if arm["eviction_policy"] != "lru":
            problems.append(f"{aid}: all arms must be lru (a2 config)")
        co = arm["cohort"]
        if co is not None:
            groups = co.get("groups")
            if not groups or not all(groups):
                problems.append(f"{aid}: empty cohort groups")
            flat = [i for g in (groups or []) for i in g]
            if sorted(flat) != list(range(len(PROMPTS))):
                problems.append(
                    f"{aid}: cohort groups must cover all 8 prompts "
                    f"exactly once, got {sorted(flat)}")
            if co.get("pad_to") != "max":
                problems.append(f"{aid}: cohort arms require pad_to=max "
                                "(identical padded inputs to c1)")
        if len(arm["prompts_json"]) not in (1, len(PROMPTS)):
            problems.append(f"{aid}: prompt count mismatch")
    # c1 must exist: it is the padded sequential reference.
    c1 = next((a for a in ARMS if a["arm_id"] == "c1"), None)
    if c1 is None or not c1.get("cohort"):
        problems.append("c1 arm missing or not cohort-mode")
    elif c1["cohort"]["groups"] != [[i] for i in range(8)]:
        problems.append("c1 must be 8 singleton groups (sequential padded "
                        "reference)")

    envs = {}
    for arm in ARMS:
        try:
            envs[arm["arm_id"]] = arm_env(arm)
        except Exception as exc:
            problems.append(f"{arm['arm_id']}: arm_env raised {exc!r}")
    run_ids = {e.get("NATIVE_RUN_ID") for e in envs.values()}
    if len(run_ids) > 1:
        problems.append(f"NATIVE_RUN_ID differs across arms: {run_ids}")
    print("[selfcheck] resolved per-arm env diff (vs c1):")
    ref_env = envs.get("c1", {})
    for aid, env in envs.items():
        knob_keys = sorted(k for k in env if k.startswith("NATIVE_"))
        diff = {k: env[k] for k in knob_keys
                if env.get(k) != ref_env.get(k)
                and k not in ("NATIVE_ARM_ID", "NATIVE_SOURCE_TREE",
                              "NATIVE_COMMIT", "NATIVE_RUN_ID")}
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
    mounts = [p for p in (
        Path("/kaggle/input/deepseek-v4-flash-0731-shards"),
        Path("/kaggle/input/datasets/nivind/deepseek-v4-flash-0731-shards"))
        if p.is_dir()]
    log(f"[p0] shard dataset mount: {mounts[0] if mounts else 'none'}"
        " (authoritative gate is the post-wait bail)")

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
    r = sh(["cmake", "--build", str(BUILD), "--target", "dee_test_assets",
            "-j", "2"], timeout=1200)
    check("dee_test_assets build", r["rc"] == 0, r["err"][-200:])
    if r["rc"] != 0:
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

    # ---------------- P2 store assembly ----------------
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
    # Shards missing => the runner would fall back to a ~153 GiB HF
    # download mid-arm and burn the session.  Bail instead.
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
            else:
                log(f"bucket {b:02d} assembled")
    check("46 segments assembled", not errors,
          "; ".join(errors[:5]) or f"{time.time() - t0:.0f}s")
    if errors:
        bail("FAIL_CLOSED"); return

    def seal_segment(entry):
        src = STORE_ROOT / entry["file"]
        h = hashlib.sha256()
        with src.open("rb") as fh:
            while True:
                block = fh.read(64 << 20)
                if not block:
                    break
                h.update(block)
        return (entry["bucket"], h.hexdigest() == entry["sha256"].lower())

    seal_bad = []
    with ThreadPoolExecutor(max_workers=CP_WORKERS) as ex:
        for b, ok in ex.map(seal_segment, segs):
            if not ok:
                seal_bad.append(f"b{int(b):02d}")
    check("46 segment seals verified", not seal_bad,
          f"bad={seal_bad[:5]}")
    if seal_bad:
        bail("FAIL_INTEGRITY"); return

    # ---------------- P3 arm loop ----------------
    # c0 first (cheap anchor), then c1 (reference), then the K sweep.
    order = sorted(ARMS, key=lambda a: (
        0 if a["arm_id"] == "c0_anchor" else
        1 if a["arm_id"] == "c1" else 2))
    for arm in order:
        report["runs"].update(run_arm(arm))
        write_report()

    # ---------------- P4 verdict ----------------
    # Load the committed Phase-4 a2 token shas for the c0 anchor.
    p4_shas = {}
    try:
        p4_shas = json.loads(P4_TOKEN_MANIFEST.read_text())
        check("p4 token manifest readable", True,
              f"{len(p4_shas)} prompts")
        # Stage it into the harvest bundle so tools/phase5/p5_verify.py
        # can resolve the c0 anchor offline.
        (OUT / "token-sha-manifest-p4ref.json").write_bytes(
            P4_TOKEN_MANIFEST.read_bytes())
    except Exception as exc:
        check("p4 token manifest readable", False, repr(exc)[:120])

    def accepted(rec):
        # rc + classification + full token budget: a systematically
        # truncated run would otherwise pass on equal shas alone.
        if not (rec.get("rc") == 0
                and rec.get("classification") == "ACCEPT_CORRECTNESS"):
            return False
        k = int(rec.get("k") or 1)
        if k > 1 or rec.get("emitted_tokens") is not None:
            return int(rec.get("emitted_tokens") or -1) == N_TOKENS * k
        return int(rec.get("n_tokens") or -1) == N_TOKENS

    all_ok = True
    # c0 anchor: unpadded q0 must reproduce the committed Phase-4 a2 sha.
    c0 = report["runs"].get("c0_anchor-q0", {})
    exp_q0 = ((p4_shas.get("q0") or {}).get("a2_fp4"))
    ok = (accepted(c0) and exp_q0
          and c0.get("row_shas", {}).get(0) == exp_q0)
    check("c0 anchor == phase4 a2 q0", ok,
          f"got={str(c0.get('row_shas', {}).get(0))[:12]} "
          f"exp={str(exp_q0)[:12]}")
    all_ok = all_ok and ok

    # c1: padded sequential reference -- every row must be complete.
    c1_rows = {}
    for ui in range(8):
        rec = report["runs"].get(f"c1-c{ui}", {})
        ok = accepted(rec)
        sha = (rec.get("row_shas") or {}).get(ui)
        check(f"c1-c{ui} accepted", ok and bool(sha),
              f"cls={rec.get('classification')} sha={str(sha)[:12]}")
        all_ok = all_ok and ok and bool(sha)
        c1_rows[ui] = sha

    # Cohort arms: every row's sha must equal c1's same-prompt sha.
    for arm in ARMS:
        aid = arm["arm_id"]
        if aid in ("c0_anchor", "c1"):
            continue
        for ci, grp in enumerate(arm["cohort"]["groups"]):
            rec = report["runs"].get(f"{aid}-c{ci}", {})
            ok = accepted(rec)
            check(f"{aid}-c{ci} accepted", ok,
                  f"cls={rec.get('classification')} K={rec.get('k')}")
            all_ok = all_ok and ok
            row_shas = rec.get("row_shas") or {}
            for prompt_index in grp:
                mine = row_shas.get(prompt_index)
                base = c1_rows.get(prompt_index)
                same = bool(mine) and bool(base) and mine == base
                check(f"{aid}-c{ci} row p{prompt_index} == c1 exact",
                      same,
                      f"cohort={str(mine)[:12]} c1={str(base)[:12]}")
                all_ok = all_ok and same

    report["verdict"] = ("PASS" if all_ok
                         and all(c["ok"] for c in report["checks"])
                         else "FAIL")
    write_report()
    print("P5 VERDICT:", report["verdict"], flush=True)


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        sys.exit(0 if selfcheck() else 1)
    try:
        main()
    except Exception as exc:
        report["verdict"] = f"ERROR: {exc!r}"[:300]
        try:
            (OUT / "p5_report.json").write_text(
                json.dumps(report, indent=1))
        except Exception:
            pass
    sys.exit(0)
