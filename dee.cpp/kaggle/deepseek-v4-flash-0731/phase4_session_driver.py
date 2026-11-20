"""Phase-4 cache-hierarchy campaign driver (dual-T4, Kaggle).

Question answered: how do the bounded cache hierarchy's arms (host-pack
bypass / fp16+rank_priority / fp4+lru / fp4+rank_priority) compare on wall
time, tier hit rates, and per-request cache events -- while every arm still
produces byte-identical routes and tokens?

Arms (same kernel, same prompts, same RUN_ID, same device split -- the
route-journal hash payload is identical across arms by construction):
  a0_bypass  host_cache_mode=bypass (1-record bounce buffer), lanes=1,
             cold reset per prompt -- the honest baseline (pack tier off)
  a1_asis    fp16 + rank_priority + lru pack 8.5 GiB/GPU -- the Phase-3
             committed config, i.e. the as-is reference for cross-arm gates
  a2_fp4     fp4 residency + lru policy -- capacity + repaired policy
  a3_fp4p    fp4 + rank_priority -- isolates the policy repair inside fp4
             (a2-vs-a3 is the pure policy delta at fp4 capacity)

Accept: per arm, per prompt: rc==0 AND classification=="ACCEPT_CORRECTNESS"
AND n_tokens==128 AND store backend==dee4_segmented with
lookup_failures==0; cross-arm journal sha256 + token-ids sha256 equality
vs the a1_asis reference (non-null on both sides).
Reject: any divergence, fail-closed miss, seal failure, or nonfinite output.

Phases:
  P0  gate: 2x T4 SM75, /tmp disk, dataset mount probe
  P1  clone research/phase4-cache-hierarchy (+ optional pin) + CUDA build +
      ctest test_dee4_segmented + pydee build_ext
  P2  store assembly: wait for async dataset mounts, symlink the 46 mounted
      segment files into /tmp/dee4-full/segments/, driver-side sha256 seal,
      mmap-vs-pread microbench
  P3  arm loop: one runner process per arm (all 8 prompts, 128 tokens)
      with a ~60s watcher heartbeat; per-arm artifacts retrieved into
      p4-out/{arm_id}/
  P4  cross-arm verdict + p4_report.json (always exits 0)

Pure-ASCII source only (Kaggle push transcodes UTF-8 -- the v47/v66
mojibake hazard); fullwidth prompt delimiters are built via chr().

--selfcheck validates ARMS/env construction + prompt assembly + artifact
glob patterns and prints the resolved per-arm env diff WITHOUT launching
anything (no clone, no GPU, no subprocess).
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
# Optional pin: empty = branch head (the arm matrix changes runner behavior,
# so the campaign pins once the parallel bindings land).
COMMIT = os.environ.get("P4_PINNED_COMMIT", "")
ROOT = Path(os.environ.get("P4_SOURCE_ROOT", "/tmp/dee-p4-src"))
DEE = ROOT / "dee.cpp"
BUILD = DEE / "build-kaggle"
WORK = Path("/kaggle/working")
OUT = WORK / "p4-out"
RUNNER = (DEE / "kaggle" / "deepseek-v4-flash-0731"
          / "deepseek_v4_native_generate.py")

INDEX_DS = "nivind/dee4-p3-full-u2081ada5e37e-index"
SEG_PREFIX = "dee4-p3-full-u2081ada5e37e-b"
N_BUCKETS = 46
STORE_ROOT = Path("/tmp/dee4-full")
N_TOKENS = 128
# 8 prompts x 128 tokens is ~75-90 min cold; cached arms are faster.
# Budget 3 h per arm -- a0's bypass path is the slowest (every VRAM miss
# is a store read through the bounce buffer).
ARM_TIMEOUT_S = int(os.environ.get("P4_ARM_TIMEOUT_S", str(3 * 3600)))
HEARTBEAT_S = 60
# run_id is inside the route-journal canonical hash payload; it MUST stay
# constant across arms (and the device split must stay dual-GPU) or the
# journals can never hash-compare.  arm_id travels separately as metadata.
RUN_ID = "p4-campaign"
REFERENCE_ARM = "a1_asis"
CP_WORKERS = 4

# Byte constants for the arm budgets.
ONE_RECORD_BYTES = 13_369_344            # one packed dee4 expert record
PACK_8_5_GIB = int(8.5 * (1 << 30))      # Phase-3 committed pack/GPU
VRAM_3_5_GIB = 3584 << 20                # Phase-3 committed VRAM arena

# ---- 8-prompt Phase-4 workload (tools/phase4/workload_spec.md) ---------
# Fullwidth delimiters built via chr() -- NEVER literal (push transcodes).
_B = ("<" + chr(0xFF5C) + "begin" + chr(0x2581) + "of" + chr(0x2581)
      + "sentence" + chr(0xFF5C) + ">")
_A = "<" + chr(0xFF5C) + "Assistant" + chr(0xFF5C) + ">"
_TEXTS = [
    # q0 factual
    "Explain how mRNA vaccines work, from injection to immune memory.",
    # q1 code
    "Write a Python function that finds the longest common subsequence "
    "of two strings, and explain its time and space complexity.",
    # q2 math/reasoning
    "A fair coin is flipped until two consecutive heads appear. What is "
    "the expected number of flips? Show your reasoning step by step.",
    # q3 long-form/open
    "Write the opening paragraph of a hard science fiction novel about "
    "a generation ship whose crew discovers the laws of physics are "
    "slightly different two light-years from Earth.",
    # r0-r2 regression (sealed Phase-3 truth)
    "Explain in one sentence why the sky is blue.",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three causes of the French Revolution.",
]
PROMPTS = [_B + t + _A for t in _TEXTS]
# rep0: verbatim repeat of q0 -- the isolated cross-prompt reuse probe.
PROMPTS.append(PROMPTS[0])

ARMS = [
    {
        # a0: honest baseline -- host pack reduced to a 1-record bounce
        # buffer (bypass) with lanes forced to 1, cold reset per prompt.
        # Isolates the host-pack tier's contribution; the VRAM arena keeps
        # the normal 3.5 GiB budget so this is "pack off", not "cache off".
        "arm_id": "a0_bypass",
        "expert_store": "dee4_segmented",
        "cache_dtype": "fp16",
        "eviction_policy": "rank_priority",
        "host_cache_mode": "bypass",
        "host_pack_gpu0_bytes": ONE_RECORD_BYTES,
        "host_pack_gpu1_bytes": ONE_RECORD_BYTES,
        "budget_bytes": VRAM_3_5_GIB,
        "lanes": 1,
        "queue_depth": 6,
        "n_tokens": N_TOKENS,
        "cache_reset": "cold",
        "trace_requests": "1",
        "prompts_json": PROMPTS,
    },
    {
        # a1: as-is reference -- the Phase-3 committed config.  This arm is
        # the cross-arm truth source for journal/token sha equality.
        "arm_id": "a1_asis",
        "expert_store": "dee4_segmented",
        "cache_dtype": "fp16",
        "eviction_policy": "rank_priority",
        "host_cache_mode": "lru",
        "host_pack_gpu0_bytes": PACK_8_5_GIB,
        "host_pack_gpu1_bytes": PACK_8_5_GIB,
        "budget_bytes": VRAM_3_5_GIB,
        "lanes": 4,
        "queue_depth": 6,
        "n_tokens": N_TOKENS,
        "cache_reset": "warm",
        "trace_requests": "1",
        "prompts_json": PROMPTS,
    },
    {
        # a2: fp4 residency + repaired (lru) eviction policy -- capacity
        # knee (~281 slots) plus pure recency.
        "arm_id": "a2_fp4",
        "expert_store": "dee4_segmented",
        "cache_dtype": "fp4",
        "eviction_policy": "lru",
        "host_cache_mode": "lru",
        "host_pack_gpu0_bytes": PACK_8_5_GIB,
        "host_pack_gpu1_bytes": PACK_8_5_GIB,
        "budget_bytes": VRAM_3_5_GIB,
        "lanes": 4,
        "queue_depth": 6,
        "n_tokens": N_TOKENS,
        "cache_reset": "warm",
        "trace_requests": "1",
        "prompts_json": PROMPTS,
    },
    {
        # a3: fp4 + rank_priority -- isolates the policy repair inside fp4:
        # a2-vs-a3 is the pure eviction-policy delta at fp4 capacity.
        "arm_id": "a3_fp4p",
        "expert_store": "dee4_segmented",
        "cache_dtype": "fp4",
        "eviction_policy": "rank_priority",
        "host_cache_mode": "lru",
        "host_pack_gpu0_bytes": PACK_8_5_GIB,
        "host_pack_gpu1_bytes": PACK_8_5_GIB,
        "budget_bytes": VRAM_3_5_GIB,
        "lanes": 4,
        "queue_depth": 6,
        "n_tokens": N_TOKENS,
        "cache_reset": "warm",
        "trace_requests": "1",
        "prompts_json": PROMPTS,
    },
]

# Artifacts the runner may leave behind in /kaggle/working; a crashed arm
# must not leave the PREVIOUS arm's outputs to be misattributed.
STALE_PATTERNS = (
    "native-generate-result*.json", "native-generate-all.json",
    "routed_experts*.jsonl", "generated_checkpoint*.jsonl",
    "cache_events*.jsonl", "arm_config*.json",
    "error.txt", "result*.json", "integrity*.json",
    "environment*.json", "run_config*.json", "profile*.json",
    "memory*.json", "progress.log", "dee4-*.json",
    "dee4-*.jsonl", "p2.2-dee4-evidence.json")

# Per-prompt artifacts retrieved into p4-out/{arm_id}/ after each arm.
PROMPT_ARTIFACT_SUFFIXES = (
    "result", "integrity", "memory", "profile",
    "run_config", "environment", "arm_config")
PROMPT_ARTIFACT_JSONL = ("routed_experts", "generated_checkpoint",
                         "cache_events")

report = {"job": "phase-4 cache-hierarchy arm-matrix campaign",
          "branch": BRANCH, "commit": COMMIT or "(branch head)",
          "run_id": RUN_ID, "reference_arm": REFERENCE_ARM,
          "checks": [], "runs": {}, "arms": {}}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [p4] {msg}", flush=True)


def write_report():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p4_report.json").write_text(json.dumps(report, indent=1))


def check(name, ok, detail=""):
    report["checks"].append({"check": name, "ok": bool(ok),
                             "detail": detail})
    log(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    write_report()


def bail(verdict):
    """Fail-closed early exit: persist verdict BEFORE returning."""
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
    """Resolve the runner environment for one arm.  Every NATIVE_* knob the
    runner honors is set explicitly so committed run_config.json values can
    never silently override the arm."""
    env = dict(os.environ)
    env.update({
        # Constant across arms AND prompts: run_id is embedded in every
        # journal record's canonical payload; per-prompt journal FILES keep
        # prompts apart, so identical routes produce identical bytes.
        "NATIVE_RUN_ID": RUN_ID,
        # arm_id is METADATA only -- the runner writes it to
        # arm_config/run_config/result/integrity but never into the journal.
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
        # Fixed-length workload: no eos early-stop, so every prompt yields
        # exactly n_tokens decode steps (acceptance gate is n_tokens==128;
        # identical across arms so exactness comparison is unaffected).
        "NATIVE_IGNORE_EOS": "1",
        # trace_requests needs profile_stages; the runner would auto-enable
        # it loudly anyway, but set it explicitly so the resolved manifest
        # shows intent rather than fallback.
        "NATIVE_PROFILE": ("1" if arm["trace_requests"] == "1" else "0"),
        "NATIVE_PROMPTS_JSON": json.dumps(arm["prompts_json"]),
        # Hand the runner THIS pinned tree + build (it would otherwise
        # re-clone its own hardcoded branch).
        "NATIVE_SOURCE_TREE": str(ROOT),
        "PYTHONPATH": str(DEE),
    })
    if COMMIT:
        env["NATIVE_COMMIT"] = COMMIT
    if arm["expert_store"] == "dee4_segmented":
        env["NATIVE_DEE4_SEGMENTED_STORE"] = str(STORE_ROOT)
        # The store is sealed ONCE in P2; per-engine-open O(146 GiB)
        # mmap-fault sealing is the v10/v11 stall.  pread reads identical
        # bytes, so lanes>1 also routes fills through materialize()/pread.
        env["DEE4_STORE_SKIP_SEAL"] = "1"
    return env


def run_arm(arm):
    """One runner process feeding ALL prompts through a single engine
    build + single store open.  Returns {run_key: rec}."""
    arm_id = arm["arm_id"]
    n_prompts = len(arm["prompts_json"])
    arm_out = OUT / arm_id
    arm_out.mkdir(parents=True, exist_ok=True)
    env = arm_env(arm)
    report["arms"][arm_id] = {
        "env": {k: v for k, v in env.items()
                if k.startswith(("NATIVE_", "DEE4_"))},
        "resolved": dict(arm, prompts_json=f"{n_prompts} prompts"),
        "prompts_json_sha256": hashlib.sha256(
            json.dumps(arm["prompts_json"]).encode()).hexdigest(),
    }
    for pat in STALE_PATTERNS:
        for stale in WORK.glob(pat):
            stale.unlink()
    log_path = arm_out / f"log-{arm_id}.txt"
    t0 = time.time()
    deadline = t0 + ARM_TIMEOUT_S
    timed_out = False
    # Watcher-friendly: stream the subprocess log to a file and emit a
    # heartbeat (elapsed + last log tail) every ~60 s of wall.
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(RUNNER)], env=env, cwd=str(WORK),
            stdout=lf, stderr=subprocess.STDOUT, text=True)
        last_hb = 0.0
        while True:
            rc = proc.poll()
            now = time.time()
            if now - last_hb >= HEARTBEAT_S:
                elapsed = now - t0
                log(f"{arm_id} heartbeat elapsed={elapsed:.0f}s "
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
                break
            time.sleep(5)
    wall = round(time.time() - t0, 1)
    if timed_out:
        log(f"{arm_id} TIMEOUT wall={wall}s")
    else:
        log(f"{arm_id} exit={proc.returncode} wall={wall}s")
    # Even after a timeout, harvest whatever per-prompt artifacts completed
    # -- partial evidence is better than a bare TIMEOUT stub.
    rc_val = proc.returncode
    default_cls = "TIMEOUT" if timed_out else "NO_RESULT"
    recs = {}
    for qi in range(n_prompts):
        tag = f"{arm_id}-q{qi}"
        rec = {"arm": arm_id, "prompt_index": qi, "rc": rc_val,
               "wall_s": wall}
        res_path = WORK / f"native-generate-result-q{qi}.json"
        journal_path = WORK / f"routed_experts-q{qi}.jsonl"
        cp_path = WORK / f"generated_checkpoint-q{qi}.jsonl"
        ce_path = WORK / f"cache_events-q{qi}.jsonl"
        # Retrieve every per-prompt evidence file into p4-out/{arm_id}/.
        for base in PROMPT_ARTIFACT_SUFFIXES:
            src = WORK / f"{base}-q{qi}.json"
            if src.is_file():
                (arm_out / src.name).write_bytes(src.read_bytes())
        for src in (cp_path, ce_path):
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
        if journal_path.is_file():
            dst = arm_out / journal_path.name
            dst.write_bytes(journal_path.read_bytes())
            rec["journal_sha256"] = sha256_path(dst)
        if result is not None:
            rec["classification"] = result.get("classification")
            rec["n_tokens"] = len(result.get("generated_token_ids") or [])
            rec["token_ids_sha256"] = hashlib.sha256(json.dumps(
                result.get("generated_token_ids")).encode()).hexdigest()
            rec["decoded_sha256"] = hashlib.sha256(
                (result.get("decoded_text") or "").encode()).hexdigest()
            rec["decoded_text"] = result.get("decoded_text")
            rec["store_backend"] = (result.get("expert_store") or {})
            rec["route_journal"] = result.get("route_journal")
            rec["cache_events"] = result.get("cache_events")
            rec["prompt_scoped_counters"] = result.get(
                "prompt_scoped_counters")
            rec["byte_accounting"] = result.get("byte_accounting")
            rec["total_wall_seconds"] = result.get("total_wall_seconds")
            rec["decode_tok_s"] = result.get("decode_tok_s")
            rec["prefill_ms"] = result.get("prefill_ms")
        elif "classification" not in rec:
            rec["classification"] = default_cls
        recs[tag] = rec
    # Process-level artifacts (same content across prompts within the arm).
    for name in ("native-generate-all.json", "integrity.json",
                 "environment.json", "run_config.json", "profile.json",
                 "memory.json", "progress.log", "error.txt",
                 "dee4-segmented-store.json"):
        src = WORK / name
        if src.is_file():
            (arm_out / src.name).write_bytes(src.read_bytes())
    return recs


def selfcheck():
    """Validate ARMS/env/prompts/artifact patterns WITHOUT launching."""
    problems = []

    # -- prompt assembly -------------------------------------------------
    if len(PROMPTS) != 8:
        problems.append(f"expected 8 prompts, got {len(PROMPTS)}")
    if PROMPTS[7] != PROMPTS[0]:
        problems.append("rep0 must be a verbatim repeat of q0")
    for i, p in enumerate(PROMPTS):
        if not p.startswith(_B) or not p.endswith(_A):
            problems.append(f"prompt {i} missing fullwidth delimiters")
        try:
            p.encode("ascii")
            problems.append(f"prompt {i} unexpectedly pure-ascii "
                            "(delimiters should be non-ascii)")
        except UnicodeEncodeError:
            pass
        if not json.dumps(p).startswith('"'):
            problems.append(f"prompt {i} not json-serializable")
    print("[selfcheck] prompt texts:")
    for i, p in enumerate(PROMPTS):
        inner = p[len(_B):-len(_A)]
        print(f"  q{i}: {len(p)} chars, {len(inner)} inner, "
              f"ascii_inner={inner.isascii()}")

    # -- ARMS validation ---------------------------------------------------
    required = {"arm_id", "expert_store", "cache_dtype", "eviction_policy",
                "host_cache_mode", "host_pack_gpu0_bytes",
                "host_pack_gpu1_bytes", "budget_bytes", "lanes",
                "queue_depth", "n_tokens", "cache_reset", "trace_requests",
                "prompts_json"}
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
        if arm["expert_store"] not in ("safetensors", "dee4", "dee4_trace",
                                      "dee4_segmented"):
            problems.append(f"{aid}: bad expert_store")
        if arm["cache_dtype"] not in ("fp16", "fp4"):
            problems.append(f"{aid}: bad cache_dtype")
        if arm["eviction_policy"] not in ("lru", "rank_priority"):
            problems.append(f"{aid}: bad eviction_policy")
        if arm["host_cache_mode"] not in ("lru", "bypass"):
            problems.append(f"{aid}: bad host_cache_mode")
        if arm["cache_reset"] not in ("warm", "cold"):
            problems.append(f"{aid}: bad cache_reset")
        if arm["trace_requests"] not in ("0", "1"):
            problems.append(f"{aid}: bad trace_requests")
        if not 1 <= arm["lanes"] <= 8:
            problems.append(f"{aid}: lanes out of [1,8]")
        if not 1 <= arm["queue_depth"] <= 256:
            problems.append(f"{aid}: queue_depth out of [1,256]")
        if arm["host_cache_mode"] == "bypass" and (
                arm["host_pack_gpu0_bytes"] != ONE_RECORD_BYTES
                or arm["host_pack_gpu1_bytes"] != ONE_RECORD_BYTES
                or arm["lanes"] != 1):
            problems.append(
                f"{aid}: bypass arm must carry 1-record pack + lanes=1 "
                "(bounce-buffer contract)")
        if len(arm["prompts_json"]) != len(PROMPTS):
            problems.append(f"{aid}: prompt count mismatch")

    if REFERENCE_ARM not in seen_ids:
        problems.append(f"reference arm {REFERENCE_ARM} not in ARMS")

    # -- env construction ----------------------------------------------------
    envs = {}
    for arm in ARMS:
        try:
            envs[arm["arm_id"]] = arm_env(arm)
        except Exception as exc:
            problems.append(f"{arm['arm_id']}: arm_env raised {exc!r}")
    run_ids = {e.get("NATIVE_RUN_ID") for e in envs.values()}
    if len(run_ids) > 1:
        problems.append(f"NATIVE_RUN_ID differs across arms: {run_ids}")
    pj = {e.get("NATIVE_PROMPTS_JSON") for e in envs.values()}
    if len(pj) > 1:
        problems.append("NATIVE_PROMPTS_JSON differs across arms")
    print("[selfcheck] resolved per-arm env diff (vs "
          f"{REFERENCE_ARM}):")
    ref_env = envs.get(REFERENCE_ARM, {})
    for aid, env in envs.items():
        knob_keys = sorted(k for k in env if k.startswith("NATIVE_"))
        diff = {k: env[k] for k in knob_keys
                if env.get(k) != ref_env.get(k)
                and k not in ("NATIVE_ARM_ID", "NATIVE_SOURCE_TREE",
                              "NATIVE_COMMIT", "NATIVE_RUN_ID")}
        print(f"  {aid}: {json.dumps(diff, sort_keys=True)}")

    # -- artifact glob expectations -----------------------------------------
    print("[selfcheck] stale-purge patterns:", len(STALE_PATTERNS))
    for arm in ARMS:
        n = len(arm["prompts_json"])
        expected = [f"native-generate-result-q{i}.json" for i in range(n)]
        expected += [f"routed_experts-q{i}.jsonl" for i in range(n)]
        expected += [f"cache_events-q{i}.jsonl" for i in range(n)]
        expected += [f"arm_config-q{i}.json" for i in range(n)]
        print(f"  {arm['arm_id']}: expects {len(expected)} per-prompt "
              f"artifacts for {n} prompts")
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
    check("tmp >= 500 GiB", free_tmp >= 500 << 30,
          f"{free_tmp / (1 << 30):.0f} GiB")
    mounts = [p for p in (
        Path("/kaggle/input/deepseek-v4-flash-0731-shards"),
        Path("/kaggle/input/datasets/nivind/deepseek-v4-flash-0731-shards"))
        if p.is_dir()]
    check("shard dataset mounted", bool(mounts),
          str(mounts[0]) if mounts else "none")

    # ---------------- P1 clone + build ----------------
    if os.environ.get("P4_SOURCE_ROOT"):
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

    # ---------------- P2 store assembly (mounted datasets -> /tmp) ----
    def ds_path(slug, name):
        # Strict: only counts as mounted when the needed FILE exists.
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
    meta = json.loads((idx_dir / "metadata.json").read_text())
    segs = meta["segments"]
    check("segment table 46", len(segs) == N_BUCKETS, f"{len(segs)}")
    if len(segs) != N_BUCKETS:
        bail("FAIL_CLOSED"); return
    (STORE_ROOT / "segments").mkdir(parents=True, exist_ok=True)
    (STORE_ROOT / "metadata.json").write_bytes(
        (idx_dir / "metadata.json").read_bytes())

    def fetch_bucket(entry):
        # SYMLINK the mounted segment into the store tree -- do NOT copy
        # (a 146 GiB /tmp copy trips the worker disk watchdog).
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
    dt = time.time() - t0
    check("46 segments assembled", not errors,
          "; ".join(errors[:5]) or f"{dt:.0f}s")
    if errors:
        bail("FAIL_CLOSED"); return

    # ---- P2b driver-side seal: sha256 every segment via sequential pread
    # (the in-engine open-time seal is skipped via DEE4_STORE_SKIP_SEAL;
    # hashing the mmap itself page-faults over FUSE).
    def seal_segment(entry):
        src = STORE_ROOT / entry["file"]
        h = hashlib.sha256()
        t0s = time.monotonic()
        with src.open("rb") as fh:
            while True:
                block = fh.read(64 << 20)
                if not block:
                    break
                h.update(block)
        dt_s = time.monotonic() - t0s
        got = h.hexdigest()
        return (entry["bucket"], got == entry["sha256"].lower(),
                f"{entry['file']} {dt_s:.0f}s "
                f"{int(entry['bytes'])/(1<<20)/max(dt_s,1e-9):.0f}MiB/s "
                f"{'match' if got == entry['sha256'].lower() else 'MISMATCH'}")

    seal_t0 = time.time()
    seal_bad = []
    with ThreadPoolExecutor(max_workers=CP_WORKERS) as ex:
        for b, ok, detail in ex.map(seal_segment, segs):
            log(f"seal b{int(b):02d}: {detail}")
            if not ok:
                seal_bad.append(f"b{int(b):02d}")
    seal_dt = time.time() - seal_t0
    check("46 segment seals verified", not seal_bad,
          f"{seal_dt/60:.1f}min bad={seal_bad[:5]}")
    if seal_bad:
        bail("FAIL_INTEGRITY"); return

    # ---- diagnostic microbench: mmap page-fault vs pread on the mount --
    try:
        import mmap as _mmap
        import random as _rnd
        segp = STORE_ROOT / "segments" / "experts-bucket-45.dee4"
        fd = os.open(str(segp), os.O_RDONLY)
        rng = _rnd.Random(0)
        offs = [rng.randrange(256) * ONE_RECORD_BYTES for _ in range(8)]
        mm = _mmap.mmap(fd, 0, access=_mmap.ACCESS_READ)
        tb = time.monotonic()
        for off in offs:
            _ = mm[off:off + ONE_RECORD_BYTES]
        mmap_ms = (time.monotonic() - tb) / len(offs) * 1000
        mm.close()
        tb = time.monotonic()
        for off in offs:
            os.pread(fd, ONE_RECORD_BYTES, off)
        pread_ms = (time.monotonic() - tb) / len(offs) * 1000
        os.close(fd)
        report["microbench"] = {
            "mmap_ms_per_record": round(mmap_ms, 1),
            "pread_ms_per_record": round(pread_ms, 1)}
        check("read-path microbench", True,
              f"mmap={mmap_ms:.0f}ms/rec pread={pread_ms:.0f}ms/rec")
    except Exception as exc:
        report["microbench"] = {"error": repr(exc)[:150]}
        log(f"read-path microbench failed: {exc!r}")

    # ---------------- P3 arm loop: one process per arm, all prompts ----
    # a1 FIRST: it is the reference truth source AND the most Phase-3-like
    # arm; if the baseline is broken we learn in the first ~90 min, not
    # after a0's slower bypass run.
    order = ([a for a in ARMS if a["arm_id"] == REFERENCE_ARM]
             + [a for a in ARMS if a["arm_id"] != REFERENCE_ARM])
    for arm in order:
        report["runs"].update(run_arm(arm))
        write_report()

    # ---------------- P4 verdict ----------------
    per_prompt = []
    ref = REFERENCE_ARM
    n_prompts = len(PROMPTS)

    def accepted(rec, want_tokens):
        # rc==0 is NOT success (the runner exits 0 on terminal rejection);
        # require the full acceptance signal.
        return (rec.get("rc") == 0
                and rec.get("classification") == "ACCEPT_CORRECTNESS"
                and rec.get("n_tokens") == want_tokens)

    for arm in ARMS:
        aid = arm["arm_id"]
        want = arm["n_tokens"]
        arm_ok = True
        for qi in range(len(arm["prompts_json"])):
            rec = report["runs"].get(f"{aid}-q{qi}", {})
            ok = accepted(rec, want)
            # Store-behavior gates -- non-vacuous: missing stats = dirty.
            sb = rec.get("store_backend") or {}
            store_ok = isinstance(sb, dict) and len(sb) > 0
            if isinstance(sb, dict):
                for key, eng in sb.items():
                    if isinstance(eng, dict):
                        if eng.get("backend") != arm["expert_store"]:
                            store_ok = False
                        if int(eng.get("lookup_failures", -1)) != 0:
                            store_ok = False
            check(f"{aid}-q{qi} accepted", ok,
                  f"cls={rec.get('classification')} "
                  f"ntok={rec.get('n_tokens')}")
            check(f"{aid}-q{qi} store clean", store_ok,
                  json.dumps(sb)[:160])
            arm_ok = arm_ok and ok and store_ok
        if aid != ref:
            # Cross-arm exactness vs the reference: identical route journals
            # and token streams (non-null on both sides so a double-failure
            # cannot compare equal).
            for qi in range(min(len(arm["prompts_json"]), n_prompts)):
                mine = report["runs"].get(f"{aid}-q{qi}", {})
                base = report["runs"].get(f"{ref}-q{qi}", {})
                same = (
                    accepted(mine, want) and accepted(base, N_TOKENS)
                    and mine.get("journal_sha256")
                    and base.get("journal_sha256")
                    and mine.get("journal_sha256")
                    == base.get("journal_sha256")
                    and mine.get("token_ids_sha256")
                    == base.get("token_ids_sha256"))
                check(f"{aid}-q{qi} == {ref} exact", same,
                      f"j={str(mine.get('journal_sha256'))[:12]} "
                      f"t={str(mine.get('token_ids_sha256'))[:12]}")
                arm_ok = arm_ok and same
        per_prompt.append(arm_ok)

    report["verdict"] = ("PASS" if all(per_prompt)
                         and all(c["ok"] for c in report["checks"])
                         else "FAIL")
    write_report()
    print("P4 VERDICT:", report["verdict"], flush=True)


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        sys.exit(0 if selfcheck() else 1)
    try:
        main()
    except Exception as exc:
        report["verdict"] = f"ERROR: {exc!r}"[:300]
        try:
            (OUT / "p4_report.json").write_text(
                json.dumps(report, indent=1))
        except Exception:
            pass
    sys.exit(0)
