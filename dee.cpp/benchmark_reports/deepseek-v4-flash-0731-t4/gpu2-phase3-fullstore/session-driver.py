"""GPU 2/2 -- Phase-3 arbitrary-prompt full-store exact inference.

Question answered: can dee.cpp serve ARBITRARY prompts over the complete
46-bucket dee4-v4-segmented expert store (11,776 records, 146.6 GiB) with
byte-exact behavior vs the authoritative safetensors backend?

Arms (same kernel, same prompts):
  A0  expert_store=safetensors   -- mounted checkpoint shards (reference)
  A1  expert_store=dee4_segmented -- the published full store under
      /tmp/dee4-full (candidate).  open() seals every segment's sha256.

Accept: per prompt, A1's generated_token_ids, decoded_text, and
routed_experts journal sha256 are IDENTICAL to A0's; A1 engine reports
backend=dee4_segmented, zero lookup_failures, all source reads
contiguous, 43 layers executed, route journal complete.
Reject: any divergence, fail-closed miss, seal-verification failure, or
nonfinite output.

Phases:
  P0  gate: 2x T4 SM75, /tmp disk, dataset mount probe
  P1  clone research/phase3-gpu2-inference @ PINNED + CUDA build +
      ctest test_dee4_segmented (segmented-reader mechanism gate)
  P2  store assembly: wait for async dataset mounts, then copy the 46
      mounted segment files into /tmp/dee4-full/segments/; size-verify
      vs the segment table
  P3  arm loop: per prompt {A0, A1} at 16 tokens
  P4  cross-arm comparison + verdict

Pure-ASCII source only (Kaggle push transcodes UTF-8 -- v47/v66 hazard);
prompts are written with  escapes.
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
BRANCH = "research/phase3-gpu2-inference"
COMMIT = "40e2050407dbf2c01a59a4cfbf937477118010ce"
ROOT = Path("/tmp/dee-src")
DEE = ROOT / "dee.cpp"
BUILD = DEE / "build-kaggle"
WORK = Path("/kaggle/working")
OUT = WORK / "gpu2-out"
RUNNER = (DEE / "kaggle" / "deepseek-v4-flash-0731"
          / "deepseek_v4_native_generate.py")

INDEX_DS = "nivind/dee4-p3-full-u2081ada5e37e-index"
SEG_PREFIX = "dee4-p3-full-u2081ada5e37e-b"
N_BUCKETS = 46
STORE_ROOT = Path("/tmp/dee4-full")
N_TOKENS = 16
# The 46 bucket datasets are MOUNTED (probe-verified 2026-09-13: all 48
# dataset_sources attach, every segment at the right size).  P2 SYMLINKS
# them into the store tree -- a 146 GiB /tmp copy trips the worker disk
# watchdog (v5/v6/v8/v9 all hard-killed that way); the C++ store mmaps
# the mounted files through the links.
CP_WORKERS = 4

_B = "<" + chr(0xFF5C) + "begin" + chr(0x2581) + "of" + chr(0x2581) \
    + "sentence" + chr(0xFF5C) + ">"
_A = "<" + chr(0xFF5C) + "Assistant" + chr(0xFF5C) + ">"
PROMPTS = [
    _B + "Explain in one sentence why the sky is blue." + _A,
    _B + "Write a Python function that returns the nth Fibonacci number."
        + _A,
    _B + "List three causes of the French Revolution." + _A,
]

report = {"job": "gpu2 phase-3 full-store arbitrary-prompt inference",
          "commit": COMMIT, "checks": [], "runs": {}}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [gpu2] {msg}", flush=True)


def write_report():
    (OUT / "gpu2_report.json").write_text(json.dumps(report, indent=1))


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


def run_arm(arm):
    """One runner process feeding ALL prompts through a single engine
    build + single store open (the multi-prompt mode).  Returns
    {run_key: rec}."""
    env = dict(os.environ)
    env.update({
        # Same RUN_ID across arms AND prompts: run_id is embedded in every
        # journal record's canonical payload; per-prompt journal FILES keep
        # prompts apart, so identical routes produce identical bytes.
        "NATIVE_RUN_ID": "gpu2-fullstore",
        "NATIVE_N_TOKENS": str(N_TOKENS),
        "NATIVE_EXPERT_STORE": ("safetensors" if arm == "a0"
                                else "dee4_segmented"),
        "NATIVE_PROMPTS_JSON": json.dumps(PROMPTS),
        "NATIVE_CACHE_DTYPE": "fp16",
        "NATIVE_PROFILE": "0",
        # Hand the runner THIS pinned tree + build (it would otherwise
        # re-clone its own hardcoded freebuff branch, which lacks the
        # dee4_segmented store entirely).
        "NATIVE_SOURCE_TREE": str(ROOT),
        "NATIVE_COMMIT": COMMIT,
        "PYTHONPATH": str(DEE),
    })
    if arm == "a0":
        # Safetensors now gather-materializes via per-region pread
        # (SafetensorsExpertStore::materialize) -- same record bytes as the
        # memcpy fill, no FUSE page faults.  v12 showed lanes=1 memcpy is
        # ~60x slower over the dataset mount (4.7s/record, 2.4 MiB/s).
        env["NATIVE_SOURCE_READ_LANES"] = "4"
    if arm == "a1":
        env["NATIVE_DEE4_SEGMENTED_STORE"] = str(STORE_ROOT)
        # lanes>1 routes fills through materialize()/pread instead of a
        # raw memcpy out of the segment mmap.  v10 showed the memcpy path
        # cannot finish even one prompt in 2h over FUSE mounts (~3264 page
        # faults per record); pread reads identical bytes.
        env["NATIVE_SOURCE_READ_LANES"] = "4"
        # The store is sealed ONCE in P2 below; the per-engine-open O(146
        # GiB) mmap-fault hash pass is the other half of the v10/v11 stall
        # (two engines => two silent seals before decode ever starts).
        env["DEE4_STORE_SKIP_SEAL"] = "1"
    # Isolation: a crashed arm must not leave the PREVIOUS arm's outputs
    # behind to be misattributed.  Delete before every run.
    for pat in ("native-generate-result*.json", "native-generate-all.json",
                "routed_experts*.jsonl", "generated_checkpoint*.jsonl",
                "error.txt", "result.json", "integrity.json",
                "environment.json", "run_config.json", "profile.json",
                "memory.json", "progress.log", "dee4-*.json",
                "dee4-*.jsonl", "p2.2-dee4-evidence.json"):
        for stale in WORK.glob(pat):
            stale.unlink()
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, str(RUNNER)], env=env,
                           capture_output=True, text=True, timeout=14400,
                           cwd=str(WORK))
    except subprocess.TimeoutExpired as exc:
        wall = round(time.time() - t0, 1)
        log(f"{arm} TIMEOUT wall={wall}s")
        # TimeoutExpired carries BYTES even with text=True (CPython quirk)
        # -- decode or the whole run log is lost (that's why v11's
        # log-a1.txt was empty).
        def _txt(x):
            if isinstance(x, bytes):
                return x.decode("utf-8", "replace")
            return x or ""
        out_txt = _txt(exc.stdout)
        err_txt = _txt(exc.stderr)
        (OUT / f"log-{arm}.txt").write_text(
            out_txt[-200000:] + "\n---STDERR---\n" + err_txt[-50000:])
        return {f"{arm}-q{qi}": {"arm": arm, "prompt_index": qi,
                                 "rc": None, "wall_s": wall,
                                 "classification": "TIMEOUT"}
                for qi in range(len(PROMPTS))}
    wall = round(time.time() - t0, 1)
    log(f"{arm} exit={p.returncode} wall={wall}s")
    (OUT / f"log-{arm}.txt").write_text(
        (p.stdout or "")[-200000:] + "\n---STDERR---\n"
        + (p.stderr or "")[-50000:])
    recs = {}
    for qi in range(len(PROMPTS)):
        tag = f"{arm}-q{qi}"
        rec = {"arm": arm, "prompt_index": qi, "rc": p.returncode,
               "wall_s": wall}
        res_path = WORK / f"native-generate-result-q{qi}.json"
        journal_path = WORK / f"routed_experts-q{qi}.jsonl"
        cp_path = WORK / f"generated_checkpoint-q{qi}.jsonl"
        if cp_path.is_file():
            (OUT / f"checkpoint-{tag}.jsonl").write_bytes(
                cp_path.read_bytes())
        if res_path.is_file():
            result = json.loads(res_path.read_text())
            (OUT / f"result-{tag}.json").write_text(
                json.dumps(result, indent=1))
            rec["classification"] = result.get("classification")
            rec["n_tokens"] = len(result.get("generated_token_ids") or [])
            rec["token_ids_sha256"] = hashlib.sha256(json.dumps(
                result.get("generated_token_ids")).encode()).hexdigest()
            rec["decoded_sha256"] = hashlib.sha256(
                (result.get("decoded_text") or "").encode()).hexdigest()
            rec["decoded_text"] = result.get("decoded_text")
            rec["store_backend"] = (result.get("expert_store") or {})
            rec["route_journal"] = result.get("route_journal")
            if journal_path.is_file():
                dst = OUT / f"routed_experts-{tag}.jsonl"
                dst.write_bytes(journal_path.read_bytes())
                rec["journal_sha256"] = sha256_path(dst)
        else:
            rec["classification"] = "NO_RESULT"
        recs[tag] = rec
    return recs


def main():
    OUT.mkdir(exist_ok=True)
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
    r = sh(["git", "clone", "--branch", BRANCH, "--single-branch", REPO,
            str(ROOT)], timeout=900)
    check("clone", r["rc"] == 0, r["err"][-200:])
    if r["rc"] != 0:
        bail("FAIL_CLOSED"); return
    r = sh(["git", "-C", str(ROOT), "checkout", "--quiet", COMMIT],
           timeout=120)
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True).strip()
    check("pinned commit", r["rc"] == 0 and head == COMMIT, head[:12])
    if r["rc"] != 0 or head != COMMIT:
        bail("FAIL_CLOSED"); return
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
    # Mounts attach asynchronously: the GPU script starts before all 48
    # dataset_sources are provisioned (v7 failed this way).  Poll for every
    # required path under either base for up to ~25 min.
    def ds_path(slug, name):
        # Strict: only counts as mounted when the needed FILE exists; a
        # half-attached dataset (dir present, blob still syncing) keeps
        # the wait loop going instead of racing into the copy phase.
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
        # SYMLINK the mounted segment into the store tree -- do NOT copy.
        # v5/v6/v8/v9 were all hard-killed by the worker disk watchdog
        # ("high /tmp usage") once ~tens of GiB landed in /tmp; links cost
        # bytes not GiB and the store mmaps through them.
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
    # and compare to the metadata table.  This is the integrity evidence
    # (replaces the in-engine open-time seal, which the a1 arm now skips
    # via DEE4_STORE_SKIP_SEAL -- hashing the mmap itself page-faults over
    # FUSE and can take far longer than sequential pread).
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
    # Decisive evidence for the v10 stall: the lanes<=1 path memcpy()s out
    # of the segment mmap; if FUSE page faults are the wall this shows it
    # before any GPU time is spent.  ~8 records x 12.75 MiB each way.
    try:
        import mmap as _mmap
        import random as _rnd
        segp = STORE_ROOT / "segments" / "experts-bucket-45.dee4"
        fd = os.open(str(segp), os.O_RDONLY)
        rng = _rnd.Random(0)
        offs = [rng.randrange(256) * 13_369_344 for _ in range(8)]
        mm = _mmap.mmap(fd, 0, access=_mmap.ACCESS_READ)
        tb = time.monotonic()
        for off in offs:
            _ = mm[off:off + 13_369_344]
        mmap_ms = (time.monotonic() - tb) / len(offs) * 1000
        mm.close()
        tb = time.monotonic()
        for off in offs:
            os.pread(fd, 13_369_344, off)
        pread_ms = (time.monotonic() - tb) / len(offs) * 1000
        os.close(fd)
        report["microbench"] = {
            "mmap_ms_per_record": round(mmap_ms, 1),
            "pread_ms_per_record": round(pread_ms, 1)}
        check("read-path microbench", True,
              f"mmap={mmap_ms:.0f}ms/rec pread={pread_ms:.0f}ms/rec")
    except Exception as exc:
        check("read-path microbench", False, repr(exc)[:150])

    # ---------------- P3 arm loop: one process per arm, all prompts ----
    # a1 FIRST: the segmented-store arm is the one under test; if it still
    # fails we learn in minutes instead of after a0's ~30 min baseline.
    for arm in ("a1", "a0"):
        report["runs"].update(run_arm(arm))
        write_report()

    # ---------------- P4 verdict ----------------
    per_prompt = []
    for qi in range(len(PROMPTS)):
        a0 = report["runs"].get(f"a0-q{qi}", {})
        a1 = report["runs"].get(f"a1-q{qi}", {})
        # rc==0 is NOT success (the runner exits 0 on terminal rejection);
        # require the full acceptance signal on both arms and non-null
        # evidence hashes so a double-failure cannot compare equal.
        def accepted(r):
            return (r.get("rc") == 0
                    and r.get("classification") == "ACCEPT_CORRECTNESS"
                    and r.get("n_tokens") == N_TOKENS)
        shas = ("token_ids_sha256", "decoded_sha256", "journal_sha256")
        same = (
            accepted(a0) and accepted(a1)
            and all(a0.get(k) and a1.get(k) for k in shas)
            and all(a0.get(k) == a1.get(k) for k in shas))
        per_prompt.append(same)
        check(f"q{qi} a0==a1 exact", same,
              f"tok={(a1.get('token_ids_sha256') or '')[:12]} "
              f"journal={(a1.get('journal_sha256') or '')[:12]}")
        # A1 store-behavior gates -- non-vacuous: missing stats = dirty.
        sb = a1.get("store_backend") or {}
        per_engine_ok = isinstance(sb, dict) and len(sb) > 0
        if isinstance(sb, dict):
            for key, eng in sb.items():
                if isinstance(eng, dict):
                    if eng.get("backend") != "dee4_segmented":
                        per_engine_ok = False
                    if int(eng.get("lookup_failures", -1)) != 0:
                        per_engine_ok = False
        check(f"q{qi} a1 store clean", per_engine_ok,
              json.dumps(sb)[:160])
    report["verdict"] = ("PASS" if all(per_prompt)
                         and all(c["ok"] for c in report["checks"])
                         else "FAIL")
    (OUT / "gpu2_report.json").write_text(json.dumps(report, indent=1))
    print("GPU2 VERDICT:", report["verdict"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report["verdict"] = f"ERROR: {exc!r}"[:300]
        try:
            (OUT / "gpu2_report.json").write_text(
                json.dumps(report, indent=1))
        except Exception:
            pass
    sys.exit(0)
