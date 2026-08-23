"""Kaggle dual-T4: real tokenizer->text generation with the native FP4 FFN.

Routed-expert FFN runs through pydee.Engine.moe_forward_experts (mmap packed
FP4 -> on-GPU dequant -> cuBLAS SwiGLU); tokenizer/attention/KV/router/shared
expert/norm/LM head stay on the sealed DS10 torch path.

Progress + errors are written to /kaggle/working (captured in the output
tarball even when the run fails) so failures are diagnosable without the live
console log.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
COMMIT = os.environ.get("NATIVE_COMMIT", "")
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
N_SHARDS = 48
# /kaggle/temp is NOT present on the dual-T4 "medium" container (verified via
# a disk probe on 2026-08-15); /tmp and / both sit on the ~8 TB root overlay
# with ~1 TiB free, so stage the 167 GB checkpoint there instead.
ROOT = Path("/tmp/dsv4-native-src")
DEE = ROOT / "dee.cpp"
BUILD = DEE / "build-kaggle"
CKPT = Path("/tmp/dsv4-checkpoint")
# When the checkpoint is published as a Kaggle dataset it mounts read-only at
# /kaggle/input/<slug>/; prefer that (no download, no disk quota). The local
# /tmp fallback remains for the 155 GiB-free case.
DATASET_DIR = Path("/kaggle/input/deepseek-v4-flash-0731-shards")
WORK = Path("/kaggle/working")
HEADERS_DIR = (DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers")
CONFIG = (DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/"
          "official-source/inference/config.json")
CANONICAL_PROMPT = (
    "<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>Who is Alan Turing?"
    "<\uFF5CAssistant\uFF5C>")
N_TOKENS = int(os.environ.get("NATIVE_N_TOKENS", "16"))
# Stage 1: raise the per-GPU VRAM expert cache from 512 MiB (~10 experts) to
# 3.5 GiB (~73 FP16 experts) — measured free VRAM after dense + engine is
# ~6.6/4.9 GiB on the two T4s, so 3.5 GiB/GPU keeps clear headroom.
BUDGET_BYTES = int(os.environ.get("NATIVE_BUDGET_BYTES", str(3584 << 20)))
# P2.3 packed FP4 residency: "fp16" (sealed exact path) or "fp4" (packed
# VRAM cache, decode-at-compute, experimental).  Env override for A/B runs.
CACHE_DTYPE = os.environ.get("NATIVE_CACHE_DTYPE", "fp16")
# Stage 1b (v8/v9/v10/v11/v12): host RAM LRU of packed FP4 expert bytes
# (12.6 MB/entry).  The 16-token working set is ~2,365 pairs ≈ 30 GiB;
# per-engine it is NOT symmetric: the v8 run measured 1,247 unique experts on
# cuda0 (layers 0-21, ≈ 16.7 GiB of packs) vs 916 on cuda1 (layers 22-42,
# ≈ 12.2 GiB).  Memory-safety history:
#   - v8: 12.87 GiB/GPU = 25.74 GiB total, NO madvise -> SURVIVED (correct
#     per-step memory, wrong tokens only because batched=True diverged).
#   - v9: 15.79/11.89 = 27.68 GiB -> OOM-killed.
#   - v11: 14.5/11.5 = 26.0 GiB + madvise ON -> OOM-killed; the DONTNEED
#     re-faulted evicted experts against the slow loop device (~4 MB/s,
#     ~3.4 s/miss) instead of hitting the page cache.
# v12 therefore returns to the v8-PROVEN ceiling (12.87 GiB/GPU symmetric)
# and leaves madvise OFF by default (env DEE_MADVISE_DONTNEED=1 opts in).
# The cuda0 shortfall (16.7 GiB set vs 12.87 budget) means cuda0 still
# evicts ~3.8 GiB of experts per pass; those re-faults now hit the page
# cache (v8 behavior) rather than the loop device.
HOST_PACK_CACHE_BYTES_GPU0 = int(os.environ.get(
    "NATIVE_HOST_PACK_GPU0_BYTES", str(int(12.87 * (1 << 30)))))
HOST_PACK_CACHE_BYTES_GPU1 = int(os.environ.get(
    "NATIVE_HOST_PACK_GPU1_BYTES", str(int(12.87 * (1 << 30)))))
# Stage 2 (v9): the pointer-batched SwiGLU path (cublasGemmBatchedEx) is a
# DIFFERENT numerical kernel than the per-expert path (cublasGemmEx per
# expert): a 1-ULP FP16 difference flips greedy tokens. v8 enabled it and
# DIVERGED from the v7 gate tokens. It stays OFF by default so the strict
# token-identity gate holds; it remains an experimental speed mode.
USE_BATCHED_EXPERTS = os.environ.get("NATIVE_BATCHED", "0") == "1"
# Stage 0 (v8): diagnostics/profiling are opt-in. Headline timing must not
# include per-layer finite checks, route serialization, checksums, or CUDA
# timing-event allocation. Run a separate diagnostic pass with both enabled.
PROFILE_STAGES = os.environ.get("NATIVE_PROFILE", "0") == "1"
DIAGNOSTICS = os.environ.get("NATIVE_DIAGNOSTICS", "0") == "1"
# v15: return to v8-PROVEN storage behavior.  v13's discard_source_pages
# (posix_fadvise + MADV_DONTNEED on the shared mmap after every pack fill)
# re-introduced the v10 behavior that v12 measured as OOM + re-fault
# thrash against the loop device (~4 MB/s): v8 (no discard, page-cache
# hits on evicted re-reads) survived 16/16 tokens at 225 s/token, while
# v14 (discard ON) OOM'd at token 9 at 369 s/token.  The engine default is
# ON, so this runtime opts OUT explicitly.  The v15 LRU cap (17 GiB)
# bounds the anonymous side regardless.
os.environ.setdefault("DEE_RELEASE_MMAP_PAGES", "0")
# P2.4 storage decision (2026-08-23 probe): the dataset mount is a ~13 MB/s
# loop device (95.7% of v15/v16 decode wall), while the /tmp root overlay
# measured 1,550-1,830 MB/s pread / ~9-11 GB/s mmap on the same GPU worker
# class.  v15/v16 preferred the mount when present, which is exactly the
# bottleneck.  NATIVE_FORCE_TMP=1 stages all shards into /tmp (copy from the
# mount when present, else HF download) so the engine's expert mmap reads
# hit the fast overlay instead of the loop device.  Default OFF until a
# clean 2-GPU run proves the 16/16 token gate holds with the staged path.
FORCE_TMP = os.environ.get("NATIVE_FORCE_TMP", "0") == "1"
# P2.4 (2026-08-23): the dual-T4 pool has been exhausted for ~12 consecutive
# launches (Kaggle hands out 1x P100 instead).  NATIVE_SINGLE_GPU=1 runs the
# full 43-layer model on one CUDA device (split=n_layers, same-device
# handoff, one engine with the full budget).  The 16/16-token gate is
# arch-independent (sm_60/sm_75 cubins, same math), so a P100 run validates
# the identical correctness contract while the T4 pool recovers; the log
# labels hardware so performance numbers stay honest.  Requires the engine
# build to include sm_60 (already default in this kernel).
SINGLE_GPU = os.environ.get("NATIVE_SINGLE_GPU", "0") == "1"
PROGRESS = WORK / "progress.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(PROGRESS, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def run(cmd, **kw):
    log("+ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        log(f"FAILED (exit {r.returncode})")
        raise RuntimeError("command failed: "
                           + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    return r


def _download_one(shard: str) -> Path:
    CKPT.mkdir(parents=True, exist_ok=True)
    dest = CKPT / shard
    url = (f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
           f"resolve/{REV}/{shard}")
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        cr = resp.headers.get("Content-Range", "")
    if not cr or "/" not in cr:
        raise RuntimeError(f"no Content-Range for {shard}")
    want = int(cr.split("/")[1])
    have = dest.stat().st_size if dest.is_file() else 0
    if have == want:
        return dest
    chunk = 16 << 20
    with open(dest, "ab") as fh:
        while have < want:
            end = min(have + chunk - 1, want - 1)
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={have}-{end}"})
            last = None
            for attempt in range(8):
                try:
                    with urllib.request.urlopen(req, timeout=600) as r:
                        data = r.read(chunk + 1)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError,
                        ConnectionError, TimeoutError) as e:
                    last = e
                    time.sleep(2.0 * (2 ** attempt))
            else:
                raise ConnectionError(f"{shard} download failed: {last!r}")
            fh.write(data)
            have += len(data)
    return dest


def _snapshot_download(shards: list[str]) -> bool:
    """Fast path: huggingface_hub + xet (~263 MB/s measured on a single T4)."""
    from huggingface_hub import snapshot_download
    t0 = time.monotonic()
    snapshot_download(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        revision=REV,
        local_dir=str(CKPT),
        allow_patterns="model-*.safetensors",
        max_workers=4,
    )
    total = sum((CKPT / s).stat().st_size for s in shards
                if (CKPT / s).is_file()) / (1 << 30)
    log(f"[download] snapshot_download {total:.1f} GiB in "
        f"{time.monotonic() - t0:.0f}s")
    return True


def _copy_mount_to_tmp(shards: list[str], dst_dir: Path) -> list[str]:
    """Sequential per-shard copy from the dataset mount to /tmp.

    The mount loop device does ~194 MB/s on large sequential reads (P2.2
    evidence) but ~13 MB/s on mmap scatter (v15/v16).  One sequential copy
    per shard amortizes to a few minutes total and makes all later expert
    reads hit the fast root overlay instead.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    total_bytes = 0
    for s in shards:
        src = DATASET_DIR / s
        dst = dst_dir / s
        if not src.is_file():
            raise FileNotFoundError(f"mount shard missing: {src}")
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            total_bytes += dst.stat().st_size
            continue
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=32 << 20)
        if dst.stat().st_size != src.stat().st_size:
            raise IOError(f"copy size mismatch for {s}")
        total_bytes += dst.stat().st_size
        mbps = total_bytes / (1 << 20) / max(time.monotonic() - t0, 1e-9)
        log(f"[download] copied {s} ({dst.stat().st_size/(1<<30):.2f} GiB) "
            f"agg {mbps:.0f} MB/s")
    dt = time.monotonic() - t0
    log(f"[download] mount->/tmp staged {total_bytes/(1<<30):.1f} GiB in "
        f"{dt:.0f}s ({total_bytes/(1<<20)/max(dt,1e-9):.0f} MB/s)")
    return [str(dst_dir / s) for s in shards]


def download_all_shards() -> list[str]:
    shards = [f"model-{i:05d}-of-00048.safetensors"
              for i in range(1, N_SHARDS + 1)]
    paths = [str(CKPT / s) for s in shards]
    # P2.4: NATIVE_FORCE_TMP=1 stages everything into /tmp (fast root
    # overlay, ~1.5 GB/s) instead of reading experts from the ~13 MB/s
    # dataset mount.  Copy from the mount when present (no re-download),
    # else snapshot_download from HF.
    if FORCE_TMP:
        CKPT.mkdir(parents=True, exist_ok=True)
        if DATASET_DIR.is_dir() and all((DATASET_DIR / s).is_file()
                                        for s in shards):
            log(f"[download] FORCE_TMP: staging mount -> {CKPT}")
            return _copy_mount_to_tmp(shards, CKPT)
        log(f"[download] FORCE_TMP: mount absent/incomplete; downloading")
        if not _snapshot_download(shards):
            raise RuntimeError("snapshot_download failed")
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            log(f"[download] {len(missing)} shards missing; range-fetch fallback")
            failures = []

            def work(p):
                try:
                    return _download_one(Path(p).name)
                except Exception as e:  # noqa: BLE001
                    failures.append((p, repr(e)))
                    return None

            with ThreadPoolExecutor(max_workers=3) as ex:
                list(ex.map(work, missing))
            if failures:
                raise RuntimeError(f"{len(failures)} downloads failed: {failures}")
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            raise RuntimeError(f"{len(missing)} shards still missing")
        return paths
    # Legacy path: dataset-mounted checkpoint, no download, no disk quota.
    if DATASET_DIR.is_dir():
        ds_paths = [str(DATASET_DIR / s) for s in shards]
        if all(Path(p).is_file() for p in ds_paths):
            log(f"[download] using dataset-mounted checkpoint at {DATASET_DIR}")
            return ds_paths
        log(f"[download] dataset dir present but incomplete; downloading")
    CKPT.mkdir(parents=True, exist_ok=True)
    if not _snapshot_download(shards):
        raise RuntimeError("snapshot_download failed")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        # Fall back to resume-capable range fetches for whatever is missing.
        log(f"[download] {len(missing)} shards missing; range-fetch fallback")
        failures = []

        def work(p):
            try:
                return _download_one(Path(p).name)
            except Exception as e:  # noqa: BLE001
                failures.append((p, repr(e)))
                return None

        with ThreadPoolExecutor(max_workers=3) as ex:
            list(ex.map(work, missing))
        if failures:
            raise RuntimeError(f"{len(failures)} downloads failed: {failures}")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} shards still missing after download")
    return paths


def host_mem_available_gib() -> float:
    """MemAvailable from /proc/meminfo, in GiB (0.0 on failure)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def host_mem_total_gib() -> float:
    """MemTotal from /proc/meminfo, in GiB (0.0 on failure)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def process_mem_gib() -> dict:
    """This process's VmRSS/VmData/VmLck/VmSwap from /proc/self/status.

    VmData = anonymous heap growth; VmLck = pinned (cudaMallocHost) growth;
    the gap between VmRSS and (VmData + VmLck) is file-backed page cache
    attributed to this process.  This is the definitive leak localizer for
    the v12/v14 decode-time host-RAM growth.
    """
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            for key in ("VmRSS", "VmData", "VmLck", "VmSwap"):
                if line.startswith(key + ":"):
                    out[key] = round(int(line.split()[1]) / (1024 * 1024), 2)
    except OSError:
        pass
    return out


def system_mem_gib() -> dict:
    """MemTotal/MemAvailable/Cached/SReclaimable, in GiB."""
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            for key in ("MemTotal", "MemAvailable", "Cached", "SReclaimable"):
                if line.startswith(key + ":"):
                    out[key] = round(int(line.split()[1]) / (1024 * 1024), 2)
    except OSError:
        pass
    return out


def gpu_memory_snapshot() -> dict:
    import torch
    out = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            out[f"cuda{i}"] = {
                "allocated_gib": round(torch.cuda.memory_allocated(i) / (1 << 30), 3),
                "reserved_gib": round(torch.cuda.memory_reserved(i) / (1 << 30), 3),
                "free_gib": round(free / (1 << 30), 3),
                "total_gib": round(total / (1 << 30), 3),
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated(i) / (1 << 30), 3),
            }
    return out


def check_gpu_allocation() -> None:
    """Fail fast (before the ~40-min build) if the worker lacks 2 GPUs.

    Kaggle's Dual-GPU pool intermittently allocates 1 GPU (v17/v18/v19 all
    hit this AFTER a full build + P2.2 repack).  Exiting early turns a wasted
    45 minutes into a 5-second failure we can re-push immediately.

    The 16/16-token gate is arch-independent (same cubin math on sm_60/sm_75),
    so any 2-GPU worker validates correctness; the log records the actual
    hardware so performance numbers are labeled correctly (T4 vs P100).
    """
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True,
                                      stderr=subprocess.STDOUT)
    except Exception as e:
        log(f"GPU_ALLOC_FAIL nvidia-smi unavailable: {e}")
        raise RuntimeError(f"expected 2 GPUs, nvidia-smi failed: {e}")
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    n_gpus = len(lines)
    log(f"GPU_ALLOC n={n_gpus}: " + " | ".join(lines))
    need = 1 if SINGLE_GPU else 2
    if n_gpus < need:
        raise RuntimeError(
            f"expected {need} GPUs, got {n_gpus}: {out.strip()}")


def main() -> int:
    check_gpu_allocation()

    log("=== clone + checkout ===")
    if ROOT.exists():
        run(["rm", "-rf", str(ROOT)])
    run(["git", "clone", "--branch", BRANCH, "--single-branch",
         REPO, str(ROOT)])
    if COMMIT:
        run(["git", "-C", str(ROOT), "checkout", "--quiet", COMMIT])
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    log(f"pinned commit {head}")

    log("=== build dee_cli + FP4 regression tests (sm_60;sm_75) ===")
    run(["cmake", "-S", str(DEE), "-B", str(BUILD),
         "-DCMAKE_CUDA_ARCHITECTURES=60;75", "-DDEE_CUDA=ON",
         "-DDEE_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(BUILD), "--target", "dee_cli",
         "-j", str(os.cpu_count() or 4)])
    for target in ("test_deepseek_v4_fp4_cuda", "test_deepseek_v4_fp4_expert"):
        run(["cmake", "--build", str(BUILD), "--target", target,
             "-j", str(os.cpu_count() or 4)])
        r = subprocess.run([str(BUILD / target)], cwd=str(DEE))
        if r.returncode != 0:
            log(f"WARNING: regression test {target} returned {r.returncode} (non-fatal)")

    log("=== build pydee ===")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--user", "pybind11"])
    run([sys.executable, "pydee/setup.py", "build_ext", "--inplace"],
        env={**os.environ, "DEE_BUILD_DIR": str(BUILD)}, cwd=str(DEE))

    log("=== download all shards ===")
    shard_paths = download_all_shards()

    # ── P2.2: DEE4 expert-major repack benchmark ──────────────────────
    log("=== P2.2 DEE4 repack benchmark ===")
    try:
        sys.path.insert(0, str(DEE / "kaggle" / "deepseek-v4-flash-0731"))
        from repack_to_dee4 import repack, benchmark_dee4_read as _b4r
        import struct as _struct
        _dee4_out = Path("/kaggle/working/dee4-test")
        _idx = Path(str(shard_paths[0]).rsplit("/", 1)[0]) / "model.safetensors.index.json"
        if not _idx.is_file():
            _idx_url = (f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
                        f"resolve/{REV}/model.safetensors.index.json")
            _idx = WORK / "model.safetensors.index.json"
            log(f"P2.2: downloading index from HF: {_idx_url}")
            _idx_data = urllib.request.urlopen(_idx_url, timeout=300).read()
            _idx.write_bytes(_idx_data)
            log(f"P2.2: index downloaded ({len(_idx_data)} bytes)")
        _t0 = time.monotonic()
        _dee4_rpt = repack(
            Path(str(shard_paths[0]).rsplit("/", 1)[0]), _dee4_out,
            index_path=_idx, start_layer=0, end_layer=3, dry_run=False)
        _dt = time.monotonic() - _t0
        _dee4_bench = _b4r(_dee4_out, n_experts=64)
        # Compare: safetensors random gather
        _idx_data = json.loads(_idx.read_text("utf-8"))
        _wm = _idx_data["weight_map"]; _hdr = {}; _sp = {}
        for _sn in sorted(set(_wm.values())):
            _p = Path(str(shard_paths[0]).rsplit("/", 1)[0]) / _sn
            _sp[_sn] = _p
            with open(_p, "rb") as _f:
                _hl = _struct.unpack("<Q", _f.read(8))[0]
                _hdr[_sn] = json.loads(_f.read(_hl))
        _st0 = time.monotonic(); _tb = 0; _rc = 0
        for _L in range(3):
            for _eid in range(min(21, 256)):  # 21 * 3 layers = 63 experts
                for _proj in ["w1","w2","w3"]:
                    for _kind in ["weight","scale"]:
                        _nm = f"layers.{_L}.ffn.experts.{_eid}.{_proj}.{_kind}"
                        if _nm not in _wm: continue
                        _sn = _wm[_nm]; _hh = _hdr[_sn]
                        _off = _hh[_nm]["data_offsets"]
                        _ln = _off[1] - _off[0]
                        with open(_sp[_sn], "rb") as _f:
                            _f.seek(8 + _off[0]); _f.read(_ln)
                        _tb += _ln; _rc += 1
                if _rc >= 64 * 6: break
            if _rc >= 64 * 6: break
        _ste = time.monotonic() - _st0
        _st_mbps = _tb / max(_ste, 0.001) / (1 << 20)
        _d4_mbps = _dee4_bench["aggregate_mbps"]
        log(f"P2.2: DEE4 contiguous {_d4_mbps:.0f} MB/s vs "
            f"safetensors scatter {_st_mbps:.0f} MB/s "
            f"({_d4_mbps/max(_st_mbps,0.01):.1f}x) "
            f"repack {_dt:.0f}s {_dee4_rpt['total_bytes_repacked']/(1<<30):.1f}GiB")
        _p22_evidence = {
            "dee4_mbps": _d4_mbps, "safetensors_mbps": _st_mbps,
            "speedup": _d4_mbps / max(_st_mbps, 0.01),
            "repack_s": _dt, "repack_gib": _dee4_rpt["total_bytes_repacked"]/(1<<30),
            "io_count_reduction": f"{_rc} random -> {len(_dee4_bench['tests'])} sequential"
        }
        (Path("/kaggle/working") / "p2.2-dee4-evidence.json").write_text(
            json.dumps(_p22_evidence, indent=2), "utf-8")
    except Exception as _e:
        log(f"P2.2 repack failed (non-fatal): {_e}")
        import traceback as _tb
        _tb.print_exc()

    sys.path.insert(0, str(DEE))
    sys.path.insert(0, str(DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/"
                          "official-source/inference"))
    import torch
    log(f"cuda devices: {torch.cuda.device_count()}")
    need = 1 if SINGLE_GPU else 2
    if torch.cuda.device_count() < need:
        raise RuntimeError(
            f"expected {need} GPUs, got {torch.cuda.device_count()}")

    from scripts import deepseek_v4_model as vm
    from scripts import deepseek_v4_encoding as enc

    cfg = vm.model_config_from_official(CONFIG)
    tokenizer = enc.load_tokenizer()
    log(f"cfg layers/exps/dim/topk: {cfg.n_layers} {cfg.n_routed} "
        f"{cfg.dim} {cfg.topk}")

    log("=== build native engines (device 0 + 1) ===")
    mem_avail = host_mem_available_gib()
    mem_total = host_mem_total_gib()
    # v15: the v12/v14 "leak" is the host-pack LRU materializing its full
    # 25.74 GiB budget on a box that physically cannot hold it.  Both runs
    # were OOM-killed with the LRU at ~21 GiB plus the ~5-8 GiB resident
    # baseline (torch + 2 CUDA contexts + python + dense host refs), and
    # the old clamp (mem_avail - 3.5 = 26.2 GiB > 25.74) never engaged.
    # Cap the TOTAL LRU budget hard at 17 GiB (8.5 GiB/GPU, below the
    # measured ~21 GiB death point with margin).  v15 also logs MemTotal +
    # per-token VmRSS/VmData/VmLck so v16 can tune the exact ceiling.
    LRU_TOTAL_CAP_GIB = 17.0
    pack_budget0 = HOST_PACK_CACHE_BYTES_GPU0
    pack_budget1 = HOST_PACK_CACHE_BYTES_GPU1
    if mem_avail > 0 or mem_total > 0:
        total_cap = int(LRU_TOTAL_CAP_GIB * (1 << 30))
        if mem_avail > 0:
            total_cap = min(total_cap, int((mem_avail - 3.5) * (1 << 30)))
        # Keep the measured 16.2:12.2 ratio when clamping under total RAM.
        ratio = HOST_PACK_CACHE_BYTES_GPU0 + HOST_PACK_CACHE_BYTES_GPU1
        if total_cap < ratio:
            share0 = HOST_PACK_CACHE_BYTES_GPU0 / max(1, ratio)
            pack_budget0 = max(0, int(total_cap * share0))
            pack_budget1 = max(0, int(total_cap * (1.0 - share0)))
        pack_budget0 = min(pack_budget0, HOST_PACK_CACHE_BYTES_GPU0)
        pack_budget1 = min(pack_budget1, HOST_PACK_CACHE_BYTES_GPU1)
    log(f"budget={BUDGET_BYTES/2**30:.2f}GiB/GPU host_pack="
        f"{pack_budget0/2**30:.2f}/{pack_budget1/2**30:.2f}GiB "
        f"batched={USE_BATCHED_EXPERTS} profile={PROFILE_STAGES} "
        f"diagnostics={DIAGNOSTICS} mem_avail={mem_avail:.1f}GiB "
        f"mem_total={mem_total:.1f}GiB lru_cap={LRU_TOTAL_CAP_GIB}GiB "
        f"cache_dtype={CACHE_DTYPE}")
    # P2.4 single-GPU mode: one engine on cuda:0 carrying the FULL budget
    # (both halves merged), all 43 layers on device0 (split=n_layers), and
    # the same-device handoff path.  eng1 is not built.  Cache budget is
    # capped below the dense+torch baseline (~7 GiB dense + ~1.5 GiB
    # torch/CUDA on a 14.5 GiB-usable 16 GB card) so the run cannot OOM.
    if SINGLE_GPU:
        single_budget = min(int(BUDGET_BYTES * 2),
                            int(4.0 * (1 << 30)))
        single_pack = pack_budget0 + pack_budget1
        eng0 = vm.build_native_engine(
            shard_paths, device_id=0, budget_bytes=single_budget,
            host_pack_cache_bytes=single_pack,
            use_batched_experts=USE_BATCHED_EXPERTS,
            profile_stages=PROFILE_STAGES,
            cache_dtype=CACHE_DTYPE)
        eng1 = eng0
        log(f"engines built SINGLE_GPU budget={single_budget/2**30:.2f}GiB "
            f"host_pack={single_pack/2**30:.2f}GiB cache_dtype={CACHE_DTYPE}")
    else:
        eng0 = vm.build_native_engine(
            shard_paths, device_id=0, budget_bytes=BUDGET_BYTES,
            host_pack_cache_bytes=pack_budget0,
            use_batched_experts=USE_BATCHED_EXPERTS,
            profile_stages=PROFILE_STAGES,
            cache_dtype=CACHE_DTYPE)
        eng1 = vm.build_native_engine(
            shard_paths, device_id=1, budget_bytes=BUDGET_BYTES,
            host_pack_cache_bytes=pack_budget1,
            use_batched_experts=USE_BATCHED_EXPERTS,
            profile_stages=PROFILE_STAGES,
            cache_dtype=CACHE_DTYPE)
        log(f"engines built (cache_dtype={CACHE_DTYPE})")

    log("=== build full model (native FFN) ===")
    t0 = time.monotonic()
    # The tensor source must read dense tensors (embed/head/norm/attention/
    # router/shared) from whichever directory actually holds the shards:
    # the dataset mount when attached, or the local /tmp download otherwise.
    shards_dir = Path(shard_paths[0]).parent
    source = vm.LocalDirTensorSource(HEADERS_DIR, shards_dir)
    provider = vm.ExpertProvider(source)
    if SINGLE_GPU:
        model = vm.DeepseekV4Model.build_candidate(
            cfg, source, device0="cuda:0", device1="cuda:0",
            cache0=None, loader0=None, cache1=None, loader1=None,
            provider=provider, ffn_backend="native",
            engine0=eng0, engine1=eng1, split=cfg.n_layers,
            diagnostics=DIAGNOSTICS)
    else:
        model = vm.DeepseekV4Model.build_candidate(
            cfg, source, device0="cuda:0", device1="cuda:1",
            cache0=None, loader0=None, cache1=None, loader1=None,
            provider=provider, ffn_backend="native",
            engine0=eng0, engine1=eng1, diagnostics=DIAGNOSTICS)
    model.reset_state()
    build_s = time.monotonic() - t0
    log(f"model build {build_s:.1f}s")

    log("=== tokenize + greedy decode ===")
    ids = tokenizer.encode(CANONICAL_PROMPT)
    input_ids = torch.tensor([ids], device="cuda:0").long()
    decode_ms: list[float] = []
    eng0.reset_external_profile()
    if not SINGLE_GPU:
        eng1.reset_external_profile()
    # v12: checkpoint every generated token to /kaggle/working so an OOM kill
    # (v9/v11 lost ALL tokens) still leaves the exact token stream + timing.
    # The checkpoint file format is a JSONL of per-token records; the final
    # RESULT block below mirrors the old single-JSON shape.
    CHECKPOINT = WORK / "generated_checkpoint.jsonl"
    cp_handle = open(CHECKPOINT, "w", encoding="utf-8")

    def _token_checkpoint(step: int, tok: int) -> None:
        mem = host_mem_available_gib()
        rec = {"step": step, "token_id": int(tok),
               "elapsed_s": round(time.monotonic() - t0, 2),
               "host_mem_available_gib": round(mem, 2)}
        # v15 diagnostics: process + system memory breakdown and engine
        # cache counters, so a v12-style OOM is attributable to a component
        # (heap vs pinned vs page cache) instead of a mystery "leak".
        rec["proc"] = process_mem_gib()
        rec["sys"] = system_mem_gib()
        try:
            rec["host_pack0"] = eng0.host_pack_stats()
            rec["host_pack1"] = eng1.host_pack_stats()
        except Exception:
            pass
        try:
            rec["bridge"] = model.bridge_counters()
        except Exception:
            pass
        cp_handle.write(json.dumps(rec) + "\n")
        cp_handle.flush()
        if step % 4 == 0 or mem < 4.0:
            log(f"[tok {step}] id={int(tok)} elapsed={rec['elapsed_s']}s "
                f"mem_avail={rec['host_mem_available_gib']}GiB")
            pm = rec.get("proc", {})
            hp0 = rec.get("host_pack0", {})
            log(f"[tok {step}] proc={pm} hp0_bytes_gib="
                f"{hp0.get('bytes', 0) / (1 << 30):.1f} "
                f"hp0_entries={hp0.get('entries', 0)} "
                f"hp0_evict={hp0.get('evictions', 0)}")

    t0 = time.monotonic()
    toks = model.generate(input_ids, max_new_tokens=N_TOKENS,
                          decode_timings_ms=decode_ms,
                          post_step_hook=_token_checkpoint)
    wall_s = time.monotonic() - t0
    cp_handle.close()
    log(f"decode done in {wall_s:.1f}s, {len(toks)} tokens")

    prefill_ms = decode_ms[0] if decode_ms else 0.0
    decode_only = decode_ms[1:]
    decode_sum_s = sum(decode_only) / 1000.0
    decode_tok_s = (len(decode_only) / decode_sum_s
                    if decode_sum_s > 0 else float("inf"))
    lat = sorted(decode_only)
    p50 = lat[len(lat) // 2] if lat else 0.0
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0
    text = tokenizer.decode(toks)

    result = {
        "commit": head,
        "host_mem_available_gib": round(mem_avail, 2),
        "host_pack_budget_gib": [round(pack_budget0 / (1 << 30), 2),
                                  round(pack_budget1 / (1 << 30), 2)],
        "model_revision": REV,
        "prompt": CANONICAL_PROMPT,
        "prompt_len": len(ids),
        "n_tokens": N_TOKENS,
        "generated_token_ids": toks,
        "decoded_text": text,
        "decoded_fragments": [tokenizer.decode([t]) for t in toks],
        "build_seconds": round(build_s, 2),
        "total_wall_seconds": round(wall_s, 2),
        "prefill_ms": round(prefill_ms, 2),
        "prefill_tokens": len(ids),
        "prefill_tok_s": round(len(ids) / (prefill_ms / 1000.0), 3)
        if prefill_ms > 0 else None,
        "decode_wall_s": round(decode_sum_s, 3),
        "decode_tokens": len(decode_only),
        "decode_tok_s": round(decode_tok_s, 3),
        "inter_token_latency_ms": {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "max": round(max(decode_only), 2) if decode_only else None,
            "median": round(float(p50), 2),
        },
        "decode_timings_ms": [round(t, 2) for t in decode_only],
        "gpu_memory": gpu_memory_snapshot(),
        "diagnostics": DIAGNOSTICS,
        "bridge_counters": model.bridge_counters(),
        "layer_count_executed": (
            len(model.execution_trace) if DIAGNOSTICS else model.cfg.n_layers),
    }

    # Stage 0 instrumentation: per-engine expert-cache + host-pack + stage
    # profile dumps so every run reports WHERE the wall went.
    try:
        result["engine_stats"] = {
            "cuda0": json.loads(eng0.last_stats_json()),
            "cuda1": json.loads(eng1.last_stats_json()),
        }
        result["host_pack"] = {
            "cuda0": eng0.host_pack_stats(),
            "cuda1": eng1.host_pack_stats(),
        }
        result["stage_profile"] = {
            "cuda0": json.loads(eng0.external_profile_json(wall_s * 1000.0)),
            "cuda1": json.loads(eng1.external_profile_json(wall_s * 1000.0)),
        }
    except Exception as exc:  # never fail the run over instrumentation
        log(f"instrumentation dump failed: {exc}")
    result["model_runtime_snapshot"] = model.runtime_snapshot()
    log("RESULT " + json.dumps(result))
    (WORK / "native-generate-result.json").write_text(
        json.dumps(result, indent=2))

    # Clean up the local download only; the dataset mount is read-only and
    # must not be touched (unlink would raise PermissionError there).
    if not (DATASET_DIR.is_dir() and Path(shard_paths[0]).parent == DATASET_DIR):
        for p in shard_paths:
            Path(p).unlink(missing_ok=True)
    log("=== VERDICT: real tokenizer->text decode measured ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        tb = traceback.format_exc()
        log("FATAL " + tb)
        (WORK / "error.txt").write_text(tb)
        (WORK / "native-generate-result.json").write_text(
            json.dumps({"status": "error", "traceback": tb}, indent=2))
        # Exit 0 so Kaggle snapshots /kaggle/working (error-exit kernels drop
        # their output/log, which is why the earlier failures were undiagnosable).
        sys.exit(0)
