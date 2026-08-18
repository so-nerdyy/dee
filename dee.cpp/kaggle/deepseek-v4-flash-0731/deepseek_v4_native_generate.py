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
# Stage 1b (v8/v9/v10): host RAM LRU of packed FP4 expert bytes (12.6 MB/
# entry).  The 16-token working set is ~2,365 pairs ≈ 30 GiB; per-engine it is
# NOT symmetric: the v8 run measured 1,247 unique experts on cuda0 (layers
# 0-21, ≈ 16.7 GiB of packs) vs 916 on cuda1 (layers 22-42, ≈ 12.2 GiB).
# v9 set budgets sized to each GPU's working set (16.2/12.2 = 28.4 GiB) but
# was OOM-KILLED during decode: host_pack filled to its 27.68 GiB cap and,
# combined with dense host tensors + the mmap page cache of the 152.8 GiB
# checkpoint, exceeded the box's ~32 GiB RAM.  v8 survived at 25.74 GiB total
# (12.87/GPU).  v10 therefore (a) caps the total at 26 GiB (14.5/11.5, the
# v8-proven-safe ceiling) and (b) the engine now madvise(MADV_DONTNEED)s the
# source mmap pages after copying a pack into the LRU, so the page cache no
# longer double-books the ~28 GiB of checkpoint pages that decode reads.
HOST_PACK_CACHE_BYTES_GPU0 = int(os.environ.get(
    "NATIVE_HOST_PACK_GPU0_BYTES", str(int(14.5 * (1 << 30)))))
HOST_PACK_CACHE_BYTES_GPU1 = int(os.environ.get(
    "NATIVE_HOST_PACK_GPU1_BYTES", str(int(11.5 * (1 << 30)))))
# Stage 2 (v9): the pointer-batched SwiGLU path (cublasGemmBatchedEx) is a
# DIFFERENT numerical kernel than the per-expert path (cublasGemmEx per
# expert): a 1-ULP FP16 difference flips greedy tokens. v8 enabled it and
# DIVERGED from the v7 gate tokens. It stays OFF by default so the strict
# token-identity gate holds; it remains an experimental speed mode.
USE_BATCHED_EXPERTS = os.environ.get("NATIVE_BATCHED", "0") == "1"
# Stage 0 (v8): enable the per-stage profiler so the next run reports WHERE
# the wall went (host wait vs H2D vs dequant vs GEMM vs dense).
PROFILE_STAGES = os.environ.get("NATIVE_PROFILE", "1") == "1"
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


def download_all_shards() -> list[str]:
    shards = [f"model-{i:05d}-of-00048.safetensors"
              for i in range(1, N_SHARDS + 1)]
    paths = [str(CKPT / s) for s in shards]
    # Dataset-mounted checkpoint: no download, no writable-disk quota.
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


def main() -> int:
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

    log("=== build dee_cli + FP4 regression tests ===")
    run(["cmake", "-S", str(DEE), "-B", str(BUILD),
         "-DCMAKE_CUDA_ARCHITECTURES=75", "-DDEE_CUDA=ON",
         "-DDEE_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(BUILD), "--target", "dee_cli",
         "-j", str(os.cpu_count() or 4)])
    for target in ("test_deepseek_v4_fp4_cuda", "test_deepseek_v4_fp4_expert"):
        run(["cmake", "--build", str(BUILD), "--target", target,
             "-j", str(os.cpu_count() or 4)])
        r = subprocess.run([str(BUILD / target)], cwd=str(DEE))
        if r.returncode != 0:
            raise RuntimeError(f"regression test failed: {target}")

    log("=== build pydee ===")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--user", "pybind11"])
    run([sys.executable, "pydee/setup.py", "build_ext", "--inplace"],
        env={**os.environ, "DEE_BUILD_DIR": str(BUILD)}, cwd=str(DEE))

    log("=== download all shards ===")
    shard_paths = download_all_shards()

    sys.path.insert(0, str(DEE))
    sys.path.insert(0, str(DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/"
                          "official-source/inference"))
    import torch
    log(f"cuda devices: {torch.cuda.device_count()}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"expected 2 GPUs, got {torch.cuda.device_count()}")

    from scripts import deepseek_v4_model as vm
    from scripts import deepseek_v4_encoding as enc

    cfg = vm.model_config_from_official(CONFIG)
    tokenizer = enc.load_tokenizer()
    log(f"cfg layers/exps/dim/topk: {cfg.n_layers} {cfg.n_routed} "
        f"{cfg.dim} {cfg.topk}")

    log("=== build native engines (device 0 + 1) ===")
    mem_avail = host_mem_available_gib()
    # Safety clamp: keep 2 GiB of headroom for torch/CUDA/python + the dense
    # host tensors that stay resident during decode.  The v8 run held 25.7 GiB
    # of host_pack with MemAvailable 29.7 GiB with no OOM, so 4 GiB was too
    # conservative and starved cuda0.  The LRU fills lazily, so this only caps
    # the box's true RAM limit.  Each GPU gets a budget sized to its OWN
    # measured working set (asymmetric: cuda0 16.7 GiB > cuda1 12.2 GiB).
    pack_budget0 = HOST_PACK_CACHE_BYTES_GPU0
    pack_budget1 = HOST_PACK_CACHE_BYTES_GPU1
    if mem_avail > 0:
        total_cap = max(0, int((mem_avail - 2.0) * (1 << 30)))
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
        f"batched={USE_BATCHED_EXPERTS} "
        f"profile={PROFILE_STAGES} mem_avail={mem_avail:.1f}GiB")
    eng0 = vm.build_native_engine(
        shard_paths, device_id=0, budget_bytes=BUDGET_BYTES,
        host_pack_cache_bytes=pack_budget0,
        use_batched_experts=USE_BATCHED_EXPERTS,
        profile_stages=PROFILE_STAGES)
    eng1 = vm.build_native_engine(
        shard_paths, device_id=1, budget_bytes=BUDGET_BYTES,
        host_pack_cache_bytes=pack_budget1,
        use_batched_experts=USE_BATCHED_EXPERTS,
        profile_stages=PROFILE_STAGES)
    log("engines built")

    log("=== build full model (native FFN) ===")
    t0 = time.monotonic()
    # The tensor source must read dense tensors (embed/head/norm/attention/
    # router/shared) from whichever directory actually holds the shards:
    # the dataset mount when attached, or the local /tmp download otherwise.
    shards_dir = Path(shard_paths[0]).parent
    source = vm.LocalDirTensorSource(HEADERS_DIR, shards_dir)
    provider = vm.ExpertProvider(source)
    model = vm.DeepseekV4Model.build_candidate(
        cfg, source, device0="cuda:0", device1="cuda:1",
        cache0=None, loader0=None, cache1=None, loader1=None,
        provider=provider, ffn_backend="native",
        engine0=eng0, engine1=eng1)
    model.reset_state()
    build_s = time.monotonic() - t0
    log(f"model build {build_s:.1f}s")

    log("=== tokenize + greedy decode ===")
    ids = tokenizer.encode(CANONICAL_PROMPT)
    input_ids = torch.tensor([ids], device="cuda:0").long()
    decode_ms: list[float] = []
    eng0.reset_external_profile()
    eng1.reset_external_profile()
    t0 = time.monotonic()
    toks = model.generate(input_ids, max_new_tokens=N_TOKENS,
                          decode_timings_ms=decode_ms)
    wall_s = time.monotonic() - t0
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
        "layer_count_executed": len(model.execution_trace),
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
