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
WORK = Path("/kaggle/working")
HEADERS_DIR = (DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers")
CONFIG = (DEE / "benchmark_reports/deepseek-v4-flash-0731-t4/"
          "official-source/inference/config.json")
CANONICAL_PROMPT = (
    "<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>Who is Alan Turing?"
    "<\uFF5CAssistant\uFF5C>")
N_TOKENS = int(os.environ.get("NATIVE_N_TOKENS", "16"))
BUDGET_BYTES = 512 << 20
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


def download_all_shards() -> list[str]:
    shards = [f"model-{i:05d}-of-00048.safetensors"
              for i in range(1, N_SHARDS + 1)]
    done = 0
    failures = []
    t0 = time.monotonic()

    def work(shard):
        nonlocal done
        try:
            p = _download_one(shard)
        except Exception as e:  # noqa: BLE001
            failures.append((shard, repr(e)))
            log(f"[download] FAILED {shard}: {e!r}")
            return None
        done += 1
        log(f"[download] {done}/{N_SHARDS} {shard} "
            f"({p.stat().st_size / (1 << 30):.2f} GiB, "
            f"{time.monotonic() - t0:.0f}s)")
        return p

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(work, shards))
    if failures:
        raise RuntimeError(f"{len(failures)} shard downloads failed: {failures}")
    total = sum(p.stat().st_size for p in results) / (1 << 30)
    log(f"[download] complete: {total:.1f} GiB in {time.monotonic() - t0:.0f}s")
    return [str(p) for p in results]


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
    eng0 = vm.build_native_engine(shard_paths, device_id=0,
                                  budget_bytes=BUDGET_BYTES)
    eng1 = vm.build_native_engine(shard_paths, device_id=1,
                                  budget_bytes=BUDGET_BYTES)
    log("engines built")

    log("=== build full model (native FFN) ===")
    t0 = time.monotonic()
    source = vm.LocalDirTensorSource(HEADERS_DIR, CKPT)
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
    log("RESULT " + json.dumps(result))
    (WORK / "native-generate-result.json").write_text(
        json.dumps(result, indent=2))

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
