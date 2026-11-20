"""Phase 6: MiMo-V2.6 native-generate runner (dee engine + torch dense path).

Modal/Kaggle-style env-driven runner — one process, one or two GPUs:

  store     NATIVE_DEE4_SEGMENTED_STORE  dee4-v4-segmented dir
  ckpt      NATIVE_MODEL_CKPT            staged safetensors dir (dense path)
  headers   NATIVE_HEADERS_DIR           shard-header JSONs (default:
                                         <store>/headers)
  engine    NATIVE_BUDGET_BYTES          per-engine VRAM arena
            NATIVE_HOST_PACK_GPU{0,1}_BYTES  host LRU caps
            NATIVE_LRU_TOTAL_CAP_GIB     total host-tier budget (report only)
            NATIVE_CACHE_DTYPE           fp4 | fp16 (default fp4)
            NATIVE_EVICTION_POLICY       lru (default) | rank_priority
            NATIVE_HOST_CACHE_MODE       lru (default) | bypass
            NATIVE_SOURCE_READ_LANES / _QUEUE_DEPTH
            NATIVE_SINGLE_GPU=1          force one engine / all layers dev0
  run       NATIVE_N_TOKENS (default 64) NATIVE_PROMPT (default builtin)
            NATIVE_CACHE_RESET  cold (default) | warm (one prefill pass first)
            NATIVE_RUN_ID, NATIVE_MAX_SEQ_LEN
  output    report.json + generated_ids.jsonl + trace jsonl in CWD

Exactness: routed experts come ONLY from the dee4 store through
pydee.Engine (mxfp4 records == DSv4 FP4 wire format).  Dense weights are
dequantized fp8-block/bf16 at load; router is the checkpoint's own
noaux_tc sigmoid+bias gate — prediction never routes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Repo clone root on sys.path so scripts.* + pydee resolve.
_SRC = os.environ.get("NATIVE_SOURCE_TREE", "/tmp/dsv4-native-src")
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "dee.cpp"))

import torch  # noqa: E402

from scripts import mimo_v2_model as mimo  # noqa: E402

import pydee  # noqa: E402


def log(line: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def build_engine(device_id: int, shard_hint: str, cfg_in: dict) -> "pydee.Engine":
    cfg = pydee.EngineConfig()
    cfg.expert_store_path = cfg_in["store"]
    cfg.shard_paths = [shard_hint]
    cfg.num_layers = cfg_in["n_buckets"]
    cfg.num_experts = cfg_in["n_experts"]
    cfg.topk = cfg_in["topk"]
    cfg.hidden = cfg_in["hidden"]
    cfg.inter = cfg_in["inter"]
    cfg.budget_bytes = cfg_in["budget_bytes"]
    cfg.host_pack_cache_bytes = cfg_in["host_pack_bytes"]
    cfg.eviction_policy = cfg_in["eviction_policy"]
    cfg.host_cache_mode = cfg_in["host_cache_mode"]
    cfg.source_read_lanes = cfg_in["read_lanes"]
    cfg.source_read_queue_depth = cfg_in["read_queue_depth"]
    cfg.cache_dtype = (pydee.DeviceCacheDType.Fp4E2m1
                       if cfg_in["cache_dtype"] == "fp4"
                       else pydee.DeviceCacheDType.Fp16)
    # Fp4E2m1 transfer is what keys the packed-record (deepseek_v4) path —
    # MiMo mxfp4 records are the identical wire format.
    cfg.transfer_dtype = pydee.WeightTransferDType.Fp4E2m1
    cfg.use_cuda = True
    cfg.device_id = device_id
    cfg.swiglu_limit = 0.0            # MiMo has no activation clamp
    cfg.use_batched_experts = True
    cfg.trace_requests = _env("NATIVE_TRACE_REQUESTS") == "1"
    cfg.profile_stages = _env("NATIVE_PROFILE") == "1"
    eng = pydee.Engine()
    if not eng.init(cfg):
        raise RuntimeError(f"engine init failed (device {device_id}): "
                           f"{eng.last_error_message()}")
    return eng


def main() -> int:
    t_start = time.monotonic()
    work = Path(os.getcwd())
    run_id = _env("NATIVE_RUN_ID", f"mimo-{int(time.time())}")
    store = _env("NATIVE_DEE4_SEGMENTED_STORE") or \
        _env("DEE4_SEGMENTED_STORE")
    ckpt = _env("NATIVE_MODEL_CKPT") or _env("DATASET_DIR") or ""
    headers = _env("NATIVE_HEADERS_DIR") or os.path.join(store, "headers")
    if not store or not ckpt:
        raise SystemExit("need NATIVE_DEE4_SEGMENTED_STORE and "
                         "NATIVE_MODEL_CKPT")

    n_gpus = torch.cuda.device_count()
    single = _env("NATIVE_SINGLE_GPU") == "1" or n_gpus < 2
    budget = int(_env("NATIVE_BUDGET_BYTES", str(3584 << 20)))
    host0 = int(_env("NATIVE_HOST_PACK_GPU0_BYTES", str(8 << 30)))
    host1 = int(_env("NATIVE_HOST_PACK_GPU1_BYTES", str(host0)))
    log(f"[env] gpus={n_gpus} single={single} run={run_id}")
    log(f"[env] store={store}")
    log(f"[env] ckpt={ckpt} headers={headers}")

    # One real safetensors shard for EngineConfig.shard_paths (mandatory
    # mmap; expert bytes come from the store, never these files).
    shards = sorted(str(p) for p in Path(ckpt).glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors under {ckpt}")

    mcfg = mimo.mimo_config_from_official(os.path.join(ckpt, "config.json"))
    engine_cfg = {
        "store": store,
        "n_buckets": 47, "n_experts": 256, "topk": mcfg.topk,
        "hidden": mcfg.hidden, "inter": mcfg.moe_inter,
        "budget_bytes": budget,
        "host_pack_bytes": host0,
        "eviction_policy": _env("NATIVE_EVICTION_POLICY", "lru"),
        "host_cache_mode": _env("NATIVE_HOST_CACHE_MODE", "lru"),
        "read_lanes": int(_env("NATIVE_SOURCE_READ_LANES", "4")),
        "read_queue_depth": int(_env("NATIVE_SOURCE_READ_QUEUE_DEPTH", "6")),
        "cache_dtype": _env("NATIVE_CACHE_DTYPE", "fp4"),
    }
    log(f"[engine] cfg={engine_cfg}")
    eng0 = build_engine(0, shards[0], engine_cfg)
    eng1 = None
    if not single:
        engine_cfg = dict(engine_cfg, host_pack_bytes=host1)
        eng1 = build_engine(1, shards[0], engine_cfg)

    source = mimo.LocalShardSource(headers, ckpt)
    log("[model] loading dense tensors (fp8-block dequant + bf16)...")
    t_load = time.monotonic()
    model = mimo.MiMoV2Model.build(
        mcfg, source,
        device0="cuda:0", device1=None if single else "cuda:1",
        engine0=eng0, engine1=eng1,
        split=None if single else (mcfg.n_layers + 1) // 2,
        max_seq_len=int(_env("NATIVE_MAX_SEQ_LEN", "8192")),
        diagnostics=True)
    log(f"[model] built in {time.monotonic() - t_load:.1f}s")

    # Tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    prompt = _env("NATIVE_PROMPT",
                  "Alan Turing (1912-1954) was an English mathematician,")
    ids = tok(prompt, return_tensors=None)["input_ids"]
    input_ids = torch.tensor([ids], dtype=torch.long)

    n_tokens = int(_env("NATIVE_N_TOKENS", "64"))
    if _env("NATIVE_CACHE_RESET", "cold") == "warm":
        log("[run] warm pass (fills engine caches)")
        model.forward(input_ids, 0)
        model.reset_state()

    decode_ms: list[float] = []
    t_gen = time.monotonic()
    result = model.generate(input_ids, n_tokens,
                            decode_timings_ms=decode_ms)
    wall = time.monotonic() - t_gen
    toks = result["tokens"]
    decode_tps = (len(decode_ms) / (sum(decode_ms) / 1000.0)
                  if decode_ms else 0.0)

    st0 = json.loads(eng0.last_stats_json())
    text = tok.decode(toks)

    report = {
        "run_id": run_id, "model": "mimo-flash",
        "repo": mimo.OFFICIAL_REPOSITORY, "revision": mimo.OFFICIAL_REVISION,
        "gpu_count": 1 if single else 2,
        "prompt": prompt, "prompt_len": result["prompt_len"],
        "generated": toks, "n_generated": len(toks), "text": text,
        "decode_tps": round(decode_tps, 4),
        "decode_ms_mean": (round(sum(decode_ms) / len(decode_ms), 2)
                           if decode_ms else None),
        "gen_wall_s": round(wall, 2),
        "total_wall_s": round(time.monotonic() - t_start, 1),
        "engine0_stats": st0,
        "engine1_stats": (json.loads(eng1.last_stats_json())
                          if eng1 else None),
        "env": {k: v for k, v in os.environ.items() if k.startswith("NATIVE_")},
    }
    (work / "report.json").write_text(json.dumps(report, indent=2))
    (work / "generated_ids.jsonl").write_text(
        "".join(json.dumps({"i": i, "id": t}) + "\n"
                for i, t in enumerate(toks)))
    log(f"[done] {len(toks)} tokens in {wall:.1f}s "
        f"decode={decode_tps:.3f} tok/s text={text[:80]!r}")
    print(f"VERDICT: {'PASS' if len(toks) == n_tokens else 'SHORT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
