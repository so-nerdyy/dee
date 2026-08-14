#!/usr/bin/env python3
"""Real-weight throughput: native FP4 FFN vs DS8 host-dequant on one official
DeepSeek-V4-Flash-0731 layer (model-00002 = layer 0, 256 routed experts).

This is the first real-TPS-relevant measurement: it streams the OFFICIAL
packed-I8 + per-block e8m0 expert weights at the real 4096x2048 dims and
measures the two candidate paths head-to-head:

  A. native  -- pydee.Engine.moe_forward_experts: mmap -> H2D of the COMPRESSED
               FP4 bytes -> on-device fp4_e2m1_to_f16_cuda dequant -> cuBLAS
               SwiGLU (this replaces the DS8 host-dequant on the critical path).
  B. host    -- the DS8/DS10 path: torch FP4->FP16 dequant on CPU -> pin ->
               H2D of FP16 -> torch cuBLAS SwiGLU.

Reports per-token wall time (6 experts, cold stream), H2D bytes, and a
projected full-model decode tok/s (43 layers x 6 experts, FFN-dominated per
the sealed DS10 trace).  This is NOT full generated-text inference -- the
dense path (embed/attention/RMSNorm/LM-head) is not run here.

Usage (on the T4, after building pydee):
    python scripts/deepseek_v4_native_throughput.py --shard /path/to/model-00002-of-00048.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_DEE_CPP_ROOT = str(Path(__file__).resolve().parent.parent)
if _DEE_CPP_ROOT not in sys.path:
    sys.path.insert(0, _DEE_CPP_ROOT)

import numpy as np
import torch

from scripts import deepseek_v4_expert_reference as ds7

HIDDEN = 4096
INTER = 2048
N_EXPERTS = 256
TOPK = 6
SWIGLU_LIMIT = 10.0
# First token's experts from the sealed DS10 trace (token 0, layer 0).
SEALED_FIRST_EXPERTS = [254, 222, 245, 200, 53, 35]


def _expert_tensors(eid: int) -> list[str]:
    return [f"layers.0.ffn.experts.{eid}.w{i}.{s}"
            for i in (1, 2, 3) for s in ("weight", "scale")]


def load_experts(shard_path: Path, eids: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    from safetensors import safe_open
    out: dict[int, dict[str, torch.Tensor]] = {}
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for eid in eids:
            t: dict[str, torch.Tensor] = {}
            for full in _expert_tensors(eid):
                if full not in keys:
                    raise KeyError(f"missing {full}")
                short = full[len(f"layers.0.ffn.experts.{eid}."):]
                t[short] = f.get_tensor(full).contiguous()
            out[eid] = t
    return out


def bench_native(engine, h_in: np.ndarray, eids: list[int],
                 per_token_budget: int) -> tuple[float, dict[str, Any]]:
    """Time one cold native forward (6 experts) via moe_forward_experts."""
    k = len(eids)
    out_np = np.empty((k * HIDDEN,), dtype=np.float32)
    engine.reset_runtime_cache()  # force a cold stream (fresh experts)
    t0 = time.perf_counter()
    ok = engine.moe_forward_experts(0, h_in, out_np, eids)
    dt = (time.perf_counter() - t0) * 1000.0
    if not ok:
        raise RuntimeError("moe_forward_experts failed")
    return dt, {"native_fwd_ms": dt}


def bench_host(experts: dict[int, dict[str, torch.Tensor]],
               eids: list[int], x: torch.Tensor) -> tuple[float, dict[str, Any]]:
    """Time the DS8 host-dequant path: FP4->FP16 on CPU, pin, H2D, matmul."""
    dev = x.device
    t_deq = t_h2d = t_mm = 0.0
    xc = x.half()
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=dev)
    for eid in eids:
        t = experts[eid]
        t0 = time.perf_counter()
        w1 = ds7.dequantize_expert_weight(t["w1.weight"], t["w1.scale"]).half()
        w2 = ds7.dequantize_expert_weight(t["w2.weight"], t["w2.scale"]).half()
        w3 = ds7.dequantize_expert_weight(t["w3.weight"], t["w3.scale"]).half()
        t_deq += time.perf_counter() - t0
        t0 = time.perf_counter()
        w1c = w1.pin_memory().to(dev, non_blocking=False)
        w2c = w2.pin_memory().to(dev, non_blocking=False)
        w3c = w3.pin_memory().to(dev, non_blocking=False)
        t_h2d += time.perf_counter() - t0
        t0 = time.perf_counter()
        gate = torch.clamp((xc @ w1c.t()).float(), max=SWIGLU_LIMIT)
        up = torch.clamp((xc @ w3c.t()).float(), min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        h = torch.nn.functional.silu(gate) * up
        out += (h.half() @ w2c.t()).float()
        t_mm += time.perf_counter() - t0
    torch.cuda.synchronize()
    return (t_deq + t_h2d + t_mm) * 1000.0, {
        "dequant_ms": t_deq * 1000.0, "h2d_ms": t_h2d * 1000.0,
        "matmul_ms": t_mm * 1000.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--tokens", type=int, default=5)
    ap.add_argument("--budget-bytes", type=int, default=8 * 3 * INTER * HIDDEN * 2)
    args = ap.parse_args()

    try:
        import pydee
        assert pydee.Engine is not None
    except Exception as exc:  # pragma: no cover
        print(f"[tp] pydee not importable ({exc!r})", flush=True)
        return 2

    shard_path = Path(args.shard)
    if not shard_path.is_file():
        print(f"[tp] missing shard {shard_path}", flush=True)
        return 2

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Expert sets: token 0 = sealed DS10 routing, then distinct cycling sets.
    rng = np.random.default_rng(0)
    token_experts: list[list[int]] = [SEALED_FIRST_EXPERTS]
    for _ in range(args.tokens - 1):
        token_experts.append([int(e) for e in rng.choice(N_EXPERTS, TOPK, replace=False)])
    all_eids = sorted({e for es in token_experts for e in es})
    experts = load_experts(shard_path, all_eids)
    print(f"[tp] loaded {len(experts)} experts from {shard_path.name}", flush=True)

    cfg = pydee.configure(
        shard_path=str(shard_path), num_experts=N_EXPERTS, num_layers=1,
        hidden=HIDDEN, inter=INTER, use_cuda=(dev == "cuda"),
        transfer_dtype="fp4", cache_dtype="fp16", topk=TOPK,
        budget_bytes=args.budget_bytes, swiglu_limit=SWIGLU_LIMIT)
    engine = pydee.new_engine(cfg)
    print(f"[tp] native engine initialized (hidden={engine.hidden_dim()}, "
          f"inter={engine.inter_dim()})", flush=True)

    x = torch.randn(1, HIDDEN, dtype=torch.float32)
    h_in = x.numpy().astype(np.float32)

    native_ms: list[float] = []
    host_ms: list[float] = []
    host_breakdown: list[dict[str, float]] = []
    for i, eids in enumerate(token_experts):
        ndt, _ = bench_native(engine, h_in, eids, args.budget_bytes)
        native_ms.append(ndt)
        hdt, hb = bench_host(experts, eids, x.to(dev))
        host_ms.append(hdt)
        host_breakdown.append(hb)
        print(f"[tp] token {i}: native {ndt:.1f} ms | host {hdt:.1f} ms "
              f"({hb['dequant_ms']:.0f}/{hb['h2d_ms']:.0f}/{hb['matmul_ms']:.0f} ms)",
              flush=True)

    native_med = float(np.median(native_ms))
    host_med = float(np.median(host_ms))
    # Compressed FP4 bytes per expert (3 projections, packed I8 + e8m0 scales).
    comp_per_expert = (3 * (INTER * HIDDEN // 2) + (INTER * HIDDEN // 32)
                       + (INTER * HIDDEN // 32) + (HIDDEN * INTER // 2)
                       + (HIDDEN * INTER // 32) + (HIDDEN * INTER // 32))
    fp16_per_expert = 3 * INTER * HIDDEN * 2
    h2d_native_per_token = comp_per_expert * TOPK
    h2d_host_per_token = fp16_per_expert * TOPK
    speedup = host_med / native_med if native_med > 0 else float("inf")
    # Projected decode: 43 layers x 6 experts, FFN-dominated (DS10 trace).
    projected_decode_s = native_med / 1000.0 * 43
    projected_tok_s = 1.0 / projected_decode_s if projected_decode_s > 0 else float("inf")

    result = {
        "shard": shard_path.name,
        "dims": {"hidden": HIDDEN, "inter": INTER},
        "topk": TOPK, "n_layers_projected": 43,
        "native_median_ms": native_med,
        "host_median_ms": host_med,
        "speedup_x": speedup,
        "h2d_native_bytes_per_token": h2d_native_per_token,
        "h2d_host_bytes_per_token": h2d_host_per_token,
        "projected_decode_s_per_token": projected_decode_s,
        "projected_decode_tok_s": projected_tok_s,
        "native_ms_per_token": native_ms,
        "host_ms_per_token": host_ms,
        "host_breakdown_ms": host_breakdown,
    }
    print("[tp] " + json.dumps(result, indent=2), flush=True)
    out = Path(args.shard).with_suffix(".throughput.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"[tp] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
