#!/usr/bin/env python3
"""Native FP4 FFN parity: DeepseekV4NativeFfn vs the trusted FP32 MoE reference.

Generates a mini DeepSeek-V4-Flash-0731 shard (4 routed experts, packed FP4
e2m1fn weights + per-block e8m0 scales) under the official
``layers.{l}.ffn.experts.{e}.w1/w2/w3.weight`` + ``.scale`` naming, builds a
pydee.Engine (transfer_dtype=fp4, FP16 device cache, CUDA), and runs
DeepseekV4NativeFfn against ``moe.moe_layer_forward`` (the DS8 FP32 trusted
reference) on identical inputs + routing.

The native path computes gate/up/down in FP16 (the harness's own precision);
the reference computes in FP32.  Per the DS7 documented policy, near-bitwise
agreement is NOT expected -- the gate is a loose FP16 tolerance.  Report
max-abs error, relative RMSE, and cosine similarity.

Usage (on the T4, after ``python setup.py build_ext --inplace`` in pydee/):
    cd dee.cpp && python scripts/deepseek_v4_native_ffn_parity.py
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

# `scripts` is a namespace package: ensure the dee.cpp root (parent of
# scripts/) is importable regardless of how this file is invoked.
_DEE_CPP_ROOT = str(Path(__file__).resolve().parent.parent)
if _DEE_CPP_ROOT not in sys.path:
    sys.path.insert(0, _DEE_CPP_ROOT)

import numpy as np
import torch

from scripts import deepseek_v4_expert_reference as ds7
from scripts import deepseek_v4_layer_reference as layer_ref
from scripts import deepseek_v4_moe_reference as moe
from scripts.deepseek_v4_layer_candidate import DeepseekV4NativeFfn

# Both 64-aligned: the fp4_e2m1_to_f16_cuda kernel requires `in % 64 == 0`.
HIDDEN = 128
INTER = 64
N_ROUTED = 4
TOPK = 2
N_TOKENS = 2
SWIGLU_LIMIT = 10.0
ROUTE_SCALE = 1.5

# Realistic ue8m0 scale bytes 0x7D..0x83 -> scales 0.25..8.0 (no overflow).
_SCALE_BYTES = (0x7D, 0x83)


def _packed_proj(rng, out, in_dim, name):
    """Packed I8 [out, in//2] + F8_E8M0 scale [out, in//32] for one projection."""
    packed = np.zeros((out, in_dim // 2), dtype=np.uint8)
    for o in range(out):
        for c in range(in_dim // 2):
            lo = int(rng.integers(0, 16))
            hi = int(rng.integers(0, 16))
            packed[o, c] = (lo | (hi << 4)) & 0xFF
    scale = rng.integers(*_SCALE_BYTES, size=(out, in_dim // 32),
                         dtype=np.uint8)
    return packed, scale


def _write_mini_shard(path: Path, routed: dict[int, dict[str, np.ndarray]]) -> None:
    """Write a mini DEEPSEEK_V4 shard (layer 0 routed experts) to disk.

    Standard safetensors layout: 8-byte LE header length, JSON header with
    the official dtype strings ("I8" packed weights, "F8_E8M0" scales), then
    the concatenated tensor bytes.  (Inlined from deepseek_v4_staging to keep
    this test free of the full model-stack import chain.)
    """
    header: dict[str, Any] = {}
    data = bytearray()
    for eid, t in routed.items():
        for kind in ("w1", "w3", "w2"):
            for suffix, dtype in (("weight", "I8"), ("scale", "F8_E8M0")):
                blob = t[f"{kind}.{suffix}"].tobytes()
                start = len(data)
                header[f"layers.0.ffn.experts.{eid}.{kind}.{suffix}"] = {
                    "dtype": dtype,
                    "shape": [int(d) for d in t[f"{kind}.{suffix}"].shape],
                    "data_offsets": [start, start + len(blob)],
                }
                data += blob
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_json)))
        fh.write(header_json)
        fh.write(bytes(data))


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    c = candidate.detach().float().flatten()
    r = reference.detach().float().flatten()
    err = (c - r).abs()
    max_abs = float(err.max())
    ss_err = float((c - r).pow(2).sum())
    ss_ref = float(r.pow(2).sum())
    rel_rmse = float((ss_err / max(ss_ref, 1e-12)) ** 0.5)
    cosine = float((c @ r) / max((c.norm() * r.norm()).item(), 1e-12))
    finite = bool(torch.isfinite(c).all() and torch.isfinite(r).all())
    return {"max_abs_err": max_abs, "rel_rmse": rel_rmse, "cosine": cosine,
            "finite": finite}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="tmp/dsv4_native_ffn_parity.safetensors")
    ap.add_argument("--budget-bytes", type=int, default=0)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    try:
        import pydee  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"[parity] pydee not importable ({exc!r}); build it first")
        return 2

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # Routed experts: packed FP4 (I8) weights + e8m0 scales.
    routed: dict[int, dict[str, np.ndarray]] = {}
    for eid in range(N_ROUTED):
        w1, w1s = _packed_proj(rng, INTER, HIDDEN, "w1")
        w3, w3s = _packed_proj(rng, INTER, HIDDEN, "w3")
        w2, w2s = _packed_proj(rng, HIDDEN, INTER, "w2")
        routed[eid] = {"w1.weight": w1, "w1.scale": w1s,
                       "w3.weight": w3, "w3.scale": w3s,
                       "w2.weight": w2, "w2.scale": w2s}

    # Router gate (score-routed layer) + input.
    gate_w = torch.randn(N_ROUTED, HIDDEN, dtype=torch.float32) * 0.1
    gate_b = torch.randn(N_ROUTED, dtype=torch.float32) * 0.05
    x = torch.randn(N_TOKENS, HIDDEN, dtype=torch.float32)

    # Reference (FP32 trusted oracle) through the DS8 MoE reference.
    routed_t = {eid: {k: torch.from_numpy(v) for k, v in t.items()}
                for eid, t in routed.items()}
    ref = moe.moe_layer_forward(
        x, gate_w, gate_b, routed_t, None, topk=TOPK,
        route_scale=ROUTE_SCALE, score_func="sqrtsoftplus",
        swiglu_limit=SWIGLU_LIMIT)

    # Write the shard and build the native engine.
    shard_path = Path(args.shard)
    _write_mini_shard(shard_path, routed)

    cfg = pydee.configure(
        shard_path=str(shard_path), num_experts=N_ROUTED, num_layers=1,
        hidden=HIDDEN, inter=INTER, use_cuda=not args.no_cuda,
        transfer_dtype="fp4", cache_dtype="fp16", topk=TOPK,
        budget_bytes=args.budget_bytes, swiglu_limit=SWIGLU_LIMIT)
    engine = pydee.new_engine(cfg)

    small_cfg = layer_ref.LayerConfig(
        hidden=HIDDEN, n_routed=N_ROUTED, topk=TOPK,
        route_scale=ROUTE_SCALE, swiglu_limit=SWIGLU_LIMIT)
    ffn = DeepseekV4NativeFfn(engine=engine, layer_id=0, cfg=small_cfg,
                              device="cuda" if not args.no_cuda else "cpu",
                              shared_payload=None)
    ffn.attach_gate(gate_w, gate_b)
    ffn.attach_hash(None)

    # Candidate: [b=1, s=N_TOKENS, H].
    xb = x.unsqueeze(0)
    cand = ffn(xb, None, None).reshape(N_TOKENS, HIDDEN)

    m = _metrics(cand, ref["moe_output"])
    print("[parity] native FFN vs FP32 reference:")
    print(f"  max_abs_err = {m['max_abs_err']:.6g}")
    print(f"  rel_rmse    = {m['rel_rmse']:.6g}")
    print(f"  cosine      = {m['cosine']:.9f}")
    print(f"  finite      = {m['finite']}")
    print(f"  routed ids  = {ref['expert_ids'].tolist()}")
    print(f"  ffn stats   = {ffn.stats}")

    result = {
        "max_abs_err": m["max_abs_err"], "rel_rmse": m["rel_rmse"],
        "cosine": m["cosine"], "finite": m["finite"],
        "native_fwd_ms": ffn.stats.get("native_fwd_ms", 0.0),
        "native_calls": ffn.stats.get("native_calls", 0),
    }
    out_path = shard_path.with_suffix(".json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[parity] wrote {out_path}")

    ok = m["finite"] and m["rel_rmse"] < 0.02 and m["cosine"] > 0.999
    print("[parity] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
