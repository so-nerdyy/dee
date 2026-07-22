#!/usr/bin/env python3
"""load_shard_swiglu_parity.py — Tier-0.5 component-level SwiGLU parity.

Validates dee.cpp's SwiGLU kernel against ONE real Ornith-1.0-35B expert loaded
directly from a safetensors shard. NO HuggingFace forward path required.

The HF pipeline (scripts/run_real_parity.py) is hard-blocked because
`transformers 4.57.6` does NOT register `qwen3_5_moe_text`. This script is the
smallest viable tier-0.5 milestone: it exercises the dee.cpp cold-load +
expert-lookup + SwiGLU forward path against REAL Ornith weights (not synthetic).

Workflow:
  1. Load /tmp/ornith_min/model-00001-of-00016.safetensors (~4.32 GB, BF16).
  2. Probe safetensors keys to determine the exact expert tensor layout
     (handles Qwen3 / Qwen3.5 / stacked / non-stacked variants).
  3. Pick layer 0, expert 0. Extract gate_proj, up_proj, down_proj as FP32.
  4. Compute torch reference:
        out = down_proj( silu(gate_proj(x)) * up_proj(x) )
  5. Build pydee.Engine pointed at the real shard.
  6. Run engine.moe_forward_experts(layer=0, h_in=x_np,
                                    experts_out=out_np, experts=[0]).
  7. Compare output (with NaN/Inf + value-range pre-metric guards):
        max_abs_err, relative RMSE, cosine sim, top-N elements.
  8. Append ledger record with PASS/FAIL + numerics + parsed engine stats
     + commit/branch/transformers metadata.

Usage:
    python3 scripts/load_shard_swiglu_parity.py
    python3 scripts/load_shard_swiglu_parity.py --expert 7 --seed 1234

Expected pass criteria:
    max_abs_err < 1e-4     (CPU SwiGLU FP32 path; tolerate vector-intrinsic drift)
    cosine_sim  > 0.99999  (matches HF reference direction)

This script is the bypass-HF tier-0.5 path. If pydee.so is not yet compiled,
build first:
    cmake --build build --parallel 4
    python3 pydee/setup.py build_ext --inplace
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
ORNITH_SHARD_0 = Path(
    os.environ.get(
        "ORNITH_SHARD_0",
        "/tmp/ornith_min/model-00001-of-00016.safetensors",
    )
)
LEDGER = REPO_ROOT / "benchmark_reports" / "real_generation_ledger.jsonl"
DEE_BUILD_DIR = Path(os.environ.get("DEE_BUILD_DIR", str(REPO_ROOT / "build")))
PYDEE_DIR = REPO_ROOT / "pydee"
MODEL_TAG = "deepreinforce-ai/Ornith-1.0-35B"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ensure_pydee_in_path() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(PYDEE_DIR))


def _git_meta() -> dict:
    """Safe git commit/branch capture. Falls back to None on failure."""
    out = {"commit": None, "branch": None}
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            out["commit"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            out["branch"] = r.stdout.strip()
    except Exception:
        pass
    return out


def _safe_transformers_version() -> str | None:
    try:
        return importlib.import_module("transformers").__version__
    except Exception:
        return None


def _probe_keys(state: dict) -> dict:
    keys = sorted(state.keys())
    layer0_keys = [k for k in keys if "layers.0." in k]
    expert_keys = [k for k in keys if ".experts." in k or ".mlp.expert" in k]
    return {
        "total": len(keys),
        "layer0_count": len(layer0_keys),
        "layer0_samples": layer0_keys[:30],
        "expert_count": len(expert_keys),
        "expert_samples": expert_keys[:30],
        "all_samples": keys[:15],
    }


def _resolve_expert_keys(
    state: dict, layer: int, expert: int
) -> dict | None:
    """Return {gate: key, up: key, down: key} or None if nothing matches.

    Order of resolution:
      1. Stacked: model.layers.{L}.mlp.experts.{gate,up,down}_proj.weight
      2. Stacked with wrappers: model.model.layers.{L}.mlp.experts.* etc
      3. Individual ModuleList:
         model.layers.{L}.mlp.experts.{N}.{g,u,d}_proj.weight
    """
    layer_seg = f"layers.{layer}."
    expert_module_seg = ".mlp.experts."
    # Stacked format candidates.
    for prefix in (
        f"model.{layer_seg}{expert_module_seg}",
        f"model.model.{layer_seg}{expert_module_seg}",
        f"model.language_model.{layer_seg}{expert_module_seg}",
        f"model.model.language_model.{layer_seg}{expert_module_seg}",
    ):
        gk = prefix + "gate_proj.weight"
        uk = prefix + "up_proj.weight"
        dk = prefix + "down_proj.weight"
        if all(k in state for k in (gk, uk, dk)):
            return {"format": "stacked", "gate": gk, "up": uk, "down": dk}
    # Individual expert ModuleList candidates.
    for prefix in (
        f"model.{layer_seg}{expert_module_seg}{expert}.",
        f"model.model.{layer_seg}{expert_module_seg}{expert}.",
        f"model.language_model.{layer_seg}{expert_module_seg}{expert}.",
        f"model.model.language_model.{layer_seg}{expert_module_seg}{expert}.",
    ):
        gk = prefix + "gate_proj.weight"
        uk = prefix + "up_proj.weight"
        dk = prefix + "down_proj.weight"
        if all(k in state for k in (gk, uk, dk)):
            return {
                "format": "individual",
                "gate": gk,
                "up": uk,
                "down": dk,
            }
    return None


def _load_real_expert(
    state: dict, layer: int, expert: int
) -> tuple:
    """Returns (W_g, W_u, W_d, layout_dict, dtype_dict). All FP32."""
    layout = _resolve_expert_keys(state, layer, expert)
    if layout is None:
        raise RuntimeError(
            f"no expert tensor format matched for layer={layer} expert={expert}"
        )
    dtype_orig = {
        layout["gate"]: str(state[layout["gate"]].dtype),
        layout["up"]:   str(state[layout["up"]].dtype),
        layout["down"]: str(state[layout["down"]].dtype),
    }
    return (
        state[layout["gate"]].float(),
        state[layout["up"]].float(),
        state[layout["down"]].float(),
        layout,
        dtype_orig,
    )


def _torch_reference(W_g: torch.Tensor,
                     W_u: torch.Tensor,
                     W_d: torch.Tensor,
                     x: torch.Tensor) -> torch.Tensor:
    """down_proj( silu(gate_proj(x)) * up_proj(x) )"""
    with torch.inference_mode():
        g = F.linear(x, W_g)
        u = F.linear(x, W_u)
        i = F.silu(g) * u
        return F.linear(i, W_d)


def _finite_check(name: str, arr: np.ndarray) -> str | None:
    """Return None if all finite, else a diagnostic string."""
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        return f"{name}: not finite (NaN={n_nan}, Inf={n_inf})"
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description="dee.cpp tier-0.5 single-expert SwiGLU parity vs torch"
    )
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--layer",  type=int, default=0)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--topn-elements", type=int, default=8)
    args = p.parse_args()

    _ensure_pydee_in_path()

    git_meta = _git_meta()
    transformers_ver = _safe_transformers_version()
    ts = _now_iso()
    print(f"[tier-0.5] ts={ts}  branch={git_meta['branch']}  "
          f"commit={git_meta['commit']}  transformers={transformers_ver}")

    print(f"[tier-0.5] shard path: {ORNITH_SHARD_0}")
    if not ORNITH_SHARD_0.exists():
        raise FileNotFoundError(
            f"safetensors shard missing at {ORNITH_SHARD_0}. "
            f"Download via `huggingface_hub.hf_hub_download("
            f"'{MODEL_TAG}', 'model-00001-of-00016.safetensors')`."
        )

    print("[tier-0.5] loading safetensors...")
    from safetensors.torch import load_file
    state = load_file(str(ORNITH_SHARD_0), device="cpu")
    print(f"[tier-0.5] loaded {len(state)} tensors "
          f"({ORNITH_SHARD_0.stat().st_size / 1024**3:.2f} GiB)")

    probe = _probe_keys(state)
    print(f"[tier-0.5] probe: layer0_count={probe['layer0_count']}, "
          f"expert_count={probe['expert_count']}, total={probe['total']}")
    print("[tier-0.5] all-sample keys (first 15):")
    for k in probe["all_samples"]:
        print(f"           {k}")
    print("[tier-0.5] layer-0 sample keys (first 30):")
    for k in probe["layer0_samples"]:
        print(f"           {k}")
    print("[tier-0.5] expert sample keys (first 30):")
    for k in probe["expert_samples"]:
        print(f"           {k}")

    try:
        W_g, W_u, W_d, layout, dtype_orig = _load_real_expert(
            state, args.layer, args.expert
        )
    except RuntimeError as exc:
        print(f"[tier-0.5] FAIL: {exc}")
        _append_ledger({
            "id": f"TIER0.5-SWIGLU-NO-FORMAT-{ts}",
            "stage": "tier0.5",
            "timestamp": ts,
            "commit": git_meta["commit"],
            "branch": git_meta["branch"],
            "transformers": transformers_ver,
            "model_revision": MODEL_TAG,
            "result": "FAIL: no expert format resolved",
            "probe": probe,
            "layer": args.layer,
            "expert": args.expert,
        })
        return 2

    H = int(W_g.shape[1])
    I = int(W_g.shape[0])
    print(f"[tier-0.5] expert layout={layout['format']}; gate={layout['gate']}; "
          f"up={layout['up']}; down={layout['down']}")
    print(f"[tier-0.5] original safetensors dtypes: gate={dtype_orig[layout['gate']]} "
          f"up={dtype_orig[layout['up']]} down={dtype_orig[layout['down']]}")
    print(f"[tier-0.5] W_g={tuple(W_g.shape)} W_u={tuple(W_u.shape)} "
          f"W_d={tuple(W_d.shape)}  H={H} I={I}")
    if W_u.shape[1] != H or W_d.shape != (H, I):
        print(
            f"[tier-0.5] FAIL: unexpected shapes "
            f"W_g={tuple(W_g.shape)} W_u={tuple(W_u.shape)} "
            f"W_d={tuple(W_d.shape)}  H={H} I={I}"
        )
        _append_ledger({
            "id": f"TIER0.5-SWIGLU-BAD-SHAPE-{ts}",
            "stage": "tier0.5",
            "timestamp": ts,
            "commit": git_meta["commit"],
            "branch": git_meta["branch"],
            "transformers": transformers_ver,
            "model_revision": MODEL_TAG,
            "result": "FAIL: unexpected weight shapes",
            "shapes": {
                "W_g": list(W_g.shape), "W_u": list(W_u.shape),
                "W_d": list(W_d.shape), "H": H, "I": I,
            },
            "dtype_orig": dtype_orig,
            "layout": layout,
        })
        return 3

    # Torch reference.
    torch.manual_seed(args.seed)
    x = torch.randn(H, dtype=torch.float32)
    ref = _torch_reference(W_g, W_u, W_d, x)
    ref_np = ref.cpu().numpy().astype(np.float32)
    print(f"[tier-0.5] ref (first 6) = {ref_np[:6].tolist()}")

    # pydee Engine init.
    import pydee
    if pydee.Engine is None:
        print(
            "[tier-0.5] FAIL: pydee compiled binding not importable. "
            f"Build first: cmake --build {DEE_BUILD_DIR} && "
            f"python3 {PYDEE_DIR}/setup.py build_ext --inplace"
        )
        _append_ledger({
            "id": f"TIER0.5-SWIGLU-NO-PYDEE-{ts}",
            "stage": "tier0.5",
            "timestamp": ts,
            "commit": git_meta["commit"],
            "branch": git_meta["branch"],
            "transformers": transformers_ver,
            "model_revision": MODEL_TAG,
            "result": "FAIL: pydee.so not importable",
            "shard_path": str(ORNITH_SHARD_0),
        })
        return 4

    cfg = pydee.EngineConfig()
    cfg.shard_path = str(ORNITH_SHARD_0)
    cfg.oracle_path = ""
    cfg.num_experts = 256
    cfg.num_layers  = 40
    cfg.hidden      = H
    cfg.inter       = I
    cfg.topk        = 8
    cfg.use_cuda    = False
    cfg.transfer_dtype = pydee.WeightTransferDType.Bf16
    cfg.cache_dtype    = pydee.DeviceCacheDType.Fp32
    cfg.verbose    = False
    engine = pydee.Engine()
    print("[tier-0.5] calling engine.init(cfg)...")
    t0 = time.time()
    if not engine.init(cfg):
        print("[tier-0.5] FAIL: engine.init returned False")
        _append_ledger({
            "id": f"TIER0.5-SWIGLU-INIT-FAIL-{ts}",
            "stage": "tier0.5",
            "timestamp": ts,
            "commit": git_meta["commit"],
            "branch": git_meta["branch"],
            "transformers": transformers_ver,
            "model_revision": MODEL_TAG,
            "result": "FAIL: engine.init returned False",
            "shard_path": str(ORNITH_SHARD_0),
            "config": {
                "num_experts": 256, "num_layers": 40,
                "hidden": H, "inter": I,
            },
        })
        return 5
    init_ms = (time.time() - t0) * 1000.0
    engine_hidden = int(engine.hidden_dim())
    engine_inter = int(engine.inter_dim())
    print(f"[tier-0.5] engine.init OK in {init_ms:.1f} ms; engine.H={engine_hidden} "
          f"engine.I={engine_inter}")
    if engine_hidden != H or engine_inter != I:
        print(f"[tier-0.5] engine dims mismatch: got "
              f"H={engine_hidden} I={engine_inter}, expected H={H} I={I}")
        return 6

    # Cold-load + SwiGLU forward via pydee.
    x_np = x.cpu().numpy().astype(np.float32)
    out_np = np.zeros(H, dtype=np.float32)
    print(f"[tier-0.5] calling moe_forward_experts(layer={args.layer}, "
          f"experts=[{args.expert}])...")
    t0 = time.time()
    ok = engine.moe_forward_experts(args.layer, x_np, out_np, [int(args.expert)])
    fwd_ms = (time.time() - t0) * 1000.0
    print(f"[tier-0.5] moe_forward_experts returned {ok} in {fwd_ms:.1f} ms")
    if not ok:
        _append_ledger({
            "id": f"TIER0.5-SWIGLU-FWD-FAIL-{ts}",
            "stage": "tier0.5",
            "timestamp": ts,
            "commit": git_meta["commit"],
            "branch": git_meta["branch"],
            "transformers": transformers_ver,
            "model_revision": MODEL_TAG,
            "result": "FAIL: moe_forward_experts returned False",
            "init_ms": init_ms,
            "fwd_ms": fwd_ms,
        })
        return 7

    # Pre-metric guards (reviewer B).
    bad = []
    for nm, arr in (("ref", ref_np), ("dee", out_np)):
        msg = _finite_check(nm, arr)
        if msg:
            bad.append(msg)
    ref_max = float(np.max(np.abs(ref_np)))
    dee_max = float(np.max(np.abs(out_np)))
    print(f"[tier-0.5] value ranges: |ref|_max={ref_max:.4e}  |dee|_max={dee_max:.4e}")
    if bad:
        print("[tier-0.5] FAIL: pre-metric guard tripped: " + "; ".join(bad))
        _append_ledger({
            "id": f"TIER0.5-SWIGLU-NONFINITE-{ts}",
            "stage": "tier0.5",
            "timestamp": ts,
            "commit": git_meta["commit"],
            "branch": git_meta["branch"],
            "transformers": transformers_ver,
            "model_revision": MODEL_TAG,
            "result": "FAIL: non-finite values in reference or dee output",
            "errors": bad,
            "ref_max_abs": ref_max,
            "dee_max_abs": dee_max,
            "init_ms": init_ms,
            "fwd_ms": fwd_ms,
        })
        return 8

    # Numeric comparison.
    elem_err = np.abs(ref_np - out_np)
    abs_err = float(elem_err.max())
    rel_rmse = float(np.sqrt(np.mean(elem_err ** 2)) /
                      (np.sqrt(np.mean(ref_np ** 2)) + 1e-12))
    denom = float(np.linalg.norm(ref_np) * np.linalg.norm(out_np))
    cos = float(np.dot(ref_np, out_np) / (denom + 1e-12))
    top_idx = elem_err.argsort()[::-1][: args.topn_elements]
    print(f"[tier-0.5] max_abs_err={abs_err:.6e}  rel_rmse={rel_rmse:.6e}  "
          f"cosine={cos:.6f}")
    print("[tier-0.5] top-N worst elements (idx, ref, dee, |err|):")
    for i in top_idx.tolist():
        print(f"           {i:5d}  ref={ref_np[i]:+.4e}  dee={out_np[i]:+.4e}  "
              f"|err|={elem_err[i]:+.4e}")
    print(f"[tier-0.5] dee (first 6) = {out_np[:6].tolist()}")
    PASS = (abs_err < 1.0e-4) and (cos > 0.99999)
    print(f"[tier-0.5] PASS = {PASS}  (gate: max_abs_err < 1e-4 && cosine > 0.99999)")

    # Parse engine stats (reviewer D).
    engine_stats_dict: dict = {}
    try:
        engine_stats_dict = json.loads(engine.last_stats_json())
    except Exception:
        engine_stats_dict = {"_raw": engine.last_stats_json()}

    _append_ledger({
        "id": f"TIER0.5-SWIGLU-{ts}",
        "stage": "tier0.5",
        "timestamp": ts,
        "commit": git_meta["commit"],
        "branch": git_meta["branch"],
        "transformers": transformers_ver,
        "model_revision": MODEL_TAG,
        "shard_path": str(ORNITH_SHARD_0),
        "shard_size_bytes": ORNITH_SHARD_0.stat().st_size,
        "layer": args.layer,
        "expert": args.expert,
        "hidden": H,
        "inter": I,
        "layout": layout,
        "dtype_orig": dtype_orig,
        "shape_g": list(W_g.shape),
        "shape_u": list(W_u.shape),
        "shape_d": list(W_d.shape),
        "init_ms": round(init_ms, 3),
        "fwd_ms": round(fwd_ms, 3),
        "ref_max_abs": ref_max,
        "dee_max_abs": dee_max,
        "max_abs_err": abs_err,
        "rel_rmse": rel_rmse,
        "cosine_sim": cos,
        "top_worst_idx": top_idx.tolist(),
        "top_worst_abs_err": elem_err[top_idx].tolist(),
        "ref_first6": ref_np[:6].tolist(),
        "dee_first6": out_np[:6].tolist(),
        "pass": bool(PASS),
        "result": "PASS" if PASS else "FAIL",
        "engine_stats": engine_stats_dict,
    })
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
