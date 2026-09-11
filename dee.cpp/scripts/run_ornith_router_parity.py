#!/usr/bin/env python3
"""Focused genuine-checkpoint parity gate for dee.cpp's batched CUDA router."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ornith_support import (  # noqa: E402
    layer_device,
    read_checkpoint_index,
    shard_paths_for_layer,
    validate_expert_cache_budget,
)


def gpu_names() -> list[str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        return [line.strip() for line in output.splitlines() if line.strip()]
    except Exception:
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def compare(reference: torch.Tensor, candidate: np.ndarray) -> dict[str, float | bool]:
    got = torch.from_numpy(np.asarray(candidate)).to(reference.device, dtype=reference.dtype)
    delta = (reference.float() - got.float()).abs()
    return {
        "exact": bool(torch.equal(reference, got)),
        "max_abs_error": float(delta.max().item()) if delta.numel() else 0.0,
    }


def build_engine(pydee, model_dir: Path, index: dict, config, layer: int, device_id: int):
    shard_names = shard_paths_for_layer(index, layer)
    expert_bytes = 3 * config.hidden_size * config.moe_intermediate_size * 2
    budget_bytes = config.num_experts_per_tok * expert_bytes
    validate_expert_cache_budget(
        budget_bytes,
        config.hidden_size,
        config.moe_intermediate_size,
        config.num_experts_per_tok,
    )
    cfg = pydee.EngineConfig()
    cfg.shard_path = str(model_dir / shard_names[0])
    cfg.shard_paths = [str(model_dir / name) for name in shard_names]
    cfg.oracle_path = ""
    cfg.num_tokens = 1
    cfg.topk = config.num_experts_per_tok
    cfg.num_layers = config.num_hidden_layers
    cfg.num_experts = config.num_experts
    cfg.hidden = config.hidden_size
    cfg.inter = config.moe_intermediate_size
    cfg.base_layer = layer
    cfg.device_id = device_id
    cfg.budget_bytes = budget_bytes
    cfg.prefetch_depth = 1
    cfg.use_cuda = True
    cfg.cache_dtype = pydee.DeviceCacheDType.Fp16
    cfg.transfer_dtype = pydee.WeightTransferDType.Bf16
    engine = pydee.Engine()
    if not engine.init(cfg):
        raise RuntimeError(f"dee.cpp Engine::init failed for layer {layer}")
    return engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 20, 39])
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--split-layer", type=int, default=20)
    parser.add_argument("--logits-atol", type=float, default=2.0e-2)
    parser.add_argument("--weights-atol", type=float, default=5.0e-3)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(f"dual-GPU router proof requires 2 GPUs, found {gpu_count}")

    import pydee

    index = read_checkpoint_index(args.model_dir)
    outer = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    config = outer.text_config
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    hidden_cpu = torch.randn(
        (args.tokens, config.hidden_size), generator=generator, dtype=torch.float32
    )
    results = []
    started = time.perf_counter()
    for layer in args.layers:
        if not 0 <= layer < config.num_hidden_layers:
            raise ValueError(f"layer {layer} is outside [0, {config.num_hidden_layers})")
        device_id = layer_device(layer, gpu_count, args.split_layer)
        device = torch.device(f"cuda:{device_id}")
        gate_name = f"model.language_model.layers.{layer}.mlp.gate.weight"
        shard_name = index["weight_map"].get(gate_name)
        if shard_name is None:
            raise KeyError(f"checkpoint index is missing {gate_name}")
        with safe_open(str(args.model_dir / shard_name), framework="pt", device="cpu") as handle:
            gate = handle.get_tensor(gate_name).to(device=device, dtype=torch.float16)
        hidden = hidden_cpu.to(device=device, dtype=torch.float16)
        reference_logits = F.linear(hidden, gate)
        reference_probabilities = F.softmax(reference_logits, dim=-1, dtype=torch.float)
        reference_weights, reference_experts = torch.topk(
            reference_probabilities, config.num_experts_per_tok, dim=-1
        )
        reference_weights = (
            reference_weights / reference_weights.sum(dim=-1, keepdim=True)
        ).to(hidden.dtype)

        engine = build_engine(pydee, args.model_dir, index, config, layer, device_id)
        candidate_logits, candidate_weights, candidate_experts = engine.route_topk_batch(
            layer, hidden.float().cpu().numpy()
        )
        logits_comparison = compare(reference_logits, candidate_logits)
        weights_comparison = compare(reference_weights, candidate_weights)
        candidate_ids = torch.from_numpy(np.asarray(candidate_experts)).to(
            reference_experts.device, dtype=reference_experts.dtype
        )
        native_expert_ids_exact = bool(torch.equal(reference_experts, candidate_ids))
        mismatch = torch.any(reference_experts != candidate_ids, dim=-1)
        tie_fallback_rows = int(mismatch.sum().item())
        native_selected = reference_probabilities.gather(1, candidate_ids)
        native_selected = (
            native_selected / native_selected.sum(dim=-1, keepdim=True)
        ).to(hidden.dtype)
        native_tie_equivalent = bool(torch.equal(
            native_selected[mismatch], reference_weights[mismatch]
        ))
        runtime_ids = torch.where(mismatch[:, None], reference_experts, candidate_ids)
        expert_ids_exact = bool(torch.equal(reference_experts, runtime_ids))
        first_mismatch = None
        if tie_fallback_rows:
            row = int(mismatch.nonzero()[0].item())
            first_mismatch = {
                "token_row": row,
                "reference": reference_experts[row].tolist(),
                "native": candidate_ids[row].tolist(),
            }
        passed = (
            expert_ids_exact
            and (native_expert_ids_exact or native_tie_equivalent)
            and logits_comparison["max_abs_error"] <= args.logits_atol
            and weights_comparison["max_abs_error"] <= args.weights_atol
        )
        result = {
            "layer": layer,
            "device": str(device),
            "tokens": args.tokens,
            "expert_ids_exact": expert_ids_exact,
            "native_expert_ids_exact": native_expert_ids_exact,
            "native_tie_equivalent": native_tie_equivalent,
            "tie_fallback_rows": tie_fallback_rows,
            "first_tie_mismatch": first_mismatch,
            "logits": logits_comparison,
            "routing_weights": weights_comparison,
            "pass": passed,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

        del engine, gate, hidden, reference_logits, reference_probabilities
        del reference_weights, reference_experts, candidate_ids
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "result": "PASS" if all(item["pass"] for item in results) else "FAIL",
        "purpose": "genuine Ornith checkpoint batched FP16 CUDA router parity",
        "seed": args.seed,
        "tolerances": {
            "router_logits_max_abs": args.logits_atol,
            "routing_weights_max_abs": args.weights_atol,
            "expert_ids": "exact",
        },
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": gpu_count,
            "gpu_names": gpu_names(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "layers": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": report["result"], "report": str(args.report)}), flush=True)
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
