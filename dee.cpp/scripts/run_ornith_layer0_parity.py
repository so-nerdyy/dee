#!/usr/bin/env python3
"""Prove dee.cpp against the genuine Ornith-1.0-35B first transformer block.

The harness deliberately does not call ``from_pretrained``. It constructs a
one-layer Qwen3.5-MoE text model on the meta device, materializes only the real
embedding and layer-0 dense/router/shared-expert tensors, and lazily reads only
the routed experts selected by the real router. The original Transformers
decoder layer owns normalization, Gated DeltaNet, residuals, routing, shared
expert execution, and the final combine. Only the routed-expert module is
swapped between the PyTorch reference and dee.cpp.

No additional checkpoint download is needed: shard 1 contains the embedding,
the complete first block, all layer-0 experts, and the LM head. Shard 16 is
needed only for the final norm after all 40 blocks and is intentionally not
downloaded for this milestone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel


DEE_CPP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path(os.environ.get("ORNITH_MODEL_DIR", "/teamspace/studios/this_studio/ornith-model"))
LEDGER = DEE_CPP_ROOT / "benchmark_reports" / "real_generation_ledger.jsonl"
SHARD_SIZES = {
    "model-00001-of-00016.safetensors": 4_324_097_448,
    "model-00016-of-00016.safetensors": 2_565_780_424,
}
DEFAULT_PROMPTS = (
    "Capital of France is",
    "7 * 6 =",
    "def fibonacci(n):",
    "Once upon a time",
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def gpu_snapshot() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        row = subprocess.check_output(query, text=True, timeout=10).strip().splitlines()[0]
        name, total, used, free = [part.strip() for part in row.split(",")]
        return {
            "name": name,
            "total_mib": int(total),
            "used_mib": int(used),
            "free_mib": int(free),
        }
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(torch.float32).cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def set_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    parent: nn.Module = module
    pieces = name.split(".")
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    setattr(parent, pieces[-1], nn.Parameter(value.contiguous(), requires_grad=False))


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().to(torch.float32).cpu()
    got = candidate.detach().to(torch.float32).cpu()
    if ref.shape != got.shape:
        return {"shape_match": False, "reference_shape": list(ref.shape), "candidate_shape": list(got.shape)}
    delta = (ref - got).abs()
    flat_index = int(delta.argmax().item()) if delta.numel() else 0
    coord = [int(item) for item in np.unravel_index(flat_index, tuple(delta.shape))] if delta.numel() else []
    denom = ref.abs().clamp_min(1e-12)
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    return {
        "shape_match": True,
        "shape": list(ref.shape),
        "max_abs_error": float(delta.max().item()) if delta.numel() else 0.0,
        "max_relative_error": float((delta / denom).max().item()) if delta.numel() else 0.0,
        "relative_l2_error": float(torch.linalg.vector_norm(ref - got).item() / max(ref_norm, 1e-12)),
        "max_error_coordinate": coord,
        "reference_at_max": float(ref.flatten()[flat_index].item()) if delta.numel() else 0.0,
        "candidate_at_max": float(got.flatten()[flat_index].item()) if delta.numel() else 0.0,
    }


class SelectiveSafetensorsExperts(nn.Module):
    """Exact Transformers expert math with lazy per-expert safetensors reads."""

    def __init__(self, shard_path: Path, layer: int = 0):
        super().__init__()
        self.shard_path = shard_path
        self.layer = layer
        self.handle = safe_open(str(shard_path), framework="pt", device="cpu")
        self.cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.last_per_expert: torch.Tensor | None = None
        self.last_combined: torch.Tensor | None = None
        self.last_selected: torch.Tensor | None = None
        self.last_weights: torch.Tensor | None = None
        self.load_seconds = 0.0

    def _keys(self, expert: int) -> tuple[str, str, str]:
        base = f"model.language_model.layers.{self.layer}.mlp.experts.{expert}"
        return (
            f"{base}.gate_proj.weight",
            f"{base}.up_proj.weight",
            f"{base}.down_proj.weight",
        )

    def _weights(self, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if expert not in self.cache:
            started = time.perf_counter()
            keys = self._keys(expert)
            self.cache[expert] = tuple(self.handle.get_tensor(key).to(torch.float32) for key in keys)  # type: ignore[assignment]
            self.load_seconds += time.perf_counter() - started
        return self.cache[expert]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        states = hidden_states.to(torch.float32)
        selected = top_k_index.detach().cpu()
        weights = top_k_weights.to(torch.float32)
        tokens, hidden = states.shape
        top_k = selected.shape[1]
        per_expert = torch.empty((tokens, top_k, hidden), dtype=torch.float32)
        combined = torch.zeros((tokens, hidden), dtype=torch.float32)
        for token in range(tokens):
            for rank in range(top_k):
                expert = int(selected[token, rank].item())
                gate, up, down = self._weights(expert)
                state = states[token : token + 1]
                output = F.linear(F.silu(F.linear(state, gate)) * F.linear(state, up), down)[0]
                per_expert[token, rank] = output
                combined[token] += output * weights[token, rank]
        self.last_per_expert = per_expert
        self.last_combined = combined
        self.last_selected = selected.clone()
        self.last_weights = weights.detach().cpu().clone()
        return combined


class DeeCppExperts(nn.Module):
    """Transformers-compatible experts module backed by dee.cpp."""

    def __init__(self, engine: Any):
        super().__init__()
        self.engine = engine
        self.last_per_expert: torch.Tensor | None = None
        self.last_combined: torch.Tensor | None = None
        self.last_selected: torch.Tensor | None = None
        self.last_weights: torch.Tensor | None = None
        self.forward_seconds = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        states = hidden_states.detach().to(torch.float32).cpu().contiguous()
        selected = top_k_index.detach().cpu()
        weights = top_k_weights.detach().to(torch.float32).cpu()
        tokens, hidden = states.shape
        top_k = selected.shape[1]
        per_expert = torch.empty((tokens, top_k, hidden), dtype=torch.float32)
        combined = torch.zeros((tokens, hidden), dtype=torch.float32)
        started = time.perf_counter()
        for token in range(tokens):
            expert_ids = selected[token].to(torch.int64).tolist()
            input_np = states[token].numpy()
            output_np = np.empty((top_k, hidden), dtype=np.float32)
            if not self.engine.moe_forward_experts(0, input_np, output_np, expert_ids):
                raise RuntimeError(f"dee.cpp moe_forward_experts failed for token={token} experts={expert_ids}")
            output = torch.from_numpy(output_np.copy())
            per_expert[token] = output
            combined[token] = torch.sum(output * weights[token, :, None], dim=0)
        self.forward_seconds += time.perf_counter() - started
        self.last_per_expert = per_expert
        self.last_combined = combined
        self.last_selected = selected.clone()
        self.last_weights = weights.clone()
        return combined


def required_shard_map(model_dir: Path) -> list[dict[str, Any]]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    shard1 = "model-00001-of-00016.safetensors"
    shard16 = "model-00016-of-00016.safetensors"
    groups = {
        "embedding": [key for key in index if key == "model.language_model.embed_tokens.weight"],
        "first_complete_transformer_block": [key for key in index if ".layers.0." in key],
        "layer0_router": [key for key in index if key == "model.language_model.layers.0.mlp.gate.weight"],
        "layer0_routed_experts": [key for key in index if ".layers.0.mlp.experts." in key],
        "layer0_shared_expert": [key for key in index if ".layers.0.mlp.shared_expert" in key],
        "layer0_output_projection": [key for key in index if key == "model.language_model.layers.0.linear_attn.out_proj.weight"],
        "final_output_norm": [key for key in index if key == "model.language_model.norm.weight"],
        "lm_head": [key for key in index if key == "lm_head.weight"],
    }
    expected = {
        "embedding": shard1,
        "first_complete_transformer_block": shard1,
        "layer0_router": shard1,
        "layer0_routed_experts": shard1,
        "layer0_shared_expert": shard1,
        "layer0_output_projection": shard1,
        "final_output_norm": shard16,
        "lm_head": shard1,
    }
    for group, keys in groups.items():
        actual = sorted({index[key] for key in keys})
        if not keys or actual != [expected[group]]:
            raise RuntimeError(f"unexpected shard map for {group}: keys={len(keys)} shards={actual}")
    entries = []
    cumulative = 0
    for filename, supplied, reason, milestone in (
        (
            shard1,
            [name for name, shard in expected.items() if shard == shard1],
            "contains the real tokenizer embedding, complete layer 0, router, all routed/shared experts, layer-0 output projection, and LM head",
            True,
        ),
        (
            shard16,
            ["final_output_norm"],
            "contains the final RMSNorm used only after all 40 transformer blocks",
            False,
        ),
    ):
        cumulative += SHARD_SIZES[filename]
        path = model_dir / filename
        entries.append(
            {
                "filename": filename,
                "tensor_groups": supplied,
                "tensor_count": len({key for name in supplied for key in groups[name]}),
                "reason": reason,
                "size_bytes": SHARD_SIZES[filename],
                "cumulative_inventory_bytes": cumulative,
                "required_for_layer0_milestone": milestone,
                "present": path.is_file(),
                "downloaded_by_this_run": False,
            }
        )
    actual_size = (model_dir / shard1).stat().st_size
    if actual_size != SHARD_SIZES[shard1]:
        raise RuntimeError(f"shard 1 size mismatch: expected {SHARD_SIZES[shard1]}, found {actual_size}")
    return entries


def build_selective_model(model_dir: Path, reference_experts: SelectiveSafetensorsExperts) -> tuple[Any, Any, list[str]]:
    root_config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    config = root_config.text_config
    config.num_hidden_layers = 1
    config.layer_types = list(config.layer_types[:1])
    with init_empty_weights():
        model = Qwen3_5MoeTextModel(config)
    model.layers[0].mlp.experts = reference_experts
    loaded: list[str] = []
    prefix = "model.language_model."
    shard_path = model_dir / "model-00001-of-00016.safetensors"
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        for name, parameter in list(model.named_parameters()):
            if name.startswith("layers.0.mlp.experts.") or name == "norm.weight":
                continue
            key = prefix + name
            if key not in handle.keys():
                raise KeyError(f"required selective-load tensor missing: {key}")
            value = handle.get_tensor(key)
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(f"shape mismatch for {key}: checkpoint={tuple(value.shape)} model={tuple(parameter.shape)}")
            set_parameter(model, name, value)
            loaded.append(key)
    remaining_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if remaining_meta != ["norm.weight"]:
        raise RuntimeError(f"unexpected parameters left on meta device: {remaining_meta}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    return model, tokenizer, loaded


def init_dee_engine(shard_path: Path) -> Any:
    sys.path.insert(0, str(DEE_CPP_ROOT / "pydee"))
    sys.path.insert(0, str(DEE_CPP_ROOT))
    import pydee

    config = pydee.EngineConfig()
    config.shard_path = str(shard_path)
    config.oracle_path = ""
    config.topk = 8
    config.num_layers = 1
    config.num_experts = 256
    config.hidden = 2048
    config.inter = 512
    config.use_cuda = False
    config.transfer_dtype = pydee.WeightTransferDType.Bf16
    config.cache_dtype = pydee.DeviceCacheDType.Fp32
    config.verbose = False
    engine = pydee.Engine()
    if not engine.init(config):
        raise RuntimeError("pydee.Engine.init rejected the genuine Ornith shard")
    if engine.hidden_dim() != 2048 or engine.inter_dim() != 512:
        raise RuntimeError(
            f"dee.cpp resolved wrong Ornith dimensions: hidden={engine.hidden_dim()} inter={engine.inter_dim()}"
        )
    return engine


def run_prompt(
    prompt: str,
    tokenizer: Any,
    model: Any,
    reference: SelectiveSafetensorsExperts,
    dee_experts: DeeCppExperts,
    max_tokens: int,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids[:, :max_tokens]
    with torch.inference_mode():
        embeddings = model.embed_tokens(input_ids)
    layer = model.layers[0]
    router_runs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def capture_router(_module: nn.Module, _args: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
        router_runs.append(tuple(item.detach().cpu().clone() for item in output))

    hook = layer.mlp.gate.register_forward_hook(capture_router)
    try:
        layer.mlp.experts = reference
        ref_started = time.perf_counter()
        with torch.inference_mode():
            ref_hidden = layer(embeddings, position_embeddings=(torch.empty(0), torch.empty(0)))
        reference_seconds = time.perf_counter() - ref_started
        ref_router_logits, ref_scores, ref_selected = router_runs[-1]
        ref_per_expert = reference.last_per_expert.clone()  # type: ignore[union-attr]
        ref_combined = reference.last_combined.clone()  # type: ignore[union-attr]

        layer.mlp.experts = dee_experts
        dee_started = time.perf_counter()
        with torch.inference_mode():
            dee_hidden = layer(embeddings, position_embeddings=(torch.empty(0), torch.empty(0)))
        dee_seconds = time.perf_counter() - dee_started
        dee_router_logits, dee_scores, dee_selected = router_runs[-1]
        dee_per_expert = dee_experts.last_per_expert.clone()  # type: ignore[union-attr]
        dee_combined = dee_experts.last_combined.clone()  # type: ignore[union-attr]
    finally:
        hook.remove()

    metrics = {
        "router_logits": comparison(ref_router_logits, dee_router_logits),
        "routing_weights": comparison(ref_scores, dee_scores),
        "expert_outputs": comparison(ref_per_expert, dee_per_expert),
        "combined_routed_moe": comparison(ref_combined, dee_combined),
        "hidden_states": comparison(ref_hidden, dee_hidden),
    }
    order_match = bool(torch.equal(ref_selected, dee_selected))
    membership_match = all(
        set(ref_selected[token].tolist()) == set(dee_selected[token].tolist())
        for token in range(ref_selected.shape[0])
    )
    tolerances = {
        "expert_outputs_max_abs": 2e-3,
        "expert_outputs_relative_l2": 2e-4,
        "combined_moe_max_abs": 2e-3,
        "combined_moe_relative_l2": 2e-4,
        "hidden_states_max_abs": 2e-3,
        "hidden_states_relative_l2": 2e-4,
    }
    passed = (
        order_match
        and membership_match
        and metrics["router_logits"]["max_abs_error"] == 0.0
        and metrics["routing_weights"]["max_abs_error"] == 0.0
        and metrics["expert_outputs"]["max_abs_error"] <= tolerances["expert_outputs_max_abs"]
        and metrics["expert_outputs"]["relative_l2_error"] <= tolerances["expert_outputs_relative_l2"]
        and metrics["combined_routed_moe"]["max_abs_error"] <= tolerances["combined_moe_max_abs"]
        and metrics["combined_routed_moe"]["relative_l2_error"] <= tolerances["combined_moe_relative_l2"]
        and metrics["hidden_states"]["max_abs_error"] <= tolerances["hidden_states_max_abs"]
        and metrics["hidden_states"]["relative_l2_error"] <= tolerances["hidden_states_relative_l2"]
    )

    first_difference = None
    checks = (
        ("layer0.router_logits", metrics["router_logits"], 0.0),
        ("layer0.routing_weights", metrics["routing_weights"], 0.0),
        ("layer0.expert_outputs", metrics["expert_outputs"], tolerances["expert_outputs_max_abs"]),
        ("layer0.combined_routed_moe", metrics["combined_routed_moe"], tolerances["combined_moe_max_abs"]),
        ("layer0.hidden_states", metrics["hidden_states"], tolerances["hidden_states_max_abs"]),
    )
    for tensor_name, metric, limit in checks:
        if not metric.get("shape_match", False) or metric.get("max_abs_error", float("inf")) > limit:
            coord = metric.get("max_error_coordinate", [])
            expert = None
            if tensor_name.endswith("expert_outputs") and len(coord) >= 2:
                expert = int(ref_selected[coord[0], coord[1]].item())
            first_difference = {"tensor_name": tensor_name, "layer": 0, "expert": expert, **metric}
            break

    return {
        "prompt": prompt,
        "token_ids": input_ids[0].tolist(),
        "token_count": int(input_ids.shape[1]),
        "embedding": {
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "sha256_fp32": tensor_digest(embeddings),
            "l2_norm": float(torch.linalg.vector_norm(embeddings.float()).item()),
        },
        "selected_experts_order": ref_selected.tolist(),
        "routing_order_match": order_match,
        "topk_membership_match": membership_match,
        "metrics": metrics,
        "tolerances": tolerances,
        "first_difference": first_difference,
        "timing_seconds": {"transformers_reference": reference_seconds, "dee_cpp_full_block": dee_seconds},
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--max-prompt-tokens", type=int, default=16)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    shard_path = model_dir / "model-00001-of-00016.safetensors"
    started = time.perf_counter()
    memory = {"rss_start_bytes": rss_bytes()}
    gpu = {"before": gpu_snapshot()}
    shard_map = required_shard_map(model_dir)
    print(json.dumps({"shard_map": shard_map}, indent=2))
    print("[ornith] additional checkpoint download required: 0 bytes")

    reference = SelectiveSafetensorsExperts(shard_path)
    load_started = time.perf_counter()
    model, tokenizer, loaded_tensors = build_selective_model(model_dir, reference)
    model_load_seconds = time.perf_counter() - load_started
    memory["rss_after_selective_model_load_bytes"] = rss_bytes()
    print(f"[ornith] selectively loaded {len(loaded_tensors)} dense/router/shared tensors in {model_load_seconds:.3f}s")

    engine_started = time.perf_counter()
    engine = init_dee_engine(shard_path)
    engine_init_seconds = time.perf_counter() - engine_started
    dee_experts = DeeCppExperts(engine)
    print(f"[dee.cpp] genuine shard initialized in {engine_init_seconds:.3f}s")

    prompts = tuple(args.prompts) if args.prompts else DEFAULT_PROMPTS
    results = []
    peak_rss = rss_bytes()
    for prompt in prompts:
        result = run_prompt(prompt, tokenizer, model, reference, dee_experts, args.max_prompt_tokens)
        results.append(result)
        peak_rss = max(peak_rss, rss_bytes())
        status = "PASS" if result["pass"] else "FAIL"
        hidden_error = result["metrics"]["hidden_states"]["max_abs_error"]
        expert_error = result["metrics"]["expert_outputs"]["max_abs_error"]
        print(f"[{status}] {prompt!r}: expert_max_abs={expert_error:.9g} hidden_max_abs={hidden_error:.9g}")

    memory["rss_peak_observed_bytes"] = peak_rss
    memory["rss_final_bytes"] = rss_bytes()
    gpu["after"] = gpu_snapshot()
    overall_pass = all(result["pass"] for result in results)
    report = {
        "schema_version": 1,
        "id": f"ORNITH-LAYER0-PARITY-{now_iso()}",
        "timestamp_utc": now_iso(),
        "pass": overall_pass,
        "model": "deepreinforce-ai/Ornith-1.0-35B",
        "execution_path": "real tokenizer -> real embedding -> real complete layer 0 -> real router/top-k -> dee.cpp real experts -> real shared expert/combine -> hidden state",
        "checkpoint": {
            "model_dir": str(model_dir),
            "shard_map": shard_map,
            "additional_download_bytes": 0,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
            "dee_cpp_backend": "CPU FP32 (real BF16 weights converted by dee.cpp)",
            "model_reference_device": "cpu",
        },
        "selective_loading": {
            "loaded_non_expert_tensor_count": len(loaded_tensors),
            "loaded_non_expert_tensors": loaded_tensors,
            "lazy_reference_experts_loaded": sorted(reference.cache),
            "lazy_reference_expert_count": len(reference.cache),
        },
        "results": results,
        "timing_seconds": {
            "model_selective_load": model_load_seconds,
            "dee_engine_init": engine_init_seconds,
            "reference_expert_weight_load": reference.load_seconds,
            "dee_cpp_expert_forward_total": dee_experts.forward_seconds,
            "end_to_end": time.perf_counter() - started,
        },
        "memory": memory,
        "gpu": gpu,
        "dee_engine_stats": json.loads(engine.last_stats_json()),
    }
    report_path = args.report
    if report_path is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = DEE_CPP_ROOT / "benchmark_reports" / f"ornith-layer0-parity-{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ledger_record = {
        "id": report["id"],
        "stage": "genuine-ornith-layer0-parity",
        "timestamp_utc": report["timestamp_utc"],
        "pass": overall_pass,
        "report": str(report_path.relative_to(DEE_CPP_ROOT)),
        "additional_download_bytes": 0,
        "prompts": len(results),
        "first_difference": next((item["first_difference"] for item in results if item["first_difference"]), None),
    }
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ledger_record) + "\n")
    print(f"[report] {report_path}")
    print(f"[result] {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
