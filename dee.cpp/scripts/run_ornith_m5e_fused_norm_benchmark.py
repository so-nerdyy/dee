#!/usr/bin/env python3
"""Paired M5E eager vs fused-normalization full-model benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ornith_generation import (  # noqa: E402
    FUSED_NORM_EXECUTION_MODE,
    compare_trace,
    engine_stats,
    load_runtime,
)
from scripts.run_ornith_m5b_execution_benchmark import (  # noqa: E402
    exact_trace_comparison,
    thermal_clock_analysis,
    tracked_source_clean,
)
from scripts.run_ornith_m5c_combined_benchmark import (  # noqa: E402
    analyze_workspace,
)
from scripts.run_ornith_router_backend_benchmark import (  # noqa: E402
    EXPECTED_TOKENS,
    MAX_PROCESS_VRAM_BYTES,
    git_revision,
    run_once,
    sha256_file,
    summarize_trials,
)


CONTROL = "native-combined-direct"
CANDIDATE = FUSED_NORM_EXECUTION_MODE
MODES = (CONTROL, CANDIDATE)
ORNITH_LAYER_COUNT = 40
ORNITH_LINEAR_ATTN_LAYER_COUNT = 30
ORNITH_FULL_ATTN_LAYER_COUNT = 10
EXPECTED_REGULAR_WRAPPERS = ORNITH_LAYER_COUNT * 2 + 1
EXPECTED_GATED_WRAPPERS = ORNITH_LINEAR_ATTN_LAYER_COUNT
EXPECTED_UNWRAPPED_HEAD_NORMS = ORNITH_FULL_ATTN_LAYER_COUNT * 2


def balanced_order(trials: int) -> list[str]:
    pattern = [
        CONTROL,
        CANDIDATE,
        CANDIDATE,
        CONTROL,
        CONTROL,
        CANDIDATE,
    ]
    order: list[str] = []
    while any(order.count(mode) < trials for mode in MODES):
        for mode in pattern:
            if order.count(mode) < trials:
                order.append(mode)
    return order


def paired_trial_analysis(sequence: list[dict[str, Any]]) -> dict[str, Any]:
    if len(sequence) % 2:
        raise ValueError("paired sequence must contain an even number of rows")
    pairs = []
    for index in range(0, len(sequence), 2):
        pair_rows = sequence[index:index + 2]
        rows = {row["execution_mode"]: row for row in pair_rows}
        if set(rows) != set(MODES):
            raise ValueError(f"pair {index // 2} does not contain both modes")
        control_tps = float(rows[CONTROL]["tokens_per_second"])
        candidate_tps = float(rows[CANDIDATE]["tokens_per_second"])
        ratio = candidate_tps / control_tps
        pairs.append({
            "pair_index": index // 2,
            "sequence_indices": [index, index + 1],
            "first_mode": pair_rows[0]["execution_mode"],
            "control_tps": control_tps,
            "candidate_tps": candidate_tps,
            "speedup_ratio": ratio,
            "speedup_percent": (ratio - 1.0) * 100.0,
            "candidate_won": candidate_tps > control_tps,
        })
    ratios = [row["speedup_ratio"] for row in pairs]
    return {
        "pair_count": len(pairs),
        "candidate_wins": sum(row["candidate_won"] for row in pairs),
        "median_speedup_ratio": float(np.median(ratios)),
        "minimum_speedup_ratio": min(ratios),
        "maximum_speedup_ratio": max(ratios),
        "pairs": pairs,
    }


def normalization_inventory_contract(inventory: dict[str, list[str]]) -> dict[str, Any]:
    """Validate the complete Ornith norm-site partition independently of wrappers."""
    regular = list(inventory.get("wrapped_regular", []))
    gated = list(inventory.get("wrapped_gated", []))
    unwrapped = list(
        inventory.get("intentionally_unwrapped_full_attention_head_norms", [])
    )
    expected_regular = [
        label
        for layer in range(ORNITH_LAYER_COUNT)
        for label in (
            f"layer={layer}:input_layernorm",
            f"layer={layer}:post_attention_layernorm",
        )
    ] + ["final_norm"]

    def layers_for_suffix(labels: list[str], suffix: str) -> list[int]:
        prefix = "layer="
        return sorted(
            int(label[len(prefix):label.index(":")])
            for label in labels
            if label.startswith(prefix) and label.endswith(suffix)
        )

    gated_layers = layers_for_suffix(gated, ":linear_attn.norm")
    q_layers = layers_for_suffix(unwrapped, ":self_attn.q_norm")
    k_layers = layers_for_suffix(unwrapped, ":self_attn.k_norm")
    full_attention_layers = sorted(set(q_layers) | set(k_layers))
    partition_layers = sorted(set(gated_layers) | set(full_attention_layers))
    checks = {
        "regular_path_list_exact": regular == expected_regular,
        "regular_wrapper_count_exact": len(regular) == EXPECTED_REGULAR_WRAPPERS,
        "gated_wrapper_count_exact": len(gated) == EXPECTED_GATED_WRAPPERS,
        "unwrapped_head_norm_count_exact": (
            len(unwrapped) == EXPECTED_UNWRAPPED_HEAD_NORMS
        ),
        "gated_paths_unique": len(gated_layers) == len(set(gated_layers)),
        "q_and_k_head_norm_layers_match": q_layers == k_layers,
        "full_attention_layer_count_exact": (
            len(full_attention_layers) == ORNITH_FULL_ATTN_LAYER_COUNT
        ),
        "attention_kinds_partition_all_layers": (
            not (set(gated_layers) & set(full_attention_layers))
            and partition_layers == list(range(ORNITH_LAYER_COUNT))
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "wrapped_regular": regular,
        "wrapped_gated": gated,
        "intentionally_unwrapped_full_attention_head_norms": unwrapped,
        "gated_layers": gated_layers,
        "full_attention_layers": full_attention_layers,
    }


def _adversarial_half_rows(torch, rows: int, dim: int, seed: int):
    """Deterministic irregular FP16 rows that expose reduction-order differences."""
    indices = torch.arange(rows * dim, dtype=torch.int64)
    values = (
        ((indices * 37 + seed * 101) % 2003).to(torch.float32) - 1001.0
    ) / 137.0
    values = values.reshape(rows, dim)
    sentinels = torch.tensor(
        [0.0, 6.0, -6.0, 2.0 ** -12, -(2.0 ** -12), 31.0, -31.0],
        dtype=torch.float32,
    )
    width = min(dim, int(sentinels.numel()))
    values[:, :width] = sentinels[:width]
    if rows > 1:
        values[1].mul_(0.03125)
    if rows > 2:
        values[2].mul_(-0.875)
    return values.to(torch.float16).contiguous()


def native_contract_probe(runtime, torch) -> dict[str, Any]:
    """Compare native kernels to saved installed-module forwards on both T4s."""
    selected: dict[tuple[str, int], Any] = {}
    for wrapper in runtime["fused_norm_wrappers"]:
        kind = (
            "regular"
            if type(wrapper).__name__ == "HybridRMSNorm"
            else "gated"
            if type(wrapper).__name__ == "HybridRMSNormGated"
            else None
        )
        if kind is None:
            continue
        device = int(wrapper.module.weight.device.index)
        selected.setdefault((kind, device), wrapper)

    required = [
        (kind, device)
        for kind in ("regular", "gated")
        for device in range(2)
    ]
    missing = [
        {"kind": kind, "device": device}
        for kind, device in required
        if (kind, device) not in selected
    ]
    records = []
    context = runtime["context"]
    saved_mode = context.mode
    saved_execution_mode = context.execution_mode
    try:
        context.mode = "dee"
        context.execution_mode = CANDIDATE
        for case_index, key in enumerate(required):
            if key not in selected:
                continue
            kind, device = key
            wrapper = selected[key]
            dim = int(wrapper.module.weight.numel())
            hidden_cpu = _adversarial_half_rows(
                torch, rows=3, dim=dim, seed=case_index + 1
            )
            gate_cpu = _adversarial_half_rows(
                torch, rows=3, dim=dim, seed=case_index + 101
            )
            streams = {
                "default": torch.cuda.default_stream(device),
                "nondefault": torch.cuda.Stream(device=device),
            }
            for stream_name, stream in streams.items():
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    hidden = hidden_cpu.to(device=f"cuda:{device}")
                    gate = gate_cpu.to(device=f"cuda:{device}")
                    if kind == "regular":
                        reference = wrapper.reference_forward(hidden)
                        candidate = wrapper.forward(hidden)
                        epsilon = float(wrapper.module.eps)
                    else:
                        reference = wrapper.reference_forward(hidden, gate)
                        candidate = wrapper.forward(hidden, gate)
                        epsilon = float(wrapper.module.variance_epsilon)
                stream.synchronize()
                reference_bits = reference.contiguous().view(torch.int16)
                candidate_bits = candidate.contiguous().view(torch.int16)
                mismatch_mask = reference_bits != candidate_bits
                mismatch_count = int(mismatch_mask.sum().item())
                max_abs_difference = float(
                    (reference.float() - candidate.float()).abs().max().item()
                )
                records.append({
                    "kind": kind,
                    "device": device,
                    "stream": stream_name,
                    "rows": 3,
                    "dim": dim,
                    "epsilon": epsilon,
                    "reference_dtype": str(reference.dtype),
                    "candidate_dtype": str(candidate.dtype),
                    "reference_contiguous": bool(reference.is_contiguous()),
                    "candidate_contiguous": bool(candidate.is_contiguous()),
                    "mismatch_count": mismatch_count,
                    "element_count": int(reference.numel()),
                    "max_abs_difference": max_abs_difference,
                    "half_bits_exact": mismatch_count == 0,
                })
    finally:
        context.mode = saved_mode
        context.execution_mode = saved_execution_mode
    expected_hidden = int(runtime["config"].hidden_size)
    real_shapes_and_epsilon = (
        not missing
        and len(records) == len(required) * 2
        and all(record["epsilon"] > 0.0 for record in records)
        and all(
            record["dim"] == expected_hidden
            for record in records
            if record["kind"] == "regular"
        )
        and all(
            0 < record["dim"] <= 4096
            for record in records
            if record["kind"] == "gated"
        )
    )
    return {
        "input_contract": (
            "Three deterministic irregular/adversarial FP16 rows per case, "
            "including mixed signs, magnitudes, and subnormal-scale values."
        ),
        "expected_regular_dim": expected_hidden,
        "all_required_wrappers_present": not missing,
        "missing": missing,
        "cases": records,
        "real_shapes_and_nonzero_epsilon": real_shapes_and_epsilon,
        "all_half_bits_exact": (
            not missing
            and len(records) == len(required) * 2
            and all(record["half_bits_exact"] for record in records)
        ),
    }


def run_m5e_once(
        runtime, tokenizer, mode: str, *, trace: bool,
        layers: int, topk: int, hidden: int) -> tuple[dict, dict]:
    row, raw = run_once(
        runtime,
        tokenizer,
        "torch-device",
        trace=trace,
        execution_mode=mode,
    )
    row["engine_workspace"] = analyze_workspace(
        engine_stats(runtime), layers=layers, topk=topk, hidden=hidden
    )
    return row, raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-experts", type=int, default=32)
    parser.add_argument("--split-layer", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--require-dual-gpu", action="store_true")
    args = parser.parse_args()
    if args.cache_experts < 8:
        parser.error("--cache-experts must be at least top-k 8")
    if args.trials < 3:
        parser.error("--trials must be at least 3")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from transformers import AutoTokenizer

    gpu_count = torch.cuda.device_count()
    if gpu_count != 2:
        raise RuntimeError(f"M5E benchmark requires exactly two GPUs, got {gpu_count}")
    if args.require_dual_gpu and any(
        "T4" not in torch.cuda.get_device_name(device) for device in range(2)
    ):
        raise RuntimeError(
            "dual-T4 benchmark found "
            + ", ".join(
                torch.cuda.get_device_name(device) for device in range(2)
            )
        )

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True
    )
    runtime = load_runtime(
        args.model_dir,
        2,
        args.split_layer,
        args.cache_experts,
        router_backend="torch-device",
    )
    load_seconds = time.perf_counter() - load_started
    import pydee.pydee_core as pydee_core

    binary = Path(pydee_core.__file__).resolve()
    runtime_binary = {
        "path": str(binary),
        "bytes": binary.stat().st_size,
        "sha256": sha256_file(binary),
    }
    source_clean = tracked_source_clean()
    layer_count = len(runtime["engines"])
    topk = 8
    hidden = int(runtime["engines"][0].hidden_dim())
    inventory_contract = normalization_inventory_contract(
        runtime["fused_norm_inventory"]
    )
    native_probe = native_contract_probe(runtime, torch)
    expected_regular_calls = (
        EXPECTED_REGULAR_WRAPPERS * len(EXPECTED_TOKENS)
    )
    expected_gated_calls = EXPECTED_GATED_WRAPPERS * len(EXPECTED_TOKENS)
    expected_output_allocations = (
        expected_regular_calls + expected_gated_calls
    )

    warmups = {}
    for mode in MODES:
        row, _ = run_m5e_once(
            runtime,
            tokenizer,
            mode,
            trace=False,
            layers=layer_count,
            topk=topk,
            hidden=hidden,
        )
        warmups[mode] = row

    order = balanced_order(args.trials)
    trials: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in MODES
    }
    sequence = []
    for sequence_index, mode in enumerate(order):
        row, _ = run_m5e_once(
            runtime,
            tokenizer,
            mode,
            trace=False,
            layers=layer_count,
            topk=topk,
            hidden=hidden,
        )
        row["sequence_index"] = sequence_index
        row["mode_trial_index"] = len(trials[mode])
        trials[mode].append(row)
        sequence.append(row)

    control_trace_row, control_trace = run_m5e_once(
        runtime,
        tokenizer,
        CONTROL,
        trace=True,
        layers=layer_count,
        topk=topk,
        hidden=hidden,
    )
    candidate_trace_row, candidate_trace = run_m5e_once(
        runtime,
        tokenizer,
        CANDIDATE,
        trace=True,
        layers=layer_count,
        topk=topk,
        hidden=hidden,
    )
    tolerance_comparison = compare_trace(
        control_trace["collector"], candidate_trace["collector"]
    )
    exact_comparison = exact_trace_comparison(
        control_trace["collector"], candidate_trace["collector"]
    )
    correctness = {
        "control_tokens": control_trace_row["generated_token_ids"],
        "candidate_tokens": candidate_trace_row["generated_token_ids"],
        "generated_token_ids_exact": (
            control_trace_row["generated_token_ids"]
            == candidate_trace_row["generated_token_ids"]
            == EXPECTED_TOKENS
        ),
        "decoded_text_exact": (
            control_trace_row["generated_text"]
            == candidate_trace_row["generated_text"]
        ),
        "all_trace_categories_passed": tolerance_comparison[
            "all_categories_passed"
        ],
        "all_trace_categories_bitwise_exact": exact_comparison[
            "all_categories_bitwise_exact"
        ],
        "tolerance_comparison": tolerance_comparison,
        "exact_comparison": exact_comparison,
    }

    summaries = {
        mode: summarize_trials(trials[mode]) for mode in MODES
    }
    control_tps = summaries[CONTROL]["tokens_per_second"]["median"]
    candidate_tps = summaries[CANDIDATE]["tokens_per_second"]["median"]
    speedup = candidate_tps / control_tps
    control_rows = trials[CONTROL]
    candidate_rows = trials[CANDIDATE]
    all_rows = control_rows + candidate_rows
    paired = paired_trial_analysis(sequence)
    thermal = thermal_clock_analysis(all_rows)

    gates = {
        "native_contract_probe_all_half_bits_exact": native_probe[
            "all_half_bits_exact"
        ],
        "native_contract_probe_real_shapes_nonzero_epsilon": native_probe[
            "real_shapes_and_nonzero_epsilon"
        ],
        "normalization_inventory_contract_exact": inventory_contract["passed"],
        "candidate_independent_median_gain_at_least_1_percent": (
            speedup >= 1.01
        ),
        "candidate_paired_median_gain_at_least_1_percent": (
            paired["median_speedup_ratio"] >= 1.01
        ),
        "candidate_wins_majority_of_pairs": (
            paired["candidate_wins"] >= paired["pair_count"] // 2 + 1
        ),
        "no_pair_regresses_more_than_2_percent": (
            paired["minimum_speedup_ratio"] >= 0.98
        ),
        "tokens_and_text_exact": (
            correctness["generated_token_ids_exact"]
            and correctness["decoded_text_exact"]
        ),
        "all_trace_categories_passed": correctness[
            "all_trace_categories_passed"
        ],
        "all_trace_categories_bitwise_exact": correctness[
            "all_trace_categories_bitwise_exact"
        ],
        "all_40_layers_executed": all(
            row["all_40_layers_executed"] for row in all_rows
        ),
        "all_expert_calls_device_resident": all(
            row["path_proof"]["device_path_share"] == 1.0
            for row in all_rows
        ),
        "zero_host_fallback_calls": all(
            row["path_proof"]["host_path_fallback_calls"] == 0
            for row in all_rows
        ),
        "same_direct_expert_path_both_modes": all(
            row["path_proof"]["native_combined_calls"] == 160
            and row["path_proof"]["native_direct_calls"] == 160
            and row["path_proof"]["native_direct_fallback_calls"] == 0
            for row in all_rows
        ),
        "control_has_zero_fused_norm_calls": all(
            row["path_proof"]["fused_rms_norm_calls"] == 0
            and row["path_proof"]["fused_rms_norm_gated_calls"] == 0
            and row["path_proof"]["fused_norm_output_allocations"] == 0
            for row in control_rows
        ),
        "candidate_regular_norm_call_count_exact": all(
            row["path_proof"]["fused_rms_norm_calls"]
            == expected_regular_calls
            for row in candidate_rows
        ),
        "candidate_gated_norm_call_count_exact": all(
            row["path_proof"]["fused_rms_norm_gated_calls"]
            == expected_gated_calls
            for row in candidate_rows
        ),
        "candidate_output_allocation_count_exact": all(
            row["path_proof"]["fused_norm_output_allocations"]
            == expected_output_allocations
            for row in candidate_rows
        ),
        "candidate_has_zero_fused_norm_failures": all(
            row["path_proof"]["fused_norm_failures"] == 0
            for row in candidate_rows
        ),
        "combined_workspace_counters_exact_and_bounded": all(
            row["engine_workspace"]["passed"] for row in all_rows
        ),
        "process_vram_at_most_8_gib_per_gpu": all(
            max(row["peak_vram_bytes"].values())
            <= MAX_PROCESS_VRAM_BYTES
            for row in all_rows
        ),
        "tracked_source_clean": source_clean,
        "runtime_binary_hashed": bool(runtime_binary["sha256"]),
        "thermal_clock_samples_present": thermal["samples_present"],
        "no_thermal_or_clock_anomaly": not thermal["anomaly_detected"],
    }
    result = "PASS" if all(gates.values()) else "FAIL"
    report = {
        "schema_version": 1,
        "result": result,
        "git_commit": git_revision(),
        "model": {
            "name": "Ornith-1.0-35B",
            "directory": str(args.model_dir),
            "config_sha256": sha256_file(args.model_dir / "config.json"),
            "index_sha256": sha256_file(
                args.model_dir / "model.safetensors.index.json"
            ),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpus": [
                {
                    "device": device,
                    "name": torch.cuda.get_device_name(device),
                    "total_memory_bytes": torch.cuda.get_device_properties(
                        device
                    ).total_memory,
                }
                for device in range(2)
            ],
        },
        "configuration": {
            "precision": "FP16 dense and expert cache; BF16 mmap transfer",
            "batch_size": 1,
            "sequence_count": 1,
            "cache_experts_per_layer": args.cache_experts,
            "split_layer": args.split_layer,
            "layer_count": layer_count,
            "topk": topk,
            "hidden": hidden,
            "router_backend": "torch-device",
            "trials_per_mode": args.trials,
            "measurement_order": order,
            "same_loaded_runtime": True,
            "warmup_per_mode": 1,
            "prompt": "Hello",
            "max_new_tokens": len(EXPECTED_TOKENS),
        },
        "runtime_load_seconds": load_seconds,
        "runtime_identity": {
            "tracked_source_clean": source_clean,
            "git_commit": git_revision(),
            "pydee_core": runtime_binary,
        },
        "warmups": warmups,
        "trial_sequence": sequence,
        "paired_analysis": paired,
        "thermal_clock_analysis": thermal,
        "mode_summaries": summaries,
        "correctness": correctness,
        "normalization_contract": {
            "scope": (
                "Partial layer/final/gated norm fusion: all layer input and "
                "post-attention norms, all linear-attention gated norms, and "
                "the final norm are fused. Full-attention q/k head norms remain "
                "official eager operations and are explicitly inventoried."
            ),
            "regular_wrappers": EXPECTED_REGULAR_WRAPPERS,
            "gated_wrappers": EXPECTED_GATED_WRAPPERS,
            "intentionally_unwrapped_head_norms":
                EXPECTED_UNWRAPPED_HEAD_NORMS,
            "expected_regular_calls_per_generation": expected_regular_calls,
            "expected_gated_calls_per_generation": expected_gated_calls,
            "expected_output_allocations_per_generation":
                expected_output_allocations,
            "inventory": inventory_contract,
            "native_contract_probe": native_probe,
        },
        "pareto": {
            "control_eager_norm_median_tps": control_tps,
            "candidate_fused_norm_median_tps": candidate_tps,
            "control_median_tpot_seconds": 1.0 / control_tps,
            "candidate_median_tpot_seconds": 1.0 / candidate_tps,
            "single_sequence_tps": candidate_tps,
            "aggregate_tps": candidate_tps,
            "speedup_ratio": speedup,
            "speedup_percent": (speedup - 1.0) * 100.0,
            "gates": gates,
        },
        "stage_timing_scope": {
            "attention_time": None,
            "expert_time": None,
            "synchronization_time": None,
            "reason": (
                "The accepted A/B is intentionally unprofiled; M5A supplies "
                "the separately instrumented stage attribution. This result "
                "does not substitute enqueue time for GPU execution time."
            ),
        },
        "implementation_boundary": {
            "control": (
                "Official Transformers eager Qwen RMSNorm and gated RMSNorm; "
                "the exact same native-combined-direct expert path."
            ),
            "candidate": (
                "Partial layer/final/gated norm fusion: one workspace-free "
                "raw-pointer CUDA launch per covered norm output; the caller "
                "allocates only the required output tensor. The launch uses "
                "the active PyTorch stream, explicit eager FP16/FP32 "
                "boundaries, and no semantic fallback. Full-attention q/k "
                "head norms remain eager and are excluded from the claim."
            ),
        },
    }
    report_path = args.output_dir / "m5e-fused-norm-benchmark.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = args.output_dir / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "result": result,
                "files": {
                    report_path.name: {
                        "bytes": report_path.stat().st_size,
                        "sha256": sha256_file(report_path),
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    runtime["pool"].close()
    del runtime
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    for device in range(2):
        torch.cuda.synchronize(device)
    print(
        "M5E_FUSED_NORM_BENCHMARK="
        + json.dumps(
            {
                "result": result,
                "control_tps": control_tps,
                "candidate_tps": candidate_tps,
                "speedup_percent": (speedup - 1.0) * 100.0,
                "output": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
