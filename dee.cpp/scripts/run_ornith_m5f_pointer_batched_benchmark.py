#!/usr/bin/env python3
"""Paired M5F direct vs pointer-batched expert full-model benchmark."""

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
    POINTER_BATCHED_EXECUTION_MODE,
    compare_trace,
    engine_stats,
    load_runtime,
    stats_delta,
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
CANDIDATE = POINTER_BATCHED_EXECUTION_MODE
MODES = (CONTROL, CANDIDATE)
POINTER_ARRAYS = 8


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


def analyze_pointer_workspace(
        stats: dict[str, Any], *, layers: int, topk: int,
        inter: int, pointer_bytes: int) -> dict[str, Any]:
    expected_host_per_layer = POINTER_ARRAYS * topk * pointer_bytes
    expected_device_per_layer = (
        3 * topk * inter * np.dtype(np.float16).itemsize
        + expected_host_per_layer
    )
    by_layer = stats["by_layer"]
    per_layer_exact = (
        len(by_layer) == layers
        and all(
            row["host_moe_pointer_table_bytes"]
            == expected_host_per_layer
            and row["device_moe_pointer_batch_workspace_bytes"]
            == expected_device_per_layer
            for row in by_layer
        )
    )
    aggregate = stats["aggregate"]
    aggregate_exact = (
        aggregate["host_moe_pointer_table_bytes"]
        == layers * expected_host_per_layer
        and aggregate["device_moe_pointer_batch_workspace_bytes"]
        == layers * expected_device_per_layer
    )
    return {
        "expected_layer_count": layers,
        "pointer_size_bytes": pointer_bytes,
        "pointer_array_count": POINTER_ARRAYS,
        "expected_host_pointer_table_bytes_per_layer":
            expected_host_per_layer,
        "expected_device_workspace_bytes_per_layer":
            expected_device_per_layer,
        "aggregate_host_pointer_table_bytes": aggregate[
            "host_moe_pointer_table_bytes"
        ],
        "aggregate_device_workspace_bytes": aggregate[
            "device_moe_pointer_batch_workspace_bytes"
        ],
        "per_layer_exact": per_layer_exact,
        "aggregate_exact": aggregate_exact,
        "passed": per_layer_exact and aggregate_exact,
    }


def run_m5f_once(
        runtime, tokenizer, mode: str, *, trace: bool,
        layers: int, topk: int, hidden: int, inter: int,
        pointer_bytes: int) -> tuple[dict, dict]:
    before = engine_stats(runtime)
    row, raw = run_once(
        runtime,
        tokenizer,
        "torch-device",
        trace=trace,
        execution_mode=mode,
    )
    after = engine_stats(runtime)
    delta = stats_delta(before, after)
    row["engine_projection_delta"] = {
        key: delta[key]
        for key in (
            "d2d_gather_copies",
            "d2d_gather_bytes",
            "d2d_scatter_copies",
            "d2d_scatter_bytes",
            "direct_row_gather_bypasses",
            "direct_row_scatter_bypasses",
            "pointer_batched_expert_calls",
            "pointer_batched_experts",
        )
    }
    row["engine_workspace"] = analyze_workspace(
        after, layers=layers, topk=topk, hidden=hidden
    )
    row["pointer_workspace"] = analyze_pointer_workspace(
        after,
        layers=layers,
        topk=topk,
        inter=inter,
        pointer_bytes=pointer_bytes,
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
        raise RuntimeError(f"M5F benchmark requires exactly two GPUs, got {gpu_count}")
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
    inter = int(runtime["engines"][0].inter_dim())
    pointer_bytes = np.dtype(np.uintp).itemsize

    warmups = {}
    for mode in MODES:
        row, _ = run_m5f_once(
            runtime,
            tokenizer,
            mode,
            trace=False,
            layers=layer_count,
            topk=topk,
            hidden=hidden,
            inter=inter,
            pointer_bytes=pointer_bytes,
        )
        warmups[mode] = row

    order = balanced_order(args.trials)
    trials: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in MODES
    }
    sequence = []
    for sequence_index, mode in enumerate(order):
        row, _ = run_m5f_once(
            runtime,
            tokenizer,
            mode,
            trace=False,
            layers=layer_count,
            topk=topk,
            hidden=hidden,
            inter=inter,
            pointer_bytes=pointer_bytes,
        )
        row["sequence_index"] = sequence_index
        row["mode_trial_index"] = len(trials[mode])
        trials[mode].append(row)
        sequence.append(row)

    control_trace_row, control_trace = run_m5f_once(
        runtime,
        tokenizer,
        CONTROL,
        trace=True,
        layers=layer_count,
        topk=topk,
        hidden=hidden,
        inter=inter,
        pointer_bytes=pointer_bytes,
    )
    candidate_trace_row, candidate_trace = run_m5f_once(
        runtime,
        tokenizer,
        CANDIDATE,
        trace=True,
        layers=layer_count,
        topk=topk,
        hidden=hidden,
        inter=inter,
        pointer_bytes=pointer_bytes,
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
    expected_layer_calls = layer_count * len(EXPECTED_TOKENS)
    expected_experts = expected_layer_calls * topk

    gates = {
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
        "control_uses_direct_combined_path": all(
            row["path_proof"]["native_combined_calls"]
            == expected_layer_calls
            and row["path_proof"]["native_direct_calls"]
            == expected_layer_calls
            and row["path_proof"]["native_pointer_batched_calls"] == 0
            for row in control_rows
        ),
        "candidate_uses_pointer_batched_path_for_all_layers": all(
            row["path_proof"]["native_combined_calls"]
            == expected_layer_calls
            and row["path_proof"]["native_direct_calls"]
            == expected_layer_calls
            and row["path_proof"]["native_pointer_batched_calls"]
            == expected_layer_calls
            and row["path_proof"][
                "native_pointer_batched_fallback_calls"
            ] == 0
            for row in candidate_rows
        ),
        "control_has_zero_pointer_batched_native_calls": all(
            row["engine_projection_delta"][
                "pointer_batched_expert_calls"
            ] == 0
            and row["engine_projection_delta"][
                "pointer_batched_experts"
            ] == 0
            for row in control_rows
        ),
        "candidate_native_pointer_batch_counts_exact": all(
            row["engine_projection_delta"][
                "pointer_batched_expert_calls"
            ] == expected_layer_calls
            and row["engine_projection_delta"][
                "pointer_batched_experts"
            ] == expected_experts
            for row in candidate_rows
        ),
        "candidate_has_zero_row_copies": all(
            row["engine_projection_delta"]["d2d_gather_copies"] == 0
            and row["engine_projection_delta"]["d2d_gather_bytes"] == 0
            and row["engine_projection_delta"]["d2d_scatter_copies"] == 0
            and row["engine_projection_delta"]["d2d_scatter_bytes"] == 0
            for row in candidate_rows
        ),
        "pointer_workspace_exact_and_bounded": all(
            row["pointer_workspace"]["passed"] for row in all_rows
        ),
        "combined_workspace_exact_and_bounded": all(
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
            "inter": inter,
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
        "launch_contract": {
            "layer_calls_per_generation": expected_layer_calls,
            "selected_experts_per_generation": expected_experts,
            "control_projection_cublas_calls_derived":
                expected_experts * 3,
            "candidate_projection_cublas_calls_derived":
                expected_layer_calls * 3,
            "control_activation_launches_derived": expected_experts,
            "candidate_activation_launches_derived": expected_layer_calls,
            "derivation_scope": (
                "Derived from proven native path counts and source contract; "
                "not a hardware-counter measurement."
            ),
        },
        "pointer_workspace_contract": candidate_rows[0][
            "pointer_workspace"
        ],
        "pareto": {
            "control_direct_median_tps": control_tps,
            "candidate_pointer_batched_median_tps": candidate_tps,
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
                "separate CUDA stage attribution. This result does not "
                "substitute enqueue time for GPU execution time."
            ),
        },
        "implementation_boundary": {
            "control": (
                "Eight one-token experts each dispatch gate, up, activation, "
                "and down operations independently; all row copies are already "
                "eliminated by the accepted direct path."
            ),
            "candidate": (
                "The same expert IDs are copied internally for cache dispatch, "
                "the same cache entries are requested and pinned, and the same "
                "exact weighted combine runs. Gate/up/down projections use "
                "three device-pointer cublasGemmBatchedEx calls and one "
                "contiguous activation launch for all eight experts."
            ),
        },
    }
    report_path = args.output_dir / "m5f-pointer-batched-benchmark.json"
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
        "M5F_POINTER_BATCHED_BENCHMARK="
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
