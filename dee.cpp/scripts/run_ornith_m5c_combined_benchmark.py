#!/usr/bin/env python3
"""Paired same-session M5C raw-Python vs native-combined benchmark."""

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
    compare_trace,
    engine_stats,
    load_runtime,
)
from scripts.run_ornith_m5b_execution_benchmark import (  # noqa: E402
    exact_trace_comparison,
    thermal_clock_analysis,
    tracked_source_clean,
)
from scripts.run_ornith_router_backend_benchmark import (  # noqa: E402
    EXPECTED_TOKENS,
    MAX_PROCESS_VRAM_BYTES,
    git_revision,
    run_once,
    sha256_file,
    summarize_trials,
)


CONTROL = "production"
CANDIDATE = "native-combined"
MODES = (CONTROL, CANDIDATE)


def analyze_workspace(
        stats: dict[str, Any], *, layers: int, topk: int,
        hidden: int) -> dict[str, Any]:
    expected_dispatch_per_layer = topk * np.dtype(np.int64).itemsize
    expected_raw_per_layer = topk * hidden * np.dtype(np.float32).itemsize
    by_layer = stats["by_layer"]
    per_layer_exact = (
        len(by_layer) == layers
        and all(
            row["host_moe_dispatch_bytes"] == expected_dispatch_per_layer
            and row["device_moe_raw_workspace_bytes"] == expected_raw_per_layer
            for row in by_layer
        )
    )
    aggregate = stats["aggregate"]
    aggregate_exact = (
        aggregate["host_moe_dispatch_bytes"]
        == layers * expected_dispatch_per_layer
        and aggregate["device_moe_raw_workspace_bytes"]
        == layers * expected_raw_per_layer
    )
    return {
        "expected_layer_count": layers,
        "expected_dispatch_bytes_per_layer": expected_dispatch_per_layer,
        "expected_raw_workspace_bytes_per_layer": expected_raw_per_layer,
        "aggregate_host_moe_dispatch_bytes": aggregate[
            "host_moe_dispatch_bytes"
        ],
        "aggregate_device_moe_raw_workspace_bytes": aggregate[
            "device_moe_raw_workspace_bytes"
        ],
        "per_layer_exact": per_layer_exact,
        "aggregate_exact": aggregate_exact,
        "passed": per_layer_exact and aggregate_exact,
    }


def run_m5c_once(
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
        raise RuntimeError(f"M5C benchmark requires exactly two GPUs, got {gpu_count}")
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
    source_clean = tracked_source_clean()
    runtime_binary = {
        "path": str(binary),
        "bytes": binary.stat().st_size,
        "sha256": sha256_file(binary),
    }
    layer_count = len(runtime["engines"])
    topk = 8
    hidden = int(runtime["engines"][0].hidden_dim())

    warmups = {}
    for mode in MODES:
        row, _ = run_m5c_once(
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
        row, _ = run_m5c_once(
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

    control_trace_row, control_trace = run_m5c_once(
        runtime,
        tokenizer,
        CONTROL,
        trace=True,
        layers=layer_count,
        topk=topk,
        hidden=hidden,
    )
    candidate_trace_row, candidate_trace = run_m5c_once(
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
    all_rows = [row for mode in MODES for row in trials[mode]]
    combined_rows = trials[CANDIDATE]
    control_rows = trials[CONTROL]
    paired = paired_trial_analysis(sequence)
    thermal = thermal_clock_analysis(all_rows)

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
        "combined_path_used_for_all_160_layer_calls": all(
            row["path_proof"]["native_combined_calls"] == 160
            for row in combined_rows
        ),
        "combined_path_has_zero_native_fallbacks": all(
            row["path_proof"]["native_combined_fallback_calls"] == 0
            for row in combined_rows
        ),
        "combined_path_has_zero_python_combines": all(
            row["path_proof"]["python_combine_calls"] == 0
            for row in combined_rows
        ),
        "combined_path_has_zero_python_raw_allocations": all(
            row["path_proof"]["raw_output_allocations"] == 0
            for row in combined_rows
        ),
        "combined_path_hands_off_all_160_streams": all(
            row["path_proof"]["native_combined_stream_handoffs"] == 160
            for row in combined_rows
        ),
        "combined_workspace_counters_exact_and_bounded": all(
            row["engine_workspace"]["passed"] for row in combined_rows
        ),
        "combined_path_labels_internal_id_d2h": all(
            row["path_proof"]["native_combined_ids_d2h_total_bytes"] > 0
            for row in combined_rows
        ),
        "control_keeps_raw_python_boundary": all(
            row["path_proof"]["native_combined_calls"] == 0
            and row["path_proof"]["python_combine_calls"] == 160
            and row["path_proof"]["raw_output_allocations"] == 160
            for row in control_rows
        ),
        "combined_trace_uses_combined_path": (
            candidate_trace_row["path_proof"]["native_combined_calls"] == 160
            and candidate_trace_row["path_proof"][
                "native_combined_fallback_calls"
            ] == 0
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
        "pareto": {
            "control_raw_python_median_tps": control_tps,
            "candidate_native_combined_median_tps": candidate_tps,
            "speedup_ratio": speedup,
            "speedup_percent": (speedup - 1.0) * 100.0,
            "gates": gates,
        },
        "implementation_boundary": {
            "control": (
                "Python allocates raw FP32 rows, launches FP32-to-FP16, "
                "multiplies/adds each selected expert, and synchronizes the "
                "native compute stream before returning."
            ),
            "candidate": (
                "Engine-owned persistent raw FP32 workspace; exact stable "
                "FP16 weighted combine kernel; completion event handed to "
                "the active PyTorch stream; no Python raw/combine work."
            ),
            "measured_residual": (
                "Expert IDs still make one bounded int64 D2H dispatch copy "
                "inside the native API. M5D targets this residual."
            ),
        },
    }
    report_path = args.output_dir / "m5c-native-combined-benchmark.json"
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
        "M5C_COMBINED_BENCHMARK="
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
