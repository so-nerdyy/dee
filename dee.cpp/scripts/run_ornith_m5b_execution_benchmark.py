#!/usr/bin/env python3
"""Paired same-session M5B production-vs-evidence execution benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ornith_generation import (  # noqa: E402
    TOLERANCES,
    compare_trace,
    load_runtime,
)
from scripts.run_ornith_router_backend_benchmark import (  # noqa: E402
    EXPECTED_TOKENS,
    MAX_PROCESS_VRAM_BYTES,
    git_revision,
    run_once,
    sha256_file,
    summarize_trials,
)


MODES = ("debug-full-logit", "production")


def balanced_order(trials: int) -> list[str]:
    pattern = [
        "debug-full-logit",
        "production",
        "production",
        "debug-full-logit",
        "debug-full-logit",
        "production",
    ]
    order: list[str] = []
    while any(order.count(mode) < trials for mode in MODES):
        for mode in pattern:
            if order.count(mode) < trials:
                order.append(mode)
    return order


def exact_trace_comparison(left, right) -> dict[str, Any]:
    required_categories = set(TOLERANCES) | {"expert_ids"}
    categories = sorted(required_categories | set(left.records) | set(right.records))
    results = {}
    for category in categories:
        expected = left.records.get(category, [])
        actual = right.records.get(category, [])
        labels_exact = [label for label, _ in expected] == [
            label for label, _ in actual
        ]
        arrays_exact = labels_exact and len(expected) == len(actual)
        maximum_absolute_error = 0.0
        if arrays_exact:
            for (_, expected_array), (_, actual_array) in zip(expected, actual):
                if (
                    expected_array.shape != actual_array.shape
                    or expected_array.dtype != actual_array.dtype
                ):
                    arrays_exact = False
                    break
                if expected_array.size:
                    maximum_absolute_error = max(
                        maximum_absolute_error,
                        float(np.max(np.abs(
                            expected_array.astype(np.float64)
                            - actual_array.astype(np.float64)
                        ))),
                    )
                if not np.array_equal(expected_array, actual_array):
                    arrays_exact = False
        results[category] = {
            "labels_exact": labels_exact,
            "arrays_exact": arrays_exact,
            "maximum_absolute_error": maximum_absolute_error,
            "record_count": len(expected),
        }
    all_required_categories_present = all(
        left.records.get(category) and right.records.get(category)
        for category in required_categories
    )
    return {
        "required_categories": sorted(required_categories),
        "all_required_categories_present": all_required_categories_present,
        "all_categories_bitwise_exact": (
            all_required_categories_present
            and all(item["arrays_exact"] for item in results.values())
        ),
        "categories": results,
    }


def tracked_source_clean() -> bool:
    return (
        subprocess.run(
            ["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False
        ).returncode == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT, check=False
        ).returncode == 0
    )


def paired_trial_analysis(sequence: list[dict[str, Any]]) -> dict[str, Any]:
    if len(sequence) % 2:
        raise ValueError("paired sequence must contain an even number of rows")
    pairs = []
    for index in range(0, len(sequence), 2):
        rows = {
            row["execution_mode"]: row
            for row in sequence[index:index + 2]
        }
        if set(rows) != set(MODES):
            raise ValueError(f"pair {index // 2} does not contain both modes")
        debug_tps = float(rows["debug-full-logit"]["tokens_per_second"])
        production_tps = float(rows["production"]["tokens_per_second"])
        pairs.append({
            "pair_index": index // 2,
            "sequence_indices": [index, index + 1],
            "first_mode": sequence[index]["execution_mode"],
            "debug_tps": debug_tps,
            "production_tps": production_tps,
            "speedup_ratio": production_tps / debug_tps,
            "speedup_percent": (production_tps / debug_tps - 1.0) * 100.0,
            "production_won": production_tps > debug_tps,
        })
    ratios = [row["speedup_ratio"] for row in pairs]
    return {
        "pair_count": len(pairs),
        "production_wins": sum(row["production_won"] for row in pairs),
        "median_speedup_ratio": float(np.median(ratios)),
        "minimum_speedup_ratio": min(ratios),
        "maximum_speedup_ratio": max(ratios),
        "pairs": pairs,
    }


def thermal_clock_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    devices: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    errors = []
    for row in rows:
        thermal = row.get("thermal_clock", {})
        if thermal.get("nvml_error"):
            errors.append(thermal["nvml_error"])
        for device, values in thermal.get("by_device", {}).items():
            devices.setdefault(device, []).append(
                (str(row.get("execution_mode", "<unspecified>")), values)
            )
    summary = {}
    for device, mode_values in sorted(devices.items()):
        values = [item[1] for item in mode_values]
        sample_count = sum(int(item.get("sample_count", 0)) for item in values)
        maximum_temperature = max(
            (
                float(item["maximum_temperature_c"])
                for item in values
                if item.get("maximum_temperature_c") is not None
            ),
            default=None,
        )
        median_clocks = [
            float(item["median_sm_clock_mhz"])
            for item in values
            if item.get("median_sm_clock_mhz") is not None
        ]
        clocks_by_mode: dict[str, list[float]] = {}
        temperatures_by_mode: dict[str, list[float]] = {}
        for mode, item in mode_values:
            if item.get("median_sm_clock_mhz") is not None:
                clocks_by_mode.setdefault(mode, []).append(
                    float(item["median_sm_clock_mhz"])
                )
            if item.get("median_temperature_c") is not None:
                temperatures_by_mode.setdefault(mode, []).append(
                    float(item["median_temperature_c"])
                )
        mode_median_clocks = {
            mode: float(np.median(mode_clocks))
            for mode, mode_clocks in clocks_by_mode.items()
        }
        mode_median_temperatures = {
            mode: float(np.median(mode_temperatures))
            for mode, mode_temperatures in temperatures_by_mode.items()
        }
        clock_values = list(mode_median_clocks.values())
        temperature_values = list(mode_median_temperatures.values())
        expected_modes = {mode for mode, _item in mode_values}
        clock_parity_ratio = (
            min(clock_values) / max(clock_values)
            if len(clock_values) >= 2 and max(clock_values) > 0
            else None
        )
        temperature_parity_delta = (
            max(temperature_values) - min(temperature_values)
            if len(temperature_values) >= 2
            else None
        )
        summary[device] = {
            "sample_count": sample_count,
            "maximum_temperature_c": maximum_temperature,
            "minimum_trial_median_sm_clock_mhz": (
                min(median_clocks) if median_clocks else None
            ),
            "mode_median_sm_clock_mhz": mode_median_clocks,
            "mode_median_temperature_c": mode_median_temperatures,
            "clock_parity_ratio": clock_parity_ratio,
            "temperature_parity_delta_c": temperature_parity_delta,
            "temperature_anomaly": (
                maximum_temperature is None or maximum_temperature > 85.0
            ),
            "clock_floor_below_900_mhz_warning": (
                bool(median_clocks) and min(median_clocks) < 900.0
            ),
            "clock_samples_missing": (
                not median_clocks
                or len(mode_median_clocks) != len(expected_modes)
            ),
            "clock_parity_anomaly": (
                clock_parity_ratio is not None
                and clock_parity_ratio < 0.95
            ),
            "temperature_parity_anomaly": (
                temperature_parity_delta is not None
                and temperature_parity_delta > 5.0
            ),
        }
    samples_present = bool(summary) and all(
        item["sample_count"] > 0 for item in summary.values()
    )
    return {
        "nvml_errors": sorted(set(errors)),
        "by_device": summary,
        "samples_present": samples_present,
        "absolute_clock_floor_warning_observed": any(
            item["clock_floor_below_900_mhz_warning"]
            for item in summary.values()
        ),
        "anomaly_detected": (
            bool(errors)
            or not samples_present
            or any(
                item["temperature_anomaly"]
                or item["clock_samples_missing"]
                or item["clock_parity_anomaly"]
                or item["temperature_parity_anomaly"]
                for item in summary.values()
            )
        ),
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
    if gpu_count < 2:
        raise RuntimeError("M5B benchmark requires the two-GPU Ornith layout")
    if args.require_dual_gpu and gpu_count != 2:
        raise RuntimeError(f"dual-T4 benchmark requires exactly 2 GPUs, got {gpu_count}")
    if args.require_dual_gpu and any(
            "T4" not in torch.cuda.get_device_name(device) for device in range(2)):
        raise RuntimeError(
            "dual-T4 benchmark found "
            + ", ".join(torch.cuda.get_device_name(device) for device in range(2))
        )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    runtime = load_runtime(
        args.model_dir,
        2,
        args.split_layer,
        args.cache_experts,
        router_backend="torch-device",
    )
    load_seconds = time.perf_counter() - started
    import pydee.pydee_core as pydee_core

    runtime_binary_path = Path(pydee_core.__file__).resolve()
    source_clean = tracked_source_clean()
    runtime_binary = {
        "path": str(runtime_binary_path),
        "bytes": runtime_binary_path.stat().st_size,
        "sha256": sha256_file(runtime_binary_path),
    }

    warmups: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        row, _ = run_once(
            runtime,
            tokenizer,
            "torch-device",
            trace=False,
            execution_mode=mode,
        )
        warmups[mode] = row

    order = balanced_order(args.trials)
    trials: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    sequence = []
    for sequence_index, mode in enumerate(order):
        row, _ = run_once(
            runtime,
            tokenizer,
            "torch-device",
            trace=False,
            execution_mode=mode,
        )
        row["sequence_index"] = sequence_index
        row["mode_trial_index"] = len(trials[mode])
        trials[mode].append(row)
        sequence.append(row)

    debug_trace_row, debug_trace = run_once(
        runtime,
        tokenizer,
        "torch-device",
        trace=True,
        execution_mode="debug-full-logit",
    )
    production_trace_row, production_trace = run_once(
        runtime,
        tokenizer,
        "torch-device",
        trace=True,
        execution_mode="production",
    )
    comparisons = compare_trace(
        debug_trace["collector"], production_trace["collector"]
    )
    exact_comparisons = exact_trace_comparison(
        debug_trace["collector"], production_trace["collector"]
    )
    correctness = {
        "debug_tokens": debug_trace_row["generated_token_ids"],
        "production_tokens": production_trace_row["generated_token_ids"],
        "generated_token_ids_exact": (
            debug_trace_row["generated_token_ids"]
            == production_trace_row["generated_token_ids"]
            == EXPECTED_TOKENS
        ),
        "decoded_text_exact": (
            debug_trace_row["generated_text"]
            == production_trace_row["generated_text"]
        ),
        "all_trace_categories_passed": comparisons["all_categories_passed"],
        "comparisons": comparisons,
        "all_trace_categories_bitwise_exact": exact_comparisons[
            "all_categories_bitwise_exact"
        ],
        "exact_comparisons": exact_comparisons,
    }

    summaries = {mode: summarize_trials(trials[mode]) for mode in MODES}
    debug_tps = summaries["debug-full-logit"]["tokens_per_second"]["median"]
    production_tps = summaries["production"]["tokens_per_second"]["median"]
    speedup = production_tps / debug_tps
    all_rows = [row for mode in MODES for row in trials[mode]]
    paired = paired_trial_analysis(sequence)
    thermal = thermal_clock_analysis(all_rows)
    gates = {
        "production_independent_median_gain_at_least_0_5_percent": (
            speedup >= 1.005
        ),
        "production_paired_median_gain_at_least_0_5_percent": (
            paired["median_speedup_ratio"] >= 1.005
        ),
        "production_wins_majority_of_pairs": (
            paired["production_wins"] >= paired["pair_count"] // 2 + 1
        ),
        "no_pair_regresses_more_than_2_percent": (
            paired["minimum_speedup_ratio"] >= 0.98
        ),
        "tokens_and_text_exact": (
            correctness["generated_token_ids_exact"]
            and correctness["decoded_text_exact"]
        ),
        "all_trace_categories_passed": correctness["all_trace_categories_passed"],
        "all_trace_categories_bitwise_exact": correctness[
            "all_trace_categories_bitwise_exact"
        ],
        "all_40_layers_executed": all(
            row["all_40_layers_executed"] for row in all_rows
        ),
        "all_expert_calls_device_resident": all(
            row["path_proof"]["device_path_share"] == 1.0 for row in all_rows
        ),
        "zero_host_fallback_calls": all(
            row["path_proof"]["host_path_fallback_calls"] == 0 for row in all_rows
        ),
        "production_has_zero_full_logits_host_records": all(
            row["full_logits_host_records"] == 0
            for row in trials["production"]
        ),
        "debug_control_captures_full_logits": all(
            row["full_logits_host_records"] == len(EXPECTED_TOKENS)
            for row in trials["debug-full-logit"]
        ),
        "process_vram_at_most_8_gib_per_gpu": all(
            max(row["peak_vram_bytes"].values()) <= MAX_PROCESS_VRAM_BYTES
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
            "debug_full_logit_median_tps": debug_tps,
            "production_median_tps": production_tps,
            "speedup_ratio": speedup,
            "speedup_percent": (speedup - 1.0) * 100.0,
            "gates": gates,
        },
        "implementation_boundary": {
            "debug_full_logit": (
                "Historical full-logit D2H copy, per-token token/mask "
                "allocations, and separate combine accumulator."
            ),
            "production": (
                "No evidence-only full-logit host copy; bounded reusable token "
                "and attention-mask buffers; combine accumulates into the "
                "already-zeroed output."
            ),
            "unchanged": (
                "Official checkpoint, torch-device router, all 40 layers, "
                "dee.cpp expert path, cache cap 32, split 20/20, FP16, greedy "
                "selection, output text, and trace tolerances."
            ),
        },
    }
    report_path = args.output_dir / "m5b-execution-benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "result": result,
        "files": {
            report_path.name: {
                "bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
            }
        },
    }
    manifest_path = args.output_dir / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    runtime["pool"].close()
    del runtime
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    for device in range(2):
        torch.cuda.synchronize(device)
    print(
        "M5B_EXECUTION_BENCHMARK="
        + json.dumps(
            {
                "result": result,
                "debug_tps": debug_tps,
                "production_tps": production_tps,
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
