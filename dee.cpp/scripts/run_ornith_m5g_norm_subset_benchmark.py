#!/usr/bin/env python3
"""Paired M5G regular-only and gated-only fused-normalization benchmark."""

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
    FUSED_GATED_NORM_EXECUTION_MODE,
    FUSED_REGULAR_NORM_EXECUTION_MODE,
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
from scripts.run_ornith_m5e_fused_norm_benchmark import (  # noqa: E402
    EXPECTED_GATED_WRAPPERS,
    EXPECTED_REGULAR_WRAPPERS,
    native_contract_probe,
    normalization_inventory_contract,
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
REGULAR = FUSED_REGULAR_NORM_EXECUTION_MODE
GATED = FUSED_GATED_NORM_EXECUTION_MODE
CANDIDATES = (REGULAR, GATED)
MODES = (CONTROL, *CANDIDATES)


def balanced_order(candidate: str, trials: int) -> list[str]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown norm subset candidate: {candidate}")
    pattern = [
        CONTROL,
        candidate,
        candidate,
        CONTROL,
        CONTROL,
        candidate,
    ]
    order: list[str] = []
    while (
        order.count(CONTROL) < trials
        or order.count(candidate) < trials
    ):
        for mode in pattern:
            if mode == CONTROL and order.count(mode) >= trials:
                continue
            if mode == candidate and order.count(mode) >= trials:
                continue
            order.append(mode)
    return order


def paired_trial_analysis(
    candidate: str, sequence: list[dict[str, Any]]
) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown norm subset candidate: {candidate}")
    if len(sequence) % 2:
        raise ValueError("paired sequence must contain an even number of rows")
    pairs = []
    for index in range(0, len(sequence), 2):
        pair_rows = sequence[index:index + 2]
        rows = {row["execution_mode"]: row for row in pair_rows}
        if set(rows) != {CONTROL, candidate}:
            raise ValueError(
                f"pair {index // 2} does not contain control and {candidate}"
            )
        control_tps = float(rows[CONTROL]["tokens_per_second"])
        candidate_tps = float(rows[candidate]["tokens_per_second"])
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


def expected_norm_calls(mode: str) -> dict[str, int]:
    generated_tokens = len(EXPECTED_TOKENS)
    regular = (
        EXPECTED_REGULAR_WRAPPERS * generated_tokens
        if mode == REGULAR
        else 0
    )
    gated = (
        EXPECTED_GATED_WRAPPERS * generated_tokens
        if mode == GATED
        else 0
    )
    return {
        "regular": regular,
        "gated": gated,
        "allocations": regular + gated,
    }


def run_subset_once(
    runtime,
    tokenizer,
    mode: str,
    *,
    trace: bool,
    layers: int,
    topk: int,
    hidden: int,
) -> tuple[dict, dict]:
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


def candidate_gates(
    *,
    candidate: str,
    sequence: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    paired: dict[str, Any],
    speedup_ratio: float,
    tolerance_comparison: dict[str, Any],
    exact_comparison: dict[str, Any],
    thermal: dict[str, Any],
    source_clean: bool,
) -> dict[str, bool]:
    rows = [*sequence, *trace_rows]
    candidate_rows = [
        row for row in rows if row["execution_mode"] == candidate
    ]
    control_rows = [
        row for row in rows if row["execution_mode"] == CONTROL
    ]
    expected = expected_norm_calls(candidate)
    return {
        "candidate_independent_median_gain_at_least_1_percent":
            speedup_ratio >= 1.01,
        "candidate_paired_median_gain_at_least_1_percent":
            paired["median_speedup_ratio"] >= 1.01,
        "candidate_wins_majority_of_pairs":
            paired["candidate_wins"] >= paired["pair_count"] // 2 + 1,
        "no_pair_regresses_more_than_2_percent":
            paired["minimum_speedup_ratio"] >= 0.98,
        "tokens_and_text_exact": all(
            row["generated_token_ids"] == EXPECTED_TOKENS
            for row in trace_rows
        ) and trace_rows[0]["generated_text"] == trace_rows[1][
            "generated_text"
        ],
        "all_trace_categories_passed":
            tolerance_comparison["all_categories_passed"],
        "all_trace_categories_bitwise_exact":
            exact_comparison["all_categories_bitwise_exact"],
        "all_40_layers_executed": all(
            row["all_40_layers_executed"] for row in rows
        ),
        "all_expert_calls_device_resident": all(
            row["path_proof"]["device_path_share"] == 1.0 for row in rows
        ),
        "zero_host_fallback_calls": all(
            row["path_proof"]["host_path_fallback_calls"] == 0
            for row in rows
        ),
        "same_direct_expert_path_both_modes": all(
            row["path_proof"]["native_combined_calls"] == 160
            and row["path_proof"]["native_direct_calls"] == 160
            and row["path_proof"]["native_direct_fallback_calls"] == 0
            for row in rows
        ),
        "control_has_zero_fused_norm_calls": all(
            row["path_proof"]["fused_rms_norm_calls"] == 0
            and row["path_proof"]["fused_rms_norm_gated_calls"] == 0
            and row["path_proof"]["fused_norm_output_allocations"] == 0
            for row in control_rows
        ),
        "candidate_regular_norm_call_count_exact": all(
            row["path_proof"]["fused_rms_norm_calls"]
            == expected["regular"]
            for row in candidate_rows
        ),
        "candidate_gated_norm_call_count_exact": all(
            row["path_proof"]["fused_rms_norm_gated_calls"]
            == expected["gated"]
            for row in candidate_rows
        ),
        "candidate_output_allocation_count_exact": all(
            row["path_proof"]["fused_norm_output_allocations"]
            == expected["allocations"]
            for row in candidate_rows
        ),
        "candidate_has_zero_fused_norm_failures": all(
            row["path_proof"]["fused_norm_failures"] == 0
            for row in candidate_rows
        ),
        "combined_workspace_counters_exact_and_bounded": all(
            row["engine_workspace"]["passed"] for row in rows
        ),
        "process_vram_at_most_8_gib_per_gpu": all(
            max(row["peak_vram_bytes"].values())
            <= MAX_PROCESS_VRAM_BYTES
            for row in rows
        ),
        "thermal_clock_samples_present": thermal["samples_present"],
        "no_thermal_or_clock_anomaly": not thermal["anomaly_detected"],
        "tracked_source_clean": source_clean,
        # A candidate may be accepted only when the optimization is an
        # execution-equivalent subset of the control: the exact same direct
        # expert path, exact trace bits, and the existing tolerance contract.
        # Keep this as one explicit fail-closed gate so an outer harness cannot
        # mistake a faster but semantically different path for a valid result.
        "execution_equivalence_fail_closed": (
            exact_comparison["all_categories_bitwise_exact"]
            and tolerance_comparison["all_categories_passed"]
            and all(
                row["generated_token_ids"] == EXPECTED_TOKENS
                for row in trace_rows
            )
            and trace_rows[0]["generated_text"]
            == trace_rows[1]["generated_text"]
            and all(
                row["path_proof"]["native_combined_calls"] == 160
                and row["path_proof"]["native_direct_calls"] == 160
                and row["path_proof"]["native_direct_fallback_calls"] == 0
                for row in rows
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
    if gpu_count != 2:
        raise RuntimeError(
            f"M5G benchmark requires exactly two GPUs, got {gpu_count}"
        )
    if args.require_dual_gpu and any(
        "T4" not in torch.cuda.get_device_name(device)
        for device in range(2)
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
    inventory = normalization_inventory_contract(
        runtime["fused_norm_inventory"]
    )
    native_probe = native_contract_probe(runtime, torch)
    shared_gates = {
        "normalization_inventory_contract_exact": inventory["passed"],
        "native_contract_probe_all_half_bits_exact":
            native_probe["all_half_bits_exact"],
        "native_contract_probe_real_shapes_nonzero_epsilon":
            native_probe["real_shapes_and_nonzero_epsilon"],
        "tracked_source_clean": source_clean,
        "runtime_binary_hashed": bool(runtime_binary["sha256"]),
    }

    warmups = {}
    for mode in MODES:
        warmups[mode], _ = run_subset_once(
            runtime,
            tokenizer,
            mode,
            trace=False,
            layers=layer_count,
            topk=topk,
            hidden=hidden,
        )

    candidate_reports = {}
    for candidate in CANDIDATES:
        order = balanced_order(candidate, args.trials)
        trials = {CONTROL: [], candidate: []}
        sequence = []
        for sequence_index, mode in enumerate(order):
            row, _ = run_subset_once(
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

        control_trace_row, control_trace = run_subset_once(
            runtime,
            tokenizer,
            CONTROL,
            trace=True,
            layers=layer_count,
            topk=topk,
            hidden=hidden,
        )
        candidate_trace_row, candidate_trace = run_subset_once(
            runtime,
            tokenizer,
            candidate,
            trace=True,
            layers=layer_count,
            topk=topk,
            hidden=hidden,
        )
        tolerance = compare_trace(
            control_trace["collector"], candidate_trace["collector"]
        )
        exact = exact_trace_comparison(
            control_trace["collector"], candidate_trace["collector"]
        )
        summaries = {
            mode: summarize_trials(rows) for mode, rows in trials.items()
        }
        control_tps = summaries[CONTROL]["tokens_per_second"]["median"]
        candidate_tps = summaries[candidate][
            "tokens_per_second"
        ]["median"]
        speedup_ratio = candidate_tps / control_tps
        paired = paired_trial_analysis(candidate, sequence)
        thermal = thermal_clock_analysis(sequence)
        trace_rows = [control_trace_row, candidate_trace_row]
        gates = candidate_gates(
            candidate=candidate,
            sequence=sequence,
            trace_rows=trace_rows,
            paired=paired,
            speedup_ratio=speedup_ratio,
            tolerance_comparison=tolerance,
            exact_comparison=exact,
            thermal=thermal,
            source_clean=source_clean,
        )
        candidate_reports[candidate] = {
            "result": "PASS" if all(gates.values()) else "FAIL",
            "candidate_accepted": all(gates.values()),
            "expected_norm_calls_per_generation":
                expected_norm_calls(candidate),
            "measurement_order": order,
            "trial_sequence": sequence,
            "mode_summaries": summaries,
            "paired_analysis": paired,
            "thermal_clock_analysis": thermal,
            "trace_rows": {
                CONTROL: control_trace_row,
                candidate: candidate_trace_row,
            },
            "correctness": {
                "tolerance_comparison": tolerance,
                "exact_comparison": exact,
            },
            "pareto": {
                "control_median_tps": control_tps,
                "candidate_median_tps": candidate_tps,
                "control_median_tpot_seconds": 1.0 / control_tps,
                "candidate_median_tpot_seconds": 1.0 / candidate_tps,
                "speedup_ratio": speedup_ratio,
                "speedup_percent": (speedup_ratio - 1.0) * 100.0,
                "gates": gates,
            },
            "execution_equivalence": {
                "verdict": (
                    "PASS"
                    if gates["execution_equivalence_fail_closed"]
                    else "REJECTED_NON_EQUIVALENT_EXECUTION_PATH"
                ),
                "fail_closed": True,
                "gate": gates["execution_equivalence_fail_closed"],
                "required": [
                    "all_trace_categories_bitwise_exact",
                    "all_trace_categories_passed",
                    "tokens_and_text_exact",
                    "same_direct_expert_path_both_modes",
                ],
            },
        }

    accepted = [
        (mode, report)
        for mode, report in candidate_reports.items()
        if report["candidate_accepted"]
    ]
    accepted_best = (
        max(
            accepted,
            key=lambda item: item[1]["pareto"]["candidate_median_tps"],
        )[0]
        if accepted
        else None
    )
    result = (
        "PASS"
        if all(shared_gates.values()) and accepted_best is not None
        else "FAIL"
    )
    if accepted_best is not None:
        verdict = "PASS"
    elif any(
        row["execution_equivalence"]["verdict"]
        == "REJECTED_NON_EQUIVALENT_EXECUTION_PATH"
        for row in candidate_reports.values()
    ):
        verdict = "REJECTED_NON_EQUIVALENT_EXECUTION_PATH"
    else:
        verdict = "FAIL"
    report = {
        "schema_version": 1,
        "result": result,
        "verdict": verdict,
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
            "same_loaded_runtime": True,
            "prompt": "Hello",
            "max_new_tokens": len(EXPECTED_TOKENS),
        },
        "runtime_load_seconds": load_seconds,
        "runtime_identity": {
            "tracked_source_clean": source_clean,
            "git_commit": git_revision(),
            "pydee_core": runtime_binary,
        },
        "shared_gates": shared_gates,
        "warmups": warmups,
        "normalization_contract": {
            "inventory": inventory,
            "native_contract_probe": native_probe,
            "candidate_modes": {
                REGULAR: (
                    "Fuse only 81 layer/final Qwen RMSNorm wrappers; keep "
                    "all 30 gated norms eager."
                ),
                GATED: (
                    "Fuse only 30 linear-attention gated RMSNorm wrappers; "
                    "keep all 81 layer/final norms eager."
                ),
            },
        },
        "candidate_reports": candidate_reports,
        "accepted_best": accepted_best,
        "stage_timing_scope": {
            "attention_time": None,
            "expert_time": None,
            "synchronization_time": None,
            "reason": (
                "M5G is an unprofiled exact subset-isolation A/B. M5A "
                "supplies the separately instrumented critical-path timing."
            ),
        },
    }
    report_path = args.output_dir / "m5g-norm-subset-benchmark.json"
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
        "M5G_NORM_SUBSET_BENCHMARK="
        + json.dumps(
            {
                "result": result,
                "accepted_best": accepted_best,
                "candidate_results": {
                    mode: candidate_reports[mode]["result"]
                    for mode in CANDIDATES
                },
                "output": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
