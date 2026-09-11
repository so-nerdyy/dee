#!/usr/bin/env python3
"""Paired exactness/Pareto benchmark for FLA recurrent linear attention."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

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


CONTROL = "torch-recurrent"
CANDIDATE = "fla-recurrent"
BASE_EXECUTION_MODE = "native-combined-direct"
LINEAR_LAYER_COUNT = 30
EXPECTED_FLA_VERSION = "0.5.2"


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
    while (
        order.count(CONTROL) < trials
        or order.count(CANDIDATE) < trials
    ):
        for mode in pattern:
            if mode == CONTROL and order.count(mode) >= trials:
                continue
            if mode == CANDIDATE and order.count(mode) >= trials:
                continue
            order.append(mode)
    return order


def paired_trial_analysis(
    sequence: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(sequence) % 2:
        raise ValueError("paired sequence must contain an even number of rows")
    pairs = []
    for index in range(0, len(sequence), 2):
        pair_rows = sequence[index:index + 2]
        rows = {row["execution_mode"]: row for row in pair_rows}
        if set(rows) != {CONTROL, CANDIDATE}:
            raise ValueError(
                f"pair {index // 2} does not contain control and candidate"
            )
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
    import statistics

    return {
        "pair_count": len(pairs),
        "candidate_wins": sum(row["candidate_won"] for row in pairs),
        "median_speedup_ratio": statistics.median(ratios),
        "minimum_speedup_ratio": min(ratios),
        "maximum_speedup_ratio": max(ratios),
        "pairs": pairs,
    }


class CountedBackend:
    """Count calls without changing the selected backend's arguments."""

    def __init__(self, name: str, function: Callable[..., Any]):
        self.name = name
        self.function = function
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.function(*args, **kwargs)

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": getattr(self.function, "__module__", None),
            "qualname": getattr(self.function, "__qualname__", None),
        }


def expected_recurrent_calls(generated_tokens: int) -> int:
    """The prefill uses the chunk path; subsequent tokens use recurrence."""

    if generated_tokens < 1:
        raise ValueError("generated_tokens must be positive")
    return LINEAR_LAYER_COUNT * (generated_tokens - 1)


def prepare_fallback_model_construction():
    """Save FLA recurrence, then force model construction onto baseline ops."""

    from transformers.models.qwen3_5_moe import (
        modeling_qwen3_5_moe as modeling,
    )

    fast_recurrent = modeling.fused_recurrent_gated_delta_rule
    if fast_recurrent is None:
        raise RuntimeError(
            "FLA recurrent kernel is unavailable; install "
            "flash-linear-attention before importing Qwen3.5-MoE"
        )
    original = {
        "FusedRMSNormGated": modeling.FusedRMSNormGated,
        "causal_conv1d_fn": modeling.causal_conv1d_fn,
        "causal_conv1d_update": modeling.causal_conv1d_update,
        "chunk_gated_delta_rule": modeling.chunk_gated_delta_rule,
        "fused_recurrent_gated_delta_rule":
            modeling.fused_recurrent_gated_delta_rule,
    }
    # Keep every non-candidate operation identical to the sealed control:
    # fallback causal convolution, fallback chunked prefill, and fallback gated
    # RMSNorm. Only the cached single-token recurrent rule is switched later.
    modeling.FusedRMSNormGated = None
    modeling.causal_conv1d_fn = None
    modeling.causal_conv1d_update = None
    modeling.chunk_gated_delta_rule = None
    modeling.fused_recurrent_gated_delta_rule = None
    return modeling, fast_recurrent, original


def restore_modeling_globals(modeling, original: dict[str, Any]) -> None:
    for name, value in original.items():
        setattr(modeling, name, value)


def linear_attention_modules(runtime) -> list[Any]:
    modules = [
        block.linear_attn
        for block in runtime["model"].model.layers
        if hasattr(block, "linear_attn")
    ]
    if len(modules) != LINEAR_LAYER_COUNT:
        raise RuntimeError(
            f"expected {LINEAR_LAYER_COUNT} linear-attention layers, "
            f"got {len(modules)}"
        )
    return modules


def select_backend(
    modules: list[Any], backend: CountedBackend
) -> dict[str, Any]:
    backend.reset()
    for module in modules:
        module.recurrent_gated_delta_rule = backend
    return {
        "backend": backend.identity(),
        "layer_count": len(modules),
        "all_layers_selected": all(
            module.recurrent_gated_delta_rule is backend
            for module in modules
        ),
        "chunk_backends": sorted({
            (
                getattr(module.chunk_gated_delta_rule, "__module__", None),
                getattr(module.chunk_gated_delta_rule, "__qualname__", None),
            )
            for module in modules
        }),
        "causal_conv_backends": sorted({
            (
                getattr(module.causal_conv1d_update, "__module__", None),
                getattr(module.causal_conv1d_update, "__qualname__", None),
            )
            for module in modules
        }),
    }


def run_backend_once(
    runtime,
    tokenizer,
    modules: list[Any],
    backend: CountedBackend,
    *,
    trace: bool,
    layers: int,
    topk: int,
    hidden: int,
) -> tuple[dict, dict]:
    selection = select_backend(modules, backend)
    row, raw = run_once(
        runtime,
        tokenizer,
        "torch-device",
        trace=trace,
        execution_mode=BASE_EXECUTION_MODE,
    )
    row["expert_execution_mode"] = row["execution_mode"]
    row["execution_mode"] = backend.name
    row["linear_attention_backend"] = backend.name
    row["recurrent_call_count"] = backend.calls
    row["backend_selection"] = selection
    row["engine_workspace"] = analyze_workspace(
        engine_stats(runtime), layers=layers, topk=topk, hidden=hidden
    )
    return row, raw


def candidate_gates(
    *,
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
        row for row in rows if row["execution_mode"] == CANDIDATE
    ]
    control_rows = [
        row for row in rows if row["execution_mode"] == CONTROL
    ]
    expected_calls = expected_recurrent_calls(len(EXPECTED_TOKENS))
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
            and row["path_proof"]["native_pointer_batched_calls"] == 0
            for row in rows
        ),
        "no_fused_norm_path_in_either_mode": all(
            row["path_proof"]["fused_rms_norm_calls"] == 0
            and row["path_proof"]["fused_rms_norm_gated_calls"] == 0
            and row["path_proof"]["fused_norm_failures"] == 0
            for row in rows
        ),
        "control_recurrent_call_count_exact": all(
            row["recurrent_call_count"] == expected_calls
            for row in control_rows
        ),
        "candidate_recurrent_call_count_exact": all(
            row["recurrent_call_count"] == expected_calls
            for row in candidate_rows
        ),
        "backend_selected_for_all_linear_layers": all(
            row["backend_selection"]["layer_count"] == LINEAR_LAYER_COUNT
            and row["backend_selection"]["all_layers_selected"]
            for row in rows
        ),
        "control_backend_is_transformers_fallback": all(
            str(row["backend_selection"]["backend"]["module"]).startswith(
                "transformers."
            )
            for row in control_rows
        ),
        "candidate_backend_is_fla_kernel": all(
            str(row["backend_selection"]["backend"]["module"]).startswith(
                "fla."
            )
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

    package_versions = {
        name: importlib.metadata.version(name)
        for name in (
            "flash-linear-attention",
            "fla-core",
            "triton",
        )
    }
    if (
        package_versions["flash-linear-attention"]
        != EXPECTED_FLA_VERSION
        or package_versions["fla-core"] != EXPECTED_FLA_VERSION
    ):
        raise RuntimeError(
            f"expected FLA {EXPECTED_FLA_VERSION}, got {package_versions}"
        )
    gpu_count = torch.cuda.device_count()
    if gpu_count != 2:
        raise RuntimeError(
            f"M5H benchmark requires exactly two GPUs, got {gpu_count}"
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

    modeling, fast_recurrent, original = prepare_fallback_model_construction()
    fallback_recurrent = modeling.torch_recurrent_gated_delta_rule
    try:
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir, local_files_only=True
        )
        runtime = load_runtime(
            args.model_dir,
            gpu_count=2,
            split_layer=args.split_layer,
            cache_experts=args.cache_experts,
            router_backend="torch-device",
        )
        load_seconds = time.perf_counter() - load_started
    finally:
        restore_modeling_globals(modeling, original)

    modules = linear_attention_modules(runtime)
    import pydee.pydee_core as pydee_core

    binary = Path(pydee_core.__file__).resolve()
    runtime_binary = {
        "path": str(binary),
        "bytes": binary.stat().st_size,
        "sha256": sha256_file(binary),
    }
    control_backend = CountedBackend(CONTROL, fallback_recurrent)
    candidate_backend = CountedBackend(CANDIDATE, fast_recurrent)
    backends = {
        CONTROL: control_backend,
        CANDIDATE: candidate_backend,
    }
    source_clean = tracked_source_clean()
    config = runtime["config"]
    layer_count = len(runtime["engines"])
    topk = int(config.num_experts_per_tok)
    hidden = int(runtime["engines"][0].hidden_dim())

    warmups = {}
    for backend in (CONTROL, CANDIDATE):
        row, _ = run_backend_once(
            runtime,
            tokenizer,
            modules,
            backends[backend],
            trace=False,
            layers=layer_count,
            topk=topk,
            hidden=hidden,
        )
        warmups[backend] = row

    order = balanced_order(args.trials)
    sequence = []
    for sequence_index, backend in enumerate(order):
        row, _ = run_backend_once(
            runtime,
            tokenizer,
            modules,
            backends[backend],
            trace=False,
            layers=layer_count,
            topk=topk,
            hidden=hidden,
        )
        row["sequence_index"] = sequence_index
        sequence.append(row)

    control_trace_row, control_trace = run_backend_once(
        runtime,
        tokenizer,
        modules,
        control_backend,
        trace=True,
        layers=layer_count,
        topk=topk,
        hidden=hidden,
    )
    candidate_trace_row, candidate_trace = run_backend_once(
        runtime,
        tokenizer,
        modules,
        candidate_backend,
        trace=True,
        layers=layer_count,
        topk=topk,
        hidden=hidden,
    )
    trace_rows = [control_trace_row, candidate_trace_row]
    tolerance_comparison = compare_trace(
        control_trace["collector"], candidate_trace["collector"]
    )
    exact_comparison = exact_trace_comparison(
        control_trace["collector"], candidate_trace["collector"]
    )

    summaries = {
        backend: summarize_trials([
            row for row in sequence if row["execution_mode"] == backend
        ])
        for backend in (CONTROL, CANDIDATE)
    }
    control_tps = summaries[CONTROL]["median_tokens_per_second"]
    candidate_tps = summaries[CANDIDATE]["median_tokens_per_second"]
    speedup_ratio = candidate_tps / control_tps
    paired = paired_trial_analysis(sequence)
    thermal = thermal_clock_analysis(sequence)
    gates = candidate_gates(
        sequence=sequence,
        trace_rows=trace_rows,
        paired=paired,
        speedup_ratio=speedup_ratio,
        tolerance_comparison=tolerance_comparison,
        exact_comparison=exact_comparison,
        thermal=thermal,
        source_clean=source_clean,
    )
    result = "PASS" if all(gates.values()) else "FAIL"

    report = {
        "schema_version": 1,
        "result": result,
        "candidate_accepted": result == "PASS",
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
            "packages": package_versions,
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
            "linear_attention_layer_count": len(modules),
            "topk": topk,
            "hidden": hidden,
            "router_backend": "torch-device",
            "expert_execution_mode": BASE_EXECUTION_MODE,
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
        "backend_contract": {
            "control": control_backend.identity(),
            "candidate": candidate_backend.identity(),
            "expected_recurrent_calls_per_generation":
                expected_recurrent_calls(len(EXPECTED_TOKENS)),
            "isolated_scope": (
                "Only the 30 cached single-token recurrent gated-delta "
                "functions switch. Model construction is forced to the "
                "Transformers fallback for causal convolution, chunked "
                "prefill, and gated RMSNorm in both modes."
            ),
        },
        "warmups": warmups,
        "balanced_sequence": sequence,
        "summaries": summaries,
        "paired_analysis": paired,
        "trace_rows": trace_rows,
        "trace_comparison": {
            "tolerance": tolerance_comparison,
            "exact": exact_comparison,
        },
        "thermal_clock_analysis": thermal,
        "gates": gates,
        "pareto": {
            "control_median_tps": control_tps,
            "candidate_median_tps": candidate_tps,
            "control_median_tpot_seconds": 1.0 / control_tps,
            "candidate_median_tpot_seconds": 1.0 / candidate_tps,
            "speedup_ratio": speedup_ratio,
            "speedup_percent": (speedup_ratio - 1.0) * 100.0,
        },
        "stage_timing_scope": {
            "attention_time": None,
            "expert_time": None,
            "synchronization_time": None,
            "reason": (
                "M5H is an unprofiled exact fast-path A/B. M5A supplies "
                "the separately instrumented critical-path timing."
            ),
        },
    }
    report_path = args.output_dir / "m5h-fla-recurrent-benchmark.json"
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
        "M5H_FLA_RECURRENT_BENCHMARK="
        + json.dumps(
            {
                "result": result,
                "candidate_accepted": result == "PASS",
                "control_median_tps": control_tps,
                "candidate_median_tps": candidate_tps,
                "speedup_percent": (speedup_ratio - 1.0) * 100.0,
                "output": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
