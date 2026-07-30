#!/usr/bin/env python3
"""Paired exactness/Pareto benchmark for per-layer CUDA Graph replay."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ornith_generation import (  # noqa: E402
    compare_trace,
    load_runtime,
)
from scripts.run_ornith_m5b_execution_benchmark import (  # noqa: E402
    exact_trace_comparison,
    thermal_clock_analysis,
    tracked_source_clean,
)
from scripts.run_ornith_m5f_pointer_batched_benchmark import (  # noqa: E402
    run_m5f_once,
)
from scripts.run_ornith_m5h_fla_recurrent_benchmark import (  # noqa: E402
    balanced_order,
    paired_trial_analysis,
)
from scripts.run_ornith_router_backend_benchmark import (  # noqa: E402
    EXPECTED_TOKENS,
    MAX_PROCESS_VRAM_BYTES,
    git_revision,
    sha256_file,
    summarize_trials,
)


CONTROL = "pointer-eager-linear-attention"
CANDIDATE = "pointer-graphed-linear-attention"
BASE_EXECUTION_MODE = "native-combined-pointer"
LINEAR_LAYER_COUNT = 30


def _cache_states(cache_params: Any, layer_idx: int) -> tuple[Any, Any]:
    layer = cache_params.layers[layer_idx]
    return layer.conv_states[0], layer.recurrent_states[0]


class LinearAttentionGraph:
    """Capture one fixed-shape cached-decode linear-attention module."""

    def __init__(self, module: Any, pool: Any):
        self.module = module
        self.layer_idx = int(module.layer_idx)
        self.original_forward = module.forward
        self.pool = pool
        self.enabled = False
        self.graph = None
        self.graph_cache = None
        self.static_hidden = None
        self.static_mask = None
        self.static_output = None
        self.capture_stream = None
        self.capture_failure: str | None = None
        self.total_captures = 0
        self.reset_counts()

    def reset_counts(self) -> None:
        self.decode_calls = 0
        self.eager_decode_calls = 0
        self.graph_decode_calls = 0
        self.replays = 0
        self.hidden_copies = 0
        self.mask_copies = 0
        self.live_cache_input_copies = 0
        self.live_cache_output_copies = 0
        self.prefill_bypasses = 0

    def _is_cached_decode(self, hidden_states: Any, cache_params: Any) -> bool:
        return (
            cache_params is not None
            and int(hidden_states.shape[1]) == 1
            and cache_params.has_previous_state(self.layer_idx)
        )

    def __call__(
        self,
        hidden_states: Any,
        cache_params: Any = None,
        attention_mask: Any = None,
        **kwargs: Any,
    ) -> Any:
        if not self._is_cached_decode(hidden_states, cache_params):
            self.prefill_bypasses += 1
            return self.original_forward(
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )

        self.decode_calls += 1
        if not self.enabled:
            self.eager_decode_calls += 1
            return self.original_forward(
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )

        self.graph_decode_calls += 1
        if self.capture_failure is not None:
            raise RuntimeError(
                f"layer {self.layer_idx} CUDA Graph capture previously failed: "
                f"{self.capture_failure}"
            )
        if self.graph is None:
            return self._capture(hidden_states, cache_params, attention_mask)
        return self._replay(hidden_states, cache_params, attention_mask)

    def _copy_dynamic_inputs(self, hidden_states: Any, attention_mask: Any) -> None:
        self.static_hidden.copy_(hidden_states)
        self.hidden_copies += 1

    def _copy_live_cache_in(self, live_cache: Any) -> None:
        if live_cache is self.graph_cache:
            return
        graph_conv, graph_recurrent = _cache_states(self.graph_cache, self.layer_idx)
        live_conv, live_recurrent = _cache_states(live_cache, self.layer_idx)
        graph_conv.copy_(live_conv)
        graph_recurrent.copy_(live_recurrent)
        self.live_cache_input_copies += 2

    def _copy_live_cache_out(self, live_cache: Any) -> None:
        if live_cache is self.graph_cache:
            return
        graph_conv, graph_recurrent = _cache_states(self.graph_cache, self.layer_idx)
        live_conv, live_recurrent = _cache_states(live_cache, self.layer_idx)
        live_conv.copy_(graph_conv)
        live_recurrent.copy_(graph_recurrent)
        self.live_cache_output_copies += 2

    def _capture(
        self,
        hidden_states: Any,
        cache_params: Any,
        attention_mask: Any,
    ) -> Any:
        import torch

        device = hidden_states.device
        with torch.cuda.device(device):
            return self._capture_on_current_device(
                hidden_states, cache_params, attention_mask
            )

    def _capture_on_current_device(
        self,
        hidden_states: Any,
        cache_params: Any,
        attention_mask: Any,
    ) -> Any:
        import torch

        device = hidden_states.device
        if attention_mask is not None and int(attention_mask.shape[0]) != 1:
            raise RuntimeError("the M5I graph specialization requires batch size one")
        self.graph_cache = cache_params
        self.static_hidden = torch.empty_like(hidden_states)
        # Qwen3.5's apply_mask_to_padding_states is deliberately a no-op for
        # batch size one. Omitting that unused argument avoids one staging
        # copy per layer without changing the executed tensor operations.
        self.static_mask = None
        self._copy_dynamic_inputs(hidden_states, attention_mask)
        conv_state, recurrent_state = _cache_states(cache_params, self.layer_idx)
        saved_conv = conv_state.clone()
        saved_recurrent = recurrent_state.clone()
        self.capture_stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        self.capture_stream.wait_stream(current)
        try:
            with torch.cuda.stream(self.capture_stream):
                for _ in range(3):
                    conv_state.copy_(saved_conv)
                    recurrent_state.copy_(saved_recurrent)
                    self.original_forward(
                        self.static_hidden,
                        cache_params=self.graph_cache,
                        attention_mask=self.static_mask,
                    )
                conv_state.copy_(saved_conv)
                recurrent_state.copy_(saved_recurrent)
            current.wait_stream(self.capture_stream)
            torch.cuda.synchronize(device)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self.graph,
                pool=self.pool,
                stream=self.capture_stream,
            ):
                self.static_output = self.original_forward(
                    self.static_hidden,
                    cache_params=self.graph_cache,
                    attention_mask=self.static_mask,
                )
            current.wait_stream(self.capture_stream)
            self.total_captures += 1
            return self.static_output
        except Exception as exc:
            self.capture_failure = f"{type(exc).__name__}: {exc}"
            self.graph = None
            raise

    def _replay(
        self,
        hidden_states: Any,
        cache_params: Any,
        attention_mask: Any,
    ) -> Any:
        import torch

        self._copy_dynamic_inputs(hidden_states, attention_mask)
        self._copy_live_cache_in(cache_params)
        with torch.cuda.device(self.static_hidden.device):
            self.graph.replay()
        self.replays += 1
        self._copy_live_cache_out(cache_params)
        return self.static_output

    def snapshot(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "enabled": self.enabled,
            "graph_ready": self.graph is not None,
            "capture_failure": self.capture_failure,
            "total_captures": self.total_captures,
            "decode_calls": self.decode_calls,
            "eager_decode_calls": self.eager_decode_calls,
            "graph_decode_calls": self.graph_decode_calls,
            "replays": self.replays,
            "hidden_copies": self.hidden_copies,
            "mask_copies": self.mask_copies,
            "live_cache_input_copies": self.live_cache_input_copies,
            "live_cache_output_copies": self.live_cache_output_copies,
            "prefill_bypasses": self.prefill_bypasses,
        }


def install_graph_controllers(runtime: dict[str, Any]) -> list[LinearAttentionGraph]:
    import torch

    modules = [
        block.linear_attn
        for block in runtime["model"].model.layers
        if hasattr(block, "linear_attn")
    ]
    if len(modules) != LINEAR_LAYER_COUNT:
        raise RuntimeError(
            f"expected {LINEAR_LAYER_COUNT} linear-attention layers, got {len(modules)}"
        )
    pools: dict[int, Any] = {}
    controllers = []
    for module in modules:
        device = int(next(module.parameters()).device.index)
        if device not in pools:
            with torch.cuda.device(device):
                pools[device] = torch.cuda.graph_pool_handle()
        controller = LinearAttentionGraph(module, pools[device])
        module.forward = controller
        controllers.append(controller)
    return controllers


def select_graph_mode(controllers: list[LinearAttentionGraph], mode: str) -> None:
    if mode not in {CONTROL, CANDIDATE}:
        raise ValueError(f"unknown graph mode {mode}")
    enabled = mode == CANDIDATE
    for controller in controllers:
        controller.enabled = enabled
        controller.reset_counts()


def graph_stats(controllers: list[LinearAttentionGraph]) -> dict[str, Any]:
    rows = [controller.snapshot() for controller in controllers]
    count_keys = (
        "total_captures",
        "decode_calls",
        "eager_decode_calls",
        "graph_decode_calls",
        "replays",
        "hidden_copies",
        "mask_copies",
        "live_cache_input_copies",
        "live_cache_output_copies",
        "prefill_bypasses",
    )
    return {
        "layer_count": len(rows),
        "graph_ready_count": sum(row["graph_ready"] for row in rows),
        "capture_failure_count": sum(
            row["capture_failure"] is not None for row in rows
        ),
        "aggregate": {key: sum(int(row[key]) for row in rows) for key in count_keys},
        "layers": rows,
    }


def run_graph_once(
    runtime: dict[str, Any],
    tokenizer: Any,
    controllers: list[LinearAttentionGraph],
    mode: str,
    *,
    trace: bool,
    layers: int,
    topk: int,
    hidden: int,
    inter: int,
    pointer_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    select_graph_mode(controllers, mode)
    row, raw = run_m5f_once(
        runtime,
        tokenizer,
        BASE_EXECUTION_MODE,
        trace=trace,
        layers=layers,
        topk=topk,
        hidden=hidden,
        inter=inter,
        pointer_bytes=pointer_bytes,
    )
    row["expert_execution_mode"] = row["execution_mode"]
    row["execution_mode"] = mode
    row["linear_attention_graph"] = graph_stats(controllers)
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
    control_rows = [row for row in rows if row["execution_mode"] == CONTROL]
    candidate_rows = [row for row in rows if row["execution_mode"] == CANDIDATE]
    expected_layer_calls = 40 * len(EXPECTED_TOKENS)
    expected_experts = expected_layer_calls * 8
    expected_decode_calls = LINEAR_LAYER_COUNT * (len(EXPECTED_TOKENS) - 1)
    return {
        "candidate_independent_median_gain_at_least_1_percent": speedup_ratio >= 1.01,
        "candidate_paired_median_gain_at_least_1_percent": paired[
            "median_speedup_ratio"
        ]
        >= 1.01,
        "candidate_wins_majority_of_pairs": paired["candidate_wins"]
        >= paired["pair_count"] // 2 + 1,
        "no_pair_regresses_more_than_2_percent": paired["minimum_speedup_ratio"]
        >= 0.98,
        "tokens_and_text_exact": all(
            row["generated_token_ids"] == EXPECTED_TOKENS for row in trace_rows
        )
        and trace_rows[0]["generated_text"] == trace_rows[1]["generated_text"],
        "all_trace_categories_passed": tolerance_comparison["all_categories_passed"],
        "all_trace_categories_bitwise_exact": exact_comparison[
            "all_categories_bitwise_exact"
        ],
        "all_40_layers_executed": all(row["all_40_layers_executed"] for row in rows),
        "all_expert_calls_device_resident": all(
            row["path_proof"]["device_path_share"] == 1.0 for row in rows
        ),
        "zero_host_fallback_calls": all(
            row["path_proof"]["host_path_fallback_calls"] == 0 for row in rows
        ),
        "same_pointer_expert_path_both_modes": all(
            row["path_proof"]["native_combined_calls"] == expected_layer_calls
            and row["path_proof"]["native_direct_calls"] == expected_layer_calls
            and row["path_proof"]["native_pointer_batched_calls"]
            == expected_layer_calls
            and row["path_proof"]["native_pointer_batched_fallback_calls"] == 0
            and row["engine_projection_delta"]["pointer_batched_expert_calls"]
            == expected_layer_calls
            and row["engine_projection_delta"]["pointer_batched_experts"]
            == expected_experts
            for row in rows
        ),
        "pointer_path_has_zero_row_copies": all(
            row["engine_projection_delta"]["d2d_gather_copies"] == 0
            and row["engine_projection_delta"]["d2d_scatter_copies"] == 0
            for row in rows
        ),
        "control_uses_eager_linear_attention": all(
            row["linear_attention_graph"]["aggregate"]["eager_decode_calls"]
            == expected_decode_calls
            and row["linear_attention_graph"]["aggregate"]["graph_decode_calls"] == 0
            for row in control_rows
        ),
        "candidate_replays_all_cached_linear_attention_calls": all(
            row["linear_attention_graph"]["aggregate"]["decode_calls"]
            == expected_decode_calls
            and row["linear_attention_graph"]["aggregate"]["graph_decode_calls"]
            == expected_decode_calls
            and row["linear_attention_graph"]["aggregate"]["replays"]
            == expected_decode_calls
            for row in candidate_rows
        ),
        "all_30_linear_attention_graphs_ready": all(
            row["linear_attention_graph"]["graph_ready_count"] == LINEAR_LAYER_COUNT
            for row in candidate_rows
        ),
        "zero_graph_capture_failures": all(
            row["linear_attention_graph"]["capture_failure_count"] == 0
            for row in candidate_rows
        ),
        "pointer_workspace_exact_and_bounded": all(
            row["pointer_workspace"]["passed"] for row in rows
        ),
        "combined_workspace_exact_and_bounded": all(
            row["engine_workspace"]["passed"] for row in rows
        ),
        "process_vram_at_most_8_gib_per_gpu": all(
            max(row["peak_vram_bytes"].values()) <= MAX_PROCESS_VRAM_BYTES
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

    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer

    gpu_count = torch.cuda.device_count()
    if gpu_count != 2:
        raise RuntimeError(f"M5I benchmark requires exactly two GPUs, got {gpu_count}")
    if args.require_dual_gpu and any(
        "T4" not in torch.cuda.get_device_name(device) for device in range(2)
    ):
        raise RuntimeError(
            "dual-T4 benchmark found "
            + ", ".join(torch.cuda.get_device_name(device) for device in range(2))
        )

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    runtime = load_runtime(
        args.model_dir,
        gpu_count=2,
        split_layer=args.split_layer,
        cache_experts=args.cache_experts,
        router_backend="torch-device",
    )
    load_seconds = time.perf_counter() - load_started
    controllers = install_graph_controllers(runtime)

    import pydee.pydee_core as pydee_core

    binary = Path(pydee_core.__file__).resolve()
    runtime_binary = {
        "path": str(binary),
        "bytes": binary.stat().st_size,
        "sha256": sha256_file(binary),
    }
    source_clean = tracked_source_clean()
    config = runtime["config"]
    layer_count = len(runtime["engines"])
    topk = int(config.num_experts_per_tok)
    hidden = int(runtime["engines"][0].hidden_dim())
    inter = int(runtime["engines"][0].inter_dim())
    pointer_bytes = np.dtype(np.uintp).itemsize

    warmups = {}
    for mode in (CONTROL, CANDIDATE):
        row, _ = run_graph_once(
            runtime,
            tokenizer,
            controllers,
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
    sequence = []
    for sequence_index, mode in enumerate(order):
        row, _ = run_graph_once(
            runtime,
            tokenizer,
            controllers,
            mode,
            trace=False,
            layers=layer_count,
            topk=topk,
            hidden=hidden,
            inter=inter,
            pointer_bytes=pointer_bytes,
        )
        row["sequence_index"] = sequence_index
        sequence.append(row)

    control_trace_row, control_trace = run_graph_once(
        runtime,
        tokenizer,
        controllers,
        CONTROL,
        trace=True,
        layers=layer_count,
        topk=topk,
        hidden=hidden,
        inter=inter,
        pointer_bytes=pointer_bytes,
    )
    candidate_trace_row, candidate_trace = run_graph_once(
        runtime,
        tokenizer,
        controllers,
        CANDIDATE,
        trace=True,
        layers=layer_count,
        topk=topk,
        hidden=hidden,
        inter=inter,
        pointer_bytes=pointer_bytes,
    )
    trace_rows = [control_trace_row, candidate_trace_row]
    tolerance_comparison = compare_trace(
        control_trace["collector"], candidate_trace["collector"]
    )
    exact_comparison = exact_trace_comparison(
        control_trace["collector"], candidate_trace["collector"]
    )
    summaries = {
        mode: summarize_trials(
            [row for row in sequence if row["execution_mode"] == mode]
        )
        for mode in (CONTROL, CANDIDATE)
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
            "linear_attention_layer_count": len(controllers),
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
        "graph_contract": {
            "scope": (
                "Each of the 30 fixed-shape cached-decode gated-delta "
                "linear-attention modules is captured independently. Dynamic "
                "hidden/mask tensors and cache state are copied into static "
                "addresses; the accepted top-8 pointer-batched expert path "
                "is identical in control and candidate."
            ),
            "prefill_is_eager": True,
            "captured_decode_sequence_length": 1,
            "shared_graph_pool_per_device": True,
            "controllers": graph_stats(controllers),
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
                "M5I is an unprofiled exact fast-path A/B. M5A supplies "
                "the separately instrumented pre-optimization timing."
            ),
        },
    }
    report_path = args.output_dir / "m5i-linear-graph-benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
    print(json.dumps(report, indent=2))
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
