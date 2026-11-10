import copy
import unittest

from scripts.analyze_milestone25_matrix import (
    CONTROL_RUN,
    EXPECTED_RUNS,
    PRIMARY_RUN,
    WARM_RUN,
    build_experiments,
    build_gpu_breakdown,
    build_host_breakdown,
    build_timing_analysis,
)


def report(run_id="run", gpu_count=2):
    per_device = {
        f"cuda:{index}": {
            "host_pinned_expert_staging_bytes": 100,
            "host_pageable_expert_staging_bytes": 200,
            "host_router_weight_bytes": 10,
            "host_hidden_buffer_bytes": 2,
            "host_prefetch_ring_bytes": 20,
            "host_prefetch_ring_slots": 1,
            "peak_transient_host_bytes": 4,
            "device_expert_cache_reserved_bytes": 300,
            "device_prefetch_staging_bytes": 30,
            "device_fixed_work_buffer_bytes": 10,
            "device_router_weight_bytes": 10,
            "device_router_dynamic_bytes": 5,
            "device_moe_batch_buffer_bytes": 5,
            "device_oracle_scratch_bytes": 1,
        }
        for index in range(gpu_count)
    }
    aggregate = {}
    for values in per_device.values():
        for key, value in values.items():
            aggregate[key] = aggregate.get(key, 0) + value
    return {
        "result": "PASS",
        "run_id": run_id,
        "machine": {"gpu_count": gpu_count},
        "configuration": {"classification": "warm"},
        "correctness": {
            "all_40_layers_executed": True,
            "baseline_tokens_exact": True,
            "warmup_tokens_exact": True,
        },
        "layout": {
            "dense_loaded_bytes": {f"cuda:{index}": 500 for index in range(gpu_count)},
            "native_engine_memory": {"aggregate": aggregate, "by_device": per_device},
            "parameter_inventory": {"groups": []},
        },
        "generation": {
            "prefill_seconds": 2.0,
            "per_token_decode_seconds": [2.0, 2.1, 2.2],
            "single_stream_decode_tokens_per_second": 0.476,
            "total_generation_seconds": 8.3,
            "generated_token_ids": [1, 2, 3, 4],
            "recurrent_or_kv_state": {
                "groups": [{"device": "cuda:0", "bytes": 20}]
            },
            "live_generation_inputs": {
                "groups": [{"device": "cuda:0", "bytes": 4}]
            },
            "resources": {
                "peak_host_rss_bytes": 1000,
                "peak_vram_bytes": {f"cuda:{index}": 1200 for index in range(gpu_count)},
            },
        },
    }


def checkpoint(gpu_count=2):
    categories = {
        "anonymous": {"rss_bytes": 600, "virtual_bytes": 700},
        "checkpoint": {"rss_bytes": 300, "virtual_bytes": 900},
        "python": {"rss_bytes": 100, "virtual_bytes": 100},
    }
    return {
        "label": "after_primary_generation",
        "psutil": {
            "memory_info": {"rss": 1000},
            "memory_full_info": {"pss": 900, "uss": 800, "swap": 0},
        },
        "proc": {
            "smaps_rollup": {
                "Rss_bytes": 1000, "Anonymous_bytes": 600,
                "Shared_Clean_bytes": 20, "Shared_Dirty_bytes": 0,
                "Private_Clean_bytes": 300, "Private_Dirty_bytes": 680,
                "Locked_bytes": 100, "Swap_bytes": 0,
            },
            "smaps_attribution": {"categories": categories},
            "maps": {"region_count": 3},
            "io": {"read_bytes": 1},
        },
        "page_faults": {"minor_faults": 2, "major_faults": 1},
        "cpu_tensors": {"unique_storage_bytes": 10},
        "cuda": {"devices": [
            {
                "index": index, "peak_allocated_bytes": 700,
                "peak_reserved_bytes": 800,
                "memory_stats": {
                    "active_bytes.all.peak": 690,
                    "inactive_split_bytes.all.peak": 20,
                },
            }
            for index in range(gpu_count)
        ]},
        "nvml": {"devices": [
            {"index": index, "process_used_bytes": 1200, "memory_used_bytes": 1200}
            for index in range(gpu_count)
        ]},
    }


def timing_bundle(run_id, gpu_count=2, profiled=True):
    rows = []
    spans = []
    cursor = 1_000_000_000
    for step in (1, 2):
        for layer in range(40):
            gpu = 0 if gpu_count == 1 or layer < 20 else 1
            rows.append({
                "step": step, "phase": "decode", "layer": layer, "gpu": gpu,
                "normalization_wall_ms": 1.0, "attention_wall_ms": 5.0,
                "router_wall_ms": 2.0, "expert_input_d2h_wall_ms": 1.0,
                "expert_native_wall_ms": 20.0, "expert_output_h2d_wall_ms": 1.0,
                "expert_output_combination_wall_ms": 1.0,
                "shared_expert_wall_ms": 2.0, "shared_expert_gate_wall_ms": 1.0,
                "inter_device_transfer_wall_ms": 0.1,
                "residual_and_unattributed_wall_ms": 1.9,
                "expert_lookup_ms": 0.1, "host_tensor_preparation_ms": 0.2,
                "pinning_ms": 0.1, "pageable_to_pinned_copy_ms": 0.2,
                "h2d_submission_ms": 0.1, "h2d_completion_cuda_ms": 1.0,
                "activation_h2d_cuda_ms": 0.1, "activation_conversion_cuda_ms": 0.1,
                "expert_compute_cuda_ms": 2.0, "expert_output_d2h_cuda_ms": 0.1,
                "synchronization_ms": 10.0, "total_layer_wall_ms": 35.0,
            })
            start = cursor
            end = start + 35_000_000
            spans.append({
                "name": "layer_total", "step": step, "gpu": gpu,
                "start_monotonic_ns": start, "end_monotonic_ns": end,
            })
            cursor = end
    steps = [
        {
            "step": step, "begin_monotonic_ns": 1_000_000_000,
            "end_monotonic_ns": cursor + 1_000_000,
            "model_wall_ms": 1450.0,
        }
        for step in (1, 2)
    ]
    sample = {
        "monotonic_ns": 1_100_000_000,
        "gpus": [{"gpu_utilization_percent": 50} for _ in range(gpu_count)],
    }
    profiles = [{
        "profile": {
            "operations": {"host_synchronizations": 8},
            "derived": {"copy_compute_overlap_ms": 0.0},
            "host_waits": {"layer_output": {"milliseconds": 10, "count": 8}},
        }
    }]
    return {
        "report": report(run_id, gpu_count),
        "memory": {"checkpoints": [checkpoint(gpu_count)]},
        "layer": {"rows": rows if profiled else []},
        "timing": {
            "steps": steps, "wall_spans": spans,
            "nvml": {"samples": [sample]}, "engine_profiles": profiles,
        },
        "utilization": {},
    }


class MatrixAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.runs = {
            run_id: timing_bundle(
                run_id,
                1 if run_id == "single-t4-warm" else 2,
                run_id != CONTROL_RUN,
            )
            for run_id in EXPECTED_RUNS
        }

    def test_host_breakdown_reconciles_smaps_and_native_allocations(self):
        result = build_host_breakdown(self.runs)
        self.assertEqual(result["primary"]["peak_rss_bytes"], 1000)
        self.assertEqual(result["primary"]["smaps_category_rss_sum_bytes"], 1000)
        self.assertGreater(result["primary"]["native_persistent_host_bytes"], 0)

    def test_gpu_breakdown_separates_torch_native_and_residual(self):
        result = build_gpu_breakdown(self.runs)
        row = result["primary"]["per_gpu"][0]
        self.assertEqual(row["peak_pytorch_allocated_bytes"], 700)
        self.assertEqual(row["peak_pytorch_reserved_bytes"], 800)
        self.assertGreater(row["native_total_device_bytes"], 0)
        self.assertEqual(
            row["peak_nvml_used_bytes"],
            row["peak_pytorch_reserved_bytes"] + row["native_total_device_bytes"] +
            row["inferred_cuda_context_libraries_and_other_bytes"],
        )

    def test_timing_analysis_uses_40_layers_and_reports_zero_pipeline_overlap(self):
        layer, concurrency = build_timing_analysis(self.runs)
        representative = layer["representative_warm_decode"]
        self.assertEqual(representative["layer_count"], 40)
        self.assertEqual(representative["cross_gpu_layer_interval_overlap_ms"], 0.0)
        self.assertEqual(concurrency["nvml_both_active_samples"], 1)

    def test_experiment_matrix_quantifies_profile_overhead(self):
        result = build_experiments(self.runs)
        overhead = result["comparisons"]["profiling_disabled_vs_enabled"]
        self.assertIsNotNone(overhead["custom_trace_overhead_fraction"])
        self.assertIn("single_vs_dual_t4", result["comparisons"])


if __name__ == "__main__":
    unittest.main()
