import unittest
from types import SimpleNamespace

import torch

from scripts.milestone25_timing import ForensicTimingRecorder
from scripts.run_ornith_forensics import (
    build_expert_events,
    build_full_token_breakdown,
    build_layer_timing,
    build_synchronization_analysis,
    build_torch_profiler_control,
)


class ForensicPackagingTests(unittest.TestCase):
    def test_profiler_attaches_to_linear_attention_layers(self):
        class Mlp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Identity()
                self.experts = torch.nn.Identity()
                self.shared_expert = torch.nn.Identity()
                self.shared_expert_gate = torch.nn.Identity()

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                class LinearAttention(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.out_proj = torch.nn.Linear(2, 2, bias=False)

                    def forward(self, value):
                        return self.out_proj(value)

                self.linear_attn = LinearAttention()
                self.input_layernorm = torch.nn.Identity()
                self.post_attention_layernorm = torch.nn.Identity()
                self.mlp = Mlp()

        block = Block()
        text_model = torch.nn.Module()
        text_model.layers = torch.nn.ModuleList([block])
        text_model.embed_tokens = torch.nn.Identity()
        text_model.norm = torch.nn.Identity()
        model = torch.nn.Module()
        model.model = text_model
        model.lm_head = torch.nn.Identity()
        context = SimpleNamespace(current_step=1, current_phase="decode")
        recorder = ForensicTimingRecorder(
            torch, gpu_count=0, sample_nvml=False
        )
        recorder.attach_model_hooks(model, context)
        block.linear_attn(torch.ones(1, 2))
        timing = recorder.stop()
        names = [row["name"] for row in timing["wall_spans"]]
        self.assertIn("attention_or_linear_attention", names)
        self.assertIn("linear_attention_out_proj", names)

    def test_synchronization_uses_representative_cumulative_delta(self):
        def snapshot(step, phase, host_syncs, stream_waits, sync_ms):
            return {
                "step": step,
                "phase": phase,
                "layers": [{
                    "profile": {
                        "operations": {
                            "host_synchronizations": host_syncs,
                            "stream_waits": stream_waits,
                        },
                        "cpu_ms": {"synchronization": sync_ms},
                    },
                }],
            }

        result = build_synchronization_analysis({
            "wall_spans": [],
            "profile_snapshots": [
                snapshot(0, "prefill", 6, 1, 1.5),
                snapshot(1, "decode", 14, 3, 2.5),
                snapshot(2, "decode", 22, 6, 4.0),
            ],
        })
        self.assertEqual(result["representative_step"], 2)
        self.assertEqual(result["host_synchronization_events_total"], 8)
        self.assertEqual(result["stream_wait_events_total"], 3)
        self.assertEqual(
            result["host_synchronization_events_breakdown"][
                "cpu_section_synchronization_ms_total"
            ],
            1.5,
        )

    def test_layer_timing_uses_cumulative_profile_deltas(self):
        timing = {
            "wall_spans": [
                {"step": 0, "layer": 0, "name": "layer_total", "cpu_wall_ms": 12.0},
                {"step": 0, "layer": 0, "name": "normalization_input", "cpu_wall_ms": 1.0},
                {"step": 0, "layer": 0, "name": "attention_or_linear_attention",
                 "cpu_wall_ms": 2.0, "cuda_event_ms": 1.5},
                {"step": 0, "layer": 0, "name": "router_module_total",
                 "cpu_wall_ms": 1.25, "cuda_event_ms": 0.5},
                {"step": 0, "layer": 0, "name": "router_native", "cpu_wall_ms": 1.0},
                {"step": 0, "layer": 0, "name": "expert_native", "cpu_wall_ms": 5.0},
                {"step": 1, "layer": 0, "name": "layer_total", "cpu_wall_ms": 10.0},
                {"step": 1, "layer": 0, "name": "expert_native", "cpu_wall_ms": 4.0},
            ],
            "engine_profiles": [{"layer": 0, "gpu": 0, "profile": {}}],
            "profile_snapshots": [
                {"step": 0, "phase": "prefill", "layers": [{"layer": 0, "profile": {
                    "cpu_ms": {"cache_lookup": 2, "host_tensor_preparation": 3,
                               "pinning": 4, "transfer_submission": 5,
                               "synchronization": 6},
                    "gpu_ms": {"h2d": 7, "gate_projection": 1, "up_projection": 1,
                               "silu_multiply": 1, "down_projection": 1},
                    "requests": {"resident_hits": 1, "inflight_hits": 0, "cold_loads": 7},
                    "transfers": {"h2d_bytes": 70},
                }}]},
                {"step": 1, "phase": "decode", "layers": [{"layer": 0, "profile": {
                    "cpu_ms": {"cache_lookup": 3, "host_tensor_preparation": 5,
                               "pinning": 5, "transfer_submission": 7,
                               "synchronization": 9},
                    "gpu_ms": {"h2d": 10, "gate_projection": 2, "up_projection": 3,
                               "silu_multiply": 2, "down_projection": 2},
                    "requests": {"resident_hits": 3, "inflight_hits": 0, "cold_loads": 13},
                    "transfers": {"h2d_bytes": 130},
                }}]},
            ],
        }
        rows = build_layer_timing(timing)["rows"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["cache_misses"], 7)
        self.assertEqual(rows[1]["cache_misses"], 6)
        self.assertEqual(rows[1]["expert_h2d_bytes"], 60)
        self.assertEqual(rows[1]["pinning_ms"], 1)
        self.assertEqual(rows[0]["router_wall_ms"], 1.25)
        self.assertEqual(rows[0]["router_component_wall_ms"], 1.0)
        self.assertEqual(rows[0]["router_cuda_event_ms"], 0.5)

    def test_full_token_breakdown_keeps_unattributed_time_unclassified(self):
        layer_timing = {
            "rows": [{
                "step": 2,
                "phase": "decode",
                "layer": 0,
                "layer_type": "linear_attention",
                "total_layer_wall_ms": 10.0,
                "attention_wall_ms": 4.0,
                "attention_cuda_event_ms": 3.0,
                "residual_and_unattributed_wall_ms": 2.0,
                "expert_d2d_gather_bytes": 4096,
                "expert_d2d_gather_copies": 1,
                "expert_d2d_scatter_bytes": 8192,
                "expert_d2d_scatter_copies": 1,
            }],
        }
        result = build_full_token_breakdown(
            {
                "torch_profiler": {
                    "captured_step": 2,
                    "operators": [{
                        "key": "cudaLaunchKernel",
                        "count": 7,
                        "self_cpu_time_total_us": 9.0,
                    }],
                    "error": None,
                },
            },
            layer_timing,
            {"pybind_device_calls": 40, "python_combine_calls": 40},
        )
        self.assertEqual(result["representative_step"], 2)
        self.assertEqual(
            result["direct_measurement"]["linear_attention_cuda_event_layers"], 1
        )
        self.assertEqual(
            result["direct_measurement"]["native_d2d_totals"][
                "expert_d2d_scatter_bytes"
            ],
            8192,
        )
        self.assertEqual(
            result["direct_measurement"]["torch_profiler"]["cuda_api_call_count"],
            7,
        )
        self.assertEqual(result["unattributed"]["wall_ms"], 2.0)
        self.assertIn("not direct recurrent", result["unattributed"]["classification"])
        self.assertEqual(result["result"], "FAIL")
        self.assertFalse(result["gates"]["all_40_layers_present"])

    def test_torch_profiler_control_passes_only_without_direct_event_hooks(self):
        timing = {
            "torch_profiler": {
                "captured_step": 2,
                "operators": [{
                    "key": "cudaLaunchKernel",
                    "count": 11,
                    "self_cpu_time_total_us": 15.0,
                }],
                "error": None,
            },
            "steps": [{
                "step": 2,
                "phase": "decode",
                "model_wall_ms": 100.0,
                "full_token_wall_ms": 101.0,
            }],
            "wall_spans": [{
                "step": 2,
                "name": "token_selection_and_item_sync",
                "cpu_wall_ms": 1.0,
            }],
            "engine_profiles": [],
        }
        result = build_torch_profiler_control(timing)
        self.assertEqual(result["result"], "PASS")
        self.assertEqual(result["cuda_api_call_count"], 11)

        timing["wall_spans"][0]["cuda_event_ms"] = 0.5
        result = build_torch_profiler_control(timing)
        self.assertEqual(result["result"], "FAIL")
        self.assertFalse(result["gates"]["per_module_cuda_event_hooks_absent"])

    def test_expert_trace_merges_routes_requests_and_cuda_transfers(self):
        profile = {
            "trace": [{
                "index": 0, "request_time_ms": 2.0, "token": 1,
                "logical_layer": 0, "resolved_shard_layer": 0, "expert": 9,
                "kind": "cold", "cache_bytes_before": 10,
                "cache_entries_before": 2, "cache_bytes_used": 20,
                "cache_bytes_after": 20, "cache_entries_after": 2,
                "evicted_layer": 0, "evicted_expert": 3,
                "evicted_generation": 4, "generation": 5, "pin_count": 1,
                "reuse_distance": 5, "distinct_reuse_distance": 4,
                "theoretical_min_cache_bytes": 30, "priority": 1,
                "source_bytes": 16, "destination_bytes": 16,
                "transfer_id": 0, "source_pinned": True,
                "transfer_launched": True, "consumed": True,
                "evicted_before_use": False,
                "eviction_reason": "capacity_lru",
            }]
        }
        timeline = {"traceEvents": [
            {"name": "h2d", "ts": 1000, "dur": 500,
             "args": {"token": 1, "logical_layer": 0, "expert": 9,
                      "bytes": 16, "transfer_id": 0}},
            {"name": "gate_projection", "ts": 1600, "dur": 100,
             "args": {"token": 1, "logical_layer": 0, "expert": 9,
                      "bytes": 0, "transfer_id": 0}},
        ]}
        timing = {
            "route_selections": [{
                "event": "route_selection", "step": 1, "phase": "decode",
                "sequence_token": 0, "layer": 0, "gpu": 0, "expert": 9,
                "routing_rank": 2, "routing_weight": 0.25,
            }],
            "engine_profiles": [{"layer": 0, "gpu": 0, "profile": profile}],
            "engine_timelines": [{"layer": 0, "gpu": 0, "timeline": timeline}],
        }
        prefix = "model.language_model.layers.0.mlp.experts.9"
        index = {"weight_map": {
            f"{prefix}.gate_proj.weight": "a.safetensors",
            f"{prefix}.up_proj.weight": "a.safetensors",
            f"{prefix}.down_proj.weight": "b.safetensors",
        }}
        events = build_expert_events("run", timing, index)
        request = next(event for event in events if event["event_type"] == "expert_request")
        transfer = next(event for event in events if event["event_type"] == "expert_transfer")
        eviction = next(event for event in events if event["event_type"] == "expert_eviction")
        self.assertEqual(request["routing_rank"], 2)
        self.assertEqual(request["source_checkpoint_shards"], ["a.safetensors", "b.safetensors"])
        self.assertEqual(request["cache_state_before"]["resident_entries"], 2)
        self.assertEqual(request["cache_state_after"]["resident_entries"], 2)
        self.assertEqual(request["residency_generation"], 5)
        self.assertEqual(request["pin_count_after"], 1)
        self.assertTrue(request["transfer_launched"])
        self.assertTrue(request["transfer_consumed"])
        self.assertFalse(request["evicted_before_use"])
        self.assertEqual(transfer["transfer_id"], 0)
        self.assertEqual(transfer["bytes"], 16)
        self.assertEqual(transfer["transfer_completion_ms"], 1.5)
        self.assertEqual(eviction["expert_id"], 3)


if __name__ == "__main__":
    unittest.main()
