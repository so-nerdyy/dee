import unittest

from scripts.run_ornith_forensics import (
    build_expert_events,
    build_layer_timing,
    build_synchronization_analysis,
)


class ForensicPackagingTests(unittest.TestCase):
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

    def test_expert_trace_merges_routes_requests_and_cuda_transfers(self):
        profile = {
            "trace": [{
                "index": 0, "request_time_ms": 2.0, "token": 1,
                "logical_layer": 0, "resolved_shard_layer": 0, "expert": 9,
                "kind": "cold", "cache_bytes_before": 10,
                "cache_entries_before": 2, "cache_bytes_used": 20,
                "evicted_layer": 0, "evicted_expert": 3,
                "reuse_distance": 5, "distinct_reuse_distance": 4,
                "theoretical_min_cache_bytes": 30, "priority": 1,
                "source_bytes": 16, "destination_bytes": 16,
                "transfer_id": 7, "source_pinned": True,
                "eviction_reason": "capacity_lru",
            }]
        }
        timeline = {"traceEvents": [
            {"name": "h2d", "ts": 1000, "dur": 500,
             "args": {"token": 1, "logical_layer": 0, "expert": 9,
                      "bytes": 16, "transfer_id": 7}},
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
        self.assertEqual(transfer["bytes"], 16)
        self.assertEqual(transfer["transfer_completion_ms"], 1.5)
        self.assertEqual(eviction["expert_id"], 3)


if __name__ == "__main__":
    unittest.main()
