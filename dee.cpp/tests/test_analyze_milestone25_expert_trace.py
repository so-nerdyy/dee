import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_milestone25_expert_trace import analyze, main, read_jsonl


def records(*events):
    return list(enumerate(events, 1))


def row(rows, **wanted):
    for candidate in rows:
        if all(candidate.get(name) == value for name, value in wanted.items()):
            return candidate
    raise AssertionError(f"row not found: {wanted!r}")


class Milestone25ExpertTraceAnalysisTests(unittest.TestCase):
    def test_cache_reuse_thrashing_and_transfer_accounting(self):
        trace = records(
            {"event_type": "run_metadata", "run_id": "warm-1"},
            {
                "event_type": "expert_request", "run_id": "warm-1",
                "event_index": 0, "token_step": 0, "token_phase": "prefill",
                "logical_layer": 0, "resolved_shard_layer": 0, "expert_id": 1,
                "routing_rank": 0, "routing_weight": 0.7,
                "source_checkpoint_shard": "model-00001.safetensors",
                "expert_bytes": 100, "cache_state_before": "absent",
                "cache_result": "miss", "gpu_destination": "cuda:0",
                "h2d_bytes": 100, "direction": "h2d",
                "transfer_duration_ms": 2, "transfer_id": "t1",
            },
            {
                "event_type": "expert_request", "run_id": "warm-1",
                "event_index": 1, "token_step": 0, "token_phase": "prefill",
                "logical_layer": 0, "resolved_shard_layer": 0, "expert_id": 1,
                "expert_bytes": 100, "cache_result": "resident_hit",
                "gpu_destination": 0,
            },
            {
                "event_type": "expert_request", "run_id": "warm-1",
                "event_index": 2, "token_step": 0, "token_phase": "prefill",
                "logical_layer": 0, "resolved_shard_layer": 0, "expert_id": 2,
                "expert_bytes": 100, "cache_result": "cold_load",
                "gpu_destination": "gpu0", "h2d_bytes": 100,
                "direction": "host_to_device", "transfer_duration_ms": 2,
                "transfer_id": "t2", "evicted_layer": 0,
                "evicted_expert": 1, "eviction_reason": "capacity",
            },
            {
                "event_type": "expert_request", "run_id": "warm-1",
                "event_index": 3, "token_step": 1, "token_phase": "decode",
                "logical_layer": 0, "resolved_shard_layer": 0, "expert_id": 1,
                "expert_bytes": 100, "cache_result": "miss",
                "gpu_destination": "cuda0", "h2d_bytes": 100,
                "direction": "h2d", "transfer_id": "t3",
            },
            {
                "event_type": "expert_transfer", "run_id": "warm-1",
                "event_index": 4, "token_step": 1, "token_phase": "decode",
                "logical_layer": 0, "resolved_shard_layer": 0, "expert_id": 1,
                "gpu_destination": "cuda:0", "direction": "h2d",
                "bytes": 100, "duration_us": 1000, "transfer_id": "t3",
            },
            {
                "event_type": "expert_request", "run_id": "warm-1",
                "event_index": 5, "token_step": 1, "token_phase": "decode",
                "logical_layer": 0, "resolved_shard_layer": 0, "expert_id": 1,
                "expert_bytes": 100, "cache_result": "hit",
                "gpu_destination": "cuda:0",
            },
            {
                "event_type": "expert_transfer", "run_id": "warm-1",
                "event_index": 6, "token_step": 1, "token_phase": "decode",
                "logical_layer": 0, "resolved_shard_layer": 0, "expert_id": 1,
                "gpu_destination": "cuda:1", "direction": "h2d",
                "bytes": 100, "duration_ms": 1, "transfer_id": "t4",
            },
        )

        cache, transfer = analyze(trace, "fixture.jsonl", short_reuse_distance=1, top_n=10)

        self.assertEqual(cache["input_summary"]["request_events"], 5)
        self.assertEqual(cache["input_summary"]["deduplicated_transfer_records"], 1)
        self.assertEqual(cache["overall"]["hits"], 2)
        self.assertEqual(cache["overall"]["misses"], 3)
        self.assertEqual(cache["request_byte_accounting"]["measured_requested_bytes"], 500)
        self.assertEqual(cache["request_byte_accounting"]["first_request_bytes_by_physical_expert"], 200)
        self.assertEqual(cache["request_byte_accounting"]["repeat_request_bytes_by_physical_expert"], 300)
        self.assertEqual(cache["same_token_repeats"]["groups"], 2)
        self.assertEqual(cache["same_token_repeats"]["repeated_events"], 2)
        self.assertEqual(cache["reuse_distance"]["previous_request_distance"]["count"], 3)
        reuse = cache["request_reuse_records"]
        self.assertEqual(reuse[3]["previous_request_distance"], 1)
        self.assertEqual(reuse[3]["previous_distinct_expert_distance"], 1)
        self.assertEqual(reuse[1]["next_request_distance"], 1)
        self.assertEqual(cache["evictions_shortly_before_reuse"]["shortly_before_reuse_count"], 1)
        self.assertEqual(cache["evictions_shortly_before_reuse"]["shortly_before_reuse_then_miss_count"], 1)
        layer = row(cache["cache_thrashing_by_layer"], run_id="warm-1", logical_layer=0)
        self.assertEqual(layer["repeated_misses"], 1)
        self.assertEqual(layer["short_post_eviction_misses"], 1)
        self.assertEqual([(item["physical_layer"], item["expert_id"])
                          for item in cache["never_reused_experts"]], [(0, 2)])

        self.assertEqual(transfer["overall"]["transfers"], 4)
        self.assertEqual(transfer["h2d"]["measured_bytes"], 400)
        self.assertAlmostEqual(transfer["h2d"]["measured_h2d_bytes_per_known_cache_miss"], 400 / 3)
        accounting = transfer["byte_accounting"]
        self.assertEqual(accounting["first_transfer_bytes_by_physical_expert"], 200)
        self.assertEqual(accounting["repeat_transfer_bytes_by_physical_expert"], 200)
        self.assertEqual(accounting["first_transfer_bytes_by_expert_and_gpu"], 300)
        self.assertEqual(accounting["repeat_transfer_bytes_to_same_expert_and_gpu"], 100)
        self.assertEqual(transfer["transfer_size_bytes"]["median"], 100)
        self.assertEqual(transfer["per_transfer_bandwidth_bytes_per_second"]["count"], 4)
        self.assertEqual(len(transfer["experts_observed_transferred_to_multiple_gpus"]), 1)
        gpu1 = row(transfer["by_gpu"], run_id="warm-1", gpu_destination="cuda:1", direction="h2d")
        self.assertEqual(gpu1["measured_bytes"], 100)
        per_miss_gpu1 = row(
            transfer["h2d_bytes_per_cache_miss"]["by_gpu"],
            run_id="warm-1", gpu_destination="cuda:1",
        )
        self.assertEqual(per_miss_gpu1["cache_misses"], 0)
        self.assertIsNone(per_miss_gpu1["measured_h2d_bytes_per_known_cache_miss"])

    def test_current_cpp_request_aliases_do_not_fabricate_transfers(self):
        trace = records(
            {
                "index": 0, "token": 2, "logical_layer": 7,
                "resolved_shard_layer": 3, "expert": 4, "kind": "cold",
                "cache_bytes_used": 0, "evicted_layer": -1,
                "evicted_expert": -1, "reuse_distance": -1,
            },
            {
                "index": 1, "token": 3, "logical_layer": 7,
                "resolved_shard_layer": 3, "expert": 4, "kind": "resident",
                "cache_bytes_used": 100, "evicted_layer": -1,
                "evicted_expert": -1, "reuse_distance": 0,
            },
        )
        cache, transfer = analyze(trace, "cpp.jsonl")
        self.assertEqual(cache["overall"]["misses"], 1)
        self.assertEqual(cache["overall"]["hits"], 1)
        self.assertEqual(cache["request_reuse_records"][1]["previous_request_distance"], 0)
        self.assertEqual(transfer["overall"]["transfers"], 0)
        self.assertEqual(transfer["h2d"]["measured_bytes"], 0)
        self.assertTrue(any("expert_bytes was not substituted" in item
                            for item in transfer["limitations"]))

    def test_engine_local_transfer_ids_and_projected_evictions_are_scoped(self):
        trace = records(
            {
                "event_type": "expert_request", "run_id": "warm-1",
                "token_step": 1, "logical_layer": 0, "resolved_shard_layer": 0,
                "expert_id": 9, "gpu_destination": "cuda:0",
                "cache_result": "miss", "evicted_expert_id": 2,
                "evicted_layer": 0, "eviction_reason": "capacity_lru",
            },
            {
                "event_type": "expert_transfer", "run_id": "warm-1",
                "token_step": 1, "logical_layer": 0, "resolved_shard_layer": 0,
                "expert_id": 9, "gpu_destination": "cuda:0",
                "direction": "h2d", "bytes": 100, "transfer_id": 7,
            },
            {
                "event_type": "expert_eviction", "run_id": "warm-1",
                "token_step": 1, "logical_layer": 0, "resolved_shard_layer": 0,
                "expert_id": 2, "triggering_expert_id": 9,
                "gpu_destination": "cuda:0", "eviction_reason": "capacity_lru",
            },
            {
                "event_type": "expert_request", "run_id": "warm-1",
                "token_step": 1, "logical_layer": 1, "resolved_shard_layer": 1,
                "expert_id": 10, "gpu_destination": "cuda:0",
                "cache_result": "miss", "evicted_expert_id": 3,
                "evicted_layer": 1, "eviction_reason": "capacity_lru",
            },
            {
                "event_type": "expert_transfer", "run_id": "warm-1",
                "token_step": 1, "logical_layer": 1, "resolved_shard_layer": 1,
                "expert_id": 10, "gpu_destination": "cuda:0",
                "direction": "h2d", "bytes": 100, "transfer_id": 7,
            },
            {
                "event_type": "expert_eviction", "run_id": "warm-1",
                "token_step": 1, "logical_layer": 1, "resolved_shard_layer": 1,
                "expert_id": 3, "triggering_expert_id": 10,
                "gpu_destination": "cuda:0", "eviction_reason": "capacity_lru",
            },
        )

        cache, transfer = analyze(trace, "engine-local-ids.jsonl")

        self.assertEqual(transfer["overall"]["transfers"], 2)
        self.assertEqual(transfer["h2d"]["measured_bytes"], 200)
        self.assertEqual(cache["input_summary"]["eviction_events"], 2)
        self.assertEqual(cache["input_summary"]["deduplicated_eviction_records"], 2)

    def test_missing_optional_fields_and_invalid_events_are_reported(self):
        trace = records(
            {"event_type": "expert_request", "logical_layer": 1, "expert_id": 9},
            {"event_type": "expert_request", "logical_layer": 1},
            {"event_type": "unrelated", "value": 4},
        )
        cache, transfer = analyze(trace, "partial.jsonl")
        self.assertEqual(cache["overall"]["cache_result_unknown"], 1)
        self.assertEqual(len(cache["input_summary"]["invalid_events"]), 1)
        self.assertEqual(cache["input_summary"]["ignored_event_types"], {"unrelated": 1})
        self.assertEqual(cache["coverage"]["expert_bytes"]["present"], 0)
        self.assertEqual(transfer["byte_accounting"]["measured_cumulative_bytes"], 0)

    def test_unknown_transfer_direction_is_not_counted_as_h2d(self):
        trace = records(
            {
                "event_type": "expert_request", "logical_layer": 1,
                "expert_id": 9, "expert_bytes": 64, "cache_result": "miss",
            },
            {
                "event_type": "expert_transfer", "logical_layer": 1,
                "expert_id": 9, "bytes": 64, "duration_us": 2,
            },
        )
        cache, transfer = analyze(trace, "unknown-direction.jsonl")
        self.assertEqual(cache["request_byte_accounting"]["measured_requested_bytes"], 64)
        self.assertEqual(transfer["byte_accounting"]["measured_cumulative_bytes"], 64)
        self.assertEqual(transfer["h2d"]["measured_bytes"], 0)
        self.assertEqual(transfer["h2d_byte_accounting"]["measured_cumulative_bytes"], 0)
        self.assertIn("unknown", transfer["byte_accounting_by_direction"])

    def test_gzip_input_and_cli_outputs(self):
        event = {
            "event_type": "expert_request", "logical_layer": 2, "expert_id": 3,
            "cache_result": "miss", "gpu_destination": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "expert-trace.jsonl.gz"
            with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            self.assertEqual(len(read_jsonl(trace_path)), 1)
            output = root / "evidence"
            self.assertEqual(main([str(trace_path), "--output-dir", str(output)]), 0)
            cache = json.loads((output / "expert-cache-analysis.json").read_text())
            transfer = json.loads((output / "transfer-analysis.json").read_text())
            self.assertEqual(cache["artifact"], "expert-cache-analysis")
            self.assertEqual(transfer["artifact"], "transfer-analysis")


if __name__ == "__main__":
    unittest.main()
