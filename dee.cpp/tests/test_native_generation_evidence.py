"""Fail-closed evidence/classification tests for the Kaggle full-model run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HARNESS = (
    Path(__file__).resolve().parents[1]
    / "kaggle"
    / "deepseek-v4-flash-0731"
    / "deepseek_v4_native_generate.py"
)
SPEC = importlib.util.spec_from_file_location("dsv4_native_generate", HARNESS)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullGenerationClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        MODULE.CACHE_DTYPE = "fp4"
        MODULE.EXPERT_STORE_BACKEND = "dee4"
        MODULE.N_TOKENS = 16

    @staticmethod
    def valid_result() -> dict:
        engine_stats = {"hidden_finite": True}
        engine_config = {"cache_dtype": "fp4-packed", "use_cuda": True}
        store = {
            "backend": "dee4",
            "integrity_identity": "a" * 64,
            "lookup_failures": 0,
            "source_reads": 10,
            "contiguous_source_reads": 10,
        }
        route_journal = {
            "artifact": "routed_experts.jsonl",
            "schema_version": 1,
            "canonical_order": MODULE.RoutedExpertJournal.CANONICAL_ORDER,
            "n_layers": 43,
            "topk": 6,
            "record_count": 16 * 43,
            "completed_forwards": 16,
            "checkpoint_link_count": 16,
            "checkpoint_steps": list(range(16)),
            "last_forward_step": 15,
            "last_layer": 42,
            "chain_sha256": "b" * 64,
            "file_sha256": "c" * 64,
            "flush_each_layer": True,
            "fsync_each_completed_forward": True,
        }
        return {
            "generated_token_ids": list(MODULE.SEALED_TOKEN_IDS),
            "decoded_text": MODULE.SEALED_DECODED_TEXT,
            "layer_count_executed": 43,
            "bridge_counters": {
                "numpy_bridge_calls": 0,
                "full_hidden_d2h_copies": 0,
                "raw_expert_output_d2h_copies": 0,
            },
            "engine_stats": {
                "cuda0": dict(engine_stats), "cuda1": dict(engine_stats)},
            "engine_config": {
                "cuda0": dict(engine_config), "cuda1": dict(engine_config)},
            "expert_store": {"cuda0": dict(store), "cuda1": dict(store)},
            "route_journal": route_journal,
            "model_runtime_snapshot": {
                "backends": {
                    "router": "torch_cuda_validated_ds9_path",
                    "cpu_expert_execution": False,
                }
            },
            "gpu_environment": {
                "gpu_count": 2,
                "single_gpu_mode": False,
                "nvidia_smi_lines": [
                    "GPU 0: Tesla T4 (UUID: test-0)",
                    "GPU 1: Tesla T4 (UUID: test-1)",
                ],
            },
        }

    def test_exact_dual_t4_run_is_correctness_accepted_and_performance_eligible(self) -> None:
        classification, gates, performance_eligible = (
            MODULE.classify_full_generation(self.valid_result()))
        self.assertEqual("ACCEPT_CORRECTNESS", classification)
        self.assertTrue(all(gates.values()))
        self.assertTrue(performance_eligible)

    def test_wrong_hardware_can_pass_correctness_but_not_performance(self) -> None:
        result = self.valid_result()
        result["gpu_environment"] = {
            "gpu_count": 1,
            "single_gpu_mode": True,
            "nvidia_smi_lines": ["GPU 0: Tesla P100 (UUID: test)"],
        }
        result["engine_stats"] = {"cuda0": {"hidden_finite": True}}
        result["engine_config"] = {
            "cuda0": {"cache_dtype": "fp4-packed", "use_cuda": True}}
        result["expert_store"] = {
            "cuda0": {
                "backend": "dee4", "integrity_identity": "b" * 64,
                "lookup_failures": 0, "source_reads": 1,
                "contiguous_source_reads": 1}}
        classification, gates, performance_eligible = (
            MODULE.classify_full_generation(result))
        self.assertEqual("ACCEPT_CORRECTNESS", classification)
        self.assertFalse(gates["required_performance_hardware"])
        self.assertFalse(performance_eligible)

    def test_non_finite_output_is_numerical_rejection(self) -> None:
        result = self.valid_result()
        result["engine_stats"]["cuda1"]["hidden_finite"] = False
        classification, _, performance_eligible = (
            MODULE.classify_full_generation(result))
        self.assertEqual("REJECT_NUMERICAL", classification)
        self.assertFalse(performance_eligible)

    def test_missing_bridge_counter_fails_integrity_closed(self) -> None:
        result = self.valid_result()
        del result["bridge_counters"]["numpy_bridge_calls"]
        classification, gates, _ = MODULE.classify_full_generation(result)
        self.assertEqual("REJECT_INTEGRITY", classification)
        self.assertFalse(gates["numpy_bridge_calls"])

    def test_dee4_scatter_read_cannot_claim_live_contiguous_backend(self) -> None:
        result = self.valid_result()
        result["expert_store"]["cuda0"]["contiguous_source_reads"] = 9
        classification, gates, _ = MODULE.classify_full_generation(result)
        self.assertEqual("REJECT_INTEGRITY", classification)
        self.assertFalse(gates["dee4_contiguous_reads"])

    def test_incomplete_route_journal_fails_integrity_closed(self) -> None:
        result = self.valid_result()
        result["route_journal"]["last_layer"] = 41
        classification, gates, performance_eligible = (
            MODULE.classify_full_generation(result))
        self.assertEqual("REJECT_INTEGRITY", classification)
        self.assertFalse(gates["route_journal_complete"])
        self.assertFalse(performance_eligible)

    def test_dee4_trace_requires_independently_validated_metadata_linkage(self) -> None:
        MODULE.EXPERT_STORE_BACKEND = "dee4_trace"
        result = self.valid_result()
        for store in result["expert_store"].values():
            store["backend"] = "dee4_trace"
        result["dee4_trace_validation"] = {
            "success": True,
            "format": "dee4-v3-trace",
            "record_indices_contiguous": True,
            "integrity_records_complete": True,
            "data_sha256": "1" * 64,
            "trace_journal_sha256": "2" * 64,
            "trace_final_chain_sha256": "3" * 64,
            "selection_sha256": "4" * 64,
        }
        classification, gates, _ = MODULE.classify_full_generation(result)
        self.assertEqual("ACCEPT_CORRECTNESS", classification)
        self.assertTrue(gates["dee4_trace_metadata_linkage"])

        del result["dee4_trace_validation"]["selection_sha256"]
        classification, gates, _ = MODULE.classify_full_generation(result)
        self.assertEqual("REJECT_INTEGRITY", classification)
        self.assertFalse(gates["dee4_trace_metadata_linkage"])

    def test_trace_bootstrap_pins_the_committed_v50_journal_and_metadata_path(self) -> None:
        self.assertEqual(
            MODULE.V50_TRACE_JOURNAL_SHA256,
            "665aac3e8db570237c6dc6acaf08dc39f2af890e8a04e400ce7154f1a858dae1",
        )
        self.assertEqual(
            MODULE.V50_TRACE_FINAL_CHAIN_SHA256,
            "086f8ca83b6a3c467cdf950096141fa9bc3e55a285d7d1fed8a0ad9913e3eb3d",
        )
        source = HARNESS.read_text("utf-8")
        self.assertIn("_repack_trace(", source)
        self.assertIn(
            "expected_journal_sha256=V50_TRACE_JOURNAL_SHA256", source)
        self.assertIn(
            "expected_final_chain_sha256=V50_TRACE_FINAL_CHAIN_SHA256", source)
        self.assertIn("dee4_store_path = (", source)
        self.assertIn("str(_trace_metadata_path)", source)


class RoutedExpertJournalTests(unittest.TestCase):
    @staticmethod
    def _payload_bytes(record: dict) -> bytes:
        payload = dict(record)
        payload.pop("chain_sha256")
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("utf-8")

    def test_canonical_hash_chain_and_token_checkpoint_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routed_experts.jsonl"
            journal = MODULE.RoutedExpertJournal(
                path, run_id="test-run", n_layers=3, topk=2)
            matrices = (
                [[9, 1], [7, 3]],
                [[8, 2], [6, 4]],
                [[5, 0], [11, 10]],
            )
            for layer, matrix in enumerate(matrices):
                journal.append_layer(
                    step=0, start_pos=0, layer=layer, device="cuda:0",
                    expert_ids=matrix)
            link = journal.checkpoint_link(0)
            journal.close()
            summary = journal.summary()

            records = [
                json.loads(line)
                for line in path.read_text("utf-8").splitlines()
            ]
            self.assertEqual(3, len(records))
            previous = MODULE.RoutedExpertJournal.GENESIS_SHA256
            for layer, (record, matrix) in enumerate(zip(records, matrices)):
                self.assertEqual(layer, record["layer"])
                self.assertEqual(matrix, record["expert_ids_rank_order"])
                self.assertEqual(previous, record["previous_chain_sha256"])
                expected = hashlib.sha256(
                    self._payload_bytes(record)).hexdigest()
                self.assertEqual(expected, record["chain_sha256"])
                previous = expected

            self.assertEqual(2, records[-1]["layer"])
            self.assertEqual(previous, link["chain_sha256"])
            self.assertEqual(3, link["record_count"])
            self.assertTrue(link["terminal_layer_fsync_complete"])
            self.assertEqual(1, summary["completed_forwards"])
            self.assertEqual([0], summary["checkpoint_steps"])
            self.assertEqual(MODULE.sha256_file(path), summary["file_sha256"])
            self.assertFalse(summary["adds_cuda_events"])
            self.assertFalse(summary["adds_device_transfers"])
            self.assertFalse(summary["adds_host_synchronizations"])

    def test_noncanonical_layer_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = MODULE.RoutedExpertJournal(
                Path(directory) / "routes.jsonl",
                run_id="test-run", n_layers=3, topk=2)
            with self.assertRaisesRegex(RuntimeError, "non-canonical"):
                journal.append_layer(
                    step=0, start_pos=0, layer=1, device="cuda:0",
                    expert_ids=[[1, 0]])
            journal.close()


if __name__ == "__main__":
    unittest.main()
