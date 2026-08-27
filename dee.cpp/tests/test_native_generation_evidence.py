"""Fail-closed evidence/classification tests for the Kaggle full-model run."""

from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
