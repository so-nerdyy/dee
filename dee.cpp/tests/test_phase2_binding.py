"""Phase-2 pydee binding tests (W1-T3, see PHASE2_PYDEE_ARMING.md).

Verifies that the Phase-2 switches and host-tier geometry are reachable from
Python, that Engine::init still fails closed on invalid combinations, and
that ``engine.phase2_metrics`` returns the TierMetrics/HostTierStats fields
documented in PHASE2_METRICS.md — including the VRAM-only arm, which must
report device fields without any host tier.

The CPU build (DEE_CUDA=OFF) cannot exercise the host tier's live path (the
engine requires packed-FP4 + CUDA for the host arm and fails closed), but
every validation rule and the VRAM-only arm run identically on the host-mock
backend.

Run:  python -m pytest dee.cpp/tests/test_phase2_binding.py
(or)  python dee.cpp/tests/test_phase2_binding.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DEE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEE_ROOT))

import pydee  # noqa: E402

REQUIRED_METRIC_KEYS = {
    # TierMetrics (PHASE2_METRICS.md)
    "host", "device_hit", "device_miss", "H2D_bytes",
    "device_evictions", "device_failures",
    "device_bytes", "device_peak_bytes", "device_budget",
    "host_capacity_wait_ms", "device_enqueue_ms",
    "device_host_wait_ms", "pageable_fallback_wait_ms",
    "device_gpu_wait_ms", "tokens", "bytes_per_token_valid",
    "SSD_bytes_per_token", "H2D_bytes_per_token", "bytes_per_token",
}
REQUIRED_HOST_KEYS = {
    # HostTierStats (PHASE2_METRICS.md, "host." prefix dropped)
    "host_hit", "host_miss", "coalesced", "SSD_bytes", "fills",
    "evictions", "failures", "budget_rejections", "pin_failures",
    "allocated_bytes", "pinned_bytes", "resident_bytes",
    "peak_resident_bytes", "leased_slots",
    "host_wait_ms", "storage_service_ms",
}


def _make_shard(directory: Path) -> Path:
    """Generate the deterministic Ornith-style shard used by the C++ tests."""
    shard = directory / "layer0_shard.safetensors"
    subprocess.run(
        [sys.executable, str(DEE_ROOT / "tests" / "gen_synthetic_shard.py"),
         str(shard)],
        check=True, capture_output=True)
    return shard


def _base_config(shard: Path):
    """Mirror of the C++ disabled_configuration() fixture (hidden=16 etc.)."""
    cfg = pydee.EngineConfig()
    cfg.shard_path = str(shard)
    cfg.hidden = 16
    cfg.inter = 8
    cfg.num_layers = 1
    cfg.num_experts = 3
    cfg.topk = 2
    return cfg


@unittest.skipIf(pydee.Engine is None, "pydee_core extension not built")
class Phase2BindingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.shard = _make_shard(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_defaults_off(self):
        cfg = pydee.EngineConfig()
        self.assertFalse(cfg.phase2.enabled)
        self.assertFalse(cfg.phase2.host_enabled)
        self.assertFalse(cfg.phase2.vram_priority_fix_enabled)
        self.assertEqual(cfg.phase2.model_identity, "")
        host = cfg.phase2.host
        self.assertEqual(host.slot_bytes, 0)
        self.assertEqual(host.alignment, 4096)
        self.assertEqual(host.policy_slots, 0)
        self.assertEqual(host.dynamic_slots, 0)
        self.assertEqual(host.budget_bytes, 0)
        self.assertTrue(host.try_pin)

    def test_field_roundtrip(self):
        cfg = pydee.EngineConfig()
        cfg.phase2.enabled = True
        cfg.phase2.host_enabled = True
        cfg.phase2.vram_priority_fix_enabled = True
        cfg.phase2.model_identity = "model-sha256:test"
        cfg.phase2.host.slot_bytes = 13369344
        cfg.phase2.host.alignment = 65536
        cfg.phase2.host.dynamic_slots = 64
        cfg.phase2.host.budget_bytes = 1 << 30
        cfg.phase2.host.try_pin = False
        self.assertTrue(cfg.phase2.enabled)
        self.assertTrue(cfg.phase2.host_enabled)
        self.assertTrue(cfg.phase2.vram_priority_fix_enabled)
        self.assertEqual(cfg.phase2.model_identity, "model-sha256:test")
        self.assertEqual(cfg.phase2.host.slot_bytes, 13369344)
        self.assertEqual(cfg.phase2.host.alignment, 65536)
        self.assertEqual(cfg.phase2.host.dynamic_slots, 64)
        self.assertEqual(cfg.phase2.host.budget_bytes, 1 << 30)
        self.assertFalse(cfg.phase2.host.try_pin)
        # A whole-object assignment must replace the member, not alias it.
        host_cfg = pydee.HostTierConfig()
        host_cfg.dynamic_slots = 7
        host_cfg.budget_bytes = 4096 * 7
        cfg.phase2.host = host_cfg
        self.assertEqual(cfg.phase2.host.dynamic_slots, 7)

    def test_fail_closed_combinations(self):
        # master OFF + host sub-switch ON
        cfg = _base_config(self.shard)
        cfg.phase2.host_enabled = True
        self.assertFalse(pydee.Engine().init(cfg))
        # master OFF + vram sub-switch ON
        cfg = _base_config(self.shard)
        cfg.phase2.vram_priority_fix_enabled = True
        self.assertFalse(pydee.Engine().init(cfg))
        # enabled + neither arm
        cfg = _base_config(self.shard)
        cfg.phase2.enabled = True
        self.assertFalse(pydee.Engine().init(cfg))
        # host arm without model_identity (also no CUDA/FP4 on this build)
        cfg = _base_config(self.shard)
        cfg.phase2.enabled = True
        cfg.phase2.host_enabled = True
        cfg.phase2.host.dynamic_slots = 4
        cfg.phase2.host.budget_bytes = 16 << 20
        self.assertFalse(pydee.Engine().init(cfg))
        # host arm WITH identity still fails closed on a CPU build — the
        # engine requires packed-FP4 + CUDA + Fp4E2m1 transfer for the tier.
        cfg = _base_config(self.shard)
        cfg.phase2.enabled = True
        cfg.phase2.host_enabled = True
        cfg.phase2.model_identity = "model-sha256:test"
        cfg.phase2.host.dynamic_slots = 4
        cfg.phase2.host.budget_bytes = 16 << 20
        self.assertFalse(pydee.Engine().init(cfg))

    def test_vram_only_arm(self):
        cfg = _base_config(self.shard)
        cfg.phase2.enabled = True
        cfg.phase2.vram_priority_fix_enabled = True
        engine = pydee.Engine()
        self.assertTrue(engine.init(cfg))
        # runtime_config echoes the armed switches.
        rc = engine.runtime_config()
        self.assertTrue(rc["phase2"]["enabled"])
        self.assertTrue(rc["phase2"]["vram_priority_fix_enabled"])
        self.assertFalse(rc["phase2"]["host_enabled"])
        # No host tier exists, but device fields must still be reported.
        metrics = engine.phase2_metrics()
        self.assertTrue(REQUIRED_METRIC_KEYS <= set(metrics))
        self.assertTrue(REQUIRED_HOST_KEYS <= set(metrics["host"]))
        self.assertGreater(metrics["device_budget"], 0)
        self.assertEqual(metrics["tokens"], 0)
        self.assertFalse(metrics["bytes_per_token_valid"])
        self.assertEqual(metrics["host"]["allocated_bytes"], 0)
        self.assertIsNone(metrics["device_gpu_wait_ms"])
        metrics = engine.phase2_metrics(7)
        self.assertEqual(metrics["tokens"], 7)
        self.assertTrue(metrics["bytes_per_token_valid"])

    def test_disabled_engine_metrics(self):
        engine = pydee.Engine()
        metrics = engine.phase2_metrics()
        self.assertTrue(REQUIRED_METRIC_KEYS <= set(metrics))
        self.assertEqual(metrics["host"]["allocated_bytes"], 0)
        self.assertFalse(metrics["bytes_per_token_valid"])
        cfg = _base_config(self.shard)
        self.assertTrue(engine.init(cfg))
        metrics = engine.phase2_metrics(3)
        # Explicit-OFF still allocates no host tier and reports no device
        # accounting (that echo is reserved for armed configs).
        self.assertEqual(metrics["host"]["allocated_bytes"], 0)
        self.assertEqual(metrics["tokens"], 0)

    def test_explicit_off_matches_default_outputs(self):
        import numpy as np
        ordinary = pydee.Engine()
        self.assertTrue(ordinary.init(_base_config(self.shard)))
        cfg = _base_config(self.shard)
        cfg.phase2.enabled = False
        cfg.phase2.host.budget_bytes = 1  # invalid IF accidentally activated
        cfg.phase2.host.dynamic_slots = 100
        disabled = pydee.Engine()
        self.assertTrue(disabled.init(cfg))
        h_in = np.asarray([0.001 * (i + 1) for i in range(16)],
                          dtype=np.float32)
        for routes in ([0, 1], [2, 0], [0, 1]):
            a = np.empty(2 * 16, dtype=np.float32)
            b = np.empty(2 * 16, dtype=np.float32)
            self.assertTrue(ordinary.moe_forward_experts(0, h_in, a, routes))
            self.assertTrue(disabled.moe_forward_experts(0, h_in, b, routes))
            self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(
            disabled.phase2_metrics()["host"]["allocated_bytes"], 0)


class Phase2ModelBuilderTest(unittest.TestCase):
    """build_native_engine kwarg validation (no GPU needed: it fails before
    or inside init on a CPU-only build, never silently arms)."""

    @unittest.skipIf(pydee.Engine is None, "pydee_core extension not built")
    def test_invalid_mode_rejected(self):
        sys.path.insert(0, str(DEE_ROOT / "scripts"))
        try:
            import deepseek_v4_model as vm
            with self.assertRaises(ValueError):
                vm.build_native_engine(
                    ["unused.safetensors"], phase2_mode="pinning")
        finally:
            sys.path.remove(str(DEE_ROOT / "scripts"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
