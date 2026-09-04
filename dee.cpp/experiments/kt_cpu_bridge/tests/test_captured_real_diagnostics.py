"""Diagnostic regression tests; these do not assert native KT acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

BRIDGE = Path(__file__).resolve().parents[1]
ROOT = BRIDGE.parents[2]
EVIDENCE = ROOT / "research" / "kt-cpu-bridge" / "evidence"
sys.path.insert(0, str(BRIDGE / "python"))
sys.path.insert(0, str(ROOT / "dee.cpp"))
sys.path.insert(0, str(EVIDENCE))

from captured_real_diagnostics import (  # noqa: E402
    bf16, boundary_variants, failed_gates, measure, row_diagnostics, swiglu,
)
from kt_cpu_bridge.reference import expert_forward_reference, kt_emulated_forward  # noqa: E402
from kt_cpu_bridge.codec import dequantize_weight  # noqa: E402
from scripts import deepseek_v4_contract as contract  # noqa: E402


def fixture():
    gen = torch.Generator().manual_seed(419)
    def packed():
        return torch.randint(0, 256, (32, 16), generator=gen, dtype=torch.uint8)
    def scale():
        return torch.full((32, 1), 121, dtype=torch.uint8)
    args = (packed(), scale(), packed(), scale(), packed(), scale())
    weights = tuple(dequantize_weight(args[i], args[i + 1]) for i in (0, 2, 4))
    x = torch.randn((1, 32), generator=gen)
    return x, args, weights


def test_diagnostic_paths_exactly_match_unchanged_implementations():
    x, args, weights = fixture()
    expected = expert_forward_reference(x, *args, routing_weight=.37)
    candidate = kt_emulated_forward(x, *args, routing_weight=.37)
    before = deepcopy(contract.DS8_TOLERANCE)
    report = row_diagnostics(x, x, weights, .37, expected, candidate)
    assert report["variants"]["emulator_existing"]["full_ds8_metrics"] == contract.compute_ds8_metrics(expected, candidate)
    assert contract.DS8_TOLERANCE == before
    with pytest.raises(ValueError, match="candidate drift"):
        row_diagnostics(x, x, weights, .37, expected, candidate + 1)


def test_down_bf16_boundary_is_distinct_before_nonunit_weight():
    x, _, weights = fixture()
    variants, _, stages = boundary_variants(x, x, *weights, .37)
    expected = bf16(bf16(stages["unweighted_down"]) * .37)
    assert torch.equal(expected, variants["emulator_plus_preweight_down_bf16"])
    assert not torch.equal(expected, variants["emulator_existing"])
    unit, _, _ = boundary_variants(x, x, *weights, 1.)
    assert torch.equal(unit["emulator_existing"], unit["emulator_plus_preweight_down_bf16"])


def test_gate_clamp_is_one_sided_and_zero_disables():
    gate = torch.tensor([[-20., 20.]])
    up = torch.tensor([[-20., 20.]])
    assert torch.equal(swiglu(gate, up), torch.nn.functional.silu(torch.tensor([[-20., 10.]])) * torch.tensor([[-10., 10.]]))
    assert torch.equal(swiglu(gate, up, 0), torch.nn.functional.silu(gate) * up)


def test_identity_still_fails_unmodified_coverage_gate():
    x = torch.tensor([[0., 1., 2., 3.]])
    report = measure(x, x)
    assert report["ds8_pass"] is False
    assert report["failed_gates"] == ["max_excluded_fraction"]


def test_finite_mask_failure_is_not_hidden():
    m = contract.compute_ds8_metrics(torch.ones(3), torch.tensor([1., float("nan"), 1.]))
    assert "sentinel_mask_exact" in failed_gates(m)
    assert not contract.ds8_gate_passed(m)


def test_archived_real_replay_has_numerical_and_independent_coverage_failures():
    report = json.loads((EVIDENCE / "captured-real-expert155-replay-20260904.json").read_text())
    assert report["contract"]["tolerance"] == contract.DS8_TOLERANCE
    for row in report["rows"]:
        assert "p99_rel_error" in failed_gates(row["kt_candidate_full_ds8_metrics"])
        assert row["cpp_reference_fixed_fp32_allclose_atol_1e-5_rtol_1e-4"]
        cpp_failed = failed_gates(row["cpp_reference_full_ds8_metrics"])
        assert cpp_failed == (["max_excluded_fraction"] if row["forward_step"] in (0, 3) else [])


def test_durable_diagnostics_preserve_baseline_and_attribute_real_boundaries():
    baseline = json.loads((EVIDENCE / "captured-real-expert155-replay-20260904.json").read_text())
    report = json.loads((EVIDENCE / "captured-real-expert155-diagnostics-20260904.json").read_text())
    assert report["contract"] == baseline["contract"]
    assert report["capture_seal"] == baseline["capture_seal"]
    assert report["record_evidence"] == baseline["record_evidence"]
    assert report["verdict"] == baseline["verdict"]
    assert len(report["rows"]) == len(baseline["rows"]) == 3
    for row, old in zip(report["rows"], baseline["rows"]):
        diagnostic = row.pop("diagnostics")
        assert row == old
        assert diagnostic["input"]["native_equal_original_count"] == 4096
        assert diagnostic["input"]["bf16_input_changed_count"] == 0
        assert diagnostic["clamp_events"]["gate"] == diagnostic["clamp_events"]["up"] == 0
        variants = diagnostic["variants"]
        assert variants["original_fp32_input"]["output_sha256"] == row["output_sha256"]["trusted_dee"]
        assert "p99_rel_error" in variants["emulator_plus_preweight_down_bf16"]["failed_gates"]
        assert "p99_rel_error" in variants["bf16_input_gate_up_intermediate"]["failed_gates"]
