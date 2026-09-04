"""Audit regression proofs; current contract behavior is observed, not changed."""
import json
from pathlib import Path

import pytest
import torch

from audit_captured_contract import compare, contract, contract_probes, decompose, grouped_dot, ordered_f32


def test_reference_coverage_is_invariant_for_finite_candidates():
    probe = contract_probes()
    fractions = [probe[k]["excluded"]["fraction"] for k in ("identity", "zero_candidate", "large_finite_candidate")]
    assert fractions[0] == fractions[1] == fractions[2]
    assert fractions[0] > .02
    assert probe["identity"]["decomposition"]["failed_checks"] == ["max_excluded_fraction"]


def test_candidate_can_change_finite_intersection_but_not_rescue_gate():
    probe = contract_probes()["candidate_nan_can_change_finite_intersection_but_fails_mask"]
    # Coverage is reference-only; a nonfinite candidate cannot alter it.
    assert probe["excluded"]["fraction"] == pytest.approx(.03)
    assert not probe["decomposition"]["actual_ds8_pass"]
    assert "sentinel_mask_exact" in probe["decomposition"]["failed_checks"]


def test_documented_nonfinite_rule_is_enforced_by_candidate_fidelity_gate():
    probes = contract_probes()
    for name in ("matching_nan", "matching_posinf", "neginf_to_nan", "neginf_to_posinf"):
        result = probes[name]["decomposition"]
        assert not result["actual_ds8_pass"]
        assert not result["checks"]["documented_no_nan_or_posinf"]["pass"]
        assert not result["candidate_fidelity"]["pass"]


def test_f32_ulp_ordering_handles_negative_neighbors_and_signed_zero():
    x = torch.tensor([-1., -0., 0., 1.])
    next_x = torch.nextafter(x, torch.full_like(x, float("inf")))
    assert (ordered_f32(next_x) - ordered_f32(x)).tolist() == [1, 1, 1, 1]
    assert ordered_f32(x)[1] == ordered_f32(x)[2]
    m = compare(torch.tensor([[1., -1.]]), torch.tensor([[1., -1.]]))
    assert m["counts"]["different_values"] == 0
    assert m["fp32_ulp_distance"]["finite_all"]["max"] == 0


def test_exact_thresholds_are_not_counted_as_greater():
    r = torch.tensor([[0., 0., 0., 0., 0.]])
    c = torch.tensor([[0., 5e-7, 5e-6, 5e-5, 5e-4]])
    counts = compare(r, c)["counts"]
    assert [counts[k] for k in ("different_values", "abs_gt_1e-6", "abs_gt_1e-5", "abs_gt_1e-4", "abs_gt_1e-3")] == [4, 3, 2, 1, 0]


def test_grouped_dot_exact_dyadic_fixture_and_both_pair_orders():
    x = torch.arange(64).float().reshape(1, -1) / 64
    w = torch.ones(3, 64) / 8
    scales = torch.full((3, 2), 124, dtype=torch.uint8)
    expected = x @ w.t()
    for natural in (False, True):
        for half_tree in (False, True):
            actual, info = grouped_dot(x, w, scales, natural, half_tree)
            assert torch.equal(actual, expected)
            assert info["products_exact_fp32"] and info["power_of_two_scaling_exact_fp32"]


def test_audit_preserves_all_real_output_hashes_and_decomposes_failures():
    root = Path(__file__).parent
    audit = json.loads((root / "captured-real-expert155-contract-audit-20260904.json").read_text())
    baseline = json.loads((root / "captured-real-expert155-diagnostics-20260904.json").read_text())
    assert audit["contract"]["tolerance"] == contract.DS8_TOLERANCE
    assert audit["contract"]["near_zero_threshold"] == .001
    assert audit["verdict"]["classification"] == "CONTRACT_REVIEW_REQUIRED_KT_STILL_FAILS"
    assert audit["verdict"]["coverage_failed_steps"] == [0, 3]
    for row, old in zip(audit["rows"], baseline["rows"], strict=True):
        cpp = row["comparisons"]["trusted_vs_cpp"]
        kt = row["comparisons"]["trusted_vs_kt_emulator"]
        assert cpp["reference_sha256"] == old["output_sha256"]["trusted_dee"]
        assert cpp["output_sha256"] == old["output_sha256"]["cpp_reference"]
        assert kt["output_sha256"] == old["output_sha256"]["kt_candidate"]
        assert cpp["metrics"] == old["cpp_reference_full_ds8_metrics"]
        assert kt["metrics"] == old["kt_candidate_full_ds8_metrics"]
        cpp_refreshed = decompose(cpp["metrics"])
        assert cpp_refreshed["sample_validity"]["pass"] == (row["forward_step"] not in (0, 3))
        assert cpp_refreshed["candidate_fidelity"]["pass"]
        assert cpp_refreshed["candidate_dependent_checks_pass_diagnostic_only"]
        assert cpp["fixed_fp32_allclose"]["pass"]
        assert "p99_rel_error" in kt["gate_decomposition"]["failed_checks"]
        assert not kt["fixed_fp32_allclose"]["pass"]
        assert row["finite_candidate_exclusion_invariant"]
        assert all(row["repeat_checks"].values())
        assert not row["bf16_output_lattice"]["strict_allclose"]["any_bf16_output_can_pass"]
        assert kt["gate_decomposition"]["checks"]["documented_no_nan_or_posinf"]["pass"]
        # The checked-in audit predates the explicit structured fields; its
        # numerical facts stay immutable while a fresh decomposition adds the
        # separated sample/candidate decisions.
        refreshed = decompose(kt["metrics"])
        assert refreshed["actual_ds8_pass"] == kt["gate_decomposition"]["actual_ds8_pass"]
        assert refreshed["candidate_fidelity"]["pass"] is False


def test_refactored_sealed_replay_separates_coverage_and_preserves_valid_decisions():
    root = Path(__file__).parent
    report = json.loads((root / "captured-real-expert155-contract-refactor-20260904.json").read_text())
    assert report["contract"]["tolerance"] == contract.DS8_TOLERANCE
    assert report["verdict"]["trusted_dee_vs_cpp_reference_passes_strict_ds8"] is False
    assert report["verdict"]["trusted_dee_vs_cpp_reference_candidate_fidelity"] is True
    assert report["verdict"]["kt_candidate_fidelity"] is False
    assert report["verdict"]["kt_cpu_remains_disabled"] is True
    by_step = {row["forward_step"]: row for row in report["rows"]}
    for step in (0, 3):
        assert by_step[step]["sample_validity"]["pass"] is False
        assert by_step[step]["cpp_reference_candidate_fidelity"]["pass"] is True
        assert by_step[step]["cpp_reference_ds8_pass"] is False
    assert by_step[10]["sample_validity"]["pass"] is True
    assert by_step[10]["cpp_reference_candidate_fidelity"]["pass"] is True
    assert by_step[10]["cpp_reference_ds8_pass"] is True
    compatibility = report["compatibility"]
    assert compatibility["previously_valid_sample_count"] == 1
    assert compatibility["candidate_decision_count"] == 2
    assert compatibility["all_previously_valid_candidate_fidelity_decisions_identical"] is True
