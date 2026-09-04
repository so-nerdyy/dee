"""Proof checks confined to research; no bridge implementation is changed."""
import json
from pathlib import Path

import pytest
import torch

from check_bf16_output_lattice import lattice_bounds


def test_representable_outputs_have_zero_lower_bound():
    result = lattice_bounds(torch.tensor([[1., -1., 0., 2., .5]]))
    assert result["reference_elements_not_representable_in_bf16"] == 0
    assert result["minimum_possible_max_abs_error"] == 0
    assert result["strict_allclose"]["any_bf16_output_can_pass"]


def test_neighbor_proof_agrees_with_exhaustive_finite_bf16_search():
    reference = torch.tensor([[1.003, -1.003, .0001, -0.0001, 13.39, -125.2345, 1e-8, -1e-8]])
    report = lattice_bounds(reference)
    bf16 = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    bf16 = bf16[torch.isfinite(bf16)].double()
    distances = (reference.flatten().double()[:, None] - bf16[None, :]).abs()
    possible = (distances <= 1e-5 + 1e-4 * bf16.abs()[None, :]).any(dim=1)
    assert report["strict_allclose"]["coordinates_no_bf16_candidate_can_satisfy"] == int((~possible).sum())
    assert report["minimum_possible_max_abs_error"] == float(distances.min(dim=1).values.max())


def test_nonfinite_reference_is_rejected():
    with pytest.raises(ValueError, match="finite reference"):
        lattice_bounds(torch.tensor([float("nan")]))


def test_captured_bounds_preserve_the_archived_reference_and_gates():
    evidence = Path(__file__).parent
    result = json.loads((evidence / "captured-real-expert155-bf16-output-lattice-20260904.json").read_text())
    original = json.loads((evidence / "captured-real-expert155-diagnostics-20260904.json").read_text())
    assert not result["native_kt_executed"]
    for row, old in zip(result["rows"], original["rows"], strict=True):
        assert row["oracle_sha256"] == old["output_sha256"]["trusted_dee"]
        assert row["nearest_bf16_ds8"] == old["diagnostics"]["variants"]["bf16_final_only"]
        assert row["fp32_identity_ds8"] == old["diagnostics"]["variants"]["fp32_reference"]
        assert row["reference_elements_not_representable_in_bf16"] > 0
        assert row["strict_allclose"]["coordinates_no_bf16_candidate_can_satisfy"] > 0
        for witness in row["strict_allclose"]["witnesses"]:
            assert witness["lower_strict_residual"] > 0
            assert witness["upper_strict_residual"] > 0
