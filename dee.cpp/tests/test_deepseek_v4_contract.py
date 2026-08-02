"""DS8 contract + corpus tests (metrics, near-zero exclusion, gates, corpus)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    sys.stderr.write("tests/test_deepseek_v4_contract.py requires pytest: pip install pytest\n")
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from scripts import deepseek_v4_contract as contract  # noqa: E402
from scripts import deepseek_v4_corpus as corpus  # noqa: E402


def test_identical_tensors_pass_every_gate() -> None:
    x = torch.randn(64, 128)
    metrics = contract.compute_ds8_metrics(x, x)
    assert contract.ds8_gate_passed(metrics)
    # cosine of identical vectors is 1.0 up to FP32 rounding
    assert metrics["cosine_similarity"] >= 1.0 - 1e-6
    assert metrics["normalized_rmse"] < 1e-6
    assert metrics["output_norm_rel_error"] < 1e-6


def test_perturbed_tensor_within_gates() -> None:
    torch.manual_seed(0)
    ref = torch.randn(64, 128)
    # Multiplicative (relative) noise: the realistic FP16-candidate error
    # model.  Additive absolute noise on N(0,1) refs would spike relative
    # error near zero -- the same artifact that justified the near-zero
    # exclusion in the DS7 evidence.
    cand = ref * (1 + torch.randn_like(ref) * 2e-3)
    metrics = contract.compute_ds8_metrics(ref, cand)
    assert contract.ds8_gate_passed(metrics)
    assert metrics["all_elements"]["mean_abs_error"] < 0.05


def test_large_perturbation_fails_gates() -> None:
    torch.manual_seed(1)
    ref = torch.randn(64, 128)
    cand = ref * 0.5  # big structured error
    metrics = contract.compute_ds8_metrics(ref, cand)
    assert not contract.ds8_gate_passed(metrics)


def test_near_zero_exclusion_prevents_rel_spike() -> None:
    # A single near-zero reference element with a small absolute error must
    # NOT spike the non-near-zero relative statistics.
    ref = torch.randn(64, 128) * 1.0
    cand = ref.clone()
    cand[0, 0] = 1e-5
    ref[0, 0] = 1e-6  # near-zero: rel error would be ~10x -> excluded
    metrics = contract.compute_ds8_metrics(ref, cand)
    # absolute error at the near-zero element is tiny
    assert metrics["all_elements"]["max_abs_error"] > 0  # still computed over all
    nnz = metrics["non_near_zero"]
    assert nnz["mean_rel_error"] < 0.01
    assert metrics["excluded"]["fraction"] > 0
    assert contract.ds8_gate_passed(metrics)


def test_worst_element_reporting() -> None:
    ref = torch.zeros(10, 10)
    cand = torch.zeros(10, 10)
    ref[3, 4] = 5.0
    cand[3, 4] = 5.5
    metrics = contract.compute_ds8_metrics(ref, cand)
    worst = metrics["worst_abs_element"]
    assert worst["abs_error"] == 0.5
    assert worst["reference"] == 5.0
    assert worst["candidate"] == 5.5
    assert worst["index"] == 34


def test_shape_mismatch_rejected() -> None:
    # Same element count, different logical shape: rejected by the pre-flatten
    # shape check.
    with pytest.raises(ValueError, match="shape mismatch"):
        contract.compute_ds8_metrics(torch.zeros(4, 8), torch.zeros(8, 4))


def test_sentinel_inf_masks_matching_give_finite_metrics() -> None:
    # DS9 v7: the indexer scores legitimately contain -inf causal sentinels on
    # BOTH sides.  Metrics must be computed on the finite intersection only;
    # cosine must NOT be NaN and the mask must be reported exact.
    torch.manual_seed(3)
    ref = torch.randn(16, 64)
    cand = ref * (1 + torch.randn_like(ref) * 1e-3)
    mask = torch.rand_like(ref) < 0.3
    ref[mask] = float("-inf")
    cand[mask] = float("-inf")  # identical sentinel positions
    metrics = contract.compute_ds8_metrics(ref, cand)
    assert metrics["sentinel_mask_exact"] is True
    assert metrics["finite_overlap"] is True
    assert metrics["mask_counts"]["both_nonfinite"] == int(mask.sum())
    assert metrics["cosine_similarity"] is not None
    assert not math.isnan(metrics["cosine_similarity"])
    assert metrics["normalized_rmse"] is not None
    assert contract.ds8_gate_passed(metrics)


def test_sentinel_mask_mismatch_fails_every_gate() -> None:
    # One side has -inf where the other is finite: a semantic failure even if
    # finite-intersection numerics are perfect.
    ref = torch.zeros(8, 8)
    cand = torch.zeros(8, 8)
    ref[0, 0] = float("-inf")
    cand[0, 0] = 0.5  # cand says finite, ref says sentinel
    metrics = contract.compute_ds8_metrics(ref, cand)
    assert metrics["sentinel_mask_exact"] is False
    assert metrics["mask_counts"]["ref_only_nonfinite"] == 1
    assert not contract.ds8_gate_passed(metrics)


def test_nan_in_candidate_fails_gates_and_reports_nan_counts() -> None:
    ref = torch.zeros(8, 8)
    cand = torch.zeros(8, 8)
    cand[3, 3] = float("nan")
    metrics = contract.compute_ds8_metrics(ref, cand)
    assert metrics["sentinel_mask_exact"] is False
    assert metrics["mask_counts"]["nan_candidate"] == 1
    assert not contract.ds8_gate_passed(metrics)


def test_no_finite_overlap_fails_explicitly() -> None:
    # All positions sentinel on both sides: no finite overlap -> cosine None,
    # gates fail (never a silent NaN pass).
    ref = torch.full((4, 4), float("-inf"))
    cand = torch.full((4, 4), float("-inf"))
    metrics = contract.compute_ds8_metrics(ref, cand)
    assert metrics["sentinel_mask_exact"] is True
    assert metrics["finite_overlap"] is False
    assert metrics["cosine_similarity"] is None
    assert not contract.ds8_gate_passed(metrics)


def test_ulp_distance_and_first_value_divergence() -> None:
    # fp32 ULP distance: identical values -> 0; adjacent values -> 1.
    assert contract.f32_ulp_distance(1.0, 1.0) == 0
    assert contract.f32_ulp_distance(1.0, 1.0 + 2.0**-23) == 1
    assert contract.f32_ulp_distance(-1.0, 1.0) > 0
    # state_mask_analysis reports the first divergent VALUE with bits/ulp.
    ref = {"buf": torch.zeros(4, 4)}
    cand = {"buf": torch.zeros(4, 4)}
    cand["buf"][1, 1] = 1.0
    out = contract.state_mask_analysis(ref, cand)
    assert out["buf"]["ok"] is False
    fd = out["buf"]["first_value_divergence"]
    assert fd is not None and fd["flat_index"] == 5
    assert fd["abs_error"] == 1.0
    assert fd["ref_bits"] != fd["cand_bits"]


def test_state_mask_analysis_sentinel_mismatch_classified() -> None:
    ref = {"score": torch.full((2, 4), float("-inf"))}
    cand = {"score": torch.full((2, 4), float("-inf"))}
    cand["score"][0, 0] = 0.0  # cand wrote a slot ref left as sentinel
    out = contract.state_mask_analysis(
        ref, cand, init_values={"score": float("-inf")})
    assert out["score"]["ok"] is False
    assert out["score"]["sentinel_mask_exact"] is False
    assert out["score"]["finite_mask_exact"] is False
    assert out["score"]["written_slot_mask_exact"] is False
    md = out["score"]["mask_divergence"]
    assert md is not None
    assert md["class"] == "ref_nonfinite_cand_finite"
    assert out["score"]["counts"]["ref_only_nonfinite"] == 1


def test_state_mask_analysis_written_slot_gate_kv_state() -> None:
    # kv_state inits to ZEROS: a candidate that writes a DIFFERENT slot than
    # the reference (but keeps the same finite mask everywhere) must fail the
    # written/untouched slot gates even though isfinite masks agree.
    ref = {"kv": torch.zeros(2, 4)}
    cand = torch.zeros(2, 4)
    ref["kv"][0, 0] = 1.0  # reference writes slot (0,0)
    cand[0, 1] = 1.0  # candidate writes slot (0,1) instead
    out = contract.state_mask_analysis(ref, {"kv": cand},
                                       init_values={"kv": 0.0})
    assert out["kv"]["sentinel_mask_exact"] is True  # all finite both sides
    assert out["kv"]["written_slot_mask_exact"] is False
    assert out["kv"]["untouched_slot_mask_exact"] is False
    assert out["kv"]["ok"] is False
    md = out["kv"]["mask_divergence"]
    assert md is not None
    # First differing flat index is 0 == slot (0,0): the reference WROTE it
    # (untouched_ref False) while the candidate left it at init zero
    # (untouched_cand True), so the class is ref_written_cand_untouched.
    assert md["class"] == "ref_written_cand_untouched"


def test_state_mask_analysis_legitimate_sentinels_pass() -> None:
    # BOTH sides leave score_state rows untouched (-inf) at the same slots and
    # write finite values at the same slots: every gate must pass.
    ref = {"score": torch.full((2, 4), float("-inf"))}
    cand = torch.full((2, 4), float("-inf"))
    ref["score"][0, 0] = 0.5
    ref["score"][0, 1] = 1.5
    cand[0, 0] = 0.5
    cand[0, 1] = 1.5
    out = contract.state_mask_analysis(ref, {"score": cand},
                                       init_values={"score": float("-inf")})
    assert out["score"]["ok"] is True
    assert out["score"]["sentinel_mask_exact"] is True
    assert out["score"]["written_slot_mask_exact"] is True
    assert out["score"]["untouched_slot_mask_exact"] is True
    assert out["score"]["mask_divergence"] is None
    assert out["score"]["first_value_divergence"] is None


def test_corpus_distributions_shapes_and_seed_repro() -> None:
    cases, meta = corpus.build_corpus(6, 64, base_seed=7)
    assert meta["n_tokens"] == 6 and meta["hidden"] == 64
    names = [name for name, _ in cases]
    assert "normal" in names and "sparse" in names and "near_zero" in names
    for name, x in cases:
        assert x.shape == (6, 64)
        assert torch.isfinite(x).all()
    # adversarial: alternating sign pattern, all |value| == 1
    adv = dict(cases)["adversarial"]
    assert torch.equal(adv.abs(), torch.ones_like(adv))
    # repeated: all rows identical
    rep = dict(cases)["repeated"]
    assert torch.equal(rep[0].unsqueeze(0).expand_as(rep), rep)
    # determinism
    cases2, _ = corpus.build_corpus(6, 64, base_seed=7)
    for (n1, x1), (n2, x2) in zip(cases, cases2):
        assert torch.equal(x1, x2)


def test_corpus_official_trace_absent_note(tmp_path) -> None:
    cases, meta = corpus.build_corpus(4, 32, base_seed=0,
                                      official_trace=tmp_path / "none.npz")
    assert "official hidden-state trace absent" in meta["official_trace_note"]
    assert not any(name == "official_trace" for name, _ in cases)


def test_corpus_official_trace_json_load() -> None:
    trace_path = Path(__file__).parents[1] / "tmp" / "ds8_trace_sample.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("[[0.5, -1.0, 2.0], [1.0, 0.25, -0.5]]", encoding="utf-8")
    try:
        cases, meta = corpus.build_corpus(2, 3, base_seed=0, official_trace=trace_path)
        official = dict(cases)["official_trace"]
        assert official.shape == (2, 3)
        assert "loaded" in meta["official_trace_note"]
    finally:
        trace_path.unlink(missing_ok=True)
