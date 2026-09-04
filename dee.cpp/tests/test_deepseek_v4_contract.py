"""DS8 contract + corpus tests (metrics, near-zero exclusion, gates, corpus)."""

from __future__ import annotations

import math
import struct
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
from scripts import deepseek_v4_expert_reference as ds7  # noqa: E402


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


def test_sample_validity_is_reference_only_and_does_not_fail_faithful_candidate() -> None:
    # Three reference near-zeros exceed the unchanged 0.02 coverage cap.  The
    # candidate is bitwise faithful, so numerical candidate fidelity must be
    # reported separately from the out-of-scope sample decision.
    ref = torch.ones(100)
    ref[:3] = 1e-4
    metrics = contract.compute_ds8_metrics(ref, ref.clone())
    report = contract.ds8_gate_report(metrics)
    assert metrics["excluded"]["threshold"] == 0.001
    assert metrics["excluded"]["fraction"] > 0.02
    assert report["sample_validity"]["pass"] is False
    assert report["candidate_fidelity"]["pass"] is True
    assert report["ds8_gate_passed"] is False
    assert contract.candidate_fidelity_passed(metrics) is True
    assert contract.ds8_gate_passed(metrics) is False


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


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_matching_forbidden_nan_or_posinf_fails_explicitly(value: float) -> None:
    # Mask equality alone cannot authorize matching NaN/+inf.  -inf remains
    # the only documented non-finite sentinel allowed on both sides.
    ref = torch.tensor([1.0, value])
    metrics = contract.compute_ds8_metrics(ref, ref.clone())
    report = contract.ds8_gate_report(metrics)
    assert metrics["sentinel_mask_exact"] is True
    assert report["sample_validity"]["checks"]["documented_no_nan_or_posinf"]["pass"] is False
    assert report["candidate_fidelity"]["checks"]["documented_no_nan_or_posinf"]["pass"] is False
    assert contract.candidate_fidelity_passed(metrics) is False
    assert contract.ds8_gate_passed(metrics) is False


def test_matching_neginf_sentinel_remains_allowed() -> None:
    ref = torch.tensor([1.0, float("-inf")])
    metrics = contract.compute_ds8_metrics(ref, ref.clone())
    report = contract.ds8_gate_report(metrics)
    assert report["sample_validity"]["pass"] is True
    assert report["candidate_fidelity"]["pass"] is True
    assert contract.ds8_gate_passed(metrics) is True


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


def test_zero_finite_overlap_gate_fails_cleanly() -> None:
    # All-(-inf) tensors (e.g. indexer_scores before any position is written)
    # must produce cosine None, finite_overlap False and a clean gate FAIL
    # (no TypeError from None >= tol, no NaN -> silent pass).
    ref = torch.full((4, 8), float("-inf"))
    cand = torch.full((4, 8), float("-inf"))
    m = contract.compute_ds8_metrics(ref, cand)
    assert m["cosine_similarity"] is None
    assert m["finite_overlap"] is False
    assert m["sentinel_mask_exact"] is True
    assert contract.ds8_gate_passed(m) is False
    # a finite/-inf mask mismatch is also a clean fail
    m2 = contract.compute_ds8_metrics(ref, torch.zeros(4, 8))
    assert m2["sentinel_mask_exact"] is False
    assert contract.ds8_gate_passed(m2) is False


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


def test_state_mask_analysis_ulp_drift_passes_structural() -> None:
    # DS9 v9: the reference (CPU fp32) and candidate (CUDA fp32) run the SAME
    # module, so cross-device reduction order legitimately drifts state
    # values by 1-7 ULP while every structural mask stays identical.  A tiny
    # value drift with identical masks must PASS the structural gate
    # (ok=True) with first_value_divergence preserved as the first-
    # divergence locator payload; real mask divergence still fails (covered
    # by the sentinel/written-slot tests above).
    ref = {"score": torch.full((2, 4), float("-inf"))}
    cand = torch.full((2, 4), float("-inf"))
    ref["score"][0, 0] = 0.5
    cand[0, 0] = 0.5 + 2.0**-24  # exactly 1 fp32 ULP above 0.5
    out = contract.state_mask_analysis(
        ref, {"score": cand}, init_values={"score": float("-inf")})
    assert out["score"]["sentinel_mask_exact"] is True
    assert out["score"]["written_slot_mask_exact"] is True
    assert out["score"]["untouched_slot_mask_exact"] is True
    assert out["score"]["mask_divergence"] is None
    assert out["score"]["ok"] is True  # v9: structural masks only
    fd = out["score"]["first_value_divergence"]
    assert fd is not None  # locator evidence preserved
    assert fd["flat_index"] == 0
    assert fd["abs_error"] == 2.0**-24
    assert fd["ulp"] == 1
    assert fd["ref_bits"] != fd["cand_bits"]


# ---------------------------------------------------------------------------
# DS9 v10: router-boundary diagnostics
# ---------------------------------------------------------------------------


def test_ulp_tensor_matches_scalar_semantics() -> None:
    a = torch.tensor([1.0, 0.5, -1.0, 1.0 + 2.0**-23, 0.0])
    b = torch.tensor([1.0 + 2.0**-23, 0.5 + 2.0**-24, -1.0, 1.0, -0.0])
    u = contract.ulp_tensor(a, b)
    assert u.tolist() == [1, 1, 0, 1, 0]  # ±0.0 have the same ordered bits


def test_router_stages_match_ds7_router_scores() -> None:
    # The diagnostic pipeline must reproduce the OFFICIAL router bitwise.
    torch.manual_seed(11)
    x = torch.randn(3, 16)
    w = torch.randn(8, 16)
    b = torch.randn(8)
    st = contract.router_stages(x, w, b, topk=6, route_scale=1.5)
    sc, ids, wts = ds7.router_scores(x, w, bias=b, score_func="sqrtsoftplus",
                                     topk=6, route_scale=1.5)
    # ds7 returns the SELECTION scores (bias-shifted); the sqrt stage is the
    # official routed score (unbiased, used for routing weights).  Note
    # (sqrt + b) - b != sqrt in fp, so the unbiased stage is cross-checked
    # via internal consistency (raw -> softplus -> sqrt) and via the routing
    # weights, which ds7 computes from the SAME unbiased sqrt gather.
    assert torch.equal(st["biased"], sc)        # selection scores bitwise
    assert torch.equal(st["ids"], ids)          # selected ids bitwise
    assert torch.equal(st["sqrt"],
                       torch.nn.functional.softplus(st["raw"]).sqrt())
    assert torch.equal(st["weights"], wts)      # weights use the sqrt stage
    assert torch.equal(st["biased"], st["sqrt"] + b)  # bias shifts selection
    # no-bias path: biased == sqrt == ds7 scores
    st2 = contract.router_stages(x, w, None, topk=6)
    sc2, ids2, wts2 = ds7.router_scores(x, w, bias=None,
                                        score_func="sqrtsoftplus",
                                        topk=6, route_scale=1.5)
    assert torch.equal(st2["biased"], st2["sqrt"])
    assert torch.equal(st2["sqrt"], sc2)
    assert torch.equal(st2["ids"], ids2)
    assert torch.equal(st2["weights"], wts2)


def test_router_boundary_metrics_identical_inputs_no_flip() -> None:
    torch.manual_seed(29)
    gw = torch.randn(8, 16)
    gb = torch.randn(8)
    x = torch.randn(2, 16)
    diag = contract.router_boundary_metrics(x, x.clone(), gw, gb, topk=6)
    assert diag["first_flip_token"] is None
    assert diag["input_summary"]["bitwise_exact"] is True
    assert diag["input_summary"]["count_diff"] == 0
    assert diag["input_summary"]["max_ulp"] == 0
    assert diag["stages"] == {}
    assert diag["sensitivity"] == {}


def test_router_boundary_metrics_near_tie_flip_explained() -> None:
    # Construct a rank-6/rank-7 near tie and perturb the input along the
    # margin gradient with 1.5x the linear reverse bound: the diagnostic must
    # find the flip token, report the crossing, and the linear sensitivity
    # estimate must reproduce the observed flip with a dx/min_reverse ratio
    # consistent with the 1.5x overshoot.
    torch.manual_seed(23)
    n_routed, hidden, topk = 8, 16, 6
    gw = torch.randn(n_routed, hidden)
    gb = torch.randn(n_routed)
    x_ref = torch.randn(1, hidden)
    st = contract.router_stages(x_ref, gw, gb, topk=topk, route_scale=1.5)
    sq, biased = st["sqrt"][0], st["biased"][0]
    rvals, ridxs = biased.sort(descending=True)
    o, i = int(ridxs[5]), int(ridxs[6])  # rank-6 / rank-7 experts
    assert float(rvals[5] - rvals[6]) > 0
    raw = st["raw"][0]
    sig = torch.sigmoid(raw)
    g = ((sig[o] / (2.0 * sq[o])) * gw[o]
         - (sig[i] / (2.0 * sq[i])) * gw[i])
    dx = -g / g.norm() * (float(rvals[5] - rvals[6]) / g.norm()) * 1.5
    x_cand = (x_ref + dx).float()
    diag = contract.router_boundary_metrics(x_ref, x_cand, gw, gb, topk=topk)
    assert diag["first_flip_token"] == 0
    assert diag["boundary"]["symmetric_difference"]  # non-empty flip
    ins = diag["input_summary"]
    assert ins["count_diff"] > 0 and ins["max_ulp"] > 0
    sens = diag["sensitivity"]
    assert sens["flip_explained"] is True
    assert sens["tightest_crossing"]["margin_ref"] > 0
    assert sens["tightest_crossing"]["margin_actual"] > 0
    assert sens["tightest_crossing"]["est_flip_matches_actual"] is True
    ratio = sens["dx_vs_min_reverse_ratio"]
    assert ratio is not None and 1.0 < ratio < 4.0
    # stage stats exist and the biased stage carries the flip
    assert diag["stages"]["biased"]["bitwise_exact"] is False
    assert diag["topk_audit"]["num_routed"] == n_routed


def _fp32_next(value: float) -> float:
    """Next fp32 value (1 ULP) in sign-magnitude order (never crosses 0)."""
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if bits == 0x00000000 or bits == 0x80000000:
        return float(value)  # +/-0.0: stay put (no direction to move)
    nbits = bits + 1 if bits < 0x80000000 else bits - 1
    return struct.unpack("<f", struct.pack("<I", nbits))[0]


def test_router_boundary_metrics_reports_ulp_level_input_drift() -> None:
    # A genuine single-element 1-ULP input change must be characterized as
    # ULP-level drift (max_ulp 1, count_diff 1) - the classification depends
    # on that (max_ulp <= 64 -> input-driven, not upstream layout/state).
    torch.manual_seed(31)
    gw = torch.randn(8, 16)
    gb = torch.randn(8)
    x = torch.randn(2, 16)
    x_c = x.clone()
    v = float(x_c[0, 5])
    assert v != 0.0 and math.isfinite(v)
    nxt = _fp32_next(v)
    assert contract.f32_ulp_distance(v, nxt) == 1
    x_c[0, 5] = nxt
    diag = contract.router_boundary_metrics(x, x_c, gw, gb, topk=6)
    ins = diag["input_summary"]
    assert ins["count_diff"] == 1
    assert ins["max_ulp"] == 1
    assert ins["max_abs"] == abs(v - nxt)


def test_router_topk_audit_detects_boundary_tie() -> None:
    scores = torch.tensor([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1],
                          dtype=torch.float32)
    audit = contract._topk_audit(scores, topk=6)
    assert audit["boundary_tie_rank6_vs_rank7"] is True
    assert [5, 6] in audit["exact_tie_groups_top10"]
    scores2 = torch.tensor([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
                           dtype=torch.float32)
    audit2 = contract._topk_audit(scores2, topk=6)
    assert audit2["boundary_tie_rank6_vs_rank7"] is False
    assert audit2["exact_tie_groups_top10"] == []
    assert audit2["sort_topk_ids_agree"] is True


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


def test_bf16_storage_bound_rounding_vs_structural() -> None:
    rng = torch.Generator().manual_seed(11)
    ref = (torch.randn(4096, generator=rng) * 0.5).clamp(-1.5, 1.5)
    # ~5% of elements shifted by exactly one bf16 ulp at the tensor scale
    # (2**-7 for magnitude in [1, 2)) -> rounding drift stays within bound.
    cand = ref.clone()
    idx = torch.randperm(4096, generator=rng)[:200]
    cand[idx] = cand[idx] + 0.0078125
    b = contract.bf16_storage_bound(ref, cand)
    assert b["within_bf16_storage_bound"] is True
    assert b["max_abs_in_bf16_steps"] <= 2.0
    assert b["fraction_within_one_bf16_step"] >= 0.9
    # near-zero element: large RELATIVE shift, tiny ABSOLUTE shift -> the
    # absolute bound at the tensor scale still holds (no structural signal).
    ref_near = ref.clone()
    ref_near[0] = 0.0005
    cand_near = ref_near.clone()
    cand_near[0] = 0.0015
    bn = contract.bf16_storage_bound(ref_near, cand_near)
    assert bn["within_bf16_storage_bound"] is True
    # structural: sign flips on the 50 largest-magnitude elements move values
    # on the order of the full range.
    top_idx = torch.topk(ref.abs(), 50).indices
    bad = ref.clone()
    bad[top_idx] = bad[top_idx] * -1.0
    bb = contract.bf16_storage_bound(ref, bad)
    assert bb["within_bf16_storage_bound"] is False
    assert bb["max_abs_in_bf16_steps"] >= 64.0
    # fraction discriminator: diffs of 4 bf16 grid steps on 85% of elements.
    # max_abs_steps sits at the 4.0 threshold (within fp32 representation), so
    # ONLY the fraction gate can reject this -> isolates the fraction gate.
    many = ref.clone()
    many[torch.randperm(4096, generator=rng)[:3500]] += 0.03125
    bm = contract.bf16_storage_bound(ref, many)
    assert bm["max_abs_in_bf16_steps"] <= 4.01
    assert bm["fraction_within_one_bf16_step"] < 0.9
    assert bm["within_bf16_storage_bound"] is False


def test_expert_ids_set_vs_tuple_exact() -> None:
    # v10/v11 proven case: identical SET, intra-set rank swap at token 4.
    a = torch.tensor([[6, 30, 78, 102, 198, 214]], dtype=torch.int64)
    b = torch.tensor([[6, 30, 78, 198, 102, 214]], dtype=torch.int64)
    set_eq, tuple_eq = contract.expert_ids_set_exact(a, b)
    assert set_eq is True
    assert tuple_eq is False
    # a genuine selection difference must fail the set gate.
    c = torch.tensor([[6, 30, 78, 102, 198, 215]], dtype=torch.int64)
    set_eq2, tuple_eq2 = contract.expert_ids_set_exact(a, c)
    assert set_eq2 is False
    assert tuple_eq2 is False
    # exact match passes both.
    set_eq3, tuple_eq3 = contract.expert_ids_set_exact(a, a)
    assert set_eq3 is True and tuple_eq3 is True


def _iso(topk_ok: bool = True, ref_ok: bool = True, cand_ok: bool = True,
         ref_ids: list[int] | None = None,
         cand_ids: list[int] | None = None) -> dict:
    ok = [True] * 16
    if not topk_ok:
        ok[4] = False
    ref_in_ok = [True] * 16
    cand_in_ok = [True] * 16
    if not ref_ok:
        ref_in_ok[4] = False
    if not cand_ok:
        cand_in_ok[4] = False
    return {
        "topk_same_scores_cpu_vs_cuda_rowwise": ok,
        "ref_in_ids_cpu_vs_cuda_rowwise": ref_in_ok,
        "cand_in_ids_cpu_vs_cuda_rowwise": cand_in_ok,
        "ref_in_cpu_ids": ref_ids if ref_ids is not None
        else [6, 30, 78, 102, 198, 214],
        "cand_in_cpu_ids": cand_ids if cand_ids is not None
        else [6, 30, 78, 198, 102, 214],
    }


def _diag(*, within: bool = True, sym_diff: list[int] | None = None,
          count_diff: int = 3595) -> dict:
    return {
        "first_flip_token": 4,
        "input_summary": {
            "count_diff": count_diff,
            "max_ulp": 264634367,
            "max_abs": 0.015625,
            "bf16_storage_bound": {
                "max_abs": 0.015625,
                "max_magnitude": 1.84375,
                "bf16_ulp_at_max_magnitude": 0.0078125,
                "max_abs_in_bf16_steps": 2.0,
                "fraction_within_one_bf16_step": 0.97,
                "within_bf16_storage_bound": within,
            },
        },
        "boundary": {
            "symmetric_difference": sym_diff if sym_diff is not None else [],
            "margin_ref": 0.0153,
            "margin_cand": 0.0146,
        },
        "sensitivity": {"flip_explained": False},
    }


def test_router_diagnosis_classify_ordering_within_set() -> None:
    out = contract.router_diagnosis_classify(_diag(), _iso())
    assert out["verdict"] == "ROUTER_IMPLEMENTATION_EXACT_INPUT_DRIVEN_FLIP"
    assert out["flip_scope"] == "ORDERING_WITHIN_SET"
    assert out["sets_identical"] is True
    assert out["flip_reproduced_by_input"] is True
    assert out["flip_token"] == 4


def test_router_diagnosis_classify_selection_flip_bounded() -> None:
    out = contract.router_diagnosis_classify(_diag(sym_diff=[198]), _iso())
    assert out["verdict"] == "ROUTER_IMPLEMENTATION_EXACT_INPUT_DRIVEN_FLIP"
    assert out["flip_scope"] == "SELECTION_FLIP"
    assert out["sets_identical"] is False


def test_router_diagnosis_classify_unfaithful_capture_reject() -> None:
    # The campaign's real structural signal: the isolation capture does not
    # reproduce the device path -> layout/lifetime/transfer/stale data.
    iso = _iso()
    iso["captured_ref_matches_cpu_recompute"] = False
    out = contract.router_diagnosis_classify(_diag(), iso)
    assert out["verdict"] == "REJECT_UPSTREAM_LAYOUT_OR_STATE"
    assert "unfaithful" in " ".join(out["reasons"])


def test_router_diagnosis_classify_structural_reject() -> None:
    # Defensive branch: input delta not storage-bounded AND the CPU router
    # does not reproduce the flip (isolation/diag inconsistency) -> the
    # operative input/path differs -> layout/lifetime/transfer.
    out = contract.router_diagnosis_classify(
        _diag(within=False, sym_diff=[198]),
        _iso(ref_ids=[6, 30, 78, 102, 198, 214],
             cand_ids=[6, 30, 78, 102, 198, 214]))
    assert out["verdict"] == "REJECT_UPSTREAM_LAYOUT_OR_STATE"


def test_router_diagnosis_classify_router_implementation_reject() -> None:
    out = contract.router_diagnosis_classify(_diag(), _iso(cand_ok=False))
    assert out["verdict"] == "REJECT_ROUTER_IMPLEMENTATION"


def test_router_diagnosis_classify_topk_semantics_reject() -> None:
    out = contract.router_diagnosis_classify(_diag(), _iso(topk_ok=False))
    assert out["verdict"] == "REJECT_TOPK_SEMANTICS"


def test_router_diagnosis_classify_identical_inputs_invalid() -> None:
    out = contract.router_diagnosis_classify(_diag(count_diff=0), _iso())
    assert out["verdict"] == "INVALID_EXPERIMENT"


def test_router_diagnosis_classify_anomalous_but_input_driven() -> None:
    # Not storage-bounded yet the flip is reproduced by the captured inputs:
    # still an input-driven flip (isolation rules out structural), flagged.
    out = contract.router_diagnosis_classify(_diag(within=False), _iso())
    assert out["verdict"] == "ROUTER_IMPLEMENTATION_EXACT_INPUT_DRIVEN_FLIP"
    assert "exceeds" in " ".join(out["reasons"])
