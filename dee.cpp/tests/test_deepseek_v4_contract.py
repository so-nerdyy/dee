"""DS8 contract + corpus tests (metrics, near-zero exclusion, gates, corpus)."""

from __future__ import annotations

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
