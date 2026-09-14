"""DS9 v13 expert-integration audit tests (CPU, no checkpoint needed).

Verifies the audit engine that decomposes the moe_out/shared_out p99 tail:

1. ``candidate_experts_replay`` is bitwise-identical to the production
   ``DeepseekV4CacheFfn._run_experts`` path for the same inputs.
2. The dequantized-FP32 replay (``fp32_experts_replay``) is bitwise-identical
   to the trusted CPU-FP32 ``moe_layer_forward`` reference on the same input.
3. FP16 storage representability: FP4 grid x e8m0 scales and E4M3 values are
   exactly representable in FP16 (so weight storage is NOT a precision source).
4. Accumulation-order audit (group vs sorted-eid) is fp32-rounding-bounded.
5. Tail locator reports over-gate elements with per-expert cancellation.
6. ``expert_integration_classify`` maps each predeclared cause branch.
7. Clone independence: the audit never mutates its inputs (DS9 v9 lesson).
8. End-to-end ``run_step_audit`` on a synthetic layer reproduces the layer's
   captured moe_out/shared_out bitwise (capture fidelity holds) -- this is the
   same fidelity gate the remote harness depends on.
"""

from __future__ import annotations

import copy
import json

import torch

from scripts import deepseek_v4_expert_audit as v4audit
from scripts import deepseek_v4_moe_reference as moe
from scripts import deepseek_v4_expert_reference as ds7
from scripts import deepseek_v4_contract as v4contract
from scripts.deepseek_v4_cache import DeepSeekExpertCache, DeepSeekExpertLoader
from scripts.deepseek_v4_layer_reference import (
    DeepseekV4Layer,
    LayerConfig,
    make_synthetic_layer_weights,
)
from scripts.deepseek_v4_layer_candidate import (
    build_fp16_payloads,
    build_shared_fp16_payload,
    make_candidate_layer,
)

# Same small synthetic geometry as test_deepseek_v4_layer.CFG so every
# quantization block divides exactly.
CFG = LayerConfig(
    hidden=64,
    n_heads=4,
    head_dim=128,
    rope_head_dim=64,
    q_lora_rank=32,
    o_lora_rank=32,
    o_groups=2,
    window_size=8,
    compress_ratio=4,
    index_n_heads=2,
    index_head_dim=128,
    index_topk=8,
    n_routed=16,
    topk=2,
    route_scale=1.5,
    swiglu_limit=10.0,
    norm_eps=1e-6,
    hc_mult=2,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_seq_len=32,
)


def _build_fixture(seed: int = 3):
    torch.manual_seed(seed)
    w, routed_raw, shared_raw = make_synthetic_layer_weights(
        CFG, seed=seed, n_experts=8)
    cache = DeepSeekExpertCache(1 << 30, device="cpu")
    loader = DeepSeekExpertLoader(cache)
    fp16 = build_fp16_payloads(routed_raw)
    shared_payload = build_shared_fp16_payload(shared_raw)
    ref_layer = DeepseekV4Layer(CFG, w, device="cpu", max_batch=1)
    cand_layer = make_candidate_layer(
        CFG, w, device="cpu", max_batch=1, cache=cache, loader=loader,
        layer_id=20, fp16_payloads=fp16, shared_payload=shared_payload)
    return w, routed_raw, shared_raw, fp16, shared_payload, cache, loader, \
        ref_layer, cand_layer


def _run_layer(layer, x, start_pos):
    cap: dict = {}
    out = layer.forward(x, start_pos, capture=cap)
    return out, cap


# ---------------------------------------------------------------------------
# 1. replay == production bitwise
# ---------------------------------------------------------------------------


def test_replay_reproduces_production_bitwise() -> None:
    w, routed_raw, shared_raw, fp16, shared_payload, cache, loader, \
        ref_layer, cand_layer = _build_fixture(seed=3)
    x = torch.randn(1, 8, CFG.hc_mult, CFG.hidden, dtype=torch.bfloat16)
    _, cc = _run_layer(cand_layer, x, 0)
    xf = cc["ffn_norm_out"].float().reshape(-1, CFG.hidden)
    ids, wts = cc["expert_ids"], cc["routing_weights"]

    replay = v4audit.candidate_experts_replay(
        xf, ids, wts, topk=CFG.topk, swiglu_limit=CFG.swiglu_limit,
        device="cpu", payload_for=lambda e: fp16[e],
        shared_payload_for=lambda: shared_payload)
    # production combined (routed+shared) == replay combined; shared == shared
    assert torch.equal(replay["combined"],
                       cc["moe_out"].reshape(-1, CFG.hidden))
    assert torch.equal(replay["shared_out"],
                       cc["shared_out"].reshape(-1, CFG.hidden))
    # per-expert buckets reconstruct the routed sum in group order
    routed_sum = torch.zeros_like(replay["combined"])
    for eid in replay["groups_order"]:
        routed_sum += replay["per_expert"][eid]
    assert torch.equal(routed_sum + replay["shared_out"], replay["combined"])


# ---------------------------------------------------------------------------
# 2. fp32 replay == trusted reference bitwise
# ---------------------------------------------------------------------------


def test_fp32_replay_matches_reference_bitwise() -> None:
    w, routed_raw, shared_raw, fp16, shared_payload, cache, loader, \
        ref_layer, cand_layer = _build_fixture(seed=3)
    x = torch.randn(1, 8, CFG.hc_mult, CFG.hidden, dtype=torch.bfloat16)
    _, rc = _run_layer(ref_layer, x, 0)
    x_ref = rc["ffn_norm_out"].float().reshape(-1, CFG.hidden)
    ids, wts = rc["expert_ids"], rc["routing_weights"]
    gw = w["ffn"]["gate_w"]
    gb = w["ffn"]["gate_b"]

    A = moe.moe_layer_forward(x_ref, gw, gb, routed_raw, shared_raw,
                              topk=CFG.topk, route_scale=CFG.route_scale,
                              score_func="sqrtsoftplus",
                              swiglu_limit=CFG.swiglu_limit,
                              keep_per_expert=True)

    def weight_for(eid):
        t = routed_raw[eid]
        return {k: ds7.dequantize_expert_weight(t[k], t[f"{k[:-7]}.scale"])
                for k in ("w1.weight", "w2.weight", "w3.weight")}

    def shared_weight_for():
        return {k: moe.dequantize_fp8_e4m3(shared_raw[k],
                                           shared_raw[f"{k[:-7]}.scale"])
                for k in ("w1.weight", "w2.weight", "w3.weight")}

    VB = v4audit.fp32_experts_replay(
        x_ref, ids, wts, topk=CFG.topk, swiglu_limit=CFG.swiglu_limit,
        device="cpu", weight_for=weight_for,
        shared_weight_for=shared_weight_for)
    assert torch.equal(VB["combined"], A["moe_output"])
    assert torch.equal(VB["shared_out"], A["shared_output"])


# ---------------------------------------------------------------------------
# 3. FP16 weight storage is exact for realistic FP4/FP8 dequant
# ---------------------------------------------------------------------------


def test_weight_storage_exact_fp16() -> None:
    w, routed_raw, shared_raw, fp16, shared_payload, cache, loader, \
        ref_layer, cand_layer = _build_fixture(seed=3)
    check = v4audit._weight_storage_check(
        sorted(routed_raw), routed_raw, fp16, shared_raw, shared_payload)
    assert check["all_exact"], json.dumps(check, indent=1)[:2000]
    assert all(v for v in check["shared"].values())
    assert all(v for v in check["routed"].values())


# ---------------------------------------------------------------------------
# 4. accumulation-order audit is fp32-rounding-bounded
# ---------------------------------------------------------------------------


def test_accumulation_order_bounded() -> None:
    w, routed_raw, shared_raw, fp16, shared_payload, cache, loader, \
        ref_layer, cand_layer = _build_fixture(seed=3)
    x = torch.randn(1, 8, CFG.hc_mult, CFG.hidden, dtype=torch.bfloat16)
    _, cc = _run_layer(cand_layer, x, 0)
    xf = cc["ffn_norm_out"].float().reshape(-1, CFG.hidden)
    ids, wts = cc["expert_ids"], cc["routing_weights"]
    replay = v4audit.candidate_experts_replay(
        xf, ids, wts, topk=CFG.topk, swiglu_limit=CFG.swiglu_limit,
        device="cpu", payload_for=lambda e: fp16[e],
        shared_payload_for=lambda: shared_payload)
    order = v4audit._order_audit(replay["combined"], replay["combined_sorted"])
    # group vs sorted-eid fp32 accumulation: fp32 rounding at tensor scale
    assert order["max_rel"] < 1e-4
    assert isinstance(order["bitwise"], bool)


# ---------------------------------------------------------------------------
# 5. tail locator
# ---------------------------------------------------------------------------


def test_tail_locator_reports_cancellation() -> None:
    w, routed_raw, shared_raw, fp16, shared_payload, cache, loader, \
        ref_layer, cand_layer = _build_fixture(seed=3)
    x = torch.randn(1, 8, CFG.hc_mult, CFG.hidden, dtype=torch.bfloat16)
    _, rc = _run_layer(ref_layer, x, 0)
    _, cc = _run_layer(cand_layer, x, 0)
    # reference per-expert buckets from the trusted recompute
    A = moe.moe_layer_forward(
        rc["ffn_norm_out"].float().reshape(-1, CFG.hidden),
        w["ffn"]["gate_w"], w["ffn"]["gate_b"], routed_raw, shared_raw,
        topk=CFG.topk, route_scale=CFG.route_scale, score_func="sqrtsoftplus",
        swiglu_limit=CFG.swiglu_limit, keep_per_expert=True)
    # candidate per-expert buckets from a replay for the breakdown
    xf = cc["ffn_norm_out"].float().reshape(-1, CFG.hidden)
    replay = v4audit.candidate_experts_replay(
        xf, cc["expert_ids"], cc["routing_weights"], topk=CFG.topk,
        swiglu_limit=CFG.swiglu_limit, device="cpu",
        payload_for=lambda e: fp16[e], shared_payload_for=lambda: shared_payload)
    tail = v4audit.tail_locator(
        rc["moe_out"], cc["moe_out"], A["per_expert"], replay["per_expert"],
        A["shared_output"], replay["shared_out"], label="moe_out",
        gate=0.05, hidden=CFG.hidden)
    assert tail["label"] == "moe_out"
    assert tail["numel"] == 8 * CFG.hidden
    if tail["worst_over"] is not None:
        worst = tail["worst_over"]
        assert set(worst) >= {"flat", "tok", "dim", "per_expert", "shared",
                              "cancellation", "sum_consistency"}
        # contributions must be self-consistent (ref sum == total value)
        assert worst["sum_consistency"]["ref_sum_matches_total"]
        assert worst["sum_consistency"]["cand_sum_matches_total"]


# ---------------------------------------------------------------------------
# 6. classifier branches (predeclared rules)
# ---------------------------------------------------------------------------


def _audit_with(steps: dict) -> dict:
    return {"steps": steps, "n_steps": len(steps)}


def _step(overrides: dict | None = None) -> dict:
    base = {
        "capture_faithful": True,
        "fidelity": {"ref_recompute_matches_capture": True,
                     "ref_shared_matches_capture": True,
                     "cand_replay_matches_capture": True,
                     "cand_shared_matches_capture": True},
        "weight_storage": {"all_exact": True, "routed": {}, "shared": {}},
        "headline": {"moe_out_p99": 0.08, "shared_out_p99": 0.07},
        "matrix": {
            "fp32exec_cand_input": {"p99": 0.0001, "cosine": 1.0},
            "kernel_ref_input": {"p99": 0.06, "cosine": 0.9999},
            "ref_input_sensitivity": {"p99": 0.001, "cosine": 1.0},
            "routing_substitution": {"p99": 0.001, "cosine": 1.0},
            "shared_isolated": {"p99": 0.065, "cosine": 0.9999},
            "routed_only": {"p99": 0.06, "cosine": 0.9999},
        },
        "accumulation_order": {"candidate": {"max_rel": 1e-7},
                               "reference": {"max_rel": 1e-7}},
    }
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v
        if "fidelity" in overrides:
            base["capture_faithful"] = bool(all(base["fidelity"].values()))
    return base


def test_classifier_fp16_execution_precision() -> None:
    v = v4contract.expert_integration_classify(_audit_with({"step0": _step()}))
    assert v["verdict"] == "FP16_EXECUTION_PRECISION"
    assert v["steps"]["step0"]["summary"]["fp32_exec_p99"] <= 0.01


def test_classifier_capture_defect() -> None:
    step = _step({"fidelity": {"ref_recompute_matches_capture": False}})
    v = v4contract.expert_integration_classify(_audit_with({"step0": step}))
    assert v["verdict"] == "HARNESS_CAPTURE_DEFECT"


def test_classifier_fp4_unpack_or_scale() -> None:
    step = _step({"weight_storage": {"all_exact": False,
                                     "routed": {"3:w1.weight": False},
                                     "shared": {}}})
    v = v4contract.expert_integration_classify(_audit_with({"step0": step}))
    assert v["verdict"] == "FP4_UNPACK_OR_SCALE"


def test_classifier_shared_semantics() -> None:
    step = _step({"weight_storage": {"all_exact": False,
                                     "routed": {},
                                     "shared": {"w1.weight": False}}})
    v = v4contract.expert_integration_classify(_audit_with({"step0": step}))
    assert v["verdict"] == "SHARED_EXPERT_SEMANTICS"


def test_classifier_shared_only_fails() -> None:
    step = _step({"matrix": {"fp32exec_cand_input": {"p99": 0.02},
                             "shared_isolated": {"p99": 0.08},
                             "routed_only": {"p99": 0.01}}})
    v = v4contract.expert_integration_classify(_audit_with({"step0": step}))
    assert v["verdict"] == "SHARED_EXPERT_SEMANTICS"


def test_classifier_input_distribution() -> None:
    step = _step({"matrix": {"fp32exec_cand_input": {"p99": 0.02},
                             "ref_input_sensitivity": {"p99": 0.09},
                             "routed_only": {"p99": 0.08}}})
    v = v4contract.expert_integration_classify(_audit_with({"step0": step}))
    assert v["verdict"] == "INTEGRATED_INPUT_DISTRIBUTION"


def test_classifier_unobserved_when_within_gate() -> None:
    step = _step({"headline": {"moe_out_p99": 0.01, "shared_out_p99": 0.02}})
    v = v4contract.expert_integration_classify(_audit_with({"step0": step}))
    assert v["verdict"] == "NO_TAIL_OBSERVED"


def test_classifier_run_level_worst_is_most_severe() -> None:
    # fail-closed: one defective step must dominate the run-level verdict
    bad = _step({"fidelity": {"ref_recompute_matches_capture": False}})
    clean = _step({"headline": {"moe_out_p99": 0.01, "shared_out_p99": 0.02}})
    v = v4contract.expert_integration_classify(
        _audit_with({"step0": bad, "step16": clean}))
    assert v["verdict"] == "HARNESS_CAPTURE_DEFECT", v
    # and the reverse ordering is also dominated by the defective step
    v2 = v4contract.expert_integration_classify(
        _audit_with({"step0": clean, "step16": bad}))
    assert v2["verdict"] == "HARNESS_CAPTURE_DEFECT", v2


# ---------------------------------------------------------------------------
# 7. clone independence (the audit never mutates its inputs)
# ---------------------------------------------------------------------------


def test_run_step_audit_clone_independence() -> None:
    w, routed_raw, shared_raw, fp16, shared_payload, cache, loader, \
        ref_layer, cand_layer = _build_fixture(seed=5)
    x = torch.randn(1, 8, CFG.hc_mult, CFG.hidden, dtype=torch.bfloat16)
    _, rc = _run_layer(ref_layer, x, 0)
    _, cc = _run_layer(cand_layer, x, 0)

    def snap(d):
        def clone_tree(v):
            if isinstance(v, torch.Tensor):
                return v.clone()
            if isinstance(v, dict):
                return {k: clone_tree(x) for k, x in v.items()}
            return copy.deepcopy(v)
        return {k: clone_tree(v) for k, v in d.items()}

    rc_pre, cc_pre = snap(rc), snap(cc)

    def run() -> dict:
        return v4audit.run_step_audit(
            x_ref=rc["ffn_norm_out"].float().reshape(-1, CFG.hidden),
            x_cand=cc["ffn_norm_out"].float().reshape(-1, CFG.hidden),
            ids_ref=rc["expert_ids"], wts_ref=rc["routing_weights"],
            ids_cand=cc["expert_ids"], wts_cand=cc["routing_weights"],
            ref_cap_moe=rc["moe_out"], ref_cap_shared=rc["shared_out"],
            cand_cap_moe=cc["moe_out"], cand_cap_shared=cc["shared_out"],
            routed_raw=routed_raw, shared_raw=shared_raw,
            fp16_payloads=fp16, shared_payload=shared_payload,
            gate_w=w["ffn"]["gate_w"], gate_b=w["ffn"]["gate_b"],
            cfg=CFG, device="cpu", cache=cache, loader=loader, layer_id=20)

    def tree_equal(a, b, path=""):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(
                tree_equal(a[k], b[k], f"{path}.{k}") for k in a)
        return a == b

    a1 = run()
    # inputs must be unmodified by the first run (recursive: captures nest)
    for k in rc_pre:
        assert tree_equal(rc_pre[k], rc[k]), f"ref capture {k} mutated"
    for k in cc_pre:
        assert tree_equal(cc_pre[k], cc[k]), f"cand capture {k} mutated"
    a2 = run()
    assert a1 == a2


# ---------------------------------------------------------------------------
# 8. end-to-end run_step_audit: capture fidelity holds on a real layer
# ---------------------------------------------------------------------------


def test_run_step_audit_end_to_end_fidelity() -> None:
    w, routed_raw, shared_raw, fp16, shared_payload, cache, loader, \
        ref_layer, cand_layer = _build_fixture(seed=7)
    x = torch.randn(1, 8, CFG.hc_mult, CFG.hidden, dtype=torch.bfloat16)
    _, rc = _run_layer(ref_layer, x, 0)
    _, cc = _run_layer(cand_layer, x, 0)

    step = v4audit.run_step_audit(
        x_ref=rc["ffn_norm_out"].float().reshape(-1, CFG.hidden),
        x_cand=cc["ffn_norm_out"].float().reshape(-1, CFG.hidden),
        ids_ref=rc["expert_ids"], wts_ref=rc["routing_weights"],
        ids_cand=cc["expert_ids"], wts_cand=cc["routing_weights"],
        ref_cap_moe=rc["moe_out"], ref_cap_shared=rc["shared_out"],
        cand_cap_moe=cc["moe_out"], cand_cap_shared=cc["shared_out"],
        routed_raw=routed_raw, shared_raw=shared_raw,
        fp16_payloads=fp16, shared_payload=shared_payload,
        gate_w=w["ffn"]["gate_w"], gate_b=w["ffn"]["gate_b"],
        cfg=CFG, device="cpu", cache=cache, loader=loader, layer_id=20)

    # the replay must bitwise reproduce the layer's captured outputs
    assert step["capture_faithful"], json.dumps(step["fidelity"])
    assert all(step["fidelity"].values())
    # weight storage exact (FP4/FP8 -> FP16 lossless)
    assert step["weight_storage"]["all_exact"]
    # matrix + per-expert + tail + order artifacts present
    assert set(step["matrix"]) >= {"kernel_ref_input", "fp32exec_cand_input",
                                   "ref_input_sensitivity",
                                   "routing_substitution", "shared_isolated",
                                   "routed_only"}
    assert "served_counts" in step["per_expert"]
    assert set(step["tail"]) == {"moe_out", "shared_out"}
    # the whole step dict is JSON-safe
    json.dumps(step)
    verdict = v4contract.expert_integration_classify(
        {"steps": {"step0": step}})
    assert verdict["verdict"] in (
        "FP16_EXECUTION_PRECISION", "NO_TAIL_OBSERVED",
        "MULTIPLE_PROVEN_CAUSES", "SHARED_EXPERT_SEMANTICS",
        "ROUTING_WEIGHT_ERROR", "INTEGRATED_INPUT_DISTRIBUTION")
