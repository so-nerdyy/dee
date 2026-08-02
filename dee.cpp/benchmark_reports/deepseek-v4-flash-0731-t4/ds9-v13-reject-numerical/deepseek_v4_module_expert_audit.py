"""DS9 v13 expert-integration audit: decompose the moe_out/shared_out p99 tail.

v12 proved the only remaining DS9 failures are ``moe_out`` (p99 ~0.074-0.099)
and ``shared_out`` (p99 ~0.068-0.082) against the sealed 0.05 gate, at BOTH
step 0 (prefill) and step 16 (decode), with the router proven exact and the
expert-ID sets identical.  This module performs the focused audit that
distinguishes among:

  1. integrated input drift        6. shared-expert integration error
  2. routing-weight drift          7. routed-plus-shared combination error
  3. FP4 unpack/dequantization     8. harness/reference mismatch
  4. FP16 execution precision
  5. accumulation-order error

Core idea: run the SAME official MoE math in several controlled
configurations over the exact device-produced inputs captured by the layer:

  A   = trusted CPU-FP32 reference on the reference FFN input (rc ffn_norm_out)
  A1  = trusted CPU-FP32 reference on the CANDIDATE FFN input (input drift in
        the FP32 world alone)
  B   = candidate FP16 kernels on the REFERENCE input + reference routing
        (kernel precision on identical input/routing: the DS8-replay analog)
  C   = candidate FP16 kernels on the CANDIDATE input + candidate routing
        (the integrated path; must reproduce the captured cc moe_out/shared_out
        bitwise -- the harness-validity / capture-fidelity gate)
  D   = candidate FP16 kernels on candidate input + REFERENCE routing
        (routing-weight substitution on the candidate input)
  E   = candidate FP16 kernels on reference input + CANDIDATE routing
  V_B = dequantized-FP32 weights + FP32 compute on the candidate input
        (same math as A but on CUDA: isolates FP16 execution precision from
        FP4/FP8 storage -- the dequant is shared with A, so any residual
        difference is compute precision)

Phase 4 cross-substitution: A1, D, E.  Phase 5: packed-byte identity +
exact FP16 storage representability (FP4 grid x e8m0 power-of-two scales and
E4M3 values are both exactly representable in FP16).  Phase 6: V_A
(production) vs V_B (fp32 exec) vs A.  Phase 7: group-order vs sorted-eid
accumulation on both sides.  Phase 8: p99-tail locator with per-expert
cancellation at the worst element.  Phase 9: capture-fidelity checks (the
replays must bitwise reproduce the captured tensors) and clone independence
(the audit never mutates its inputs).

``candidate_experts_replay`` replicates ``DeepseekV4CacheFfn._run_experts``
op-for-op (same grouping order, same fp16 cast points, same clamps, same
weight-before-w2 placement, same fp32 accumulator) so the per-expert capture
is device-authentic and bitwise-identical to production for the same inputs.

Every returned value is JSON-safe (scalars / lists / dicts of primitives,
tensor hashes, bounded element samples).  Full tensors are never written to
the evidence.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Optional

import torch

from scripts import deepseek_v4_contract as v4contract
from scripts import deepseek_v4_expert_reference as ds7
from scripts import deepseek_v4_moe_reference as moe

NEAR_ZERO = v4contract.NEAR_ZERO_THRESHOLD


def _f32_bits(value: float) -> str:
    import struct
    return format(struct.unpack("<I", struct.pack("<f", float(value)))[0], "08x")


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Shape-checked DS8 metrics (finite-intersection aware)."""
    return v4contract.compute_ds8_metrics(a, b)


def _p99(m: dict[str, Any]) -> float:
    return float(m["non_near_zero"]["p99_rel_error"])


def _p95(m: dict[str, Any]) -> float:
    return float(m["non_near_zero"]["p95_rel_error"])


def group_tokens(ids: torch.Tensor, weights: torch.Tensor,
                 topk: int) -> dict[int, list[tuple[int, float]]]:
    """Token-major, position-major grouping (identical to production)."""
    groups: dict[int, list[tuple[int, float]]] = {}
    for tok in range(int(ids.shape[0])):
        for pos in range(topk):
            eid = int(ids[tok, pos])
            w = float(weights[tok, pos])
            if w == 0.0:
                continue
            groups.setdefault(eid, []).append((tok, w))
    return groups


def candidate_experts_replay(
    xf: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    topk: int,
    swiglu_limit: float,
    device: str,
    payload_for: Callable[[int], dict[str, torch.Tensor]],
    shared_payload_for: Callable[[], dict[str, torch.Tensor]],
    leader_eid: Optional[int] = None,
) -> dict[str, Any]:
    """Bitwise replication of ``DeepseekV4CacheFfn._run_experts`` with capture.

    ``xf``: [n, hidden] FP32 (the pre-cast FFN input).  ``payload_for(eid)``
    returns the resident FP16 payload dict (w1/w2/w3.weight) on ``device``;
    ``shared_payload_for()`` the shared FP16 payload.  Returns:

      per_expert        dict[eid -> [n, hidden] FP32 weighted output]
      shared_out        [n, hidden] FP32 shared-only contribution
      combined          routed group-order sum + shared (production order)
      combined_sorted   routed sum accumulated in sorted-eid order + shared
      groups_order      eid insertion order (production order)
      leader            stage capture for ``leader_eid`` (gate/up/h/h_pre/out
                        as CPU tensors + served token ids), or None
    """
    n, d = xf.shape
    groups = group_tokens(ids, weights, topk)
    moe = torch.zeros(n, d, dtype=torch.float32, device=device)
    per_expert: dict[int, torch.Tensor] = {}
    leader: dict[str, Any] | None = None
    for eid, pairs in groups.items():
        payload = payload_for(eid)
        toks = [pair[0] for pair in pairs]
        ws = torch.tensor([[pair[1]] for pair in pairs],
                          dtype=torch.float32).reshape(-1, 1)
        # production cast points: fp16 on host, then to device
        xc = xf[toks].half().to(device)
        gate = torch.clamp((xc @ payload["w1.weight"].t()).float(),
                           max=swiglu_limit)
        up = torch.clamp((xc @ payload["w3.weight"].t()).float(),
                         min=-swiglu_limit, max=swiglu_limit)
        h_pre = torch.nn.functional.silu(gate) * up
        h = ws.to(device) * h_pre  # official: weight before w2
        out = (h.half() @ payload["w2.weight"].t()).float()
        moe[toks] += out
        bucket = per_expert.setdefault(
            eid, torch.zeros(n, d, dtype=torch.float32, device=device))
        bucket[toks] += out
        if eid == leader_eid:
            leader = {
                "eid": int(eid),
                "toks": toks,
                "xc": xc.detach().cpu(),
                "gate": gate.detach().cpu(),
                "up": up.detach().cpu(),
                "h_pre": h_pre.detach().cpu(),
                "h": h.detach().cpu(),
                "out": out.detach().cpu(),
            }
    sp = shared_payload_for()
    xc = xf.half().to(device)
    gate = torch.clamp((xc @ sp["w1.weight"].t()).float(), max=swiglu_limit)
    up = torch.clamp((xc @ sp["w3.weight"].t()).float(),
                     min=-swiglu_limit, max=swiglu_limit)
    h = torch.nn.functional.silu(gate) * up
    shared_out = (h.half() @ sp["w2.weight"].t()).float()
    combined = moe + shared_out
    moe_sorted = torch.zeros(n, d, dtype=torch.float32, device=device)
    for eid in sorted(groups):
        moe_sorted += per_expert[eid]
    combined_sorted = moe_sorted + shared_out
    return {
        "per_expert": per_expert,
        "shared_out": shared_out,
        "combined": combined,
        "combined_sorted": combined_sorted,
        "groups_order": list(groups.keys()),
        "leader": leader,
    }


def fp32_experts_replay(
    xf: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    topk: int,
    swiglu_limit: float,
    device: str,
    weight_for: Callable[[int], dict[str, torch.Tensor]],
    shared_weight_for: Callable[[], dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Dequantized-FP32 weights + FP32 compute (no FP16 anywhere).

    The exact reference math (A) executed on the candidate device: isolates
    FP16 execution precision from FP4/FP8 storage (the dequant is the same
    function the reference uses).  Returns per_expert/shared_out/combined
    like the candidate replay.
    """
    n, d = xf.shape
    groups = group_tokens(ids, weights, topk)
    moe = torch.zeros(n, d, dtype=torch.float32, device=device)
    per_expert: dict[int, torch.Tensor] = {}
    for eid, pairs in groups.items():
        wts = weight_for(eid)
        toks = [pair[0] for pair in pairs]
        ws = torch.tensor([[pair[1]] for pair in pairs],
                          dtype=torch.float32).reshape(-1, 1).to(device)
        xc = xf[toks].float().to(device)
        gate = torch.clamp(xc @ wts["w1.weight"].t(), max=swiglu_limit)
        up = torch.clamp(xc @ wts["w3.weight"].t(),
                         min=-swiglu_limit, max=swiglu_limit)
        h = ws * (torch.nn.functional.silu(gate) * up)
        out = h @ wts["w2.weight"].t()
        moe[toks] += out
        bucket = per_expert.setdefault(
            eid, torch.zeros(n, d, dtype=torch.float32, device=device))
        bucket[toks] += out
    sw = shared_weight_for()
    xc = xf.float().to(device)
    gate = torch.clamp(xc @ sw["w1.weight"].t(), max=swiglu_limit)
    up = torch.clamp(xc @ sw["w3.weight"].t(),
                     min=-swiglu_limit, max=swiglu_limit)
    h = torch.nn.functional.silu(gate) * up
    shared_out = h @ sw["w2.weight"].t()
    combined = moe + shared_out
    moe_sorted = torch.zeros(n, d, dtype=torch.float32, device=device)
    for eid in sorted(groups):
        moe_sorted += per_expert[eid]
    combined_sorted = moe_sorted + shared_out
    return {"per_expert": per_expert, "shared_out": shared_out,
            "combined": combined, "combined_sorted": combined_sorted,
            "groups_order": list(groups.keys())}


def reference_leader_stages(
    x_ref: torch.Tensor,
    ids_ref: torch.Tensor,
    weights_ref: torch.Tensor,
    leader_eid: int,
    routed_raw: dict[int, dict[str, torch.Tensor]],
    *,
    topk: int,
    swiglu_limit: float,
) -> Optional[dict[str, Any]]:
    """Reference (CPU FP32) stage capture for one routed expert.

    Recomputes gate / up / h_pre / h (weighted) / out for the leader expert's
    served tokens so the candidate leader stages can be compared stage by
    stage (first divergent intermediate localization).
    """
    groups = group_tokens(ids_ref, weights_ref, topk)
    pairs = groups.get(leader_eid)
    if not pairs:
        return None
    t = routed_raw[leader_eid]
    w1 = ds7.dequantize_expert_weight(t["w1.weight"], t["w1.scale"])
    w2 = ds7.dequantize_expert_weight(t["w2.weight"], t["w2.scale"])
    w3 = ds7.dequantize_expert_weight(t["w3.weight"], t["w3.scale"])
    toks = [pair[0] for pair in pairs]
    ws = torch.tensor([[pair[1]] for pair in pairs],
                      dtype=torch.float32).reshape(-1, 1)
    xc = x_ref[toks]
    gate = torch.clamp(xc @ w1.transpose(0, 1), max=swiglu_limit)
    up = torch.clamp(xc @ w3.transpose(0, 1),
                     min=-swiglu_limit, max=swiglu_limit)
    h_pre = torch.nn.functional.silu(gate) * up
    h = ws * h_pre
    out = h @ w2.transpose(0, 1)
    return {"gate": gate, "up": up, "h_pre": h_pre, "h": h, "out": out}


def _order_audit(combined_group: torch.Tensor,
                 combined_sorted: torch.Tensor) -> dict[str, Any]:
    d = (combined_group - combined_sorted).abs()
    max_abs = float(d.max()) if d.numel() else 0.0
    scale = float(combined_group.abs().max()) if d.numel() else 0.0
    return {
        "bitwise": bool(torch.equal(combined_group, combined_sorted)),
        "max_abs": max_abs,
        "max_rel": max_abs / (scale + 1e-12),
    }


def _stage_compare(ref: dict[str, torch.Tensor], cand: dict[str, torch.Tensor],
                   names: tuple[str, ...]) -> dict[str, Any]:
    """Per-stage metrics + first divergent element for the leader expert."""
    out: dict[str, Any] = {}
    for name in names:
        r = ref[name].float().reshape(-1)
        c = cand[name].float().reshape(-1)
        if r.numel() != c.numel():
            out[name] = {"shape_mismatch": True}
            continue
        nd = int((r.view(torch.int32) != c.view(torch.int32)).sum())
        ulps = v4contract.ulp_tensor(r, c)
        m = _metrics(r, c)
        row: dict[str, Any] = {
            "numel": int(r.numel()),
            "bitwise": nd == 0,
            "count_diff": nd,
            "max_ulp": int(ulps.max().item()) if nd else 0,
            "max_abs": float((r - c).abs().max()) if nd else 0.0,
            "p99_rel": _p99(m),
            "cosine": m["cosine_similarity"],
        }
        if nd:
            flat = int((r.view(torch.int32) != c.view(torch.int32))
                       .nonzero().flatten()[0])
            rv, cv = float(r[flat]), float(c[flat])
            row["first_diff"] = {
                "flat": flat, "ref_bits": _f32_bits(rv),
                "cand_bits": _f32_bits(cv), "reference": rv, "candidate": cv,
                "abs_error": float(abs(rv - cv)),
                "rel_error": float(abs(rv - cv) / (abs(rv) + 1e-12)),
                "ulp": int(v4contract.ulp_tensor(r[flat:flat + 1],
                                                 c[flat:flat + 1]).item()),
            }
        out[name] = row
    return out


def tail_locator(
    ref: torch.Tensor,
    cand: torch.Tensor,
    per_expert_ref: dict[int, torch.Tensor],
    per_expert_cand: dict[int, torch.Tensor],
    shared_ref: torch.Tensor,
    shared_cand: torch.Tensor,
    *,
    label: str,
    gate: float = 0.05,
    hidden: int = 0,
    max_records: int = 6,
) -> dict[str, Any]:
    """Phase 8: p99-tail localization with per-expert cancellation.

    For the failing combined output: elements over the sealed relative-error
    gate, first/worst elements, and at the worst over-gate element the
    per-expert routed contributions + shared contribution from BOTH sides,
    with the cancellation ratio sum(|contrib|)/|total| (cancellation
    amplifies relative error when the routed sum nearly cancels).
    """
    rf = ref.reshape(-1).float()
    cf = cand.reshape(-1).float()
    numel = int(rf.numel())
    hidden = hidden or int(rf.numel() / max(1, int(ref.shape[0])))
    nnz = rf.abs() >= NEAR_ZERO
    rel = (rf - cf).abs() / (rf.abs() + 1e-8)
    over = nnz & (rel > gate)
    n_over = int(over.sum())
    frac = float(n_over / max(1, int(nnz.sum())))
    out: dict[str, Any] = {
        "label": label,
        "numel": numel,
        "nnz": int(nnz.sum()),
        "over_gate": n_over,
        "fraction_over": frac,
    }
    if n_over:
        first = int(over.nonzero().flatten()[0])
        out["first_over"] = {
            "flat": first, "tok": first // hidden, "dim": first % hidden,
            "reference": float(rf[first]), "candidate": float(cf[first]),
            "rel_error": float(rel[first]), "abs_error": float(abs(rf[first] - cf[first])),
            "ref_bits": _f32_bits(float(rf[first])),
            "cand_bits": _f32_bits(float(cf[first])),
        }
        # worst over-gate element (largest rel among the failing set)
        worst = int(rel[over].argmax())
        flat = int(over.nonzero().flatten()[worst])
        worst_row = _element_breakdown(
            flat, rf, cf, per_expert_ref, per_expert_cand,
            shared_ref, shared_cand, hidden)
        out["worst_over"] = worst_row
        # sample a few more over-gate elements for the error spread
        samples = []
        for sub in over.nonzero().flatten()[:max_records]:
            f = int(sub)
            samples.append({
                "flat": f, "tok": f // hidden, "dim": f % hidden,
                "reference": float(rf[f]), "candidate": float(cf[f]),
                "rel_error": float(rel[f]),
            })
        out["samples"] = samples
    else:
        out["worst_over"] = None
        out["samples"] = []
    # broad distribution view: worst rel over ALL nnz (even when under gate)
    if bool(nnz.any()):
        w = int(rel[nnz].argmax())
        wf = int(nnz.nonzero().flatten()[w])
        out["worst_nnz_rel"] = {
            "flat": wf, "tok": wf // hidden, "dim": wf % hidden,
            "reference": float(rf[wf]), "candidate": float(cf[wf]),
            "rel_error": float(rel[wf]),
        }
    return out


def _element_breakdown(
    flat: int,
    rf: torch.Tensor,
    cf: torch.Tensor,
    per_expert_ref: dict[int, torch.Tensor],
    per_expert_cand: dict[int, torch.Tensor],
    shared_ref: torch.Tensor,
    shared_cand: torch.Tensor,
    hidden: int,
) -> dict[str, Any]:
    tok, dim = flat // hidden, flat % hidden
    row: dict[str, Any] = {
        "flat": flat, "tok": tok, "dim": dim,
        "reference": float(rf[flat]), "candidate": float(cf[flat]),
        "abs_error": float(abs(rf[flat] - cf[flat])),
        "rel_error": float(abs(rf[flat] - cf[flat]) / (abs(rf[flat]) + 1e-8)),
        "ref_bits": _f32_bits(float(rf[flat])),
        "cand_bits": _f32_bits(float(cf[flat])),
    }
    contribs = []
    for eid in sorted(set(per_expert_ref) | set(per_expert_cand)):
        v_ref = float(per_expert_ref[eid][tok, dim]) if eid in per_expert_ref else 0.0
        v_cand = float(per_expert_cand[eid][tok, dim]) if eid in per_expert_cand else 0.0
        contribs.append({"expert": eid, "ref_contrib": v_ref,
                         "cand_contrib": v_cand,
                         "abs_delta": float(abs(v_ref - v_cand))})
    row["per_expert"] = contribs
    s_ref = float(shared_ref.reshape(-1)[flat])
    s_cand = float(shared_cand.reshape(-1)[flat])
    row["shared"] = {"ref_contrib": s_ref, "cand_contrib": s_cand}
    routed_ref = sum(c["ref_contrib"] for c in contribs)
    routed_cand = sum(c["cand_contrib"] for c in contribs)
    total_ref = routed_ref + s_ref
    total_cand = routed_cand + s_cand
    row["routed_sum"] = {"ref": routed_ref, "cand": routed_cand}
    row["cancellation"] = {
        "ref_ratio": abs(routed_ref + s_ref) / (
            sum(abs(c["ref_contrib"]) for c in contribs) + abs(s_ref) + 1e-12),
        "cand_ratio": abs(total_cand) / (
            sum(abs(c["cand_contrib"]) for c in contribs) + abs(s_cand) + 1e-12),
    }
    row["sum_consistency"] = {
        "ref_sum_matches_total": bool(
            math.isclose(total_ref, rf[flat].item(), rel_tol=1e-5, abs_tol=1e-5)),
        "cand_sum_matches_total": bool(
            math.isclose(total_cand, cf[flat].item(), rel_tol=1e-5, abs_tol=1e-5)),
    }
    return row


def _weight_storage_check(
    eids: list[int],
    routed_raw: dict[int, dict[str, torch.Tensor]],
    fp16_payloads: dict[int, dict[str, torch.Tensor]],
    shared_raw: dict[str, torch.Tensor],
    shared_payload: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Phase 5: exact FP16 representability of dequantized weights."""
    out: dict[str, Any] = {"routed": {}, "shared": {}, "all_exact": True}
    for eid in eids:
        for key in ("w1.weight", "w2.weight", "w3.weight"):
            fp32 = ds7.dequantize_expert_weight(
                routed_raw[eid][key], routed_raw[eid][f"{key[:-7]}.scale"])
            fp16 = fp16_payloads[eid][key]
            exact = bool(torch.equal(fp16.float(), fp32))
            out["routed"][f"{eid}:{key}"] = exact
            out["all_exact"] = out["all_exact"] and exact
    for key in ("w1.weight", "w2.weight", "w3.weight"):
        fp32 = moe.dequantize_fp8_e4m3(shared_raw[key],
                                       shared_raw[f"{key[:-7]}.scale"])
        fp16 = shared_payload[key]
        exact = bool(torch.equal(fp16.float(), fp32))
        out["shared"][key] = exact
        out["all_exact"] = out["all_exact"] and exact
    return out


def run_step_audit(
    *,
    x_ref: torch.Tensor,
    x_cand: torch.Tensor,
    ids_ref: torch.Tensor,
    wts_ref: torch.Tensor,
    ids_cand: torch.Tensor,
    wts_cand: torch.Tensor,
    ref_cap_moe: torch.Tensor,
    ref_cap_shared: torch.Tensor,
    cand_cap_moe: torch.Tensor,
    cand_cap_shared: torch.Tensor,
    routed_raw: dict[int, dict[str, torch.Tensor]],
    shared_raw: dict[str, torch.Tensor],
    fp16_payloads: dict[int, dict[str, torch.Tensor]],
    shared_payload: dict[str, torch.Tensor],
    gate_w: torch.Tensor,
    gate_b: torch.Tensor,
    cfg: Any,
    device: str,
    cache: Any,
    loader: Any,
    layer_id: int,
) -> dict[str, Any]:
    """Phase 1-9 decomposition for ONE step (token boundary)."""
    topk, sll = int(cfg.topk), float(cfg.swiglu_limit)
    n, d = x_ref.shape
    x_ref = x_ref.reshape(n, d).float()
    x_cand = x_cand.reshape(n, d).float()
    gw = gate_w.float()
    gb = gate_b.float() if gate_b is not None else None

    def routed_kwargs() -> dict[str, Any]:
        return dict(topk=topk, route_scale=float(cfg.route_scale),
                    score_func="sqrtsoftplus", swiglu_limit=sll,
                    keep_per_expert=True)

    A = moe.moe_layer_forward(x_ref, gw, gb, routed_raw, shared_raw,
                              **routed_kwargs())
    A1 = moe.moe_layer_forward(x_cand, gw, gb, routed_raw, shared_raw,
                               **routed_kwargs())

    # device-authentic payload access through the sealed cache/loader
    def payload_for(eid: int) -> dict[str, torch.Tensor]:
        entry = cache.get(layer_id, eid)
        if entry is None:
            entry = loader.stage(layer_id, eid, fp16_payloads[eid],
                                 metadata={"expert_type": "routed"})
        loader.wait(entry)
        return entry.payload

    def shared_payload_for() -> dict[str, torch.Tensor]:
        entry = cache.get(layer_id, -1)
        if entry is None:
            entry = loader.stage(layer_id, -1, shared_payload,
                                 metadata={"expert_type": "shared"})
        loader.wait(entry)
        return entry.payload

    def weight_for(eid: int) -> dict[str, torch.Tensor]:
        t = routed_raw[eid]
        wts = {k: ds7.dequantize_expert_weight(t[k], t[f"{k[:-7]}.scale"])
               for k in ("w1.weight", "w2.weight", "w3.weight")}
        return {k: v.to(device) for k, v in wts.items()}

    def shared_weight_for() -> dict[str, torch.Tensor]:
        wts = {k: moe.dequantize_fp8_e4m3(shared_raw[k],
                                          shared_raw[f"{k[:-7]}.scale"])
               for k in ("w1.weight", "w2.weight", "w3.weight")}
        return {k: v.to(device) for k, v in wts.items()}

    leader_eid = next(iter(group_tokens(ids_cand, wts_cand, topk)))

    def relocate_replay(r: dict[str, Any]) -> dict[str, Any]:
        """Move device replay tensors to CPU for the host-side metrics
        (the reference runs on CPU; compute_ds8_metrics is not mixed-device)."""
        r = dict(r)
        r["per_expert"] = {e: v.detach().cpu()
                           for e, v in r["per_expert"].items()}
        r["shared_out"] = r["shared_out"].detach().cpu()
        r["combined"] = r["combined"].detach().cpu()
        r["combined_sorted"] = r["combined_sorted"].detach().cpu()
        return r

    B = relocate_replay(candidate_experts_replay(
        x_ref, ids_ref, wts_ref, topk=topk, swiglu_limit=sll, device=device,
        payload_for=payload_for, shared_payload_for=shared_payload_for,
        leader_eid=leader_eid))
    C = relocate_replay(candidate_experts_replay(
        x_cand, ids_cand, wts_cand, topk=topk, swiglu_limit=sll,
        device=device, payload_for=payload_for,
        shared_payload_for=shared_payload_for, leader_eid=leader_eid))
    D = relocate_replay(candidate_experts_replay(
        x_cand, ids_ref, wts_ref, topk=topk, swiglu_limit=sll, device=device,
        payload_for=payload_for, shared_payload_for=shared_payload_for))
    E = relocate_replay(candidate_experts_replay(
        x_ref, ids_cand, wts_cand, topk=topk, swiglu_limit=sll, device=device,
        payload_for=payload_for, shared_payload_for=shared_payload_for))
    VB = relocate_replay(fp32_experts_replay(
        x_cand, ids_cand, wts_cand, topk=topk, swiglu_limit=sll,
        device=device, weight_for=weight_for,
        shared_weight_for=shared_weight_for))

    A_flat = A["moe_output"].reshape(-1)
    A_sh_flat = A["shared_output"].reshape(-1)
    C_flat = C["combined"].reshape(-1)
    C_sh_flat = C["shared_out"].reshape(-1)

    # ---- Phase 9: capture fidelity (harness-validity) ----------------------
    fidelity = {
        "ref_recompute_matches_capture": bool(
            torch.equal(A_flat, ref_cap_moe.reshape(-1))),
        "ref_shared_matches_capture": bool(
            torch.equal(A_sh_flat, ref_cap_shared.reshape(-1))),
        "cand_replay_matches_capture": bool(
            torch.equal(C_flat, cand_cap_moe.reshape(-1))),
        "cand_shared_matches_capture": bool(
            torch.equal(C_sh_flat, cand_cap_shared.reshape(-1))),
    }
    faithful = bool(all(fidelity.values()))

    # ---- Phase 5: weight storage ------------------------------------------
    storage = _weight_storage_check(
        sorted(set(routed_raw) & set(fp16_payloads)), routed_raw,
        fp16_payloads, shared_raw, shared_payload)

    # ---- headline metrics (captured tensors, identical to the gates) -------
    m_prod = _metrics(ref_cap_moe.reshape(-1), cand_cap_moe.reshape(-1))
    m_shared = _metrics(ref_cap_shared.reshape(-1), cand_cap_shared.reshape(-1))
    headline = {
        "moe_out_p99": _p99(m_prod),
        "moe_out_cosine": m_prod["cosine_similarity"],
        "shared_out_p99": _p99(m_shared),
        "shared_out_cosine": m_shared["cosine_similarity"],
    }

    # ---- three-way / cross-substitution matrix -----------------------------
    m_kernel = _metrics(A_flat, B["combined"].reshape(-1))     # B vs A
    m_fp32exec = _metrics(A_flat, VB["combined"].reshape(-1))  # V_B vs A
    m_input_sens = _metrics(A_flat, A1["moe_output"].reshape(-1))  # A1 vs A
    m_routing_sub = _metrics(C_flat, D["combined"].reshape(-1))  # D vs C
    # E vs B: routing-substitution on the REFERENCE input (candidate world)
    m_routing_sub_ref = _metrics(B["combined"].reshape(-1),
                                 E["combined"].reshape(-1))
    m_shared_isol = _metrics(A_sh_flat, C_sh_flat)  # shared: C vs A
    # routed-only reference: sum the per-expert buckets exactly (the fp32
    # (moe+shared)-shared subtraction is not guaranteed bitwise-exact).
    routed_a = torch.zeros_like(A["moe_output"])
    for eid in A["per_expert"]:
        routed_a += A["per_expert"][eid]
    routed_c = C["combined"] - C["shared_out"]
    m_routed = _metrics(routed_a.reshape(-1), routed_c.reshape(-1))
    matrix = {
        "kernel_ref_input": {
            "p99": _p99(m_kernel), "p95": _p95(m_kernel),
            "cosine": m_kernel["cosine_similarity"],
            "norm_rmse": m_kernel["normalized_rmse"],
            "max_abs": m_kernel["all_elements"]["max_abs_error"]},
        "fp32exec_cand_input": {
            "p99": _p99(m_fp32exec), "p95": _p95(m_fp32exec),
            "cosine": m_fp32exec["cosine_similarity"],
            "norm_rmse": m_fp32exec["normalized_rmse"],
            "max_abs": m_fp32exec["all_elements"]["max_abs_error"]},
        "ref_input_sensitivity": {
            "p99": _p99(m_input_sens), "p95": _p95(m_input_sens),
            "cosine": m_input_sens["cosine_similarity"],
            "max_abs": m_input_sens["all_elements"]["max_abs_error"]},
        "routing_substitution": {
            "p99": _p99(m_routing_sub), "p95": _p95(m_routing_sub),
            "cosine": m_routing_sub["cosine_similarity"],
            "max_abs": m_routing_sub["all_elements"]["max_abs_error"]},
        "routing_substitution_ref_input": {
            "p99": _p99(m_routing_sub_ref), "p95": _p95(m_routing_sub_ref),
            "cosine": m_routing_sub_ref["cosine_similarity"],
            "max_abs": m_routing_sub_ref["all_elements"]["max_abs_error"]},
        "shared_isolated": {
            "p99": _p99(m_shared_isol), "p95": _p95(m_shared_isol),
            "cosine": m_shared_isol["cosine_similarity"],
            "max_abs": m_shared_isol["all_elements"]["max_abs_error"]},
        "routed_only": {
            "p99": _p99(m_routed), "p95": _p95(m_routed),
            "cosine": m_routed["cosine_similarity"],
            "max_abs": m_routed["all_elements"]["max_abs_error"]},
    }

    # ---- per-expert canonicalized table (by expert ID) ---------------------
    per_expert = {}
    union = sorted(set(A["per_expert"]) | set(C["per_expert"]))
    for eid in union:
        a_bucket = A["per_expert"].get(eid)
        c_bucket = C["per_expert"].get(eid)
        if a_bucket is None:
            per_expert[f"{eid}"] = {"served": "cand_only",
                                    "cand_p99_vs_zero": _p99(
                                        _metrics(torch.zeros_like(c_bucket),
                                                 c_bucket))}
            continue
        if c_bucket is None:
            per_expert[f"{eid}"] = {"served": "ref_only"}
            continue
        m = _metrics(a_bucket.reshape(-1), c_bucket.reshape(-1))
        per_expert[f"{eid}"] = {
            "served": "both",
            "p99": _p99(m), "p95": _p95(m),
            "cosine": m["cosine_similarity"],
            "max_abs": m["all_elements"]["max_abs_error"],
            "ref_norm": float(a_bucket.norm()),
            "cand_norm": float(c_bucket.norm()),
        }
    per_expert["served_counts"] = {
        "ref_only": sorted(eid for eid in union
                           if eid not in C["per_expert"]),
        "cand_only": sorted(eid for eid in union
                            if eid not in A["per_expert"]),
    }

    # ---- Phase 7: accumulation-order audit --------------------------------
    order = {
        "candidate": _order_audit(C["combined"], C["combined_sorted"]),
        "reference": _order_audit(
            A["moe_output"], _sorted_ref_combined(A, shared_raw is not None)),
    }

    # ---- leader stage localization -----------------------------------------
    # ``compare`` = C leader (candidate input, candidate routing) vs reference
    # stages (reference input, reference routing): the FULL divergence
    # profile (input drift + fp16 compute combined).
    # ``compare_compute_only`` = B leader (candidate kernels on the REFERENCE
    # input with reference routing) vs the same reference stages: isolates
    # the fp16-compute-only stage profile (identical input + routing).
    leader = None
    if C["leader"] is not None:
        ref_stages = reference_leader_stages(
            x_ref, ids_ref, wts_ref, leader_eid, routed_raw,
            topk=topk, swiglu_limit=sll)
        if ref_stages is not None:
            cand_stages = {k: C["leader"][k] for k in
                           ("gate", "up", "h_pre", "h", "out")}
            leader = {
                "eid": leader_eid,
                "toks": C["leader"]["toks"],
                "compare": _stage_compare(ref_stages, cand_stages,
                                          ("gate", "up", "h_pre", "h", "out")),
            }
            if B["leader"] is not None:
                b_stages = {k: B["leader"][k] for k in
                            ("gate", "up", "h_pre", "h", "out")}
                leader["compare_compute_only"] = _stage_compare(
                    ref_stages, b_stages, ("gate", "up", "h_pre", "h", "out"))

    # ---- Phase 8: tail locator --------------------------------------------
    tail_moe = tail_locator(
        ref_cap_moe, cand_cap_moe, A["per_expert"], C["per_expert"],
        A["shared_output"], C["shared_out"], label="moe_out",
        gate=0.05, hidden=d)
    tail_shared = tail_locator(
        ref_cap_shared, cand_cap_shared, {}, {},
        A["shared_output"], C["shared_out"], label="shared_out",
        gate=0.05, hidden=d)

    # routing-weight delta summary
    wt_delta = float((wts_ref.float() - wts_cand.float()).abs().max()) \
        if wts_ref.shape == wts_cand.shape else None
    wt_norm_ref = float(wts_ref.float().abs().max())
    wt_norm_cand = float(wts_cand.float().abs().max())

    return {
        "fidelity": fidelity,
        "capture_faithful": faithful,
        "weight_storage": storage,
        "headline": headline,
        "matrix": matrix,
        "per_expert": per_expert,
        "accumulation_order": order,
        "leader": leader,
        "tail": {"moe_out": tail_moe, "shared_out": tail_shared},
        "routing_delta": {
            "max_abs_wt": wt_delta,
            "ref_max_abs_wt": wt_norm_ref,
            "cand_max_abs_wt": wt_norm_cand,
        },
        "input_ulp": v4contract.router_boundary_metrics(
            x_ref, x_cand, gw, gb, topk=topk, route_scale=float(cfg.route_scale),
            max_tokens=x_ref.shape[0]).get("input_summary"),
    }


def _sorted_ref_combined(A: dict[str, Any], has_shared: bool) -> torch.Tensor:
    """Reference combined re-accumulated in sorted-eid order (Phase 7)."""
    n, d = A["moe_output"].shape
    moe = torch.zeros(n, d, dtype=torch.float32)
    for eid in sorted(A["per_expert"]):
        moe += A["per_expert"][eid]
    if has_shared:
        moe = moe + A["shared_output"]
    return moe


def run_expert_integration_audit(
    *,
    step_captures: list[tuple[dict[str, Any], dict[str, Any]]],
    routed_raw: dict[int, dict[str, torch.Tensor]],
    shared_raw: dict[str, torch.Tensor],
    fp16_payloads: dict[int, dict[str, torch.Tensor]],
    shared_payload: dict[str, torch.Tensor],
    gate_w: torch.Tensor,
    gate_b: torch.Tensor,
    cfg: Any,
    device: str,
    cache: Any,
    loader: Any,
    layer_id: int,
) -> dict[str, Any]:
    """Full DS9 v13 expert-integration audit across the captured steps."""
    hidden = int(cfg.hidden)
    steps: dict[str, Any] = {}
    for i, (rc, cc) in enumerate(step_captures):
        x_ref = rc["ffn_norm_out"].float().reshape(-1, hidden)
        x_cand = cc["ffn_norm_out"].float().reshape(-1, hidden)
        step = run_step_audit(
            x_ref=x_ref, x_cand=x_cand,
            ids_ref=rc["expert_ids"], wts_ref=rc["routing_weights"],
            ids_cand=cc["expert_ids"], wts_cand=cc["routing_weights"],
            ref_cap_moe=rc["moe_out"].reshape(-1, hidden),
            ref_cap_shared=rc["shared_out"].reshape(-1, hidden),
            cand_cap_moe=cc["moe_out"].reshape(-1, hidden),
            cand_cap_shared=cc["shared_out"].reshape(-1, hidden),
            routed_raw=routed_raw, shared_raw=shared_raw,
            fp16_payloads=fp16_payloads, shared_payload=shared_payload,
            gate_w=gate_w, gate_b=gate_b, cfg=cfg, device=device,
            cache=cache, loader=loader, layer_id=layer_id)
        start_pos = int(rc.get("start_pos", i))
        steps[f"step{start_pos}"] = step
    return {
        "steps": steps,
        "n_steps": len(step_captures),
        "device": device,
        "layer_id": layer_id,
        "cfg_hidden": hidden,
        "cfg_topk": int(cfg.topk),
        "cfg_swiglu_limit": float(cfg.swiglu_limit),
        "cfg_route_scale": float(cfg.route_scale),
    }
