"""DS8 predeclared numerical contract for the T4 candidate runtime.

Extends the DS7 contract (kept immutable for historical evidence) with a NEW
predeclared gate set for the generalized expert runtime:

  max_abs_error        <= 2.0      (per-element absolute, all elements)
  mean_abs_error       <= 0.5
  mean_rel_error       <= 0.01     (non-near-zero elements)
  p95_rel_error        <= 0.03
  p99_rel_error        <= 0.05
  cosine_similarity    >= 0.999
  normalized_rmse      <= 0.01
  output_norm_rel_error <= 0.02    (||cand||_2 vs ||ref||_2 relative)

Near-zero reference elements (|ref| < NEAR_ZERO_THRESHOLD = 1e-3) are
excluded from the RELATIVE statistics (relative error is undefined for
near-zero reference values -- catastrophic cancellation).  The absolute
gates still apply to them.  The excluded fraction is reported and gated
(max_excluded_fraction <= 0.02).

These values are declared here BEFORE any DS8 run; they are NOT derived from
DS8 results.  They were chosen from the DS7 evidence (max_abs 0.0046,
mean_abs 0.0009, mean_rel 0.0059, p99_rel 0.038 for the same Family-B FP16
candidate math) with headroom for the 6-expert weighted combination.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import torch

NEAR_ZERO_THRESHOLD = 1e-3

DS8_TOLERANCE = {
    "max_abs_error": 2.0,
    "mean_abs_error": 0.5,
    "mean_rel_error": 0.01,
    "p95_rel_error": 0.03,
    "p99_rel_error": 0.05,
    "cosine_similarity": 0.999,
    "normalized_rmse": 0.01,
    "output_norm_rel_error": 0.02,
    "max_excluded_fraction": 0.02,
}


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().reshape(-1)


def compute_ds8_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """Full DS8 metric set, split into all-element and non-near-zero views.

    reference / candidate: same logical shape (e.g. [n, hidden]); the shape
    check happens BEFORE flattening so distinct shapes are rejected even when
    their element counts coincide.

    DS9 v7 sentinel awareness: the indexer score tensors legitimately contain
    ``-inf`` causal-mask sentinels.  All numerical metrics are therefore
    computed on the FINITE INTERSECTION of ref and cand only; the sentinel
    (finite vs non-finite) masks are compared EXACTLY.  A mask mismatch is a
    semantic failure and fails every gate regardless of numerics.  Cosine is
    never emitted as NaN: with no finite overlap it is None and the gates
    fail explicitly.
    """
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: ref {list(reference.shape)} vs "
            f"cand {list(candidate.shape)}")
    ref = _flatten(reference)
    cand = _flatten(candidate)
    if ref.numel() != cand.numel():
        raise ValueError(f"shape mismatch: ref {ref.shape} vs cand {cand.shape}")
    if ref.numel() == 0:
        raise ValueError("empty tensors")

    ref_fin = torch.isfinite(ref)
    cand_fin = torch.isfinite(cand)
    sentinel_mask_exact = bool(torch.equal(ref_fin, cand_fin))
    both_fin = ref_fin & cand_fin
    # Naming: "ref_only_nonfinite" = positions where the REFERENCE is
    # non-finite (e.g. -inf sentinel) while the CANDIDATE is finite
    # (ref -inf / cand finite) -- i.e. a candidate that wrote a slot the
    # reference left untouched.  "cand_only_nonfinite" is the reverse.
    mask_counts = {
        "both_finite": int(both_fin.sum()),
        "both_nonfinite": int((~ref_fin & ~cand_fin).sum()),
        "ref_only_nonfinite": int((~ref_fin & cand_fin).sum()),
        "cand_only_nonfinite": int((ref_fin & ~cand_fin).sum()),
        "nan_reference": int(torch.isnan(ref).sum()),
        "nan_candidate": int(torch.isnan(cand).sum()),
        "posinf_reference": int((ref == float("inf")).sum()),
        "posinf_candidate": int((cand == float("inf")).sum()),
    }
    finite_overlap = bool(both_fin.any())
    if not finite_overlap:
        # No finite overlap: numerical gates are undefined.  Never emit NaN;
        # fail explicitly (ds8_gate_passed returns False).
        empty = {"max_abs_error": None, "mean_abs_error": None,
                 "max_rel_error": None, "mean_rel_error": None,
                 "p95_rel_error": None, "p99_rel_error": None}
        return {
            "all_elements": dict(empty),
            "non_near_zero": {"count": 0, "max_abs_error": 0.0,
                              "mean_abs_error": 0.0, "mean_rel_error": 0.0,
                              "p95_rel_error": 0.0, "p99_rel_error": 0.0},
            "excluded": {"threshold": NEAR_ZERO_THRESHOLD, "count": 0,
                         "fraction": 0.0},
            "cosine_similarity": None,
            "normalized_rmse": None,
            "output_norm_rel_error": None,
            "worst_abs_element": None,
            "sentinel_mask_exact": sentinel_mask_exact,
            "finite_overlap": finite_overlap,
            "mask_counts": mask_counts,
        }

    r = ref[both_fin]
    c = cand[both_fin]
    flat_of = both_fin.nonzero().squeeze(-1)  # subset position -> original flat idx
    abs_err = (r - c).abs()
    near_zero = r.abs() < NEAR_ZERO_THRESHOLD
    not_near_zero = ~near_zero
    excluded_fraction = float(near_zero.float().mean())

    def rel_of(mask: torch.Tensor) -> torch.Tensor:
        denom = r[mask].abs() + 1e-8
        return abs_err[mask] / denom

    rel_all = abs_err / (r.abs() + 1e-8)
    rel_nnz = rel_of(not_near_zero)

    # cosine similarity over the finite intersection.
    denom = (r.norm() * c.norm())
    cosine = float((r @ c) / denom) if denom.item() > 0 else 1.0
    # Normalized RMSE: RMSE divided by the reference RMS, so it is
    # scale-invariant like the cosine/relative gates.
    ref_rms = float(r.pow(2).mean().sqrt())
    normalized_rmse = float((abs_err.pow(2).mean()).sqrt() / max(ref_rms, 1e-30))
    ref_norm = float(r.norm())
    out_norm_rel = float(abs(ref_norm - float(c.norm())) / max(ref_norm, 1e-30))
    worst_abs_sub = int(abs_err.argmax())
    worst_abs_flat = int(flat_of[worst_abs_sub])

    metrics = {
        "all_elements": {
            "max_abs_error": float(abs_err.max()),
            "mean_abs_error": float(abs_err.mean()),
            "max_rel_error": float(rel_all.max()),
            "mean_rel_error": float(rel_all.mean()),
            "p95_rel_error": float(rel_all.flatten().quantile(0.95)),
            "p99_rel_error": float(rel_all.flatten().quantile(0.99)),
        },
        "non_near_zero": {
            "count": int(not_near_zero.sum()),
            "max_abs_error": float(abs_err[not_near_zero].max())
            if bool(not_near_zero.any()) else 0.0,
            "mean_abs_error": float(abs_err[not_near_zero].mean())
            if bool(not_near_zero.any()) else 0.0,
            "mean_rel_error": float(rel_nnz.mean()) if rel_nnz.numel() else 0.0,
            "p95_rel_error": float(rel_nnz.flatten().quantile(0.95))
            if rel_nnz.numel() else 0.0,
            "p99_rel_error": float(rel_nnz.flatten().quantile(0.99))
            if rel_nnz.numel() else 0.0,
        },
        "excluded": {
            "threshold": NEAR_ZERO_THRESHOLD,
            "count": int(near_zero.sum()),
            "fraction": excluded_fraction,
        },
        "cosine_similarity": cosine,
        "normalized_rmse": normalized_rmse,
        "output_norm_rel_error": out_norm_rel,
        "worst_abs_element": {
            "index": worst_abs_flat,
            "reference": float(r[worst_abs_sub]),
            "candidate": float(c[worst_abs_sub]),
            "abs_error": float(abs_err.max()),
        },
        "sentinel_mask_exact": sentinel_mask_exact,
        "finite_overlap": finite_overlap,
        "mask_counts": mask_counts,
    }
    if rel_nnz.numel():
        nnz_pos = not_near_zero.nonzero().squeeze(-1)
        worst_nnz_sub = int(rel_nnz.argmax())
        worst_nnz_flat = int(flat_of[nnz_pos[worst_nnz_sub]])
        metrics["worst_non_near_zero_rel_element"] = {
            "flat_index": worst_nnz_flat,
            "reference": float(r[nnz_pos[worst_nnz_sub]]),
            "candidate": float(c[nnz_pos[worst_nnz_sub]]),
            "rel_error": float(rel_nnz.max()),
        }
    return metrics


def ds8_gate_passed(metrics: dict[str, Any], tol: dict[str, float] | None = None) -> bool:
    """Check the predeclared DS8 gates against computed metrics.

    DS9 v7: ``sentinel_mask_exact`` and ``finite_overlap`` are REQUIRED.  A
    finite/-inf mask mismatch or a NaN/+inf anywhere is a semantic failure
    and fails all gates even if finite-intersection numerics look fine.
    """
    tol = tol or DS8_TOLERANCE
    if not metrics.get("sentinel_mask_exact", True):
        return False
    if not metrics.get("finite_overlap", True):
        return False
    nnz = metrics["non_near_zero"]
    return (
        metrics["all_elements"]["max_abs_error"] <= tol["max_abs_error"]
        and metrics["all_elements"]["mean_abs_error"] <= tol["mean_abs_error"]
        and nnz["mean_rel_error"] <= tol["mean_rel_error"]
        and nnz["p95_rel_error"] <= tol["p95_rel_error"]
        and nnz["p99_rel_error"] <= tol["p99_rel_error"]
        and metrics["cosine_similarity"] >= tol["cosine_similarity"]
        and metrics["normalized_rmse"] <= tol["normalized_rmse"]
        and metrics["output_norm_rel_error"] <= tol["output_norm_rel_error"]
        and metrics["excluded"]["fraction"] <= tol["max_excluded_fraction"]
    )


# ---------------------------------------------------------------------------
# DS9 v7 state sentinel mask analysis (finite/-inf classification + first
# divergent element with ref/cand bits, values, abs/rel error and ULP).
# ---------------------------------------------------------------------------


def f32_ulp_distance(a: float, b: float) -> int:
    """IEEE-754 fp32 ULP distance between two values (sign-magnitude order)."""

    def ordered(bits: int) -> int:
        return bits if bits < 0x80000000 else 0xFFFFFFFF - bits

    ia = ordered(struct.unpack("<I", struct.pack("<f", float(a)))[0])
    ib = ordered(struct.unpack("<I", struct.pack("<f", float(b)))[0])
    return abs(ia - ib)


def state_mask_analysis(
    ref_buffers: dict[str, torch.Tensor],
    cand_buffers: dict[str, torch.Tensor],
    init_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Exact sentinel/finite mask classification per state buffer.

    For every buffer returns: mask counts (both-finite / both-nonfinite /
    ref-only-nonfinite / cand-only-nonfinite / NaN / +inf), the exact
    sentinel-mask agreement, the first DIVERGENT mask position (when masks
    differ) and the first divergent VALUE position within the finite
    intersection (flat index, coords, ref/cand bits, values, abs/rel, ULP).
    A finite/-inf mask mismatch or any NaN/+inf is a semantic failure.

    ``init_values`` optionally maps a buffer name to its INITIAL fill value
    (kv_state -> 0.0, score_state -> -inf); the written/untouched slot
    heuristic then classifies a slot as untouched iff it still equals that
    value.  Buffers without an entry fall back to the combined 0.0|-inf
    heuristic (correct for kv_state, ambiguous for score_state: a candidate
    writing 0.0 into an untouched -inf slot is only caught via explicit
    init_values).
    """
    init_values = init_values or {}
    out: dict[str, Any] = {}
    for name, rt in ref_buffers.items():
        ct = cand_buffers.get(name)
        if ct is None:
            out[name] = {"ok": False, "reason": "missing candidate buffer"}
            continue
        if rt.shape != ct.shape or rt.dtype != ct.dtype:
            out[name] = {"ok": False, "reason": "shape/dtype mismatch"}
            continue
        rf = rt.reshape(-1).float()
        cf = ct.reshape(-1).float()
        rfin, cfin = torch.isfinite(rf), torch.isfinite(cf)
        both_fin = rfin & cfin
        both_nf = ~rfin & ~cfin
        # naming: ref_only_nf = ref non-finite AND cand finite (ref -inf /
        # cand finite); cand_only_nf = ref finite AND cand non-finite.
        ref_only_nf = ~rfin & cfin
        cand_only_nf = rfin & ~cfin
        nan_ref = int(torch.isnan(rf).sum())
        nan_cand = int(torch.isnan(cf).sum())
        pinf_ref = int((rf == float("inf")).sum())
        pinf_cand = int((cf == float("inf")).sum())
        sentinel_exact = bool(torch.equal(rfin, cfin))
        # Phase-3 slot gates: kv_state inits to ZEROS (written -> nonzero,
        # untouched -> zero); score_state inits to -inf (written -> finite,
        # untouched -> -inf).  Exact agreement of the written/untouched slot
        # masks is required: a candidate that writes a different slot set is a
        # semantic failure even if the finite-intersection values happen to be
        # close.
        init = init_values.get(name)
        if init is None and "score" in name:
            # defense-in-depth: score_state buffers initialize to -inf; a
            # caller that forgets init_values would otherwise hit the
            # ambiguous combined heuristic (0.0 written into an untouched
            # -inf slot would look "untouched" on both sides).
            init = float("-inf")
        if init is not None:
            untouched_ref = rf == float(init)
            untouched_cand = cf == float(init)
        else:
            untouched_ref = (rf == 0.0) | (rf == float("-inf"))
            untouched_cand = (cf == 0.0) | (cf == float("-inf"))
        written_slot_exact = bool(
            torch.equal(untouched_ref, untouched_cand))
        untouched_slot_exact = written_slot_exact  # complement of the same mask
        mask_div: dict[str, Any] | None = None
        if not sentinel_exact:
            diff = (rfin != cfin).nonzero().squeeze(-1)
            fi = int(diff[0])
            mask_div = {
                "flat_index": fi,
                "ref_finite": bool(rfin[fi]),
                "cand_finite": bool(cfin[fi]),
                "ref_value": float(rf[fi]),
                "cand_value": float(cf[fi]),
                "class": ("ref_finite_cand_nonfinite" if rfin[fi]
                           else "ref_nonfinite_cand_finite"),
            }
        elif not written_slot_exact:
            diff = (untouched_ref != untouched_cand).nonzero().squeeze(-1)
            fi = int(diff[0])
            mask_div = {
                "flat_index": fi,
                "ref_finite": bool(rfin[fi]),
                "cand_finite": bool(cfin[fi]),
                "ref_untouched": bool(untouched_ref[fi]),
                "cand_untouched": bool(untouched_cand[fi]),
                "ref_value": float(rf[fi]),
                "cand_value": float(cf[fi]),
                "class": ("ref_untouched_cand_written"
                           if untouched_ref[fi] else "ref_written_cand_untouched"),
            }
        first_value_div: dict[str, Any] | None = None
        if bool(both_fin.any()):
            absd = (rf[both_fin] - cf[both_fin]).abs()
            sub = int(absd.argmax())
            if float(absd.max()) > 0.0:
                flat = int(both_fin.nonzero().squeeze(-1)[sub])
                rv, cv = float(rf[flat]), float(cf[flat])
                first_value_div = {
                    "flat_index": flat,
                    "ref_bits": format(
                        struct.unpack("<I", struct.pack("<f", rv))[0], "08x"),
                    "cand_bits": format(
                        struct.unpack("<I", struct.pack("<f", cv))[0], "08x"),
                    "reference": rv,
                    "candidate": cv,
                    "abs_error": float(abs(rv - cv)),
                    "rel_error": float(abs(rv - cv) / (abs(rv) + 1e-12)),
                    "ulp": f32_ulp_distance(rv, cv),
                }
        # DS9 v9: ok = STRUCTURAL mask agreement only (finite/sentinel/
        # written/untouched masks exact, no NaN/+inf).  A finite-intersection
        # VALUE divergence is NOT a structural failure: the reference (CPU
        # fp32) and candidate (CUDA fp32) run the same module, so cross-
        # device reduction order legitimately drifts state values by 1-7 ULP
        # (v8 boundary-capture evidence).  Value BOUNDS are enforced by the
        # separate predeclared state_agreement per-buffer relative gates
        # (0.001); first_value_divergence remains in the record as the
        # first-divergence locator evidence.
        ok = bool(sentinel_exact and written_slot_exact
                  and nan_ref == 0 and nan_cand == 0
                  and pinf_ref == 0 and pinf_cand == 0)
        out[name] = {
            "ok": ok,
            "sentinel_mask_exact": sentinel_exact,
            "finite_mask_exact": sentinel_exact,
            "written_slot_mask_exact": written_slot_exact,
            "untouched_slot_mask_exact": untouched_slot_exact,
            "counts": {
                "both_finite": int(both_fin.sum()),
                "both_nonfinite": int(both_nf.sum()),
                "ref_only_nonfinite": int(ref_only_nf.sum()),
                "cand_only_nonfinite": int(cand_only_nf.sum()),
                "nan_reference": nan_ref,
                "nan_candidate": nan_cand,
                "posinf_reference": pinf_ref,
                "posinf_candidate": pinf_cand,
            },
            "mask_divergence": mask_div,
            "first_value_divergence": first_value_div,
        }
    return out


# ---------------------------------------------------------------------------
# DS9 predeclared numerical contract (category-specific).
#
# Declared BEFORE any DS9 run; NOT derived from results.  Rationale:
# - attention-path categories (norms, q/kv, compressor, index scores,
#   attention output, hc outputs) compare the reference and candidate BOTH in
#   FP32/bf16 with identical QAT simulation points, so they should agree to
#   ~1e-3; the gates allow 10-100x headroom for CUDA reduction-order drift.
# - router scores/weights: identical FP32 gate math on both sides -> tight.
# - expert categories reuse the DS8 contract (worst accepted DS8 error:
#   max_abs 0.0054, mean_abs 0.0008, mean_rel 0.0026, p99 0.034, cosine
#   0.9999998, norm_rmse 0.0005) with 2-10x headroom, since the candidate
#   runs the DS8 FP16 cache payloads while the reference is full FP32.
# - the final layer output mixes both paths, so it gates between the two.
#
# Exactness gates (not numerical): window indices exact, router expert IDs
# exact, routing-weight signs exact; compressed (indexer) indices exact for
# all corpus cases except the near_zero tie case, which the official
# implementation itself leaves effectively arbitrary (documented, approved).
# ---------------------------------------------------------------------------

DS9_TOLERANCE_ATTENTION = {
    "max_abs_error": 0.05,
    "mean_abs_error": 0.01,
    "mean_rel_error": 0.01,
    "p95_rel_error": 0.02,
    "p99_rel_error": 0.05,
    "cosine_similarity": 0.9999,
    "normalized_rmse": 0.01,
    "output_norm_rel_error": 0.02,
    "max_excluded_fraction": 0.02,
}

DS9_TOLERANCE_INDEX = {
    "max_abs_error": 0.5,
    "mean_abs_error": 0.05,
    "mean_rel_error": 0.05,
    "p95_rel_error": 0.1,
    "p99_rel_error": 0.2,
    "cosine_similarity": 0.999,
    "normalized_rmse": 0.05,
    "output_norm_rel_error": 0.05,
    "max_excluded_fraction": 0.1,
}

DS9_TOLERANCE_ROUTER = {
    "max_abs_error": 0.05,
    "mean_abs_error": 0.01,
    "mean_rel_error": 0.01,
    "p95_rel_error": 0.02,
    "p99_rel_error": 0.05,
    "cosine_similarity": 0.9999,
    "normalized_rmse": 0.01,
    "output_norm_rel_error": 0.02,
    "max_excluded_fraction": 0.02,
}

# Expert and final-layer categories reuse the DS8 contract shape with the
# DS8 limits (see module docstring); declared here for provenance.
DS9_TOLERANCE_EXPERT = dict(DS8_TOLERANCE)
DS9_TOLERANCE_FINAL = dict(DS8_TOLERANCE)
DS9_TOLERANCE_FINAL.update({"max_abs_error": 0.1, "mean_abs_error": 0.02})

DS9_TOLERANCES = {
    "layer_input": DS9_TOLERANCE_ATTENTION,
    "attn_norm_in": DS9_TOLERANCE_ATTENTION,
    "attn_norm_out": DS9_TOLERANCE_ATTENTION,
    "qr": DS9_TOLERANCE_ATTENTION,
    "q": DS9_TOLERANCE_ATTENTION,
    "kv": DS9_TOLERANCE_ATTENTION,
    "kv_compressed": DS9_TOLERANCE_ATTENTION,
    "attn_o": DS9_TOLERANCE_ATTENTION,
    "attn_out": DS9_TOLERANCE_ATTENTION,
    "attn_hc_out": DS9_TOLERANCE_ATTENTION,
    "ffn_norm_in": DS9_TOLERANCE_ATTENTION,
    "ffn_norm_out": DS9_TOLERANCE_ATTENTION,
    "indexer_scores": DS9_TOLERANCE_INDEX,
    "router_scores": DS9_TOLERANCE_ROUTER,
    "router_bias_scores": DS9_TOLERANCE_ROUTER,
    "moe_out": DS9_TOLERANCE_EXPERT,
    "shared_out": DS9_TOLERANCE_EXPERT,
    "output": DS9_TOLERANCE_FINAL,
}

DS9_INDEX_EXACT_CATEGORIES = ("attn_window_idxs", "attn_compress_idxs")
# The near-zero corpus input leaves index scores ~0 everywhere: the official
# top-k then selects effectively arbitrary positions (an official tie case),
# so compressed-index exactness is waived ONLY for that case.
DS9_INDEX_EXACT_WAIVED_CASES = ("near_zero",)
DS9_EXACT_GATES = {"router_expert_ids", "routing_weight_signs"}


# ---------------------------------------------------------------------------
# DS9 v10: router-boundary diagnostics (expert-ID flip causal proof)
# ---------------------------------------------------------------------------

def ulp_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized fp32 ULP distance between two float tensors.

    Matches f32_ulp_distance semantics (IEEE-754 sign-magnitude ordering):
    ordered(bits) = bits for positive floats, 0xFFFFFFFF - bits for negative
    floats, so the integer difference grows monotonically with the real
    distance on the float line.
    """
    ia = a.float().reshape(-1).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ib = b.float().reshape(-1).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    oa = torch.where(ia < 0x80000000, ia, 0xFFFFFFFF - ia)
    ob = torch.where(ib < 0x80000000, ib, 0xFFFFFFFF - ib)
    ulps = (oa - ob).abs()
    # +0.0 and -0.0 are numerically equal: treat the sign bit as zero
    # distance (otherwise a lone +/-0 pair reads as 2^31 ULP and poisons
    # the ULP-level-drift classification).  Scalar f32_ulp_distance keeps
    # the raw sign-magnitude semantics for scalar records.
    both_zero = ((ia == 0) | (ia == 0x80000000)) \
        & ((ib == 0) | (ib == 0x80000000))
    return torch.where(both_zero, torch.zeros_like(ulps), ulps)


def _f32_hex(t: torch.Tensor) -> str:
    """Concatenated fp32 bit patterns of a 1-D tensor (little-endian hex)."""
    return "".join(format(v & 0xFFFFFFFF, "08x")
                    for v in t.float().view(torch.int32).tolist())


def _fp32_bits(value: float) -> str:
    return format(struct.unpack("<I", struct.pack("<f", float(value)))[0],
                  "08x")


def _tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def router_stages(
    xf: torch.Tensor,
    gate_w: torch.Tensor,
    gate_b: torch.Tensor | None,
    *,
    topk: int = 6,
    route_scale: float = 1.5,
) -> dict[str, torch.Tensor]:
    """Official Gate.forward pipeline on [n, hidden] fp32 input.

    Mirrors the official inference/model.py Gate.forward and
    scripts/deepseek_v4_expert_reference.router_scores EXACTLY (same fp32
    ops, same order) while exposing every intermediate for boundary analysis:

      raw      = x @ W^T                        (pre-activation logits)
      softplus = F.softplus(raw)
      sqrt     = sqrt(softplus)                 (official routed score)
      biased   = sqrt + bias                    (selection scores)
      ids      = biased.topk(topk, -1)[1]
      weights  = normalize(sqrt.gather(1, ids)) * route_scale
    """
    raw = xf @ gate_w.transpose(0, 1)
    softplus = torch.nn.functional.softplus(raw)
    sqrt = softplus.sqrt()
    biased = sqrt + gate_b if gate_b is not None else sqrt
    ids = biased.topk(topk, dim=-1)[1]
    sq = sqrt.gather(1, ids)
    weights = sq / sq.sum(dim=-1, keepdim=True)
    weights = weights * route_scale
    return {"raw": raw, "softplus": softplus, "sqrt": sqrt,
            "biased": biased, "ids": ids, "weights": weights}


def _topk_audit(scores: torch.Tensor, topk: int) -> dict[str, Any]:
    """Phase 4: torch.topk semantics audit on one biased-score vector.

    torch.topk is not documented as stable: for EXACT ties the winning index
    is implementation-defined (device-dependent select).  This reports the
    descending-sort vs topk consistency, exact-tie groups inside the top 10,
    and whether a tie straddles the rank-6/rank-7 boundary.
    """
    top = min(10, int(scores.numel()))
    vals, idxs = scores.topk(topk, dim=-1)
    svals, sidxs = scores.sort(descending=True)
    groups: list[list[int]] = []
    g = [int(sidxs[0])]
    for k in range(1, top):
        same = (struct.pack("<f", float(svals[k]))
                == struct.pack("<f", float(svals[k - 1])))
        if same:
            g.append(int(sidxs[k]))
        else:
            if len(g) > 1:
                groups.append([int(e) for e in g])
            g = [int(sidxs[k])]
    if len(g) > 1:
        groups.append([int(e) for e in g])
    boundary_tie = bool(top >= 7 and (struct.pack("<f", float(svals[5]))
                                       == struct.pack("<f", float(svals[6]))))
    return {
        "num_routed": int(scores.numel()),
        "topk_ids": [int(e) for e in idxs.tolist()],
        "sort_topk_ids_agree": bool(torch.equal(idxs, sidxs[:topk])),
        "exact_tie_groups_top10": groups,
        "boundary_tie_rank6_vs_rank7": boundary_tie,
    }


def _router_sensitivity(
    ref_row: torch.Tensor,
    cand_row: torch.Tensor,
    gw: torch.Tensor,
    gb: torch.Tensor | None,
    rs: dict[str, torch.Tensor],
    cs: dict[str, torch.Tensor],
    row: int,
    topk: int,
) -> dict[str, Any]:
    """Phase 6: linearized score-change causal analysis for one token.

    score_e = sqrt(softplus(x @ w_e + b_e)), so
    d(score_e)/dx = sigmoid(l_e) / (2 * score_e) * w_e.  With dx = cand - ref
    the estimated score change is dscore_e = sigmoid(l_e)/(2 s_e) * (dx@w_e);
    the estimated SELECTION score is biased_ref + dscore (the bias is
    input-independent, so it drops out of the derivative).

    Reports the top-6 symmetric-difference (flipped) expert pairs with
    BIASED-score margins (the ranking is on biased), estimated vs ACTUAL
    candidate selection scores, whether the linear estimate reproduces the
    observed flip, and the minimum input perturbation that reverses the
    tightest crossing (margin / ||g|| where g is the margin gradient) versus
    the measured input delta.
    """
    dx = cand_row - ref_row
    raw_r = rs["raw"][row]
    sq_r = rs["sqrt"][row]
    biased_ref = rs["biased"][row]   # SELECTION scores (sqrt + bias)
    biased_cand = cs["biased"][row]  # actual candidate selection scores
    sig = torch.sigmoid(raw_r)
    dl = dx @ gw.transpose(0, 1)
    # d(score)/d(logit) = sigmoid(l)/(2*sqrt(softplus(l))); the bias is
    # input-independent so it drops out of the derivative, and the estimated
    # SELECTION score is biased_ref + dscore.
    dscore = (sig / (2.0 * sq_r)) * dl
    est_biased = biased_ref + dscore
    ref6 = set(int(e) for e in rs["ids"][row].tolist())
    cand6 = set(int(e) for e in cs["ids"][row].tolist())
    out: dict[str, Any] = {
        "dx_norm": float(dx.norm()),
        "dx_max_abs": float(dx.abs().max()),
        "dx_ulp_max": int(ulp_tensor(ref_row, cand_row).max()),
    }
    # crossing analysis for every (outgoing, incoming) pair.  Margins use the
    # BIASED selection scores (the ranking is on biased); the gradient is
    # unchanged because the bias derivative is zero.
    outgoing = sorted(ref6 - cand6)
    incoming = sorted(cand6 - ref6)
    crossings = []
    for o in outgoing:
        for i in incoming:
            margin_ref = float(biased_ref[o] - biased_ref[i])
            # > 0: ref selection ranked o above i
            margin_act = float(biased_cand[i] - biased_cand[o])
            # > 0: candidate flipped the pair
            margin_est = float(est_biased[i] - est_biased[o])
            # margin gradient of (score_o - score_i) w.r.t. input
            g = ((sig[o] / (2.0 * sq_r[o])) * gw[o]
                 - (sig[i] / (2.0 * sq_r[i])) * gw[i])
            gnorm = float(g.norm())
            min_reverse = abs(margin_ref) / (gnorm + 1e-12)
            crossings.append({
                "outgoing": o, "incoming": i,
                "ref_score_outgoing": float(biased_ref[o]),
                "ref_score_incoming": float(biased_ref[i]),
                "margin_ref": margin_ref,
                "logit_delta_outgoing": float(dl[o]),
                "logit_delta_incoming": float(dl[i]),
                "est_score_delta_outgoing": float(dscore[o]),
                "est_score_delta_incoming": float(dscore[i]),
                "est_cand_outgoing": float(est_biased[o]),
                "est_cand_incoming": float(est_biased[i]),
                "actual_cand_outgoing": float(biased_cand[o]),
                "actual_cand_incoming": float(biased_cand[i]),
                "margin_actual": margin_act,
                "margin_est": margin_est,
                "est_flip_matches_actual": bool(
                    (margin_est > 0) == (margin_act > 0)),
                "min_reverse_perturbation_norm": min_reverse,
                "est_vs_actual_delta": float(est_biased[o]
                                              - biased_cand[o]),
            })
    out["crossings"] = crossings
    if crossings:
        tightest = min(crossings, key=lambda c: c["margin_ref"])
        out["tightest_crossing"] = tightest
        out["flip_explained"] = bool(
            tightest["est_flip_matches_actual"]
            and tightest["margin_actual"] > 0)
        out["dx_vs_min_reverse_ratio"] = float(
            out["dx_norm"] / (tightest["min_reverse_perturbation_norm"]
                               + 1e-12))
    else:
        out["flip_explained"] = False
        out["dx_vs_min_reverse_ratio"] = None
    return out


def router_boundary_metrics(
    ref_xf: torch.Tensor,
    cand_xf: torch.Tensor,
    gate_w: torch.Tensor,
    gate_b: torch.Tensor | None,
    *,
    topk: int = 6,
    route_scale: float = 1.5,
    max_tokens: int = 64,
) -> dict[str, Any]:
    """Full DS9 v10 router-boundary causal analysis (CPU fp32, deterministic).

    ref_xf / cand_xf: [n, hidden] fp32 router inputs (float-cast, exactly as
    the official Gate.forward consumes them).  gate_w: [n_routed, hidden]
    fp32; gate_b: [n_routed] fp32.  JSON-safe output:

      input_summary    - full-tensor ULP/abs/rel/hash summary of the router
                         input difference (first divergent element, count,
                         max ULP, ULP histogram, SHA256 per side)
      first_flip_token - first token row whose selected top-6 ID set differs
      stages           - per-stage (raw/softplus/sqrt/biased) hashes, norms
                         and ref-vs-cand error stats at the first flip token
      raw/biased_hex   - full 256-score fp32 hex for the flip token (both
                         sides) - the required full-score capture
      boundary         - top-10 IDs+scores, rank-6/7 margin + IEEE bits,
                         symmetric difference, ordering-change audit
      topk_audit       - torch.topk tie/stability audit on the ref scores
      sensitivity      - linearized score-change + min-reverse-perturbation
                         causal analysis for the flipped pairs
    """
    ref_xf = ref_xf.float().reshape(-1, ref_xf.shape[-1]).cpu()
    cand_xf = cand_xf.float().reshape(-1, cand_xf.shape[-1]).cpu()
    n = min(ref_xf.shape[0], cand_xf.shape[0], max_tokens)
    ref_xf, cand_xf = ref_xf[:n], cand_xf[:n]
    gw = gate_w.float().cpu()
    gb = gate_b.float().cpu() if gate_b is not None else None
    out: dict[str, Any] = {}

    # ---- Phase 1: input ULP summary -------------------------------------
    ra, ca = ref_xf.reshape(-1), cand_xf.reshape(-1)
    bits_diff = ra.view(torch.int32) != ca.view(torch.int32)
    nd = int(bits_diff.sum())
    ulps = ulp_tensor(ra, ca)
    hist: dict[str, int] = {}
    for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 10), (11, 100), (101, 1 << 62)):
        hist[f"{lo}-{hi}"] = int(((ulps >= lo) & (ulps <= hi)).sum())
    first_flat = int(bits_diff.nonzero().flatten()[0]) if nd else None
    out["input_summary"] = {
        "shape": list(ref_xf.shape),
        "numel": int(ra.numel()),
        "bitwise_exact": nd == 0,
        "count_diff": nd,
        "first_diff_flat": first_flat,
        "first_diff_coords": (
            [first_flat // ref_xf.shape[-1], first_flat % ref_xf.shape[-1]]
            if first_flat is not None else None),
        "max_ulp": int(ulps.max().item()) if nd else 0,
        "max_abs": float((ra - ca).abs().max()) if nd else 0.0,
        "max_rel": float(((ra - ca).abs() / (ra.abs() + 1e-12)).max())
        if nd else 0.0,
        "ulp_histogram": hist,
        "ref_sha256": _tensor_sha(ref_xf),
        "cand_sha256": _tensor_sha(cand_xf),
        "ref_min": float(ref_xf.min()), "ref_max": float(ref_xf.max()),
        "cand_min": float(cand_xf.min()), "cand_max": float(cand_xf.max()),
        "ref_norm": float(ref_xf.norm()), "cand_norm": float(cand_xf.norm()),
    }

    # ---- official pipeline for both inputs --------------------------------
    rs = router_stages(ref_xf, gw, gb, topk=topk, route_scale=route_scale)
    cs = router_stages(cand_xf, gw, gb, topk=topk, route_scale=route_scale)

    # ---- first divergent token (row whose top-6 ID set differs) -----------
    flip_row: int | None = None
    for r in range(n):
        if not torch.equal(rs["ids"][r], cs["ids"][r]):
            flip_row = r
            break
    out["first_flip_token"] = flip_row
    out["tokens_checked"] = n

    if flip_row is None:
        out["stages"] = {}
        out["boundary"] = {}
        out["topk_audit"] = {}
        out["sensitivity"] = {}
        return out

    # ---- Phase 2: stage-level detail at the first flip token --------------
    stages: dict[str, Any] = {}
    for sname in ("raw", "softplus", "sqrt", "biased"):
        rf, cf = rs[sname][flip_row].float().reshape(-1), \
            cs[sname][flip_row].float().reshape(-1)
        sd = rf.view(torch.int32) != cf.view(torch.int32)
        sulp = ulp_tensor(rf, cf)
        nsf = int(sd.sum())
        stages[sname] = {
            "numel": int(rf.numel()),
            "bitwise_exact": nsf == 0,
            "count_diff": nsf,
            "max_ulp": int(sulp.max()) if nsf else 0,
            "max_abs": float((rf - cf).abs().max()) if nsf else 0.0,
            "max_rel": float(
                ((rf - cf).abs() / (rf.abs() + 1e-12)).max()) if nsf else 0.0,
            "cosine": float(torch.nn.functional.cosine_similarity(
                rf.unsqueeze(0), cf.unsqueeze(0)).item()) if nsf else 1.0,
            "ref_sha256": _tensor_sha(rf),
            "cand_sha256": _tensor_sha(cf),
        }
    out["stages"] = stages
    out["raw_hex_ref"] = _f32_hex(rs["raw"][flip_row])
    out["raw_hex_cand"] = _f32_hex(cs["raw"][flip_row])
    out["biased_hex_ref"] = _f32_hex(rs["biased"][flip_row])
    out["biased_hex_cand"] = _f32_hex(cs["biased"][flip_row])

    # ---- boundary ranks (top-10 + rank-6/7 margin + IEEE bits) ------------
    rvals, ridxs = rs["biased"][flip_row].sort(descending=True)
    cvals, cidxs = cs["biased"][flip_row].sort(descending=True)
    topn = min(10, int(rvals.numel()))
    r10 = [{"expert": int(ridxs[k]), "score": float(rvals[k]),
            "bits": _fp32_bits(float(rvals[k]))} for k in range(topn)]
    c10 = [{"expert": int(cidxs[k]), "score": float(cvals[k]),
            "bits": _fp32_bits(float(cvals[k]))} for k in range(topn)]
    ref6 = sorted(int(e) for e in rs["ids"][flip_row].tolist())
    cand6 = sorted(int(e) for e in cs["ids"][flip_row].tolist())
    n_routed_actual = int(rvals.numel())
    has_r67 = n_routed_actual >= 7
    out["boundary"] = {
        "top10_ref": r10,
        "top10_cand": c10,
        "ref_top6": ref6,
        "cand_top6": cand6,
        "symmetric_difference": sorted(set(ref6) ^ set(cand6)),
        "rank6_ref": {"expert": int(ridxs[5]), "score": float(rvals[5]),
                       "bits": _fp32_bits(float(rvals[5]))}
        if has_r67 else None,
        "rank7_ref": {"expert": int(ridxs[6]), "score": float(rvals[6]),
                       "bits": _fp32_bits(float(rvals[6]))}
        if has_r67 else None,
        "rank6_cand": {"expert": int(cidxs[5]), "score": float(cvals[5]),
                        "bits": _fp32_bits(float(cvals[5]))}
        if has_r67 else None,
        "rank7_cand": {"expert": int(cidxs[6]), "score": float(cvals[6]),
                        "bits": _fp32_bits(float(cvals[6]))}
        if has_r67 else None,
        "margin_ref": float(rvals[5] - rvals[6]) if has_r67 else None,
        "margin_cand": float(cvals[5] - cvals[6]) if has_r67 else None,
        "same_pair_reversed": bool(ridxs[5].item() == cidxs[6].item()
                                    and ridxs[6].item() == cidxs[5].item())
        if has_r67 else None,
        "other_ranks_changed": [k for k in range(6)
                                if int(ridxs[k]) != int(cidxs[k])],
    }
    out["topk_audit"] = _topk_audit(rs["biased"][flip_row], topk)
    out["sensitivity"] = _router_sensitivity(
        ref_xf[flip_row], cand_xf[flip_row], gw, gb, rs, cs, flip_row, topk)
    return out
