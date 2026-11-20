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

import math
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

    abs_err = (ref - cand).abs()
    near_zero = ref.abs() < NEAR_ZERO_THRESHOLD
    not_near_zero = ~near_zero
    excluded_fraction = float(near_zero.float().mean())

    def rel_of(mask: torch.Tensor) -> torch.Tensor:
        denom = ref[mask].abs() + 1e-8
        return abs_err[mask] / denom

    rel_all = abs_err / (ref.abs() + 1e-8)
    rel_nnz = rel_of(not_near_zero)

    # cosine similarity over all elements.
    denom = (ref.norm() * cand.norm())
    cosine = float((ref @ cand) / denom) if denom.item() > 0 else 1.0
    # Normalized RMSE: RMSE divided by the reference RMS, so it is
    # scale-invariant like the cosine/relative gates.
    ref_rms = float(ref.pow(2).mean().sqrt())
    normalized_rmse = float((abs_err.pow(2).mean()).sqrt() / max(ref_rms, 1e-30))
    ref_norm = float(ref.norm())
    out_norm_rel = float(abs(ref_norm - float(cand.norm())) / max(ref_norm, 1e-30))

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
            "index": int(abs_err.argmax()),
            "reference": float(ref[abs_err.argmax()]),
            "candidate": float(cand[abs_err.argmax()]),
            "abs_error": float(abs_err.max()),
        },
    }
    if rel_nnz.numel():
        worst_nnz_idx = int(rel_nnz.argmax())
        metrics["worst_non_near_zero_rel_element"] = {
            "flat_index": int(not_near_zero.nonzero()[worst_nnz_idx]),
            "reference": float(ref[not_near_zero.nonzero()[worst_nnz_idx]]),
            "candidate": float(cand[not_near_zero.nonzero()[worst_nnz_idx]]),
            "rel_error": float(rel_nnz.max()),
        }
    return metrics


def ds8_gate_passed(metrics: dict[str, Any], tol: dict[str, float] | None = None) -> bool:
    """Check the predeclared DS8 gates against computed metrics."""
    tol = tol or DS8_TOLERANCE
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
