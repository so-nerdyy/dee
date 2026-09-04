"""Offline, diagnostic-only boundary ablations; never an acceptance backend.

The caller must validate the capture and real record first. Every variant is
measured against the SAME unmodified DS8 contract and reference output. Removing
a BF16 boundary is an attribution experiment, not a proposed native-KT fix.
"""
from __future__ import annotations

import hashlib

import torch

from kt_cpu_bridge.codec import compact_scales_e8, dequantize_weight, expand_e8_scales
from scripts import deepseek_v4_contract as contract
from scripts import deepseek_v4_expert_reference as trusted


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def tensor_sha(x: torch.Tensor) -> str:
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


def failed_gates(metrics: dict) -> list[str]:
    """Explain DS8 without editing its tolerances or overriding its verdict."""
    tol = contract.DS8_TOLERANCE
    failures = []
    for name in ("sentinel_mask_exact", "finite_overlap"):
        if not metrics[name]:
            failures.append(name)
    if not metrics["finite_overlap"]:
        return failures
    checks = {name: metrics["all_elements"][name]
              for name in ("max_abs_error", "mean_abs_error")}
    checks.update({name: metrics["non_near_zero"][name]
                   for name in ("mean_rel_error", "p95_rel_error", "p99_rel_error")})
    checks.update({name: metrics[name]
                   for name in ("normalized_rmse", "output_norm_rel_error")})
    checks["max_excluded_fraction"] = metrics["excluded"]["fraction"]
    failures.extend(name for name, value in checks.items() if value > tol[name])
    if metrics["cosine_similarity"] < tol["cosine_similarity"]:
        failures.append("cosine_similarity")
    assert bool(not failures) == contract.ds8_gate_passed(metrics)
    return failures


def measure(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    metrics = contract.compute_ds8_metrics(reference, candidate)
    return {"ds8_pass": contract.ds8_gate_passed(metrics),
            "failed_gates": failed_gates(metrics),
            "output_sha256": tensor_sha(candidate),
            "full_ds8_metrics": metrics}


def codec_diagnostics(tensors: dict) -> dict:
    result = {}
    for name in ("w1", "w3", "w2"):
        packed, scale = tensors[name + ".weight"], tensors[name + ".scale"]
        reference = trusted.dequantize_expert_weight(packed, scale)
        decoded = dequantize_weight(packed, scale)
        adapted = (scale.to(torch.int16) << 7).view(torch.bfloat16).float()
        values, counts = torch.unique(scale, return_counts=True)
        result[name] = {
            "packed_shape": list(packed.shape), "scale_shape": list(scale.shape),
            "packed_sha256": tensor_sha(packed), "scale_sha256": tensor_sha(scale),
            "scale_byte_histogram": {str(int(v)): int(n) for v, n in zip(values, counts)},
            "weight_decode_exact": bool(torch.equal(reference, decoded)),
            "dequantized_weight_bf16_exact": bool(torch.equal(bf16(reference), reference)),
            "loader_scale_bf16_exact": bool(torch.equal(adapted, trusted.decode_e8m0(scale))),
            "scale_compaction_byte_exact": bool(torch.equal(compact_scales_e8(scale), scale)),
            "scale_expansion_exact": bool(torch.equal(expand_e8_scales(scale), trusted.decode_e8m0(scale))),
        }
        for key in ("weight_decode_exact", "dequantized_weight_bf16_exact",
                    "loader_scale_bf16_exact", "scale_compaction_byte_exact", "scale_expansion_exact"):
            if not result[name][key]:
                raise ValueError(f"real record codec mismatch: {name}.{key}")
    return result


def boundary_variants(x: torch.Tensor, original: torch.Tensor, w1: torch.Tensor,
                      w2: torch.Tensor, w3: torch.Tensor, weight: float,
                      limit: float = 10.0) -> tuple[dict, dict, dict]:
    """Return independent/cumulative interventions and intermediate tensors."""
    gate, up = x @ w1.t(), x @ w3.t()
    h = swiglu(gate, up, limit)
    xb = bf16(x)
    gx, ux = xb @ w1.t(), xb @ w3.t()
    gb, ub = bf16(gx), bf16(ux)
    hb = bf16(swiglu(gb, ub, limit))
    down = hb @ w2.t()
    reference = (h * weight) @ w2.t()
    gc = gate.clamp(max=limit) if limit > 0 else gate
    uc = up.clamp(min=-limit, max=limit) if limit > 0 else up
    variants = {
        "fp32_reference": reference,
        "original_fp32_input": (swiglu(original @ w1.t(), original @ w3.t(), limit) * weight) @ w2.t(),
        "routing_weight_late_only": (h @ w2.t()) * weight,
        "bf16_input_only": (swiglu(gx, ux, limit) * weight) @ w2.t(),
        "bf16_gate_only": (swiglu(bf16(gate), up, limit) * weight) @ w2.t(),
        "bf16_up_only": (swiglu(gate, bf16(up), limit) * weight) @ w2.t(),
        "bf16_gate_up_only": (swiglu(bf16(gate), bf16(up), limit) * weight) @ w2.t(),
        "bf16_intermediate_only": (bf16(h) * weight) @ w2.t(),
        "bf16_final_only": bf16(reference),
        "bf16_preweight_down_only": bf16(h @ w2.t()) * weight,
        "bf16_input_gate_up_intermediate": (hb * weight) @ w2.t(),
        "emulator_existing": bf16(down * weight),
        "emulator_plus_preweight_down_bf16": bf16(bf16(down) * weight),
        "emulator_without_gate_up_rounding": bf16((bf16(swiglu(gx, ux, limit)) @ w2.t()) * weight),
        "emulator_without_intermediate_rounding": bf16((swiglu(gb, ub, limit) @ w2.t()) * weight),
        "emulator_without_final_rounding": down * weight,
        "silu_div_exp_only": ((gc / (1 + torch.exp(-gc)) * uc) * weight) @ w2.t(),
    }
    ref_stages = {"input": x, "gate": gate, "up": up, "intermediate": h,
                  "unweighted_down": h @ w2.t()}
    kt_stages = {"input": xb, "gate": gb, "up": ub, "intermediate": hb,
                 "unweighted_down": down}
    return variants, ref_stages, kt_stages


def row_diagnostics(x: torch.Tensor, original: torch.Tensor, weights: tuple,
                    weight: float, expected: torch.Tensor, candidate: torch.Tensor) -> dict:
    w1, w2, w3 = weights
    variants, ref_stages, kt_stages = boundary_variants(x, original, w1, w2, w3, weight)
    if not torch.equal(expected, variants["fp32_reference"]):
        raise ValueError("diagnostic FP32 reference drift")
    if not torch.equal(candidate, variants["emulator_existing"]):
        raise ValueError("diagnostic candidate drift")
    original_bf = bf16(original)
    flat_ref, flat_cand = expected.flatten(), candidate.flatten()
    eligible = (flat_ref.abs() >= contract.NEAR_ZERO_THRESHOLD).nonzero().flatten()
    relative = (flat_ref[eligible] - flat_cand[eligible]).abs() / (flat_ref[eligible].abs() + 1e-8)
    worst = eligible[relative.argsort(descending=True)[:3]]
    cancellations = []
    for coord in worst.tolist():
        ref_terms = (ref_stages["intermediate"] * weight).flatten() * w2[coord]
        kt_terms = (kt_stages["intermediate"] * weight).flatten() * w2[coord]
        raw_weighted = kt_stages["unweighted_down"][0, coord] * weight
        cancellations.append({
            "index": coord, "reference": float(flat_ref[coord]), "candidate": float(flat_cand[coord]),
            "reference_sum_abs_terms": float(ref_terms.abs().sum()),
            "reference_cancellation_ratio": float(ref_terms.abs().sum() / flat_ref[coord].abs()),
            "intermediate_error_projected_at_coordinate": float((kt_terms - ref_terms).sum()),
            "candidate_before_final_bf16": float(raw_weighted),
            "final_bf16_error_at_coordinate": float(flat_cand[coord] - raw_weighted),
        })
    return {
        "scope": "diagnostic ablations, no native KT; no removed boundary is an accepted fix",
        "input": {
            "original_fp32_sha256": tensor_sha(original), "native_as_fp32_sha256": tensor_sha(x),
            "original_exactly_bf16": bool(torch.equal(original, original_bf)),
            "native_equal_original_count": int((x == original).sum()),
            "native_vs_original_max_abs": float((x - original).abs().max()),
            "bf16_native_vs_original_changed_count": int((bf16(x) != original_bf).sum()),
            "bf16_input_changed_count": int((bf16(x) != x).sum()),
        },
        "clamp_events": {"gate": int((ref_stages["gate"] > 10).sum()),
                         "up": int((ref_stages["up"].abs() > 10).sum()),
                         "gate_min": float(ref_stages["gate"].min()),
                         "gate_max": float(ref_stages["gate"].max()),
                         "up_min": float(ref_stages["up"].min()),
                         "up_max": float(ref_stages["up"].max())},
        "stages": {name: {"reference_sha256": tensor_sha(value),
                           "candidate_sha256": tensor_sha(kt_stages[name]),
                           "changed_count": int((value != kt_stages[name]).sum()),
                           "max_abs": float((value - kt_stages[name]).abs().max()),
                           "rmse": float((value - kt_stages[name]).square().mean().sqrt())}
                   for name, value in ref_stages.items()},
        "variants": {name: measure(expected, value) for name, value in variants.items()},
        "preweight_down_boundary_changed_output_count": int(
            (candidate != variants["emulator_plus_preweight_down_bf16"]).sum()),
        "worst_relative_cancellation": cancellations,
    }
