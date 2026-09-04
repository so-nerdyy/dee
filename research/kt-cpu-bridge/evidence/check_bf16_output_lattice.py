"""Evidence-only representability proof, not an implementation of native KT.

The pinned TP merge returns BF16. Recompute each sealed FP32 oracle, then
test its closest BF16 neighbors. A coordinate that fails both bracketing
neighbors cannot be rescued by any other BF16 output under the strict
allclose inequality, because rtol < 1 makes the residual increase away
from the reference on either side. DS8 is evaluated separately, unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import replay_captured_real_expert as replay
from captured_real_diagnostics import measure, tensor_sha


def lattice_bounds(reference: torch.Tensor) -> dict:
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("This proof requires finite reference values")
    nearest_bf16 = reference.to(torch.bfloat16)
    nearest = nearest_bf16.float()
    predecessor = torch.nextafter(nearest_bf16, torch.full_like(nearest_bf16, -float("inf"))).float()
    successor = torch.nextafter(nearest_bf16, torch.full_like(nearest_bf16, float("inf"))).float()
    lower = torch.where(nearest <= reference, nearest, predecessor)
    upper = torch.where(nearest >= reference, nearest, successor)
    if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        raise ValueError("This proof requires finite bracketing BF16 values")
    assert bool((lower <= reference).all() and (reference <= upper).all())
    # Float64 here evaluates the mathematical inequality on exact FP32
    # values. It does not recalculate the FP32 oracle in higher precision.
    ref64, low64, high64 = reference.double(), lower.double(), upper.double()
    atol, rtol = 1e-5, 1e-4
    residual_lower = (ref64 - low64).abs() - (atol + rtol * low64.abs())
    residual_upper = (ref64 - high64).abs() - (atol + rtol * high64.abs())
    impossible = (residual_lower > 0) & (residual_upper > 0)
    error = (reference - nearest).abs()
    # This equality verifies that the BF16 conversion selected an absolute-
    # error-minimizing neighbor for every observed coordinate.
    assert torch.equal(error, torch.minimum((reference - lower).abs(), (reference - upper).abs()))
    indices = impossible.flatten().nonzero().flatten()
    witnesses = []
    for index in indices[:3].tolist():
        witnesses.append({
            "index": index, "reference": float(reference.flatten()[index]),
            "lower_bf16": float(lower.flatten()[index]), "upper_bf16": float(upper.flatten()[index]),
            "lower_strict_residual": float(residual_lower.flatten()[index]),
            "upper_strict_residual": float(residual_upper.flatten()[index]),
        })
    return {
        "output_elements": reference.numel(),
        "reference_elements_not_representable_in_bf16": int((nearest != reference).sum()),
        "bitwise_fp32_oracle_equality_possible_for_bf16_output": bool(torch.equal(reference, nearest)),
        "minimum_possible_max_abs_error": float(error.max()),
        "minimum_possible_mean_abs_error": float(error.mean()),
        "strict_allclose": {
            "atol": atol, "rtol": rtol, "relative_term_uses_candidate": True,
            "coordinates_no_bf16_candidate_can_satisfy": int(impossible.sum()),
            "any_bf16_output_can_pass": not bool(impossible.any()),
            "nearest_bf16_torch_allclose": bool(torch.allclose(reference, nearest, atol=atol, rtol=rtol)),
            "witnesses": witnesses,
        },
        "nearest_bf16_ds8": measure(reference, nearest),
        "fp32_identity_ds8": measure(reference, reference),
    }


def run(diagnostics_path: Path) -> dict:
    baseline = replay.read_json(diagnostics_path)
    source = {key: Path(value) for key, value in baseline["diagnostics"]["sources"].items()}
    capture, seal = replay.provenance(source["capture_bundle"], source["validation"],
                                      source["sealed_bundle"], source["terminal_seal"])
    tensors, _, record = replay.load_real_expert(source["shard"], source["sealed_bundle"],
                                                source["terminal_seal"], 0, 155)
    replay.require(seal == baseline["capture_seal"], "capture seal drift")
    # load_real_expert returns occurrence tuples; the sealed JSON stores
    # those same ordered triples as arrays. Compare the JSON representation.
    replay.require(json.loads(json.dumps(record)) == baseline["record_evidence"], "source record drift")
    replay.require(baseline["contract"]["tolerance"] == replay.contract.DS8_TOLERANCE, "contract drift")
    contract_sha = replay.sha(Path(replay.contract.__file__).read_bytes())
    replay.require(contract_sha == baseline["diagnostics"]["contract_source_sha256"], "contract source drift")
    args = [tensors[name] for name in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")]
    rows = []
    for item, old in zip(capture["rows"], baseline["rows"], strict=True):
        for name in ("native_input_fp16", "input_fp32", "router_weights_fp32"):
            artifact = item["artifacts"][name]
            raw = (source["capture_bundle"] / artifact["path"]).read_bytes()
            replay.require(replay.sha(raw) == artifact["sha256"], f"capture artifact drift: {name}")
        identity = {key: item[key] for key in ("forward_step", "token_row", "topk_rank")}
        replay.require(identity == {key: old[key] for key in identity}, "row identity drift")
        native = item["artifacts"]["native_input_fp16"]
        original = item["artifacts"]["input_fp32"]
        router = item["artifacts"]["router_weights_fp32"]
        x = replay.load_f16(source["capture_bundle"] / native["path"], 4096).float().reshape(1, 4096)
        x_original = replay.load_f32(source["capture_bundle"] / original["path"], 4096).reshape_as(x)
        replay.require(torch.equal(x, x_original), "original/native activation mismatch")
        weights = replay.load_f32(source["capture_bundle"] / router["path"], 6)
        replay.require(weights.tolist() == item["official_router_weights_rank_order"], "router vector drift")
        replay.require(item["expert_ids_rank_order"][item["topk_rank"]] == 155, "wrong expert")
        weight = float(weights[item["topk_rank"]])
        replay.require(weight == item["selected_expert_weight"] == old["router_weight"], "router weight drift")
        expected = replay.expert_forward_reference(x, *args, routing_weight=weight)
        replay.require(tensor_sha(expected) == old["output_sha256"]["trusted_dee"], "FP32 oracle hash drift")
        bounds = lattice_bounds(expected)
        replay.require(bounds["nearest_bf16_ds8"] == old["diagnostics"]["variants"]["bf16_final_only"],
                       "nearest BF16 diagnostic drift")
        replay.require(bounds["fp32_identity_ds8"] == old["diagnostics"]["variants"]["fp32_reference"],
                       "identity diagnostic drift")
        rows.append({**identity, "router_weight": weight, "oracle_sha256": tensor_sha(expected), **bounds})
    return {
        "schema": "kt-bf16-output-lattice-v1",
        "scope": "Output-representability lower bound only; not a proposed candidate or native KT execution",
        "native_kt_executed": False,
        "upstream_commit": "31985f40bcc40da08107efdb1f81bf88cb38c6b2",
        "source_baseline_sha256": replay.sha(diagnostics_path.read_bytes()),
        "script_sha256": replay.sha(Path(__file__).read_bytes()),
        "contract_source_sha256": contract_sha,
        "capture_manifest_sha256": seal["capture_manifest_sha256"],
        "record_sha256": record["record_sha256"],
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
        "rows": rows,
        "verdict": {
            "bf16_final_output_cannot_equal_the_fp32_oracle_on_any_captured_row":
                all(not row["bitwise_fp32_oracle_equality_possible_for_bf16_output"] for row in rows),
            "bf16_final_output_cannot_pass_strict_allclose_on_any_captured_row":
                all(not row["strict_allclose"]["any_bf16_output_can_pass"] for row in rows),
            "all_row_ds8_rejected_independently_of_candidate_by_reference_coverage":
                any("max_excluded_fraction" in row["fp32_identity_ds8"]["failed_gates"] for row in rows),
            "runtime_change_justified": False,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=Path(__file__).with_name("captured-real-expert155-diagnostics-20260904.json"))
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    report = run(options.diagnostics)
    options.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], sort_keys=True))
