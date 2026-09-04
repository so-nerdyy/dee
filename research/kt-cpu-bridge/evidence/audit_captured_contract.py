"""Audit-only decomposition of the sealed real replay; no policy/executor edits.

Revalidates the capture and official weights, reproduces archived output hashes,
runs the portable FP32 probe, and separates coverage from numerical diagnostics.
Grouped reductions are controlled software interventions, NOT native KT results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile

import torch

import replay_captured_real_expert as replay
from captured_real_diagnostics import bf16, boundary_variants, codec_diagnostics, swiglu, tensor_sha
from check_bf16_output_lattice import lattice_bounds

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
contract = replay.contract


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def decompose(metrics: dict) -> dict:
    gate = contract.ds8_gate_report(metrics)
    sample_checks = gate["sample_validity"]["checks"]
    candidate_checks = gate["candidate_fidelity"]["checks"]
    # The flat field preserves the prior report-reader convenience; structured
    # fields below are the contract authority and avoid conflating coverage
    # with candidate numerics.
    checks = dict(candidate_checks)
    checks["max_excluded_fraction"] = sample_checks["max_excluded_fraction"]
    failed = [name for name, value in sample_checks.items() if not value["pass"]]
    failed.extend(name for name, value in candidate_checks.items()
                  if not value["pass"] and name not in failed)
    assert gate["ds8_gate_passed"] == contract.ds8_gate_passed(metrics)
    return {"checks": checks, "actual_ds8_pass": gate["ds8_gate_passed"],
            "sample_validity": gate["sample_validity"],
            "candidate_fidelity": gate["candidate_fidelity"],
            "failed_checks": failed,
            # Retained name for old readers.  It now means the explicit
            # candidate-fidelity decision, not a coverage-excluded proxy.
            "candidate_dependent_checks_pass_diagnostic_only": gate["candidate_fidelity"]["pass"],
            "mask_counts": metrics["mask_counts"]}


def ordered_f32(x: torch.Tensor) -> torch.Tensor:
    bits = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    magnitude = bits & 0x7FFFFFFF
    # Collapse signed zeros, preserve numeric order on both sides of zero.
    return torch.where(bits >= 0x80000000, 0x80000000 - magnitude, 0x80000000 + magnitude)


def percentiles(x: torch.Tensor) -> dict:
    x = x.double().flatten()
    return {"count": x.numel(), "max": float(x.max()) if x.numel() else None,
            "mean": float(x.mean()) if x.numel() else None,
            **{name: float(x.quantile(q)) if x.numel() else None
               for name, q in (("p50", .5), ("p95", .95), ("p99", .99))}}


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    m = contract.compute_ds8_metrics(reference, candidate)
    err = (reference.double() - candidate.double()).abs()
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    same_sign_nnz = finite & (reference.abs() >= contract.NEAR_ZERO_THRESHOLD) & ((reference > 0) == (candidate > 0))
    ulp = (ordered_f32(reference) - ordered_f32(candidate)).abs()
    close = torch.isclose(reference, candidate, atol=1e-5, rtol=1e-4)
    out = {"metrics": m, "gate_decomposition": decompose(m),
           "output_sha256": tensor_sha(candidate), "reference_sha256": tensor_sha(reference),
           "counts": {"elements": reference.numel(), "different_values": int((reference != candidate).sum()),
                      "different_bits": int((reference.view(torch.int32) != candidate.view(torch.int32)).sum()),
                      **{f"abs_gt_{threshold}": int((err > float(threshold)).sum())
                         for threshold in ("1e-6", "1e-5", "1e-4", "1e-3")}},
           "fixed_fp32_allclose": {"pass": bool(close.all()), "failed_coordinates": int((~close).sum()),
                                    "atol": 1e-5, "rtol": 1e-4, "relative_term_uses_candidate": True,
                                    "formula": "abs(reference-candidate) <= 1e-5 + 1e-4*abs(candidate)"},
           "fp32_ulp_distance": {"finite_all": percentiles(ulp[finite]),
                                  "same_sign_reference_non_near_zero": percentiles(ulp[same_sign_nnz]),
                                  "sign_disagreement_count": int((finite & ((reference > 0) != (candidate > 0))).sum()),
                                  "note": "ULP is representable-FP32 step distance, not a relative-error gate; signed zeros collapse."}}
    if torch.equal(bf16(reference), reference) and torch.equal(bf16(candidate), candidate):
        out["bf16_ulp_distance"] = percentiles(ulp[finite].double() / 65536)
    return out


def legacy_metric_view(metrics: dict) -> dict:
    """Remove additive v2 report facts before comparing immutable v1 evidence."""
    copied = json.loads(json.dumps(metrics))
    copied.pop("sample_validity", None)
    return copied


def contract_probes() -> dict:
    ref = torch.ones(100)
    ref[:3] = 1e-4
    results = {}
    for name, cand in (("identity", ref.clone()), ("zero_candidate", torch.zeros_like(ref)),
                       ("large_finite_candidate", ref * 100)):
        m = contract.compute_ds8_metrics(ref, cand)
        results[name] = {"excluded": m["excluded"], "decomposition": decompose(m)}
    nonfinite = ref.clone()
    nonfinite[:3] = float("nan")
    m = contract.compute_ds8_metrics(ref, nonfinite)
    results["candidate_nan_can_change_finite_intersection_but_fails_mask"] = {
        "excluded": m["excluded"], "decomposition": decompose(m)}
    for name, r, c in (("matching_nan", [1., float("nan")], [1., float("nan")]),
                       ("matching_posinf", [1., float("inf")], [1., float("inf")]),
                       ("neginf_to_nan", [1., -float("inf")], [1., float("nan")]),
                       ("neginf_to_posinf", [1., -float("inf")], [1., float("inf")])):
        m = contract.compute_ds8_metrics(torch.tensor(r), torch.tensor(c))
        results[name] = {"decomposition": decompose(m)}
    return results


def serial_dot(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    product = w * x
    acc = torch.zeros(w.shape[0])
    for column in range(w.shape[1]):
        acc = acc + product[:, column]
    return acc.reshape(1, -1)


def grouped_dot(x: torch.Tensor, w: torch.Tensor, scales: torch.Tensor,
                natural: bool, half_tree: bool) -> tuple[torch.Tensor, dict]:
    """Source-inspired 32-value/16-lane diagnostic; no compiler/ISA claim.

    dpbf16 has zero source accumulator in pinned kernel. BF16*FP4 products
    are exact FP32 here. Pair addition, exact power-of-two scaling, group
    accumulation and horizontal trees are kept distinct and asserted.
    """
    assert torch.equal(x, bf16(x)) and torch.equal(w, bf16(w))
    n, k = w.shape
    scale = replay.trusted.decode_e8m0(scales).reshape(n, k // 32)
    raw_w = w.reshape(n, k // 32, 32) / scale[:, :, None]
    acts = x.reshape(1, k // 32, 32)
    if natural:
        order = torch.tensor(list(range(0, 32, 2)) + list(range(1, 32, 2)))
        raw_w, acts = raw_w[:, :, order], acts[:, :, order]
    product = raw_w * acts
    assert torch.equal(product.double(), raw_w.double() * acts.double())
    pair = product[:, :, 0::2] + product[:, :, 1::2]
    scaled = pair * scale[:, :, None]
    assert torch.equal(scaled.double(), pair.double() * scale[:, :, None].double())
    assert not bool(((scaled != 0) & (scaled.abs() < torch.finfo(torch.float32).tiny)).any())
    acc = torch.zeros((n, 16))
    for group in range(k // 32):
        acc = acc + scaled[:, group, :]
    while acc.shape[1] > 1:
        if half_tree:
            half = acc.shape[1] // 2
            acc = acc[:, :half] + acc[:, half:]
        else:
            acc = acc[:, 0::2] + acc[:, 1::2]
    return acc.reshape(1, -1), {"products_exact_fp32": True, "power_of_two_scaling_exact_fp32": True,
                               "scaled_pair_subnormals": 0, "pair_order": "natural_even_then_odd" if natural else "logical_adjacent",
                               "horizontal_tree": "halves" if half_tree else "adjacent"}


def reduction_diagnostics(x: torch.Tensor, weights: tuple, tensors: dict, weight: float,
                          expected: torch.Tensor, restored: torch.Tensor) -> dict:
    w1, w2, w3 = weights
    result = {}
    for mode in ("fp64_dots_then_fp32", "serial_fp32_dots"):
        dot = (lambda a, w: (a.double() @ w.double().t()).float()) if mode.startswith("fp64") else serial_dot
        gate, up = dot(x, w1), dot(x, w3)
        fp32_out = dot(swiglu(gate, up) * weight, w2)
        h = bf16(swiglu(bf16(gate), bf16(up)))
        kt_out = bf16(bf16(dot(h, w2)) * weight)
        result[mode] = {"fp32_path_vs_trusted": compare(expected, fp32_out),
                        "bf16_boundaries_vs_trusted": compare(expected, kt_out),
                        "bf16_boundaries_vs_blas_restored": compare(restored, kt_out)}
    for natural in (False, True):
        for half_tree in (False, True):
            gate, gm = grouped_dot(x, w1, tensors["w1.scale"], natural, half_tree)
            up, um = grouped_dot(x, w3, tensors["w3.scale"], natural, half_tree)
            h = bf16(swiglu(bf16(gate), bf16(up)))
            down, dm = grouped_dot(h, w2, tensors["w2.scale"], natural, half_tree)
            out = bf16(bf16(down) * weight)
            name = ("natural" if natural else "logical") + ("_halves_tree" if half_tree else "_adjacent_tree")
            result[name] = {"scope": "controlled pair/group/reduction sensitivity; PyTorch SiLU retained; TP=1/single contribution assumption; not native KT",
                            "stage_assertions": {"gate": gm, "up": um, "down": dm},
                            "vs_trusted": compare(expected, out), "vs_blas_restored": compare(restored, out),
                            "gate_rounding_changed_vs_blas": int((bf16(gate) != bf16(x @ w1.t())).sum()),
                            "up_rounding_changed_vs_blas": int((bf16(up) != bf16(x @ w3.t())).sum())}
    return result


def run() -> dict:
    torch.set_num_threads(1)
    baseline_path = EVIDENCE / "captured-real-expert155-diagnostics-20260904.json"
    baseline = replay.read_json(baseline_path)
    source = {k: Path(v) for k, v in baseline["diagnostics"]["sources"].items()}
    capture, seal = replay.provenance(source["capture_bundle"], source["validation"], source["sealed_bundle"], source["terminal_seal"])
    tensors, record, record_evidence = replay.load_real_expert(source["shard"], source["sealed_bundle"], source["terminal_seal"], 0, 155)
    assert seal == baseline["capture_seal"]
    assert json.loads(json.dumps(record_evidence)) == baseline["record_evidence"]
    assert contract.DS8_TOLERANCE == baseline["contract"]["tolerance"]
    assert contract.NEAR_ZERO_THRESHOLD == 0.001
    executor = Path(baseline["executor"]["path"])
    assert replay.sha(executor.read_bytes()) == baseline["executor"]["sha256"]
    subprocess.run([str(executor), "--self-test"], check=True, capture_output=True, timeout=30)
    args = [tensors[name] for name in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")]
    weights = tuple(replay.trusted.dequantize_expert_weight(tensors[f"{p}.weight"], tensors[f"{p}.scale"]) for p in ("w1", "w2", "w3"))
    rows = []
    with tempfile.TemporaryDirectory(prefix="contract-audit-", dir=ROOT / "tmp") as scratch_dir:
        scratch = Path(scratch_dir)
        (scratch / "record.bin").write_bytes(record)
        for item, old in zip(capture["rows"], baseline["rows"], strict=True):
            identity = {k: item[k] for k in ("forward_step", "token_row", "topk_rank")}
            assert identity == {k: old[k] for k in identity}
            art = item["artifacts"]
            for key in ("native_input_fp16", "input_fp32", "router_weights_fp32"):
                assert replay.sha((source["capture_bundle"] / art[key]["path"]).read_bytes()) == art[key]["sha256"]
            x = replay.load_f16(source["capture_bundle"] / art["native_input_fp16"]["path"], 4096).float().reshape(1, -1)
            original = replay.load_f32(source["capture_bundle"] / art["input_fp32"]["path"], 4096).reshape_as(x)
            assert torch.equal(x, original) and torch.equal(x, bf16(x))
            router = replay.load_f32(source["capture_bundle"] / art["router_weights_fp32"]["path"], 6)
            assert router.tolist() == item["official_router_weights_rank_order"]
            assert item["expert_ids_rank_order"][item["topk_rank"]] == 155
            weight = float(router[item["topk_rank"]])
            assert weight == old["router_weight"] == item["selected_expert_weight"]
            expected = replay.expert_forward_reference(x, *args, routing_weight=weight)
            candidate = replay.kt_emulated_forward(x, *args, routing_weight=weight)
            assert tensor_sha(expected) == old["output_sha256"]["trusted_dee"]
            assert tensor_sha(candidate) == old["output_sha256"]["kt_candidate"]
            (scratch / "input.bin").write_bytes(x.numpy().tobytes())
            command = [str(executor), str(scratch / "record.bin"), str(scratch / "input.bin"), str(scratch / "cpp.bin"), repr(weight)]
            subprocess.run(command, check=True, capture_output=True, timeout=120)
            cpp = replay.load_f32(scratch / "cpp.bin", 4096).reshape_as(x)
            assert tensor_sha(cpp) == old["output_sha256"]["cpp_reference"]
            subprocess.run(command, check=True, capture_output=True, timeout=120)
            assert torch.equal(cpp, replay.load_f32(scratch / "cpp.bin", 4096).reshape_as(x))
            variants, _, _ = boundary_variants(x, original, *weights, weight)
            assert torch.equal(expected, variants["fp32_reference"]) and torch.equal(candidate, variants["emulator_existing"])
            restored = variants["emulator_plus_preweight_down_bf16"]
            comparisons = {"trusted_vs_cpp": compare(expected, cpp), "trusted_vs_kt_emulator": compare(expected, candidate),
                           "trusted_vs_kt_restored_down_boundary": compare(expected, restored),
                           "trusted_vs_ideal_bf16": compare(expected, bf16(expected)),
                           "cpp_vs_kt_emulator": compare(cpp, candidate),
                           "ideal_bf16_vs_kt_emulator": compare(bf16(expected), candidate)}
            assert legacy_metric_view(comparisons["trusted_vs_cpp"]["metrics"]) == old["cpp_reference_full_ds8_metrics"]
            assert legacy_metric_view(comparisons["trusted_vs_kt_emulator"]["metrics"]) == old["kt_candidate_full_ds8_metrics"]
            decomposition = {name: compare(expected, value) for name, value in variants.items()}
            for name, value in decomposition.items():
                assert legacy_metric_view(value["metrics"]) == old["diagnostics"]["variants"][name]["full_ds8_metrics"]
            identity_metrics = contract.compute_ds8_metrics(expected, expected)
            finite_candidate_exclusions = {name: val["metrics"]["excluded"] for name, val in decomposition.items()}
            assert all(ex == identity_metrics["excluded"] for ex in finite_candidate_exclusions.values())
            rows.append({**identity, "router_weight": weight, "comparisons": comparisons,
                         "reference_only_sample_coverage": identity_metrics["excluded"],
                         "fp32_identity_gate_decomposition": decompose(identity_metrics),
                         "finite_candidate_exclusion_invariant": True,
                         "bf16_output_lattice": lattice_bounds(expected),
                         "boundary_ablations": decomposition,
                         "archived_stage_and_cancellation_diagnostics": {k: old["diagnostics"][k] for k in ("input", "clamp_events", "stages", "worst_relative_cancellation")},
                         "reduction_diagnostics": reduction_diagnostics(x, weights, tensors, weight, expected, restored),
                         "repeat_checks": {"cpp_bitwise": True,
                                           "trusted_bitwise": torch.equal(expected, replay.expert_forward_reference(x, *args, routing_weight=weight)),
                                           "kt_emulator_bitwise": torch.equal(candidate, replay.kt_emulated_forward(x, *args, routing_weight=weight))}})
            print(f"Audited step {item['forward_step']}; archived hashes and metrics reproduced", flush=True)
    probes = contract_probes()
    assert all(row["comparisons"]["trusted_vs_cpp"]["gate_decomposition"]["candidate_fidelity"]["pass"] for row in rows)
    assert all(not row["comparisons"]["trusted_vs_kt_emulator"]["gate_decomposition"]["candidate_fidelity"]["pass"] for row in rows)
    assert all(not row["bf16_output_lattice"]["strict_allclose"]["any_bf16_output_can_pass"] for row in rows)
    return {"schema": "kt-correctness-contract-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "audit_base_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "scope": "three real captured inputs, one expert; trusted output is recomputed FP32 CPU oracle, not GPU output dump; no native KT",
            "contract": {"near_zero_threshold": contract.NEAR_ZERO_THRESHOLD, "tolerance": contract.DS8_TOLERANCE,
                         "source_sha256": replay.sha(Path(contract.__file__).read_bytes()),
                         "baseline_source_sha256": baseline["diagnostics"]["contract_source_sha256"],
                         "numerical_tolerances_modified": False,
                         "gate_structure_refactored": True},
            "source_baseline_sha256": replay.sha(baseline_path.read_bytes()),
            "script_sha256": replay.sha(Path(__file__).read_bytes()), "capture_seal": seal, "record_evidence": record_evidence,
            "codec_diagnostics": codec_diagnostics(tensors), "source_paths": {k: str(v) for k, v in source.items()},
            "executor": baseline["executor"], "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
            "rows": rows, "contract_probes": probes,
            "verdict": {"classification": "CONTRACT_REVIEW_REQUIRED_KT_STILL_FAILS",
                        "strict_fp32_status": "KT_NUMERICALLY_INCOMPATIBLE_WITH_STRICT_FP32_CONTRACT",
                        "coverage_failed_steps": [r["forward_step"] for r in rows if not r["fp32_identity_gate_decomposition"]["actual_ds8_pass"]],
                        "native_kt_ds8_fidelity": "unresolved; no native execution or instruction-exact emulator",
                        "native_kt_executed": False, "production_cpu_enabled": False, "policy_changed": False,
                        "structural_separation_is_diagnostic_only": False,
                        "candidate_fidelity_is_separate_from_sample_validity": True}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    report = run()
    options.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], sort_keys=True))
