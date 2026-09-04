"""Evidence-only replay of the guarded real layer-0/expert-155 capture.

This deliberately lives under research evidence, not the dee runtime.  It
does not mount, modify, or execute a full model.  It fail-closes on capture
provenance, all listed capture hashes, the sealed real expert record, and the
predeclared DS8 numerical contract before reporting the C++ reference and the
KT-emulated candidate on the three captured activation/router samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[3]
BRIDGE = ROOT / "dee.cpp" / "experiments" / "kt_cpu_bridge"
sys.path.insert(0, str(BRIDGE / "python"))
sys.path.insert(0, str(BRIDGE / "bench"))
sys.path.insert(0, str(ROOT / "dee.cpp"))

from kt_cpu_bridge.reference import expert_forward_reference, kt_emulated_forward
from scripts import deepseek_v4_contract as contract
from scripts import deepseek_v4_expert_reference as trusted
from verify_real_expert import load_real_expert


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_f32(path: Path, n: int) -> torch.Tensor:
    raw = path.read_bytes()
    require(len(raw) == n * 4, f"wrong FP32 byte count: {path.name}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def load_f16(path: Path, n: int) -> torch.Tensor:
    raw = path.read_bytes()
    require(len(raw) == n * 2, f"wrong FP16 byte count: {path.name}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float16).clone()


def compact_metrics(metrics: dict) -> dict:
    """Requested metric names, retaining the full contract metrics separately."""
    all_values, nnz = metrics["all_elements"], metrics["non_near_zero"]
    finite = bool(metrics["finite_overlap"] and metrics["sentinel_mask_exact"]
                  and metrics["mask_counts"]["nan_reference"] == 0
                  and metrics["mask_counts"]["nan_candidate"] == 0
                  and metrics["mask_counts"]["posinf_reference"] == 0
                  and metrics["mask_counts"]["posinf_candidate"] == 0)
    return {
        "max_abs": all_values["max_abs_error"],
        "mean_abs": all_values["mean_abs_error"],
        "mean_rel": nnz["mean_rel_error"],
        "p95_rel": nnz["p95_rel_error"],
        "p99_rel": nnz["p99_rel_error"],
        "cosine": metrics["cosine_similarity"],
        "finite": finite,
    }


def provenance(bundle: Path, validation_path: Path, sealed_bundle: Path,
               terminal_seal: Path) -> tuple[dict, dict]:
    validation = read_json(validation_path)
    capture = read_json(bundle / "kt-input-capture.json")
    require(validation["status"] == "VALIDATED_REAL_INPUT_CAPTURE", "capture not validated")
    require(sha((bundle / "kt-input-capture.json").read_bytes()) ==
            validation["capture_manifest_sha256"], "capture manifest hash mismatch")
    require(capture["run_id"] == validation["run_id"], "capture run id mismatch")
    require(capture["layer"] == validation["layer0_expert155_source_record"]["layer"] == 0,
            "capture layer mismatch")
    require(capture["expert"] == validation["layer0_expert155_source_record"]["expert"] == 155,
            "capture expert mismatch")
    require(capture["row_count"] == validation["rows"] == 3, "capture row count mismatch")
    require(validation["all_18_generation_gates"], "one or more generation gates false")
    require(validation["all_688_route_semantics_match_v65"], "route semantic comparison failed")
    require(capture["unchanged_generation_gates"] and
            all(capture["unchanged_generation_gates"].values()), "capture gate false")
    for name, expected in validation["verified_artifacts"].items():
        require(sha((bundle / name).read_bytes()) == expected, f"capture artifact hash mismatch: {name}")

    environment = read_json(bundle / "environment.json")
    result = read_json(bundle / "result.json")
    terminal = read_json(terminal_seal)
    index_path = sealed_bundle / "model.safetensors.index.json"
    require(sha(index_path.read_bytes()) == terminal["raw_sha256"]["model.safetensors.index.json"],
            "sealed checkpoint index hash mismatch")
    index = read_json(index_path)
    dee4 = read_json(bundle / "dee4-trace-validation.json")
    dee4_import = read_json(bundle / "dee4-import-validation.json")
    p22 = read_json(bundle / "p2.2-dee4-evidence.json")
    route_lines = (bundle / "routed_experts.jsonl").read_text(encoding="utf-8").splitlines()
    route_journal = result["route_journal"]
    require(len(route_lines) == route_journal["record_count"] == 688, "route record count mismatch")
    require(route_journal["chain_sha256"] == validation["route_chain_sha256"], "route chain mismatch")
    require(route_journal["file_sha256"] == validation["verified_artifacts"]["routed_experts.jsonl"],
            "route file hash mismatch")
    require(result["execution_terminal"] == {"phase": "complete", "layers_executed": 43},
            "generation not terminal at all 43 layers")
    require(len(result["generated_token_ids"]) == 16 and route_journal["completed_forwards"] == 16,
            "generation did not preserve the 16-token/forward gate")
    require(result["bridge_counters"]["native_calls"] == 688 and
            result["bridge_counters"]["native_batch_calls"] == 688, "native route call count mismatch")
    require(environment["storage_mount"] == "/kaggle/input/datasets/nivind/deepseek-v4-flash-0731-shards",
            "wrong capture storage mount")
    require(environment["gpu"]["gpu_count"] == 2 and environment["cuda_device_names"] == ["Tesla T4", "Tesla T4"],
            "capture is not 2x Tesla T4")
    require(p22["safetensors_storage"]["mountpoint"] == environment["storage_mount"] and
            p22["safetensors_storage"]["filesystem"] == "nfs" and
            "ro" in p22["safetensors_storage"]["mount_options"], "mount evidence mismatch")
    shard_names = set(index["weight_map"].values())
    require(len(shard_names) == 48 and "model-00002-of-00048.safetensors" in shard_names,
            "checkpoint does not declare 48 shards")
    data_hash = result["dee4_trace_validation"]["data_sha256"]
    require(dee4["success"] and dee4_import["success"] and p22["data_sha256"] == data_hash and
            dee4["data_sha256"] == data_hash, "DEE4 hash/validation mismatch")
    require(dee4["trace_record_count"] == 688 and dee4["integrity_records_complete"], "DEE4 record gate failed")
    return capture, {
        "capture_manifest_sha256": validation["capture_manifest_sha256"],
        "source_commit": validation["source_commit"],
        "model_revision": validation["model_revision"],
        "mount": {"path": environment["storage_mount"], "filesystem": p22["safetensors_storage"]["filesystem"],
                  "read_only": "ro" in p22["safetensors_storage"]["mount_options"]},
        "checkpoint_shards": len(shard_names),
        "hardware": environment["gpu"],
        "generation_gates": {"all_18": validation["all_18_generation_gates"], "tokens": len(result["generated_token_ids"]),
                             "layers": result["execution_terminal"]["layers_executed"], "routes": len(route_lines)},
        "dee4": {"data_sha256": data_hash, "integrity_sha256": validation["source_expert_integrity_file_sha256"],
                 "trace_record_count": dee4["trace_record_count"]},
    }


def replay(shard: Path, capture_bundle: Path, validation: Path, sealed_bundle: Path,
           terminal_seal: Path, executor: Path, diagnostics: bool = False) -> dict:
    capture, seal = provenance(capture_bundle, validation, sealed_bundle, terminal_seal)
    subprocess.run([str(executor), "--self-test"], check=True, timeout=30,
                   capture_output=True, text=True)
    tensors, record, record_evidence = load_real_expert(shard, sealed_bundle, terminal_seal, 0, 155)
    validated_record = read_json(validation)["layer0_expert155_source_record"]
    require(record_evidence["record_sha256"] == validated_record["record_sha256"], "real record hash differs from capture")
    require(record_evidence["component_sha256"] == validated_record["component_sha256"],
            "real component hashes differ from capture")
    args = [tensors[name] for name in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")]
    w1, w2, w3 = [trusted.dequantize_expert_weight(tensors[f"{p}.weight"], tensors[f"{p}.scale"])
                  for p in ("w1", "w2", "w3")]
    if diagnostics:
        from captured_real_diagnostics import codec_diagnostics, row_diagnostics
        codec_report = codec_diagnostics(tensors)
    rows = []
    with tempfile.TemporaryDirectory(prefix="dee-kt-captured-real-") as scratch_dir:
        scratch = Path(scratch_dir)
        (scratch / "record.bin").write_bytes(record)
        for item in capture["rows"]:
            native = item["artifacts"]["native_input_fp16"]
            original = item["artifacts"]["input_fp32"]
            router = item["artifacts"]["router_weights_fp32"]
            for artifact in (native, original, router):
                raw = (capture_bundle / artifact["path"]).read_bytes()
                require(sha(raw) == artifact["sha256"], f"row hash mismatch: {artifact['path']}")
            x_fp32 = load_f32(capture_bundle / original["path"], 4096)
            x_native = load_f16(capture_bundle / native["path"], 4096)
            require(torch.equal(x_fp32.to(torch.float16), x_native), "native FP16 capture is not the FP32 conversion")
            router_values = load_f32(capture_bundle / router["path"], 6)
            require(router_values.tolist() == item["official_router_weights_rank_order"],
                    "captured full router weight vector mismatch")
            rank = item["topk_rank"]
            require(item["expert_ids_rank_order"][rank] == 155, "captured rank does not select expert 155")
            weight = float(router_values[rank])
            require(weight == float(item["selected_expert_weight"]), "captured router weight mismatch")
            x = x_native.to(torch.float32).reshape(1, 4096)
            # The trusted dee oracle applies the selected router weight at the
            # same pre-down-projection point as the bridge reference.
            expected = expert_forward_reference(x, *args, routing_weight=weight)
            manual = ((torch.nn.functional.silu(torch.clamp(x @ w1.t(), max=10.0)) *
                       torch.clamp(x @ w3.t(), min=-10.0, max=10.0)) * weight) @ w2.t()
            require(torch.equal(expected, manual), "trusted dee/reference semantic mismatch")
            candidate = kt_emulated_forward(x, *args, routing_weight=weight)
            candidate_repeat = kt_emulated_forward(x, *args, routing_weight=weight)
            (scratch / "x.bin").write_bytes(x.numpy().tobytes())
            output = scratch / "cpp.bin"
            command = [str(executor), str(scratch / "record.bin"), str(scratch / "x.bin"), str(output), repr(weight)]
            subprocess.run(command, check=True, timeout=120)
            first = torch.frombuffer(bytearray(output.read_bytes()), dtype=torch.float32).clone().reshape(1, 4096)
            subprocess.run(command, check=True, timeout=120)
            second = torch.frombuffer(bytearray(output.read_bytes()), dtype=torch.float32).clone().reshape(1, 4096)
            cpp_metrics = contract.compute_ds8_metrics(expected, first)
            kt_metrics = contract.compute_ds8_metrics(expected, candidate)
            rows.append({
                "forward_step": item["forward_step"], "token_row": item["token_row"], "topk_rank": rank,
                "router_weight": weight, "input_fp16_sha256": native["sha256"],
                "trusted_dee_repeat_bitwise": bool(torch.equal(expected, expert_forward_reference(x, *args, routing_weight=weight))),
                "cpp_reference_repeat_bitwise": bool(torch.equal(first, second)),
                "kt_candidate_repeat_bitwise": bool(torch.equal(candidate, candidate_repeat)),
                "cpp_reference_fixed_fp32_allclose_atol_1e-5_rtol_1e-4": bool(torch.allclose(expected, first, atol=1e-5, rtol=1e-4)),
                "cpp_reference_ds8_pass": contract.ds8_gate_passed(cpp_metrics),
                "kt_candidate_ds8_pass": contract.ds8_gate_passed(kt_metrics),
                "trusted_dee_vs_cpp_reference": compact_metrics(cpp_metrics),
                "trusted_dee_vs_kt_candidate": compact_metrics(kt_metrics),
                "cpp_reference_full_ds8_metrics": cpp_metrics,
                "kt_candidate_full_ds8_metrics": kt_metrics,
                "output_sha256": {"trusted_dee": sha(expected.numpy().tobytes()), "cpp_reference": sha(first.numpy().tobytes()),
                                  "kt_candidate": sha(candidate.numpy().tobytes())},
            })
            if diagnostics:
                rows[-1]["diagnostics"] = row_diagnostics(
                    x, x_fp32.reshape(1, 4096), (w1, w2, w3), weight, expected, candidate)
    cpp_ok = all(row["cpp_reference_ds8_pass"] and row["cpp_reference_repeat_bitwise"] and
                 row["trusted_dee_repeat_bitwise"] and row["trusted_dee_vs_cpp_reference"]["finite"] for row in rows)
    kt_ok = all(row["kt_candidate_ds8_pass"] and row["kt_candidate_repeat_bitwise"] and
                row["trusted_dee_vs_kt_candidate"]["finite"] for row in rows)
    cpp_coverage_failure = any(row["cpp_reference_full_ds8_metrics"]["excluded"]["fraction"] >
                               contract.DS8_TOLERANCE["max_excluded_fraction"] for row in rows)
    report = {
        "schema": "kt-captured-real-expert-replay-v1", "scope": "evidence-only; one captured expert, no full-model or native KT execution",
        "capture_seal": seal, "record_evidence": record_evidence,
        "executor": {"path": str(executor), "sha256": sha(executor.read_bytes()), "self_test_pass": True},
        "contract": {"name": "predeclared DS8_TOLERANCE; unmodified", "tolerance": contract.DS8_TOLERANCE},
        "rows": rows,
        "verdict": {"trusted_dee_vs_cpp_reference_passes_strict_ds8": cpp_ok,
                    "cpp_reference_strict_gate_note": None if cpp_ok else
                    ("coverage gate only: captured rows exceed unchanged DS8 max_excluded_fraction=0.02; "
                     "all C++ numerical, finite-mask, and repeat checks passed" if cpp_coverage_failure else
                     "one or more unchanged DS8 numerical/semantic reference gates failed"),
                    "kt_candidate_passes_strict_ds8": kt_ok,
                    "kt_cpu_remains_disabled": not kt_ok,
                    "failure_classification": None if kt_ok else "numerical: deterministic finite KT emulation exceeds predeclared relative-error gates; no routing/provenance/finite-mask semantic failure observed",
                    "interface_cherry_pick_justified": False},
    }
    if diagnostics:
        import captured_real_diagnostics
        report["diagnostics"] = {
            "codec": codec_report, "native_kt_executed": False,
            "script_sha256": sha(Path(captured_real_diagnostics.__file__).read_bytes()),
            "replay_script_sha256": sha(Path(__file__).read_bytes()),
            "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
            "contract_source_sha256": sha(Path(contract.__file__).read_bytes()),
            "sources": {"shard": str(shard), "capture_bundle": str(capture_bundle),
                        "validation": str(validation), "sealed_bundle": str(sealed_bundle),
                        "terminal_seal": str(terminal_seal)},
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shard", "capture_bundle", "validation", "sealed_bundle", "terminal_seal", "executor", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--diagnostics", action="store_true", help="Record independent BF16 boundary ablations")
    options = vars(parser.parse_args())
    out = options.pop("out")
    report = replay(**options)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], sort_keys=True))
    return 0 if report["verdict"]["trusted_dee_vs_cpp_reference_passes_strict_ds8"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
