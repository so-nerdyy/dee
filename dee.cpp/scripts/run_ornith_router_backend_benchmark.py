#!/usr/bin/env python3
"""Measure the native-host and device-resident Ornith router backends.

The model and expert caches are loaded once.  After one warmup per backend,
trials alternate backend order so initialization and drift do not masquerade
as a throughput gain.  The optimized backend changes only routing placement:
the official checkpoint router stays on its layer GPU while dee.cpp continues
to execute every selected expert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ornith_generation import (  # noqa: E402
    compare_trace,
    fresh_engine_path_proof,
    load_runtime,
    run_generation,
)


BACKENDS = ("native-host", "torch-device")
EXPECTED_TOKENS = [11, 271, 40, 1044]
MAX_PROCESS_VRAM_BYTES = 8 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def summarize_trials(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty trials")
    rates = [float(row["tokens_per_second"]) for row in rows]
    decode_seconds = [
        float(value)
        for row in rows
        for value in row["per_token_decode_seconds"]
    ]
    peak_by_device: dict[str, int] = {}
    for row in rows:
        for device, value in row["peak_vram_bytes"].items():
            peak_by_device[device] = max(peak_by_device.get(device, 0), int(value))
    return {
        "trial_count": len(rows),
        "tokens_per_second": {
            "minimum": min(rates),
            "median": statistics.median(rates),
            "maximum": max(rates),
            "all": rates,
        },
        "decode_seconds": {
            "sample_count": len(decode_seconds),
            "median": statistics.median(decode_seconds),
            "all": decode_seconds,
        },
        "peak_process_vram_per_gpu_bytes": dict(sorted(peak_by_device.items())),
        "all_tokens_exact": all(row["tokens_exact"] for row in rows),
        "all_40_layers_executed": all(row["all_40_layers_executed"] for row in rows),
        "all_expert_calls_device_resident": all(
            row["path_proof"]["device_path_share"] == 1.0 for row in rows
        ),
        "host_fallback_calls": sum(
            int(row["path_proof"]["host_path_fallback_calls"]) for row in rows
        ),
        "router_native_host_calls": sum(
            int(row["path_proof"]["router_native_host_calls"]) for row in rows
        ),
        "router_torch_device_calls": sum(
            int(row["path_proof"]["router_torch_device_calls"]) for row in rows
        ),
        "router_hidden_d2h_total_bytes": sum(
            int(row["path_proof"]["router_hidden_d2h_total_bytes"]) for row in rows
        ),
        "router_outputs_h2d_total_bytes": sum(
            int(row["path_proof"]["router_outputs_h2d_total_bytes"]) for row in rows
        ),
        "router_scalar_sync_calls": sum(
            int(row["path_proof"]["router_scalar_sync_calls"]) for row in rows
        ),
    }


def proof_with_shares(context) -> dict[str, Any]:
    proof = dict(context.engine_path_proof)
    total = proof["device_path_calls"] + proof["host_path_fallback_calls"]
    proof["device_path_share"] = proof["device_path_calls"] / total if total else 0.0
    proof["host_fallback_share"] = (
        proof["host_path_fallback_calls"] / total if total else 0.0
    )
    proof["router_backend"] = context.router_backend
    return proof


def run_once(runtime, tokenizer, backend: str, *, trace: bool) -> tuple[dict, dict]:
    context = runtime["context"]
    context.router_backend = backend
    context.engine_path_proof = fresh_engine_path_proof()
    result = run_generation(
        runtime,
        tokenizer,
        "Hello",
        len(EXPECTED_TOKENS),
        False,
        "dee",
        trace=trace,
    )
    proof = proof_with_shares(context)
    serializable = {
        "backend": backend,
        "generated_token_ids": result["generated_token_ids"],
        "generated_text": result["generated_text"],
        "tokens_exact": result["generated_token_ids"] == EXPECTED_TOKENS,
        "all_40_layers_executed": result["executed_router_layers"] == list(range(40)),
        "tokens_per_second": result["single_stream_decode_tokens_per_second"],
        "per_token_decode_seconds": result["per_token_decode_seconds"],
        "prefill_seconds": result["prefill_seconds"],
        "total_generation_seconds": result["total_generation_seconds"],
        "peak_vram_bytes": result["resources"]["peak_vram_bytes"],
        "peak_host_rss_bytes": result["resources"]["peak_host_rss_bytes"],
        "path_proof": proof,
    }
    if not serializable["tokens_exact"]:
        raise RuntimeError(
            f"{backend} tokens changed: {result['generated_token_ids']} "
            f"!= {EXPECTED_TOKENS}"
        )
    if not serializable["all_40_layers_executed"]:
        raise RuntimeError(
            f"{backend} did not route every layer: {result['executed_router_layers']}"
        )
    if proof["device_path_share"] != 1.0 or proof["host_path_fallback_calls"] != 0:
        raise RuntimeError(f"{backend} expert path regressed: {proof}")
    if max(result["resources"]["peak_vram_bytes"].values()) > MAX_PROCESS_VRAM_BYTES:
        raise RuntimeError(f"{backend} exceeded the 8 GiB/GPU process VRAM gate")
    return serializable, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-experts", type=int, default=32)
    parser.add_argument("--split-layer", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--require-dual-gpu", action="store_true")
    args = parser.parse_args()
    if args.cache_experts < 8:
        parser.error("--cache-experts must be at least top-k 8")
    if args.trials < 3:
        parser.error("--trials must be at least 3")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from transformers import AutoTokenizer

    gpu_count = torch.cuda.device_count()
    if args.require_dual_gpu and gpu_count != 2:
        raise RuntimeError(f"dual-T4 benchmark requires exactly 2 GPUs, got {gpu_count}")
    if gpu_count < 2:
        raise RuntimeError("router backend benchmark requires the two-GPU Ornith layout")
    if args.require_dual_gpu and any(
        "T4" not in torch.cuda.get_device_name(device) for device in range(2)
    ):
        raise RuntimeError(
            "dual-T4 benchmark found "
            + ", ".join(torch.cuda.get_device_name(device) for device in range(2))
        )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    runtime = load_runtime(
        args.model_dir,
        2,
        args.split_layer,
        args.cache_experts,
        router_backend="native-host",
    )
    load_seconds = time.perf_counter() - started

    # Warm both backends before measurement.  Native first also materializes
    # its separate baseline router buffers; neither warmup contributes a row.
    warmups: dict[str, dict[str, Any]] = {}
    for backend in BACKENDS:
        row, _raw = run_once(runtime, tokenizer, backend, trace=False)
        warmups[backend] = row

    # Pair-balanced alternating order for three trials, repeated if requested.
    base_order = [
        "native-host",
        "torch-device",
        "torch-device",
        "native-host",
        "native-host",
        "torch-device",
    ]
    order: list[str] = []
    while any(order.count(backend) < args.trials for backend in BACKENDS):
        for backend in base_order:
            if order.count(backend) < args.trials:
                order.append(backend)
    trials: dict[str, list[dict[str, Any]]] = {backend: [] for backend in BACKENDS}
    sequence: list[dict[str, Any]] = []
    for sequence_index, backend in enumerate(order):
        row, _raw = run_once(runtime, tokenizer, backend, trace=False)
        row["sequence_index"] = sequence_index
        row["backend_trial_index"] = len(trials[backend])
        trials[backend].append(row)
        sequence.append(row)

    # One traced A/B pair proves that the routing placement change preserves
    # every existing trace category, selected expert output, intermediate
    # hidden state, token ID, and decoded text.
    native_trace_row, native_trace = run_once(
        runtime, tokenizer, "native-host", trace=True
    )
    device_trace_row, device_trace = run_once(
        runtime, tokenizer, "torch-device", trace=True
    )
    comparisons = compare_trace(
        native_trace["collector"], device_trace["collector"]
    )
    correctness = {
        "native_tokens": native_trace_row["generated_token_ids"],
        "optimized_tokens": device_trace_row["generated_token_ids"],
        "generated_token_ids_exact": (
            native_trace_row["generated_token_ids"]
            == device_trace_row["generated_token_ids"]
            == EXPECTED_TOKENS
        ),
        "decoded_text_exact": (
            native_trace_row["generated_text"]
            == device_trace_row["generated_text"]
        ),
        "all_trace_categories_passed": comparisons["all_categories_passed"],
        "comparisons": comparisons,
    }

    summaries = {
        backend: summarize_trials(trials[backend]) for backend in BACKENDS
    }
    baseline_tps = summaries["native-host"]["tokens_per_second"]["median"]
    optimized_tps = summaries["torch-device"]["tokens_per_second"]["median"]
    speedup = optimized_tps / baseline_tps
    vram_ok = all(
        max(summary["peak_process_vram_per_gpu_bytes"].values())
        <= MAX_PROCESS_VRAM_BYTES
        for summary in summaries.values()
    )
    path_ok = (
        summaries["native-host"]["router_native_host_calls"] > 0
        and summaries["native-host"]["router_torch_device_calls"] == 0
        and summaries["torch-device"]["router_torch_device_calls"] > 0
        and summaries["torch-device"]["router_native_host_calls"] == 0
        and summaries["torch-device"]["router_hidden_d2h_total_bytes"] == 0
        and summaries["torch-device"]["router_outputs_h2d_total_bytes"] == 0
        and summaries["torch-device"]["router_scalar_sync_calls"] == 0
    )
    pareto = {
        "throughput_improved_by_at_least_2_percent": speedup >= 1.02,
        "tokens_and_text_exact": (
            correctness["generated_token_ids_exact"]
            and correctness["decoded_text_exact"]
        ),
        "all_trace_categories_passed": correctness["all_trace_categories_passed"],
        "all_expert_calls_device_resident": all(
            summary["all_expert_calls_device_resident"]
            for summary in summaries.values()
        ),
        "zero_host_fallback_calls": all(
            summary["host_fallback_calls"] == 0 for summary in summaries.values()
        ),
        "optimized_router_has_zero_host_roundtrips_and_scalar_syncs": path_ok,
        "process_vram_at_most_8_gib_per_gpu": vram_ok,
    }
    result = "PASS" if all(pareto.values()) else "FAIL"
    report = {
        "schema_version": 1,
        "result": result,
        "git_commit": git_revision(),
        "model": {
            "directory": str(args.model_dir),
            "config_sha256": sha256_file(args.model_dir / "config.json"),
            "index_sha256": sha256_file(
                args.model_dir / "model.safetensors.index.json"
            ),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpus": [
                {
                    "device": device,
                    "name": torch.cuda.get_device_name(device),
                    "total_memory_bytes": torch.cuda.get_device_properties(
                        device
                    ).total_memory,
                }
                for device in range(2)
            ],
        },
        "configuration": {
            "cache_experts_per_layer": args.cache_experts,
            "split_layer": args.split_layer,
            "trials_per_backend": args.trials,
            "measurement_order": order,
            "prompt": "Hello",
            "max_new_tokens": len(EXPECTED_TOKENS),
            "greedy": True,
            "load_once": True,
            "warmup_per_backend": 1,
        },
        "runtime_load_seconds": load_seconds,
        "warmups": warmups,
        "trial_sequence": sequence,
        "backend_summaries": summaries,
        "correctness": correctness,
        "pareto": {
            "baseline_native_host_median_tps": baseline_tps,
            "optimized_torch_device_median_tps": optimized_tps,
            "speedup_ratio": speedup,
            "speedup_percent": (speedup - 1.0) * 100.0,
            "gates": pareto,
        },
        "implementation_boundary": {
            "changed": (
                "Router placement only: official Transformers checkpoint router "
                "moves from native-host roundtrip/canonicalization to its resident "
                "layer GPU module."
            ),
            "unchanged": (
                "dee.cpp device-resident expert selection inputs, expert caches, "
                "SwiGLU execution, weighted combination, dense layers, recurrent "
                "state, token selection, precision, and greedy semantics."
            ),
        },
    }
    report_path = args.output_dir / "router-backend-benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "result": result,
        "files": {
            report_path.name: {
                "bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
            }
        },
    }
    manifest_path = args.output_dir / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    runtime["pool"].close()
    del runtime
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    for device in range(2):
        torch.cuda.synchronize(device)
    print(
        "ROUTER_BACKEND_BENCHMARK="
        + json.dumps(
            {
                "result": result,
                "baseline_tps": baseline_tps,
                "optimized_tps": optimized_tps,
                "speedup_percent": report["pareto"]["speedup_percent"],
                "output": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
