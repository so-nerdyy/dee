#!/usr/bin/env python3
"""Tests for analyze_milestone3_matrix classifier functions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_milestone3_matrix import (  # noqa: E402
    M25_BASELINES,
    build_before_after,
    classify_defect_1_sequential_dual_gpu,
    classify_defect_2_compute_transfer_overlap,
    classify_defect_3_expert_residency,
    classify_defect_4_repeated_transfers,
    classify_defect_5_duplicate_staging,
    classify_defect_6_python_orchestration,
    classify_defect_7_multi_gpu_utilization,
    collect_run_metrics,
    main,
    parse_args,
)


def _sync(stream_waits: int = 0, host_synchronizations: int = 0):
    return {
        "schema_version": 2,
        "wall_spans_by_name": {},
        "wall_ms_by_name": {
            "cudaHostAlloc": 0.0,
            "mmap_to_pinned": 0.0,
            "pinning": 0.0,
            "pinning_ms": 0.0,
            "cudaStreamSynchronize": 0.0,
            "cublas_gemm": 0.0,
            "h2d_completion": 0.0,
        },
        "stream_wait_events_total": stream_waits,
        "host_synchronization_events_total": host_synchronizations,
        "host_synchronization_events_breakdown": {
            "cpu_section_synchronization_ms_total": host_synchronizations,
            "dedicated_host_synchronization_counter_total": 0,
        },
        "step_breakdown": {},
        "model_level_excluded_spans_count": 0,
    }


def _overlap(both_busy_us: int = 0, one_busy_us: int = 0,
              neither_busy_us: int = 0, samples_total: int = 0,
              samples_both_busy: int = 0):
    return {
        "schema_version": 2,
        "samples_total": samples_total,
        "both_busy_us": both_busy_us,
        "one_busy_us": one_busy_us,
        "neither_busy_us": neither_busy_us,
        "samples_with_both_busy": samples_both_busy,
        "nvml_both_gpus_active_samples": samples_both_busy,
        "intersample_interval_us_estimated": 50_000,
        "both_busy_fraction_of_active":
            both_busy_us / (both_busy_us + one_busy_us)
            if (both_busy_us + one_busy_us) else 0.0,
        "both_busy_fraction_of_total":
            both_busy_us / (both_busy_us + one_busy_us + neither_busy_us)
            if (both_busy_us + one_busy_us + neither_busy_us) else 0.0,
    }


def test_defect_1_full_active_overlap_fixes():
    overlap = _overlap(both_busy_us=20_000, one_busy_us=5_000,
                        neither_busy_us=0, samples_total=40,
                        samples_both_busy=32)
    result = classify_defect_1_sequential_dual_gpu(overlap)
    assert result["status"] == "FULLY_FIXED", result
    assert result["samples_both_busy"] == 32


def test_defect_1_baseline_or_below_returns_not_fixed():
    overlap = _overlap(both_busy_us=2_000, one_busy_us=18_000,
                        neither_busy_us=0, samples_total=20,
                        samples_both_busy=1)  # matches M2.5 1/18
    result = classify_defect_1_sequential_dual_gpu(overlap)
    assert result["status"] == "NOT_FIXED", result


def test_defect_1_missing_returns_inconclusive():
    assert classify_defect_1_sequential_dual_gpu({})["status"] == "INCONCLUSIVE"


def test_defect_2_stream_wait_plus_overlap_fixes():
    sync = _sync(stream_waits=120, host_synchronizations=80)
    overlap = _overlap(both_busy_us=8_000, one_busy_us=2_000,
                        neither_busy_us=0, samples_total=20,
                        samples_both_busy=10)
    transfer = {"all_h2d": {"measured_overlap_with_compute_seconds": 0.001}}
    result = classify_defect_2_compute_transfer_overlap(sync, overlap, transfer)
    assert result["status"] == "FULLY_FIXED", result


def test_defect_2_zero_waits_returns_not_fixed():
    sync = _sync(stream_waits=0, host_synchronizations=2488)
    overlap = _overlap(both_busy_us=0, one_busy_us=20_000,
                        samples_total=20, samples_both_busy=0)
    transfer = {"all_h2d": {"measured_overlap_with_compute_seconds": 0.0}}
    result = classify_defect_2_compute_transfer_overlap(sync, overlap, transfer)
    assert result["status"] in {"NOT_FIXED", "PARTIALLY_FIXED"}, result


def test_defect_3_capacity_sweep_proves_unavoidable():
    rows = {
        "dual-warm-profiled": {"cache_hit_rate": 0.0},
        "dual-cache-capacity-4": {"cache_hit_rate": 0.001},
        "dual-cache-disabled": {"cache_hit_rate": 0.0},
    }
    result = classify_defect_3_expert_residency(rows)
    assert result["status"] == "UNAVOIDABLE", result


def test_defect_3_high_hit_rate_fixes():
    rows = {"dual-warm-profiled": {"cache_hit_rate": 0.6}}
    result = classify_defect_3_expert_residency(rows)
    assert result["status"] == "FULLY_FIXED", result


def test_defect_4_full_fix_when_below_lower_bound_and_no_repeats():
    path = {"expert_ids_d2h_total_bytes": 10240}
    transfer = {
        "by_component": [{
            "component": "expert_weight",
            "direction": "h2d",
            "measured_bytes": (
                M25_BASELINES["expert_h2d_bytes_total"].value * 0.80
            ),
        }],
        "same_token_repeat_transfers": {"repeated_events": 0},
    }
    result = classify_defect_4_repeated_transfers(path, transfer)
    assert result["status"] == "FULLY_FIXED", result


def test_defect_4_partial_when_under_baseline():
    path = {"expert_ids_d2h_total_bytes": 10240}
    transfer = {
        "by_component": [{
            "component": "expert_weight",
            "direction": "h2d",
            "measured_bytes": int(
                M25_BASELINES["expert_h2d_bytes_total"].value * 0.97
            ),
        }],
        "same_token_repeat_transfers": {"repeated_events": 1},
    }
    result = classify_defect_4_repeated_transfers(path, transfer)
    assert result["status"] == "PARTIALLY_FIXED", result


def test_defect_5_zero_pinning_fixes():
    run = {"pinning_ms_total": 1.0, "mmap_to_pinned_ms_total": 2.0}
    result = classify_defect_5_duplicate_staging(run)
    assert result["status"] == "FULLY_FIXED", result


def test_defect_6_all_device_calls_fixes():
    path = {
        "device_path_calls": 160,
        "host_path_fallback_calls": 0,
        "device_path_share": 1.0,
        "host_fallback_share": 0.0,
        "fp32_to_fp16_conversion_ms_total": 1.5,
        "expert_ids_d2h_total_bytes": 640,
    }
    result = classify_defect_6_python_orchestration(path)
    assert result["status"] == "FULLY_FIXED", result


def test_defect_7_low_syncs_and_both_active_fixes():
    sync = _sync(stream_waits=120, host_synchronizations=200)
    overlap = _overlap(samples_total=20, samples_both_busy=4)  # > baseline 1
    result = classify_defect_7_multi_gpu_utilization(sync, overlap)
    assert result["status"] == "FULLY_FIXED", result


def test_build_before_after_handles_synthetic_run(tmp_path: Path):
    canonical_dir = tmp_path / "runs" / "dual-warm-profiled"
    canonical_dir.mkdir(parents=True)
    canonical_dir.joinpath("run-report.json").write_text(json.dumps({
        "result": "PASS",
        "git_commit": "deadbeef",
        "machine": {"pytorch": "2.5", "cuda_runtime": "12.4",
                    "transformers": "5.14"},
        "warmup_generation": {},
        "generation": {
            "generated_token_ids": [11, 271, 40, 1044],
            "per_token_decode_seconds": [2.0, 2.0, 2.0, 2.0],
            "single_stream_decode_tokens_per_second": 0.5,
            "time_to_first_token_seconds": 2.37,
        },
        "correctness": {"all_40_layers_executed": True,
                        "reference_parity": None},
    }))
    canonical_dir.joinpath("path-proof.json").write_text(json.dumps({
        "schema_version": 2,
        "device_path_calls": 160,
        "host_path_fallback_calls": 0,
        "device_path_share": 1.0,
        "host_fallback_share": 0.0,
        "fp32_to_fp16_conversion_ms_total": 1.0,
        "expert_ids_d2h_total_bytes": 256,
        "expert_native_device_calls_total_ms": 30.0,
        "expert_native_host_calls_total_ms": 0.0,
    }))
    canonical_dir.joinpath("synchronization-analysis.json").write_text(json.dumps(_sync(
        stream_waits=120, host_synchronizations=160)))
    canonical_dir.joinpath("overlap-analysis.json").write_text(json.dumps(_overlap(
        both_busy_us=8000, one_busy_us=4000, neither_busy_us=0,
        samples_total=40, samples_both_busy=20)))
    rows = {"dual-warm-profiled": collect_run_metrics(canonical_dir)}
    table = build_before_after(rows, "dual-warm-profiled")
    by_metric = {row["metric"]: row for row in table}
    assert "warm_decode_tps" in by_metric, by_metric
    assert "device_path_calls" in by_metric, by_metric
    assert by_metric["device_path_calls"]["milestone_3"] == 160, by_metric
    assert by_metric["warm_decode_tps"]["milestone_3"] == 0.5, by_metric
    assert "evidence_artifact" in by_metric["warm_decode_tps"], by_metric


def test_main_fails_when_no_canonical_run(tmp_path: Path, capsys):
    (tmp_path / "runs" / "dual-cache-disabled").mkdir(parents=True)
    # Force parse_args to use tmp_path's runs/ dir by patching via sys.argv
    backup = sys.argv[:]
    sys.argv = ["analyze_milestone3_matrix.py",
                 "--m3-dir", str(tmp_path),
                 "--output-dir", str(tmp_path / "out")]
    try:
        with pytest.raises(RuntimeError, match="exactly the seven declared runs"):
            main()
    finally:
        sys.argv = backup


def test_args_parser_smoke(tmp_path: Path):
    sys.argv = ["analyze_milestone3_matrix.py", "--m3-dir", str(tmp_path)]
    args = parse_args()
    assert args.m3_dir == tmp_path
