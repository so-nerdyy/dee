#!/usr/bin/env python3
"""Authoritative dual-T4 M5G fused-normalization subset experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import psutil
import torch


RUN_ID = "20260730T220000Z-m5g-v2-execution-equivalent"
EXPECTED_COMMIT = "1289bd25d21a69ec69b0d43fd203520ffea5a4cf"
BRANCH = "codex/phase2-cap32-matrix"
ROOT = Path("/kaggle/temp/dee-source")
EVIDENCE = Path(f"/kaggle/working/ornith-m5g-evidence-{RUN_ID}")
ARCHIVE_BASE = Path(
    f"/kaggle/working/ornith-m5g-evidence-{RUN_ID}"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_logged(
    command: list[str],
    log_path: Path,
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {"stage": log_path.name, "state": "START", "command": command},
            sort_keys=True,
        ),
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        return_code = process.wait()
    print(
        json.dumps(
            {
                "stage": log_path.name,
                "state": "END",
                "return_code": return_code,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if check and return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


EVIDENCE.mkdir(parents=True, exist_ok=True)
validation_failures: list[dict[str, Any]] = []
required_paths: list[Path] = []
fatal_error: dict[str, Any] | None = None
candidate_summary: dict[str, Any] = {
    "accepted_best": None,
    "candidate_results": {},
    "report_result": None,
}
commit = None
model = None


def require(name: str, condition: bool, details: Any = None) -> None:
    if not condition:
        validation_failures.append({"name": name, "details": details})


try:
    bootstrap = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cpu_count": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            {
                "index": device,
                "name": torch.cuda.get_device_name(device),
                "memory_bytes": torch.cuda.get_device_properties(
                    device
                ).total_memory,
            }
            for device in range(torch.cuda.device_count())
        ],
    }
    write_json(EVIDENCE / "bootstrap-environment.json", bootstrap)
    required_paths.append(EVIDENCE / "bootstrap-environment.json")
    print(json.dumps(bootstrap, indent=2), flush=True)
    run_logged(
        ["nvidia-smi", "-q"],
        EVIDENCE / "logs/nvidia-smi-before.log",
        EVIDENCE,
    )
    required_paths.append(EVIDENCE / "logs/nvidia-smi-before.log")
    if torch.cuda.device_count() != 2 or not all(
        "T4" in torch.cuda.get_device_name(device) for device in range(2)
    ):
        raise RuntimeError(
            "expected exactly two T4 GPUs, got "
            + ", ".join(
                torch.cuda.get_device_name(device)
                for device in range(torch.cuda.device_count())
            )
        )

    run_logged(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-q",
            "transformers==5.14.1",
            "safetensors==0.8.0",
            "pybind11==3.0.1",
            "psutil==7.0.0",
            "nvidia-ml-py",
        ],
        EVIDENCE / "logs/pip-install.log",
        EVIDENCE,
    )
    required_paths.append(EVIDENCE / "logs/pip-install.log")

    if ROOT.exists():
        resolved_root = ROOT.resolve()
        if not str(resolved_root).startswith("/kaggle/temp/"):
            raise RuntimeError(
                f"refusing to remove unexpected path {resolved_root}"
            )
        shutil.rmtree(ROOT)
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            BRANCH,
            "--single-branch",
            "https://github.com/so-nerdyy/dee.git",
            str(ROOT),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", EXPECTED_COMMIT], cwd=ROOT, check=True
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
    ).strip()
    tracked_status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=ROOT,
        text=True,
    )
    if commit != EXPECTED_COMMIT or tracked_status:
        raise RuntimeError({
            "commit": commit,
            "expected_commit": EXPECTED_COMMIT,
            "tracked_status": tracked_status,
        })

    dee = ROOT / "dee.cpp"
    candidates = []
    for index_path in Path("/kaggle/input").rglob(
        "model.safetensors.index.json"
    ):
        config_path = index_path.parent / "config.json"
        if (
            config_path.is_file()
            and json.loads(config_path.read_text()).get("model_type")
            == "qwen3_5_moe"
        ):
            candidates.append(index_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one Ornith checkpoint, got {candidates}"
        )
    model = candidates[0]
    index = json.loads(
        (model / "model.safetensors.index.json").read_text()
    )
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 16 or not all(
        (model / name).is_file() for name in shards
    ):
        raise RuntimeError("official checkpoint shard inventory is incomplete")
    environment = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "commit": commit,
        "source_tree": source_tree,
        "branch": BRANCH,
        "tracked_source_clean_at_checkout": tracked_status == "",
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "transformers",
                "safetensors",
                "pybind11",
                "psutil",
                "nvidia-ml-py",
            )
        },
        "model_dir": str(model),
        "model_config_sha256": sha256_file(model / "config.json"),
        "model_index_sha256": sha256_file(
            model / "model.safetensors.index.json"
        ),
        "model_shards": shards,
        "model_bytes": sum((model / name).stat().st_size for name in shards),
        "gpus": bootstrap["gpus"],
        "candidate_claim_scope": (
            "Disjoint regular-only and gated-only fused RMSNorm modes are "
            "measured against native-combined-direct in one loaded runtime. "
            "Each subset must preserve exact tokens, text, all trace "
            "categories, all 40 layers, device residency, and the cap-32 "
            "VRAM envelope."
        ),
    }
    write_json(EVIDENCE / "environment.json", environment)
    required_paths.append(EVIDENCE / "environment.json")

    build = dee / "build-kaggle-cuda"
    for stale_extension in (dee / "pydee").glob("pydee_core*.so"):
        stale_extension.unlink()
    run_logged(
        [
            "cmake",
            "-S",
            str(dee),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DDEE_CUDA=ON",
            "-DDEE_BUILD_TESTS=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=75",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        EVIDENCE / "logs/configure.log",
        dee,
    )
    run_logged(
        ["cmake", "--build", str(build), "--parallel", "4"],
        EVIDENCE / "logs/build.log",
        dee,
    )
    run_logged(
        [
            "ctest",
            "--test-dir",
            str(build),
            "--output-on-failure",
        ],
        EVIDENCE / "logs/ctest.log",
        dee,
    )
    run_logged(
        [str(build / "test_rmsnorm_cuda")],
        EVIDENCE / "logs/test-rmsnorm-cuda-direct.log",
        dee,
    )
    run_logged(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(dee / "tests"),
        ],
        EVIDENCE / "logs/pytest.log",
        dee,
    )
    required_paths.extend([
        EVIDENCE / "logs/configure.log",
        EVIDENCE / "logs/build.log",
        EVIDENCE / "logs/ctest.log",
        EVIDENCE / "logs/test-rmsnorm-cuda-direct.log",
        EVIDENCE / "logs/pytest.log",
    ])

    build_environment = os.environ.copy()
    build_environment["DEE_BUILD_DIR"] = str(build)
    run_logged(
        [
            sys.executable,
            str(dee / "pydee/setup.py"),
            "build_ext",
            "--inplace",
            "--force",
        ],
        EVIDENCE / "logs/pydee-build.log",
        dee,
        env=build_environment,
    )
    required_paths.append(EVIDENCE / "logs/pydee-build.log")
    extensions = list((dee / "pydee").glob("pydee_core*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"expected one pydee extension, got {extensions}")
    extension = extensions[0]
    build_manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "commit": commit,
        "source_tree": source_tree,
        "cuda_architectures": "75",
        "extension": str(extension),
        "extension_bytes": extension.stat().st_size,
        "extension_sha256": sha256_file(extension),
        "compile_commands_sha256": sha256_file(
            build / "compile_commands.json"
        ),
    }
    write_json(EVIDENCE / "build-manifest.json", build_manifest)
    required_paths.append(EVIDENCE / "build-manifest.json")

    benchmark_dir = EVIDENCE / "m5g"
    benchmark_rc = run_logged(
        [
            sys.executable,
            "-u",
            "-X",
            "faulthandler",
            str(dee / "scripts/run_ornith_m5g_norm_subset_benchmark.py"),
            "--model-dir",
            str(model),
            "--output-dir",
            str(benchmark_dir),
            "--cache-experts",
            "32",
            "--split-layer",
            "20",
            "--trials",
            "3",
            "--require-dual-gpu",
        ],
        EVIDENCE / "logs/m5g.log",
        dee,
        check=False,
    )
    benchmark_required = [
        benchmark_dir / "m5g-norm-subset-benchmark.json",
        benchmark_dir / "artifact-manifest.json",
    ]
    required_paths.extend([
        EVIDENCE / "logs/m5g.log",
        *benchmark_required,
    ])
    if all(path.is_file() for path in benchmark_required):
        report = json.loads(
            (
                benchmark_dir / "m5g-norm-subset-benchmark.json"
            ).read_text()
        )
        candidate_reports = report.get("candidate_reports")
        expected_candidate_ids = {
            "native-combined-direct-fused-regular-norm",
            "native-combined-direct-fused-gated-norm",
        }
        reports_are_mapping = isinstance(candidate_reports, dict)
        candidate_ids_exact = (
            reports_are_mapping
            and set(candidate_reports) == expected_candidate_ids
        )
        require(
            "m5g_candidate_reports_mapping",
            reports_are_mapping,
            type(candidate_reports).__name__,
        )
        require(
            "m5g_candidate_reports_complete",
            candidate_ids_exact,
            sorted(candidate_reports) if reports_are_mapping else None,
        )
        safe_candidate_reports = (
            candidate_reports if candidate_ids_exact else {}
        )
        accepted_best = report.get("accepted_best")
        candidate_results = {
            mode: row.get("result")
            for mode, row in safe_candidate_reports.items()
            if isinstance(row, dict)
        }
        candidate_summary = {
            "accepted_best": accepted_best,
            "candidate_results": candidate_results,
            "report_result": report.get("result"),
            "verdict": report.get("verdict"),
            "execution_equivalence": {
                mode: row.get("execution_equivalence")
                for mode, row in safe_candidate_reports.items()
                if isinstance(row, dict)
            },
            "pareto": {
                mode: row.get("pareto")
                for mode, row in safe_candidate_reports.items()
                if isinstance(row, dict)
            },
        }
        candidate_rows_are_mappings = (
            candidate_ids_exact
            and all(
                isinstance(row, dict)
                for row in safe_candidate_reports.values()
            )
        )
        require(
            "m5g_candidate_rows_mappings",
            candidate_rows_are_mappings,
            candidate_summary,
        )
        required_equivalence_gates = {
            "all_trace_categories_bitwise_exact",
            "all_trace_categories_passed",
            "tokens_and_text_exact",
            "same_direct_expert_path_both_modes",
        }
        def candidate_report_is_valid(row: Any) -> bool:
            if not isinstance(row, dict):
                return False
            pareto = row.get("pareto")
            equivalence = row.get("execution_equivalence")
            if not isinstance(pareto, dict):
                return False
            gates = pareto.get("gates")
            if not isinstance(gates, dict) or not gates:
                return False
            if not all(isinstance(value, bool) for value in gates.values()):
                return False
            if not isinstance(row.get("candidate_accepted"), bool):
                return False
            if row["candidate_accepted"] != all(gates.values()):
                return False
            if row.get("result") != (
                "PASS" if row["candidate_accepted"] else "FAIL"
            ):
                return False
            if not isinstance(equivalence, dict):
                return False
            if equivalence.get("fail_closed") is not True:
                return False
            equivalence_gate = equivalence.get("gate")
            if not isinstance(equivalence_gate, bool):
                return False
            expected_equivalence_gate = all(
                gates.get(name) is True for name in required_equivalence_gates
            )
            if equivalence_gate != expected_equivalence_gate:
                return False
            return equivalence.get("verdict") == (
                "PASS"
                if equivalence_gate
                else "REJECTED_NON_EQUIVALENT_EXECUTION_PATH"
            )

        candidate_gate_reports_valid = (
            candidate_rows_are_mappings
            and all(
                candidate_report_is_valid(row)
                for row in safe_candidate_reports.values()
            )
        )
        require(
            "m5g_candidate_acceptance_matches_all_gates",
            candidate_gate_reports_valid,
            candidate_summary,
        )
        accepted_modes = {
            mode
            for mode, row in safe_candidate_reports.items()
            if candidate_gate_reports_valid and row["candidate_accepted"]
        }
        accepted_best_valid = (
            accepted_best is None
            and not accepted_modes
        ) or (
            accepted_best in expected_candidate_ids
            and accepted_best in accepted_modes
        )
        require(
            "m5g_accepted_best_matches_validated_rows",
            accepted_best_valid,
            {
                "accepted_best": accepted_best,
                "accepted_modes": sorted(accepted_modes),
            },
        )
        candidate_accepted = bool(accepted_modes)
        require(
            "m5g_result_matches_acceptance",
            (report.get("result") == "PASS") == candidate_accepted,
            candidate_summary,
        )
        require(
            "m5g_execution_equivalence_verdict_present",
            report.get("verdict") in {
                "PASS",
                "FAIL",
                "REJECTED_NON_EQUIVALENT_EXECUTION_PATH",
            },
            report.get("verdict"),
        )
        equivalence_rows_valid = candidate_gate_reports_valid
        require(
            "m5g_execution_equivalence_gates_fail_closed",
            equivalence_rows_valid,
            candidate_summary.get("execution_equivalence"),
        )
        expected_verdict = (
            "PASS"
            if accepted_modes
            else (
                "REJECTED_NON_EQUIVALENT_EXECUTION_PATH"
                if equivalence_rows_valid
                and any(
                    row["execution_equivalence"]["verdict"]
                    == "REJECTED_NON_EQUIVALENT_EXECUTION_PATH"
                    for row in safe_candidate_reports.values()
                )
                else "FAIL"
            )
        )
        require(
            "m5g_top_level_verdict_consistent",
            report.get("verdict") == expected_verdict,
            {"actual": report.get("verdict"), "expected": expected_verdict},
        )
        require(
            "m5g_return_code_matches_acceptance",
            benchmark_rc == (0 if candidate_accepted else 1),
            {"return_code": benchmark_rc, **candidate_summary},
        )
        require(
            "m5g_commit_exact",
            report.get("git_commit") == EXPECTED_COMMIT
            and report.get("runtime_identity", {}).get("git_commit")
            == EXPECTED_COMMIT,
            report.get("runtime_identity"),
        )
    else:
        require(
            "m5g_required_artifacts_present",
            False,
            [
                str(path)
                for path in benchmark_required
                if not path.is_file()
            ],
        )

    run_logged(
        ["nvidia-smi", "-q"],
        EVIDENCE / "logs/nvidia-smi-after.log",
        EVIDENCE,
    )
    required_paths.append(EVIDENCE / "logs/nvidia-smi-after.log")
except Exception as exc:
    fatal_error = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    print(fatal_error["traceback"], file=sys.stderr, flush=True)
    write_json(EVIDENCE / "fatal-error.json", fatal_error)
finally:
    required_status = [
        {
            "path": (
                path.relative_to(EVIDENCE).as_posix()
                if path.is_relative_to(EVIDENCE)
                else str(path)
            ),
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        for path in required_paths
    ]
    missing_required = [
        row["path"] for row in required_status if not row["present"]
    ]
    if missing_required:
        validation_failures.append({
            "name": "required_artifacts_present",
            "details": missing_required,
        })
    final_result = (
        "PASS"
        if fatal_error is None and not validation_failures
        else "FAIL"
    )
    artifacts = [
        {
            "path": path.relative_to(EVIDENCE).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(EVIDENCE.rglob("*"))
        if path.is_file()
        and path != EVIDENCE / "artifact-manifest.json"
    ]
    manifest = {
        "schema_version": 1,
        "result": final_result,
        "run_id": RUN_ID,
        "commit": commit,
        "expected_commit": EXPECTED_COMMIT,
        "model_dir": str(model) if model is not None else None,
        "fatal_error": fatal_error,
        "validation_failures": validation_failures,
        "candidate_summary": candidate_summary,
        "required_paths": required_status,
        "artifacts": artifacts,
    }
    write_json(EVIDENCE / "artifact-manifest.json", manifest)
    archive = shutil.make_archive(
        str(ARCHIVE_BASE),
        "gztar",
        root_dir=EVIDENCE.parent,
        base_dir=EVIDENCE.name,
    )
    print(
        json.dumps(
            {
                "final_status": final_result,
                "run_id": RUN_ID,
                "commit": commit,
                "candidate_summary": candidate_summary,
                "validation_failure_count": len(validation_failures),
                "fatal_error": fatal_error,
                "artifact_count": len(artifacts),
                "archive": archive,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if fatal_error is not None or validation_failures:
        raise SystemExit(1)
