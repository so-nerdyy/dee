#!/usr/bin/env python3
"""Authoritative dual-T4 M5A attribution and M5B execution-path A/B."""

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


RUN_ID = "20260729T195432Z-m5ab"
EXPECTED_COMMIT = "7244b42fff1665bb170c72d7cb4c54d989f5916f"
BRANCH = "codex/phase2-cap32-matrix"
ROOT = Path("/kaggle/temp/dee-source")
EVIDENCE = Path(f"/kaggle/working/ornith-m5-ab-evidence-{RUN_ID}")
ARCHIVE_BASE = Path("/kaggle/working/ornith-m5-ab-evidence")
MAX_VRAM_BYTES = 8 * 1024**3


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


def run_tee(
        command: list[str],
        log_path: Path,
        cwd: Path,
        *,
        env: dict[str, str] | None = None,
        check: bool = True,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if check and return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


def tool_inventory(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    row: dict[str, Any] = {
        "name": name,
        "available": path is not None,
        "path": path,
        "version_return_code": None,
        "version_output": None,
    }
    if path is not None:
        result = subprocess.run(
            [path, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        row["version_return_code"] = result.returncode
        row["version_output"] = result.stdout.strip()
    return row


EVIDENCE.mkdir(parents=True, exist_ok=True)
validation_failures: list[dict[str, Any]] = []
required_paths: list[Path] = []
fatal_error: dict[str, Any] | None = None
candidate_summary: dict[str, Any] = {
    "m5b_candidate_accepted": None,
    "m5b_report_result": None,
}
commit = None
model = None


def require(name: str, condition: bool, details: Any = None) -> None:
    if not condition:
        validation_failures.append({
            "name": name,
            "details": details,
        })


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
    subprocess.run(["nvidia-smi"], check=True)
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"expected exactly two visible GPUs, got {torch.cuda.device_count()}"
        )
    if not all(
        "T4" in torch.cuda.get_device_name(device) for device in range(2)
    ):
        raise RuntimeError(
            "expected dual T4, got "
            + ", ".join(
                torch.cuda.get_device_name(device) for device in range(2)
            )
        )

    run_tee(
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
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "transformers",
            "safetensors",
            "pybind11",
            "psutil",
            "nvidia-ml-py",
        )
    }

    if ROOT.exists():
        resolved_root = ROOT.resolve()
        if not str(resolved_root).startswith("/kaggle/temp/"):
            raise RuntimeError(f"refusing to remove unexpected path {resolved_root}")
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
    subprocess.run(["git", "checkout", EXPECTED_COMMIT], cwd=ROOT, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
    ).strip()
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
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
        raise RuntimeError(f"expected one Ornith checkpoint, got {candidates}")
    model = candidates[0]
    index = json.loads(
        (model / "model.safetensors.index.json").read_text()
    )
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 16 or not all(
        (model / name).is_file() for name in shards
    ):
        raise RuntimeError("official checkpoint shard inventory is incomplete")

    tools = {
        name: tool_inventory(name)
        for name in ("ncu", "nsys")
    }
    environment = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "commit": commit,
        "source_tree": source_tree,
        "branch": BRANCH,
        "tracked_source_clean_at_checkout": tracked_status == "",
        "packages": packages,
        "model_dir": str(model),
        "model_config_sha256": sha256_file(model / "config.json"),
        "model_index_sha256": sha256_file(
            model / "model.safetensors.index.json"
        ),
        "model_shards": shards,
        "model_bytes": sum((model / name).stat().st_size for name in shards),
        "gpu_count": torch.cuda.device_count(),
        "gpus": bootstrap["gpus"],
        "hardware_counter_tools": tools,
        "hardware_counter_claim": (
            "inventory only; this run does not claim a counter-based "
            "impossibility proof"
        ),
    }
    write_json(EVIDENCE / "environment.json", environment)
    required_paths.append(EVIDENCE / "environment.json")

    build = dee / "build-kaggle-cuda"
    for stale_extension in (dee / "pydee").glob("pydee_core*.so"):
        stale_extension.unlink()
    run_tee(
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
    run_tee(
        ["cmake", "--build", str(build), "--parallel", "4"],
        EVIDENCE / "logs/build.log",
        dee,
    )
    run_tee(
        ["ctest", "--test-dir", str(build), "--output-on-failure"],
        EVIDENCE / "logs/ctest.log",
        dee,
    )
    run_tee(
        [sys.executable, "-m", "pytest", "-q", str(dee / "tests")],
        EVIDENCE / "logs/pytest.log",
        dee,
    )
    required_paths.extend([
        EVIDENCE / "logs/configure.log",
        EVIDENCE / "logs/build.log",
        EVIDENCE / "logs/ctest.log",
        EVIDENCE / "logs/pytest.log",
    ])

    build_environment = os.environ.copy()
    build_environment["DEE_BUILD_DIR"] = str(build)
    run_tee(
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

    direct_dir = EVIDENCE / "m5a-direct"
    direct_rc = run_tee(
        [
            sys.executable,
            "-u",
            "-X",
            "faulthandler",
            str(dee / "scripts/run_ornith_forensics.py"),
            "--model-dir",
            str(model),
            "--output-dir",
            str(direct_dir),
            "--run-id",
            f"{RUN_ID}-direct",
            "--prompt",
            "Hello",
            "--max-new-tokens",
            "4",
            "--split-layer",
            "20",
            "--cache-experts",
            "32",
            "--router-backend",
            "torch-device",
            "--classification",
            "warm",
            "--profile",
            "--trace-requests",
            "--profile-timeline",
            "--torch-profiler-step",
            "2",
            "--warmup-generation",
            "--require-dual-gpu",
        ],
        EVIDENCE / "logs/m5a-direct.log",
        dee,
        check=False,
    )
    direct_required = [
        direct_dir / "run-report.json",
        direct_dir / "m5a-full-token-breakdown.json",
        direct_dir / "path-proof.json",
        direct_dir / "timing-raw.json",
        direct_dir / "layer-timing.json",
        direct_dir / "torch-profiler-trace.json.gz",
    ]
    required_paths.extend(
        [EVIDENCE / "logs/m5a-direct.log", *direct_required]
    )
    require(
        "m5a_direct_process_passed",
        direct_rc == 0,
        {"return_code": direct_rc},
    )
    if all(path.is_file() for path in direct_required):
        direct_report = json.loads(
            (direct_dir / "run-report.json").read_text()
        )
        direct_breakdown = json.loads(
            (direct_dir / "m5a-full-token-breakdown.json").read_text()
        )
        direct_path = json.loads(
            (direct_dir / "path-proof.json").read_text()
        )
        require(
            "m5a_direct_report_passed",
            direct_report.get("result") == "PASS",
            direct_report.get("instrumentation"),
        )
        require(
            "m5a_direct_breakdown_passed",
            direct_breakdown.get("result") == "PASS"
            and all(direct_breakdown.get("gates", {}).values()),
            direct_breakdown.get("gates"),
        )
        require(
            "m5a_direct_commit_exact",
            direct_report.get("git_commit") == EXPECTED_COMMIT
            and direct_path.get("git_commit") == EXPECTED_COMMIT,
            {
                "report": direct_report.get("git_commit"),
                "path": direct_path.get("git_commit"),
            },
        )
        require(
            "m5a_direct_correctness",
            direct_report.get("correctness", {}).get(
                "baseline_tokens_exact"
            ) is True
            and direct_report.get("correctness", {}).get(
                "all_40_layers_executed"
            ) is True
            and direct_report.get("correctness", {}).get(
                "warmup_tokens_exact"
            ) is True,
            direct_report.get("correctness"),
        )
        require(
            "m5a_direct_path_exact",
            direct_path.get("device_path_share") == 1.0
            and direct_path.get("host_path_fallback_calls") == 0
            and direct_path.get("pybind_host_fallback_calls") == 0
            and direct_path.get("pybind_device_calls") == 160
            and direct_path.get("python_combine_calls") == 160
            and direct_path.get("raw_output_allocations") == 160
            and direct_path.get("router_torch_device_calls") == 160
            and direct_path.get("router_scalar_sync_calls") == 0,
            direct_path,
        )
        direct_vram = direct_report["generation"]["resources"][
            "peak_vram_bytes"
        ]
        require(
            "m5a_direct_vram_within_bound",
            max(direct_vram.values()) <= MAX_VRAM_BYTES,
            direct_vram,
        )
    else:
        require(
            "m5a_direct_required_artifacts_present",
            False,
            [str(path) for path in direct_required if not path.is_file()],
        )

    control_dir = EVIDENCE / "m5a-control"
    control_rc = run_tee(
        [
            sys.executable,
            "-u",
            "-X",
            "faulthandler",
            str(dee / "scripts/run_ornith_forensics.py"),
            "--model-dir",
            str(model),
            "--output-dir",
            str(control_dir),
            "--run-id",
            f"{RUN_ID}-control",
            "--prompt",
            "Hello",
            "--max-new-tokens",
            "4",
            "--split-layer",
            "20",
            "--cache-experts",
            "32",
            "--router-backend",
            "torch-device",
            "--classification",
            "warm",
            "--torch-profiler-control",
            "--torch-profiler-step",
            "2",
            "--warmup-generation",
            "--require-dual-gpu",
        ],
        EVIDENCE / "logs/m5a-control.log",
        dee,
        check=False,
    )
    control_required = [
        control_dir / "run-report.json",
        control_dir / "m5a-profiler-control.json",
        control_dir / "timing-raw.json",
        control_dir / "torch-profiler-trace.json.gz",
    ]
    required_paths.extend(
        [EVIDENCE / "logs/m5a-control.log", *control_required]
    )
    require(
        "m5a_control_process_passed",
        control_rc == 0,
        {"return_code": control_rc},
    )
    if all(path.is_file() for path in control_required):
        control_report = json.loads(
            (control_dir / "run-report.json").read_text()
        )
        control = json.loads(
            (control_dir / "m5a-profiler-control.json").read_text()
        )
        require(
            "m5a_control_report_passed",
            control_report.get("result") == "PASS",
            control_report.get("instrumentation"),
        )
        require(
            "m5a_control_gate_passed",
            control.get("result") == "PASS"
            and all(control.get("gates", {}).values()),
            control.get("gates"),
        )
        control_vram = control_report["generation"]["resources"][
            "peak_vram_bytes"
        ]
        require(
            "m5a_control_vram_within_bound",
            max(control_vram.values()) <= MAX_VRAM_BYTES,
            control_vram,
        )
    else:
        require(
            "m5a_control_required_artifacts_present",
            False,
            [str(path) for path in control_required if not path.is_file()],
        )

    m5b_dir = EVIDENCE / "m5b"
    m5b_rc = run_tee(
        [
            sys.executable,
            "-u",
            "-X",
            "faulthandler",
            str(dee / "scripts/run_ornith_m5b_execution_benchmark.py"),
            "--model-dir",
            str(model),
            "--output-dir",
            str(m5b_dir),
            "--cache-experts",
            "32",
            "--split-layer",
            "20",
            "--trials",
            "3",
            "--require-dual-gpu",
        ],
        EVIDENCE / "logs/m5b.log",
        dee,
        check=False,
    )
    m5b_required = [
        m5b_dir / "m5b-execution-benchmark.json",
        m5b_dir / "artifact-manifest.json",
    ]
    required_paths.extend([EVIDENCE / "logs/m5b.log", *m5b_required])
    if all(path.is_file() for path in m5b_required):
        m5b = json.loads(
            (m5b_dir / "m5b-execution-benchmark.json").read_text()
        )
        gates = m5b.get("pareto", {}).get("gates", {})
        performance_gate_names = {
            "production_independent_median_gain_at_least_0_5_percent",
            "production_paired_median_gain_at_least_0_5_percent",
            "production_wins_majority_of_pairs",
            "no_pair_regresses_more_than_2_percent",
        }
        validity_gates = {
            name: value
            for name, value in gates.items()
            if name not in performance_gate_names
        }
        candidate_accepted = (
            all(gates.get(name) is True for name in performance_gate_names)
            and all(validity_gates.values())
        )
        candidate_summary = {
            "m5b_candidate_accepted": candidate_accepted,
            "m5b_report_result": m5b.get("result"),
            "debug_full_logit_median_tps": m5b.get("pareto", {}).get(
                "debug_full_logit_median_tps"
            ),
            "production_median_tps": m5b.get("pareto", {}).get(
                "production_median_tps"
            ),
            "speedup_percent": m5b.get("pareto", {}).get(
                "speedup_percent"
            ),
            "performance_gates": {
                name: gates.get(name)
                for name in sorted(performance_gate_names)
            },
        }
        require(
            "m5b_validity_gates_passed",
            bool(validity_gates) and all(validity_gates.values()),
            validity_gates,
        )
        require(
            "m5b_result_matches_acceptance",
            (m5b.get("result") == "PASS") == candidate_accepted,
            candidate_summary,
        )
        require(
            "m5b_return_code_matches_acceptance",
            m5b_rc == (0 if candidate_accepted else 1),
            {"return_code": m5b_rc, **candidate_summary},
        )
        require(
            "m5b_commit_exact",
            m5b.get("git_commit") == EXPECTED_COMMIT
            and m5b.get("runtime_identity", {}).get("git_commit")
            == EXPECTED_COMMIT,
            m5b.get("runtime_identity"),
        )
    else:
        require(
            "m5b_required_artifacts_present",
            False,
            [str(path) for path in m5b_required if not path.is_file()],
        )

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
        if path.is_file() and path != EVIDENCE / "artifact-manifest.json"
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
