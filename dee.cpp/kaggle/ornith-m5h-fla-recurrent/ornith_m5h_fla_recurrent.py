#!/usr/bin/env python3
"""Authoritative dual-T4 M5H FLA recurrent-path experiment."""

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


RUN_ID = "20260730T203000Z-m5h-v1"
EXPECTED_COMMIT = "901a2fe20c68a3dd5f9f65bc63a394bab09ae0b1"
BRANCH = "codex/phase2-cap32-matrix"
ROOT = Path("/kaggle/temp/dee-source")
EVIDENCE = Path(f"/kaggle/working/ornith-m5h-evidence-{RUN_ID}")
ARCHIVE_BASE = Path("/kaggle/working/ornith-m5h-evidence")
EXPECTED_FLA_VERSION = "0.5.2"


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
    "candidate_accepted": None,
    "report_result": None,
    "pareto": None,
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
            "flash-linear-attention==0.5.2",
            "fla-core==0.5.2",
        ],
        EVIDENCE / "logs/pip-install.log",
        EVIDENCE,
    )
    required_paths.append(EVIDENCE / "logs/pip-install.log")
    package_versions = {
        name: importlib.metadata.version(name)
        for name in (
            "transformers",
            "safetensors",
            "pybind11",
            "psutil",
            "nvidia-ml-py",
            "flash-linear-attention",
            "fla-core",
            "triton",
        )
    }
    if (
        package_versions["flash-linear-attention"]
        != EXPECTED_FLA_VERSION
        or package_versions["fla-core"] != EXPECTED_FLA_VERSION
    ):
        raise RuntimeError(f"unexpected FLA packages: {package_versions}")

    if ROOT.exists():
        resolved_root = ROOT.resolve()
        if not str(resolved_root).startswith("/kaggle/temp/"):
            raise RuntimeError(
                f"refusing to remove unexpected path {resolved_root}"
            )
        shutil.rmtree(ROOT)
    run_logged(
        [
            "git",
            "clone",
            "--branch",
            BRANCH,
            "--single-branch",
            "https://github.com/so-nerdyy/dee.git",
            str(ROOT),
        ],
        EVIDENCE / "logs/git-clone.log",
        EVIDENCE,
    )
    required_paths.append(EVIDENCE / "logs/git-clone.log")
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
        "packages": package_versions,
        "model_dir": str(model),
        "model_config_sha256": sha256_file(model / "config.json"),
        "model_index_sha256": sha256_file(
            model / "model.safetensors.index.json"
        ),
        "model_shards": shards,
        "model_bytes": sum((model / name).stat().st_size for name in shards),
        "gpus": bootstrap["gpus"],
        "candidate_claim_scope": (
            "Switch only the 30 cached single-token recurrent gated-delta "
            "functions from the Transformers PyTorch fallback to the pinned "
            "FLA 0.5.2 kernel. Keep causal convolution, chunked prefill, "
            "gated RMSNorm, experts, router, model, tokens, and runtime fixed."
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
        [str(build / "test_pointer_batched_cuda")],
        EVIDENCE / "logs/test-pointer-batched-cuda-direct.log",
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
        EVIDENCE / "logs/test-pointer-batched-cuda-direct.log",
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

    benchmark_dir = EVIDENCE / "m5h"
    benchmark_rc = run_logged(
        [
            sys.executable,
            "-u",
            "-X",
            "faulthandler",
            str(dee / "scripts/run_ornith_m5h_fla_recurrent_benchmark.py"),
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
        EVIDENCE / "logs/m5h.log",
        dee,
        check=False,
    )
    benchmark_required = [
        benchmark_dir / "m5h-fla-recurrent-benchmark.json",
        benchmark_dir / "artifact-manifest.json",
    ]
    required_paths.extend([
        EVIDENCE / "logs/m5h.log",
        *benchmark_required,
    ])
    if all(path.is_file() for path in benchmark_required):
        report = json.loads(
            (
                benchmark_dir / "m5h-fla-recurrent-benchmark.json"
            ).read_text()
        )
        candidate_accepted = report.get("candidate_accepted") is True
        candidate_summary = {
            "candidate_accepted": candidate_accepted,
            "report_result": report.get("result"),
            "pareto": report.get("pareto"),
        }
        require(
            "m5h_result_matches_acceptance",
            (report.get("result") == "PASS") == candidate_accepted,
            candidate_summary,
        )
        require(
            "m5h_return_code_matches_acceptance",
            benchmark_rc == (0 if candidate_accepted else 1),
            {"return_code": benchmark_rc, **candidate_summary},
        )
        require(
            "m5h_commit_exact",
            report.get("git_commit") == EXPECTED_COMMIT
            and report.get("runtime_identity", {}).get("git_commit")
            == EXPECTED_COMMIT,
            report.get("runtime_identity"),
        )
        require(
            "m5h_fla_version_exact",
            report.get("environment", {}).get("packages", {}).get(
                "flash-linear-attention"
            ) == EXPECTED_FLA_VERSION
            and report.get("environment", {}).get("packages", {}).get(
                "fla-core"
            ) == EXPECTED_FLA_VERSION,
            report.get("environment", {}).get("packages"),
        )
    else:
        require(
            "m5h_required_artifacts_present",
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
