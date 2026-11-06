#!/usr/bin/env python3
"""Authoritative dual-T4 device-resident router Pareto benchmark."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import torch


RUN_ID = "20260728T235600Z-router-pareto"
EXPECTED_COMMIT = "9aaf727873cd63198c03f76492ade98f32b5d75c"
BRANCH = "codex/phase2-cap32-matrix"
ROOT = Path("/kaggle/temp/dee-source")
EVIDENCE = Path(f"/kaggle/working/ornith-m4-router-pareto-evidence-{RUN_ID}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_tee(command: list[str], log_path: Path, cwd: Path, *, env=None) -> None:
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
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


print(
    json.dumps(
        {
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
        },
        indent=2,
    ),
    flush=True,
)
subprocess.run(["nvidia-smi"], check=True)
assert torch.cuda.device_count() == 2
assert all("T4" in torch.cuda.get_device_name(device) for device in range(2))

subprocess.run(
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
    check=True,
)
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
    assert str(ROOT.resolve()).startswith("/kaggle/temp/")
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
assert commit == EXPECTED_COMMIT, (commit, EXPECTED_COMMIT)
DEE = ROOT / "dee.cpp"
EVIDENCE.mkdir(parents=True, exist_ok=True)

candidates = []
for index_path in Path("/kaggle/input").rglob("model.safetensors.index.json"):
    config_path = index_path.parent / "config.json"
    if (
        config_path.is_file()
        and json.loads(config_path.read_text()).get("model_type") == "qwen3_5_moe"
    ):
        candidates.append(index_path.parent)
assert len(candidates) == 1, candidates
model = candidates[0]
index = json.loads((model / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
assert len(shards) == 16 and all((model / name).is_file() for name in shards)

environment = {
    "schema_version": 1,
    "run_id": RUN_ID,
    "commit": commit,
    "branch": BRANCH,
    "packages": packages,
    "model_dir": str(model),
    "model_shards": shards,
    "model_bytes": sum((model / name).stat().st_size for name in shards),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [
        {
            "index": device,
            "name": torch.cuda.get_device_name(device),
            "memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
        for device in range(torch.cuda.device_count())
    ],
}
(EVIDENCE / "environment.json").write_text(
    json.dumps(environment, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

build = DEE / "build-kaggle-cuda"
for stale_extension in (DEE / "pydee").glob("pydee_core*.so"):
    stale_extension.unlink()
run_tee(
    [
        "cmake",
        "-S",
        str(DEE),
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
    DEE,
)
run_tee(
    ["cmake", "--build", str(build), "--parallel", "4"],
    EVIDENCE / "logs/build.log",
    DEE,
)
run_tee(
    ["ctest", "--test-dir", str(build), "--output-on-failure"],
    EVIDENCE / "logs/ctest.log",
    DEE,
)
run_tee(
    [
        sys.executable,
        "-m",
        "pytest",
        str(DEE / "tests/test_run_ornith_router_backend.py"),
        str(DEE / "tests/test_run_ornith_router_backend_benchmark.py"),
        str(DEE / "tests/test_milestone25_matrix_driver.py"),
        str(DEE / "tests/test_run_ornith_forensics.py"),
        "-q",
    ],
    EVIDENCE / "logs/pytest.log",
    DEE,
)
build_environment = os.environ.copy()
build_environment["DEE_BUILD_DIR"] = str(build)
run_tee(
    [
        sys.executable,
        str(DEE / "pydee/setup.py"),
        "build_ext",
        "--inplace",
        "--force",
    ],
    EVIDENCE / "logs/pydee-build.log",
    DEE,
    env=build_environment,
)
extensions = list((DEE / "pydee").glob("pydee_core*.so"))
assert len(extensions) == 1, extensions
(EVIDENCE / "build-manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "commit": commit,
            "cuda_architectures": "75",
            "extension": str(extensions[0]),
            "extension_bytes": extensions[0].stat().st_size,
            "extension_sha256": sha256_file(extensions[0]),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

benchmark_dir = EVIDENCE / "benchmark"
run_tee(
    [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        str(DEE / "scripts/run_ornith_router_backend_benchmark.py"),
        "--model-dir",
        str(model),
        "--output-dir",
        str(benchmark_dir),
        "--cache-experts",
        "32",
        "--trials",
        "3",
        "--require-dual-gpu",
    ],
    EVIDENCE / "logs/router-backend-benchmark.log",
    DEE,
)
report_path = benchmark_dir / "router-backend-benchmark.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
assert report["result"] == "PASS", report.get("pareto")
assert report["git_commit"] == EXPECTED_COMMIT
assert report["configuration"]["cache_experts_per_layer"] == 32
assert report["pareto"]["speedup_ratio"] >= 1.02
assert all(report["pareto"]["gates"].values())
assert report["correctness"]["generated_token_ids_exact"] is True
assert report["correctness"]["decoded_text_exact"] is True
assert report["correctness"]["all_trace_categories_passed"] is True
optimized = report["backend_summaries"]["torch-device"]
assert optimized["router_hidden_d2h_total_bytes"] == 0
assert optimized["router_outputs_h2d_total_bytes"] == 0
assert optimized["router_scalar_sync_calls"] == 0
assert optimized["host_fallback_calls"] == 0
assert max(optimized["peak_process_vram_per_gpu_bytes"].values()) <= 8 * 1024**3

required = [
    EVIDENCE / "environment.json",
    EVIDENCE / "build-manifest.json",
    EVIDENCE / "logs/configure.log",
    EVIDENCE / "logs/build.log",
    EVIDENCE / "logs/ctest.log",
    EVIDENCE / "logs/pytest.log",
    EVIDENCE / "logs/pydee-build.log",
    EVIDENCE / "logs/router-backend-benchmark.log",
    benchmark_dir / "router-backend-benchmark.json",
    benchmark_dir / "artifact-manifest.json",
]
assert all(path.is_file() and path.stat().st_size > 0 for path in required)
artifacts = [
    {
        "path": path.relative_to(EVIDENCE).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    for path in sorted(EVIDENCE.rglob("*"))
    if path.is_file() and path.name != "artifact-manifest.json"
]
manifest = {
    "schema_version": 1,
    "result": "PASS",
    "run_id": RUN_ID,
    "commit": commit,
    "required_paths": [
        path.relative_to(EVIDENCE).as_posix() for path in required
    ],
    "artifacts": artifacts,
}
(EVIDENCE / "artifact-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
archive = shutil.make_archive(
    "/kaggle/working/ornith-m4-router-pareto-evidence",
    "gztar",
    root_dir=EVIDENCE.parent,
    base_dir=EVIDENCE.name,
)
print(
    json.dumps(
        {
            "final_status": "PASS",
            "run_id": RUN_ID,
            "commit": commit,
            "baseline_tps": report["pareto"][
                "baseline_native_host_median_tps"
            ],
            "optimized_tps": report["pareto"][
                "optimized_torch_device_median_tps"
            ],
            "speedup_percent": report["pareto"]["speedup_percent"],
            "archive": archive,
            "artifact_count": len(artifacts),
        },
        indent=2,
    ),
    flush=True,
)
