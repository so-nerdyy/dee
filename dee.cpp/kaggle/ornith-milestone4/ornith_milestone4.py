#!/usr/bin/env python3
"""Authoritative dual-T4 Milestone 4 Phase 2 cap-32 lifecycle matrix."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch


RUN_ID = "20260727T024309Z-capacity-sweep"
EXPECTED_COMMIT = "726fbfce22d676e9c86e1af1e1cb197d21ea8612"
ROOT = Path("/kaggle/temp/dee-source")
EVIDENCE = Path(f"/kaggle/working/ornith-milestone4-phase2-evidence-{RUN_ID}")


def run_tee(command: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
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


print(json.dumps({
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
            "memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
        for device in range(torch.cuda.device_count())
    ],
}, indent=2), flush=True)
subprocess.run(["nvidia-smi"], check=True)
assert torch.cuda.device_count() == 2
assert all("T4" in torch.cuda.get_device_name(device) for device in range(2))

subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
    "transformers==5.14.1", "safetensors==0.8.0", "pybind11==3.0.1",
    "psutil==7.0.0", "nvidia-ml-py",
], check=True)
packages = {
    name: importlib.metadata.version(name)
    for name in ("transformers", "safetensors", "pybind11", "psutil", "nvidia-ml-py")
}

if ROOT.exists():
    assert str(ROOT.resolve()).startswith("/kaggle/temp/")
    shutil.rmtree(ROOT)
subprocess.run([
    "git", "clone", "--branch", "opt/real-model-t1", "--single-branch",
    "https://github.com/so-nerdyy/dee.git", str(ROOT),
], check=True)
subprocess.run(
    ["git", "checkout", "origin/opt/real-model-t1"], cwd=ROOT, check=True
)
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
MODEL = candidates[0]
index = json.loads((MODEL / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
assert len(shards) == 16 and all((MODEL / name).is_file() for name in shards)

environment = {
    "schema_version": 1,
    "run_id": RUN_ID,
    "commit": commit,
    "harness_nonce": globals().get("HARNESS_NONCE", "unstaged"),
    "packages": packages,
    "model_dir": str(MODEL),
    "model_shards": len(shards),
    "model_bytes": sum((MODEL / name).stat().st_size for name in shards),
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
    json.dumps(environment, indent=2, sort_keys=True) + "\n"
)

BUILD = DEE / "build-kaggle-cuda"
for stale_extension in (DEE / "pydee").glob("pydee_core*.so"):
    stale_extension.unlink()
subprocess.run([
    "cmake", "-S", str(DEE), "-B", str(BUILD), "-G", "Ninja",
    "-DDEE_CUDA=ON", "-DDEE_BUILD_TESTS=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_BUILD_TYPE=Release",
], check=True)
subprocess.run(["cmake", "--build", str(BUILD), "--parallel", "4"], check=True)
subprocess.run(
    ["ctest", "--test-dir", str(BUILD), "--output-on-failure"], check=True
)
subprocess.run([
    sys.executable, "-m", "pytest",
    str(DEE / "tests/test_analyze_milestone25_expert_trace.py"),
    str(DEE / "tests/test_milestone25_matrix_driver.py"),
    str(DEE / "tests/test_milestone4_seven_variant_cap32_matrix.py"),
    str(DEE / "tests/test_run_ornith_forensics.py"),
    "-q",
], cwd=DEE, check=True)
build_environment = os.environ.copy()
build_environment["DEE_BUILD_DIR"] = str(BUILD)
subprocess.run([
    sys.executable, str(DEE / "pydee/setup.py"),
    "build_ext", "--inplace", "--force",
], cwd=DEE, env=build_environment, check=True)

extensions = list((DEE / "pydee").glob("pydee_core*.so"))
assert len(extensions) == 1, extensions
extension_sha256 = hashlib.sha256(extensions[0].read_bytes()).hexdigest()
(EVIDENCE / "build-manifest.json").write_text(json.dumps({
    "schema_version": 1,
    "run_id": RUN_ID,
    "commit": commit,
    "harness_nonce": globals().get("HARNESS_NONCE", "unstaged"),
    "cuda_architectures": "75",
    "extension": str(extensions[0]),
    "extension_sha256": extension_sha256,
}, indent=2, sort_keys=True) + "\n")

run_tee([
    sys.executable, "-u", "-X", "faulthandler",
    str(DEE / "scripts/run_milestone4_seven_variant_cap32_matrix.py"),
    "--seal-dir", str(DEE / "benchmark_reports/milestone-4/phase1-seal"),
    "--model-dir", str(MODEL),
    "--output-dir", str(EVIDENCE),
    "--launch-engine",
], EVIDENCE / "logs/phase2-driver.log", DEE)

summary = json.loads(
    (EVIDENCE / "matrix-summary.json").read_text(encoding="utf-8")
)
expected_runs = [
    "dual-cold-primary",
    "dual-warm-profiled",
    "dual-warm-control",
    "dual-warm-reference-present",
    "dual-cache-disabled",
    "dual-cache-capacity-4",
    "dual-long-prompt",
]
assert summary["selected_run_ids"] == expected_runs
assert len(summary["experiments"]) == 7
assert all(item.get("result") == "PASS" for item in summary["experiments"])
phase2 = json.loads(
    (EVIDENCE / "phase2-final-report.json").read_text(encoding="utf-8")
)
assert phase2["result"] == "PASS"
assert phase2["selected_run_ids"] == expected_runs
assert phase2["configuration_fingerprints_unique"] is True
assert len({row["configuration_fingerprint_sha256"] for row in phase2["rows"]}) == 7
assert all(row["lifetime"]["result"] == "NO_TRACE_ABORT" for row in phase2["rows"])
assert all(
    max(row["peak_process_vram_per_gpu_bytes"].values()) <= 8 * 1024**3
    for row in phase2["rows"]
)

required = [
    EVIDENCE / "environment.json",
    EVIDENCE / "build-manifest.json",
    EVIDENCE / "phase2-contract.json",
    EVIDENCE / "matrix-summary.json",
    EVIDENCE / "matrix-progress.jsonl",
    EVIDENCE / "phase2-final-report.json",
    EVIDENCE / "phase2-artifact-manifest.json",
    EVIDENCE / "raw-allocation-trace.log",
]
for run_id in expected_runs:
    run_dir = EVIDENCE / "runs" / run_id
    required.extend([
        run_dir / "run-report.json",
        run_dir / "memory-timeline.json",
        run_dir / "layer-timing.json",
        run_dir / "timing-raw.json",
        run_dir / "gpu-utilization-summary.json",
        run_dir / "synchronization-analysis.json",
        run_dir / "overlap-analysis.json",
        run_dir / "multi-gpu-timeline.json",
        run_dir / "path-proof.json",
        run_dir / "configuration-fingerprint.json",
        run_dir / "expert-trace.jsonl",
        run_dir / "warmup-expert-trace.jsonl",
    ])
    if run_id != "dual-warm-control":
        required.extend([
            run_dir / "expert-cache-analysis.json",
            run_dir / "transfer-analysis.json",
        ])
empty_allowed = {
    EVIDENCE / "runs" / "dual-cold-primary" / "warmup-expert-trace.jsonl",
    EVIDENCE / "runs" / "dual-warm-control" / "expert-trace.jsonl",
    EVIDENCE / "runs" / "dual-warm-control" / "warmup-expert-trace.jsonl",
}
missing = [
    str(path.relative_to(EVIDENCE))
    for path in required
    if not path.is_file() or (path.stat().st_size == 0 and path not in empty_allowed)
]
assert not missing, missing

artifacts = []
for artifact in sorted(
    path for path in EVIDENCE.rglob("*")
    if path.is_file() and path.name != "artifact-manifest.json"
):
    artifacts.append({
        "path": artifact.relative_to(EVIDENCE).as_posix(),
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    })
manifest = {
    "schema_version": 1,
    "result": "PASS",
    "run_id": RUN_ID,
    "commit": commit,
    "expected_runs": expected_runs,
    "required_paths": [str(path.relative_to(EVIDENCE)) for path in required],
    "artifacts": artifacts,
}
(EVIDENCE / "artifact-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
archive = shutil.make_archive(
    "/kaggle/working/ornith-milestone4-phase2-evidence",
    "gztar",
    root_dir=EVIDENCE.parent,
    base_dir=EVIDENCE.name,
)
print(json.dumps({
    "final_status": "PASS",
    "run_id": RUN_ID,
    "commit": commit,
    "archive": archive,
    "artifact_count": len(artifacts),
}, indent=2), flush=True)
