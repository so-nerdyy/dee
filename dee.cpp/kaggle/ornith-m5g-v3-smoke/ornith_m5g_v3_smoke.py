#!/usr/bin/env python3
"""Authoritative dual-T4 M5G-v3 regular-norm diagnostic smoke harness."""
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

RUN_ID = "20260730T235000Z-m5g-v3-regular-norm-smoke"
# Replaced with the immutable diagnostic-infrastructure commit before push.
EXPECTED_RUNTIME_COMMIT = "3f03930d78596d457732de9cbab1f50f7878abb1"
ROOT = Path("/kaggle/temp/dee-source")
EVIDENCE = Path(f"/kaggle/working/ornith-m5g-evidence-{RUN_ID}")
ARCHIVE_BASE = Path(f"/kaggle/working/ornith-m5g-evidence-{RUN_ID}")
REPOSITORY = "https://github.com/so-nerdyy/dee.git"
BRANCH = "codex/phase2-cap32-matrix"
def is_accepted_harness_filename(actual_name: str, expected_name: str | None) -> bool:
    return actual_name in {expected_name, "script.py"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_logged(
    command: list[str],
    log_path: Path,
    cwd: Path,
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
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
            log.write(line)
            log.flush()
        rc = process.wait()
    if check and rc:
        raise subprocess.CalledProcessError(rc, command)
    return rc


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    required_paths: list[Path] = []
    failures: list[dict[str, Any]] = []
    fatal_error: dict[str, Any] | None = None
    smoke_rc: int | None = None
    commit: str | None = None
    source_tree: str | None = None
    model: Path | None = None
    harness_sha: str | None = None

    try:
        bootstrap = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
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
        write_json(EVIDENCE / "bootstrap-environment.json", bootstrap)
        required_paths.append(EVIDENCE / "bootstrap-environment.json")
        if torch.cuda.device_count() != 2 or any(
            "T4" not in torch.cuda.get_device_name(device) for device in range(torch.cuda.device_count())
        ):
            raise RuntimeError(f"expected dual T4 topology, got {bootstrap['gpus']}")

        run_logged(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
             "transformers==5.14.1", "safetensors==0.8.0", "pybind11==3.0.1",
             "psutil==7.0.0", "nvidia-ml-py"],
            EVIDENCE / "logs/pip-install.log", EVIDENCE,
        )
        required_paths.append(EVIDENCE / "logs/pip-install.log")

        if ROOT.exists():
            resolved_root = ROOT.resolve()
            if not str(resolved_root).startswith("/kaggle/temp/"):
                raise RuntimeError(f"refusing to remove unexpected path {resolved_root}")
            shutil.rmtree(ROOT)
        subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", REPOSITORY, str(ROOT)], check=True)
        # Script kernels upload only code_file. Read the sidecar from the harness
        # commit before switching the clone to the separately pinned runtime
        # commit, which intentionally does not contain Kaggle packaging files.
        repository_identity_path = ROOT / "dee.cpp/kaggle/ornith-m5g-v3-smoke/harness-identity.json"
        if not repository_identity_path.is_file():
            raise RuntimeError(f"missing repository harness identity {repository_identity_path}")
        harness_identity = json.loads(repository_identity_path.read_text(encoding="utf-8"))
        harness_sha = sha256_file(Path(__file__).resolve())
        expected_harness_file = harness_identity.get("harness_file")
        actual_harness_file = Path(__file__).name
        if not is_accepted_harness_filename(actual_harness_file, expected_harness_file):
            raise RuntimeError({
                "harness_file": actual_harness_file,
                "expected_harness_file": expected_harness_file,
            })
        if harness_identity.get("harness_sha256") != harness_sha:
            raise RuntimeError({
                "harness_sha256": harness_sha,
                "expected_harness_sha256": harness_identity.get("harness_sha256"),
            })
        subprocess.run(["git", "checkout", EXPECTED_RUNTIME_COMMIT], cwd=ROOT, check=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        source_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
        tracked_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        )
        if commit != EXPECTED_RUNTIME_COMMIT or tracked_status:
            raise RuntimeError({"commit": commit, "expected": EXPECTED_RUNTIME_COMMIT, "status": tracked_status})

        candidates = []
        for index_path in Path("/kaggle/input").rglob("model.safetensors.index.json"):
            config_path = index_path.parent / "config.json"
            if config_path.is_file() and json.loads(config_path.read_text()).get("model_type") == "qwen3_5_moe":
                candidates.append(index_path.parent)
        if len(candidates) != 1:
            raise RuntimeError(f"expected one Ornith checkpoint, got {candidates}")
        model = candidates[0]
        write_json(EVIDENCE / "environment.json", {
            "schema_version": 1,
            "run_id": RUN_ID,
            "runtime_commit": commit,
            "source_tree": source_tree,
            "harness_sha256": sha256_file(Path(__file__).resolve()),
            "model_config_sha256": sha256_file(model / "config.json"),
            "model_index_sha256": sha256_file(model / "model.safetensors.index.json"),
            "packages": {name: importlib.metadata.version(name) for name in (
                "transformers", "safetensors", "pybind11", "psutil", "nvidia-ml-py"
            )},
            "performance_comparable": False,
        })
        required_paths.append(EVIDENCE / "environment.json")

        dee = ROOT / "dee.cpp"
        build = dee / "build-kaggle-cuda"
        for stale_extension in (dee / "pydee").glob("pydee_core*.so"):
            stale_extension.unlink()
        run_logged(
            ["cmake", "-S", str(dee), "-B", str(build), "-G", "Ninja", "-DDEE_CUDA=ON",
             "-DDEE_BUILD_TESTS=ON", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_BUILD_TYPE=Release"],
            EVIDENCE / "logs/configure.log", dee,
        )
        run_logged(["cmake", "--build", str(build), "--parallel", "4"], EVIDENCE / "logs/build.log", dee)
        run_logged(["ctest", "--test-dir", str(build), "--output-on-failure"], EVIDENCE / "logs/ctest.log", dee)
        run_logged([str(build / "test_rmsnorm_cuda")], EVIDENCE / "logs/test-rmsnorm-cuda-direct.log", dee)
        build_env = os.environ.copy()
        build_env["DEE_BUILD_DIR"] = str(build)
        run_logged(
            [sys.executable, str(dee / "pydee/setup.py"), "build_ext", "--inplace", "--force"],
            EVIDENCE / "logs/pydee-build.log", dee, check=True, env=build_env,
        )
        required_paths.extend(EVIDENCE / "logs" / name for name in (
            "configure.log", "build.log", "ctest.log", "test-rmsnorm-cuda-direct.log", "pydee-build.log"
        ))

        smoke_dir = EVIDENCE / "smoke"
        assert harness_sha is not None
        smoke_rc = run_logged(
            [sys.executable, "-u", str(dee / "scripts/m5g_v3_cuda_smoke.py"),
             "--model-dir", str(model), "--output-dir", str(smoke_dir),
             "--runtime-sha", EXPECTED_RUNTIME_COMMIT, "--harness-sha", harness_sha,
             "--split-layer", "20", "--cache-experts", "32"],
            EVIDENCE / "logs/smoke.log", dee, check=False,
        )
        required_paths.extend([EVIDENCE / "logs/smoke.log", smoke_dir / "smoke-report.json", smoke_dir / "smoke-status.json"])
        if smoke_rc != 0:
            failures.append({"name": "smoke_driver_return_code", "details": smoke_rc})

    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(EVIDENCE / "fatal-error.json", fatal_error)

    finally:
        required_status = []
        for path in required_paths:
            required_status.append({
                "path": path.relative_to(EVIDENCE).as_posix() if path.is_relative_to(EVIDENCE) else str(path),
                "present": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
            })
        missing = [row["path"] for row in required_status if not row["present"]]
        if missing:
            failures.append({"name": "required_artifacts_present", "details": missing})
        result = "PASS" if fatal_error is None and not failures else "FAIL"
        artifacts = [
            {"path": path.relative_to(EVIDENCE).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(EVIDENCE.rglob("*"))
            if path.is_file() and path.name != "artifact-manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "result": result,
            "terminal_verdict": "INVALID_EXPERIMENT",
            "runtime_commit": commit,
            "expected_runtime_commit": EXPECTED_RUNTIME_COMMIT,
            "source_tree": source_tree,
            "smoke_return_code": smoke_rc,
            "fatal_error": fatal_error,
            "validation_failures": failures,
            "required_paths": required_status,
            "artifacts": artifacts,
            "performance_comparable": False,
        }
        write_json(EVIDENCE / "artifact-manifest.json", manifest)
        archive = shutil.make_archive(str(ARCHIVE_BASE), "gztar", root_dir=EVIDENCE.parent, base_dir=EVIDENCE.name)
        write_json(EVIDENCE / "archive-metadata.json", {
            "archive": archive,
            "archive_sha256": sha256_file(Path(archive)),
            "manifest_sha256": sha256_file(EVIDENCE / "artifact-manifest.json"),
            "excluded_from_archive": ["archive-metadata.json"],
        })
        print(json.dumps({"run_id": RUN_ID, "result": result, "archive": archive, "fatal_error": fatal_error}, sort_keys=True), flush=True)
        if result != "PASS":
            raise SystemExit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
