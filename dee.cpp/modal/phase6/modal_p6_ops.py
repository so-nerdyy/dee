"""Phase 6 — Modal CPU ops: checkpoint staging + DEE4 store builds.

Split from modal_p6.py: GPU functions require a payment method on the
workspace; this CPU-only app runs without one.  Both apps share the same
volumes and MODELS registry.

  stage_ckpt   snapshot_download HF checkpoint -> dee-models volume
  build_store  clone pinned commit -> p3 manifest -> segmented build via
               RemoteRangeSource (HF byte-range reads; the checkpoint is
               never downloaded for repack) -> dee-stores volume.
               Resume-safe: build.journal.jsonl + *.partial survive restarts.
  store_status p3 build-journal summary for a store dir

Usage
  modal run modal/phase6/modal_p6_ops.py::stage_ckpt --model dsv4-flash
  modal run --detach modal/phase6/modal_p6_ops.py::build_store --model dsv4-flash
  modal run modal/phase6/modal_p6_ops.py::store_status --model dsv4-flash
"""

from __future__ import annotations

import modal

REPOSITORY = "https://github.com/so-nerdyy/dee.git"
BRANCH = "research/phase6-modal"
PINNED_COMMIT = ""  # empty = branch head; set for sealed runs

VOL_MODELS = "/vol/models"
VOL_STORES = "/vol/stores"

vol_models = modal.Volume.from_name("dee-models", create_if_missing=True)
vol_stores = modal.Volume.from_name("dee-stores", create_if_missing=True)

MODELS = {
    "dsv4-flash": {
        "hf_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "hf_rev": "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
        "ckpt_dir": "DeepSeek-V4-Flash-0731",
        "store_dir": "dsv4-flash",
        "headers": "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
                   "shard-headers",
    },
}

image_cpu = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("huggingface_hub", "hf_transfer", "requests")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("dee-p6-ops")


def _clone() -> str:
    commit = PINNED_COMMIT
    lines = [
        "set -e",
        f"git clone --branch {BRANCH} --single-branch "
        f"{REPOSITORY} /tmp/dsv4-native-src",
    ]
    if commit:
        lines.append(f"cd /tmp/dsv4-native-src && git checkout --quiet {commit}")
    lines.append("cd /tmp/dsv4-native-src && git rev-parse HEAD")
    return " && ".join(lines)


@app.function(
    image=image_cpu,
    cpu=4,
    memory=16384,
    timeout=8 * 3600,
    volumes={VOL_MODELS: vol_models},
)
def stage_ckpt(model: str) -> dict:
    import os
    import time
    from pathlib import Path

    from huggingface_hub import snapshot_download

    spec = MODELS[model]
    dest = os.path.join(VOL_MODELS, spec["ckpt_dir"])
    os.makedirs(dest, exist_ok=True)
    t0 = time.monotonic()
    snap = snapshot_download(
        repo_id=spec["hf_repo"],
        revision=spec["hf_rev"],
        local_dir=dest,
        allow_patterns=["*.safetensors", "*.json", "*.py", "*.txt",
                        "*.model", "tokenizer*"],
        max_workers=8,
    )
    vol_models.commit()
    files = [p for p in Path(snap).rglob("*") if p.is_file()]
    gb = sum(p.stat().st_size for p in files) / (1 << 30)
    return {"model": model, "snap": snap, "files": len(files),
            "gib": round(gb, 1), "wall_s": round(time.monotonic() - t0, 1)}


@app.function(
    image=image_cpu,
    cpu=8,
    memory=32768,
    timeout=12 * 3600,
    volumes={VOL_STORES: vol_stores},
)
def build_store(model: str, prefetch: int = 8, buckets: int = 0) -> dict:
    import os
    import subprocess
    import time

    spec = MODELS[model]
    src = "/tmp/dsv4-native-src"
    t0 = time.monotonic()
    subprocess.run(_clone(), shell=True, check=True)

    build_dir = os.path.join(VOL_STORES, spec["store_dir"])
    os.makedirs(build_dir, exist_ok=True)
    cmd = [
        "python", "tools/phase3/p3_kaggle_job.py", "build",
        "--build-dir", build_dir,
        "--headers", os.path.join(src, spec["headers"]),
        "--source", "remote",
        "--prefetch", str(prefetch),
        "--publisher", "none",
    ]
    if buckets:
        cmd += ["--buckets", str(buckets)]
    proc = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
    vol_stores.commit()
    tail = (proc.stdout + "\n---STDERR---\n" + proc.stderr)[-8000:]
    return {"model": model, "rc": proc.returncode, "tail": tail,
            "wall_s": round(time.monotonic() - t0, 1)}


@app.function(
    image=image_cpu,
    cpu=2,
    memory=8192,
    timeout=3600,
    volumes={VOL_STORES: vol_stores},
)
def store_status(model: str) -> str:
    import subprocess

    spec = MODELS[model]
    src = "/tmp/dsv4-native-src"
    subprocess.run(_clone(), shell=True, check=True)
    r = subprocess.run(
        ["python", "tools/phase3/p3_kaggle_job.py", "status",
         "--build-dir", f"{VOL_STORES}/{spec['store_dir']}"],
        cwd=src, capture_output=True, text=True)
    return r.stdout + r.stderr


@app.local_entrypoint()
def main() -> None:
    print("dee-p6-ops functions: stage_ckpt, build_store, store_status")
    print("Models:", ", ".join(MODELS))
