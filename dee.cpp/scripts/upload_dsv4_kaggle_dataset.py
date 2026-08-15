#!/usr/bin/env python3
"""Download DeepSeek-V4-Flash-0731 shards and publish them as a Kaggle dataset.

Run this ONCE on a machine with >= 165 GB free disk and internet (home machine,
a cloud VM, or a Lightning Studio -- NOT a Kaggle kernel, which has a ~50 GB
writable-disk quota). It:

  1. downloads the 48 safetensors shards (~155 GiB) via huggingface_hub + xet
     into a local directory (resume-capable);
  2. writes the Kaggle dataset-metadata.json;
  3. creates the dataset with `kaggle datasets create` (or, if it already
     exists, appends/updates it with `kaggle datasets version`).

The resulting dataset mounts read-only at:
    /kaggle/input/deepseek-v4-flash-0731-shards/
inside any kernel that lists it as a data source, which is how the native
tokenizer->text runtime reads the checkpoint without hitting the disk quota.

Usage:
    pip install -U "huggingface_hub[hf_transfer]" kaggle
    # put your kaggle.json in ~/.kaggle/ (or `kaggle` CLI already logged in)
    python scripts/upload_dsv4_kaggle_dataset.py --dir /path/to/shard-store \
        [--owner YOUR_KAGGLE_USER] [--skip-download]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
N_SHARDS = 48
DATASET_SLUG = "deepseek-v4-flash-0731-shards"
LICENSE = "deepseek"  # adjust if a more specific license name is required


def download_shards(dest: Path) -> None:
    from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] {REPO_ID}@{REVISION} -> {dest}", flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=str(dest),
        allow_patterns="model-*.safetensors",
        max_workers=4,
    )
    total = sum((dest / f"model-{i:05d}-of-00048.safetensors").stat().st_size
                for i in range(1, N_SHARDS + 1)) / (1 << 30)
    print(f"[download] {total:.1f} GiB in {time.time() - t0:.0f}s", flush=True)


def verify_shards(dest: Path) -> None:
    missing = [i for i in range(1, N_SHARDS + 1)
               if not (dest / f"model-{i:05d}-of-00048.safetensors").is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} shards: {missing[:5]}...")
    print(f"[verify] all {N_SHARDS} shards present", flush=True)


def write_metadata(dest: Path, owner: str) -> None:
    meta = dest / "dataset-metadata.json"
    meta.write_text(
        '{\n'
        f'  "title": "DeepSeek-V4-Flash-0731 official safetensors shards",\n'
        f'  "id": "{owner}/{DATASET_SLUG}",\n'
        f'  "licenses": [{{"name": "{LICENSE}"}}]\n'
        '}\n',
        encoding="utf-8",
    )
    print(f"[metadata] wrote {meta}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="local directory to store the shards (needs ~165 GiB)")
    ap.add_argument("--owner", default="nivind",
                    help="Kaggle username that will own the dataset")
    ap.add_argument("--skip-download", action="store_true",
                    help="skip the download (shards already in --dir)")
    args = ap.parse_args()

    dest = Path(args.dir).resolve()
    if not args.skip_download:
        download_shards(dest)
    verify_shards(dest)
    write_metadata(dest, args.owner)

    print("[upload] creating/updating the Kaggle dataset (this uploads ~155 GiB)",
          flush=True)
    # `kaggle datasets create` creates a new dataset; if it already exists it
    # errors with a 409, in which case `version` appends the files.
    r = subprocess.run(["kaggle", "datasets", "create", "-p", str(dest)])
    if r.returncode != 0:
        print("[upload] create returned non-zero; trying version update",
              flush=True)
        subprocess.run(["kaggle", "datasets", "version", "-p", str(dest),
                        "-m", "upload official shards"], check=True)
    print("[upload] done. Dataset at "
          f"https://www.kaggle.com/datasets/{args.owner}/{DATASET_SLUG}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
