"""Assemble the DS5 trace kernel payload and write a payload manifest.

Usage:
    python tools/build_ds5_kernel.py [--out tmp/ds5-kernel-payload]

The DS5 kernel follows the sealed DS9 repo-clone pattern: the push payload is
ONLY the harness (``code_file``) + kernel-metadata. At runtime the harness
clones the pinned repository commit into /kaggle/temp/dsv4-source, verifies
the harness + module SHA-256s against harness-identity-ds5.json, and imports
the official inference stack and tokenizer assets from the checked-out tree.

The payload manifest (payload-manifest.json) records every file's SHA-256 so
the pushed payload can be pinned before launch (immutable-SHA discipline).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KAGGLEDIR = REPO / "kaggle" / "deepseek-v4-flash-0731"

PAYLOAD_FILES: dict[str, Path] = {
    "ds5_trace_runtime.py": KAGGLEDIR / "ds5_trace_runtime.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    manifest: dict[str, str] = {}
    for rel, src in PAYLOAD_FILES.items():
        if not src.is_file():
            raise FileNotFoundError(f"missing payload source: {src}")
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        manifest[rel] = sha256(src)
    metadata = json.loads((KAGGLEDIR / "kernel-metadata-ds5.json").read_text())
    metadata["code_file"] = "ds5_trace_runtime.py"
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=1), encoding="utf-8")
    manifest["kernel-metadata.json"] = sha256(out_dir / "kernel-metadata.json")
    (out_dir / "payload-manifest.json").write_text(
        json.dumps({"files": manifest,
                    "payload_sha256": hashlib.sha256(
                        json.dumps(manifest, sort_keys=True).encode()).hexdigest()},
                   indent=1),
        encoding="utf-8")
    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"payload written to {out_dir} ({total / 1e6:.2f} MB, "
          f"{len(manifest)} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "tmp" / "ds5-kernel-payload")
    args = parser.parse_args()
    main(args.out)
