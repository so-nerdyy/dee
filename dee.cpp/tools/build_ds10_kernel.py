"""Assemble the DS10 model-runtime kernel payload + manifest.

Usage:
    python tools/build_ds10_kernel.py [--out tmp/ds10-kernel-payload]

Mirrors tools/build_ds5_kernel.py: the push payload is ONLY the harness +
kernel-metadata. At runtime the harness clones the pinned repository commit,
verifies harness + module SHAs against harness-identity-ds10.json, and
imports the campaign runtime from the checked-out tree.

DS10_STAGE is an environment variable read by the harness at runtime
(v1 load-smoke default; v2 partition0; v3 handoff; v4 logits; v5 first
token; v6 decode 4; final decode 16).
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
    "deepseek_v4_model_runtime.py": KAGGLEDIR / "deepseek_v4_model_runtime.py",
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
    metadata = json.loads((KAGGLEDIR / "kernel-metadata-ds10.json").read_text())
    metadata["code_file"] = "deepseek_v4_model_runtime.py"
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
    for rel, h in manifest.items():
        print(f"  {rel}  {h[:16]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tmp/ds10-kernel-payload")
    args = parser.parse_args()
    main(Path(args.out))
