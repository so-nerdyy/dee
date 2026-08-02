"""Assemble the DS5 trace kernel payload and write a payload manifest.

Usage:
    python tools/build_ds5_kernel.py [--out tmp/ds5-kernel-payload]

Builds a Kaggle-pushable directory containing:

- ds5_trace_runtime.py            (harness, `code_file`)
- deepseek_v4_encoding.py         (flat copy: official tokenizer/encoding)
- deepseek_v4_trace_spec.py       (flat copy: DS5 trace contract)
- model.py, kernel.py, convert.py, generate.py, requirements.txt
                                  (official inference stack, flat)
- config.json, generation_config.json, inference/config.json
                                  (pinned official configs)
- encoding/encoding_dsv4.py       (official encoder)
- tokenizer-assets/               (official tokenizer, SHA-pinned)
- kernel-metadata.json            (copied from kernel-metadata-ds5.json)

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
CAMPAIGN = REPO / "benchmark_reports" / "deepseek-v4-flash-0731-t4"
KAGGLEDIR = REPO / "kaggle" / "deepseek-v4-flash-0731"
OFFICIAL = CAMPAIGN / "official-source"

PAYLOAD_FILES: dict[str, Path] = {
    "ds5_trace_runtime.py": KAGGLEDIR / "ds5_trace_runtime.py",
    "deepseek_v4_encoding.py": REPO / "scripts" / "deepseek_v4_encoding.py",
    "deepseek_v4_trace_spec.py": REPO / "scripts" / "deepseek_v4_trace_spec.py",
    "model.py": OFFICIAL / "inference" / "model.py",
    "kernel.py": OFFICIAL / "inference" / "kernel.py",
    "convert.py": OFFICIAL / "inference" / "convert.py",
    "generate.py": OFFICIAL / "inference" / "generate.py",
    "requirements.txt": OFFICIAL / "inference" / "requirements.txt",
    "config.json": OFFICIAL / "config.json",
    "generation_config.json": OFFICIAL / "generation_config.json",
    "inference/config.json": OFFICIAL / "inference" / "config.json",
    "encoding/encoding_dsv4.py": OFFICIAL / "encoding" / "encoding_dsv4.py",
    "tokenizer-assets/tokenizer.json": CAMPAIGN / "tokenizer-assets" / "tokenizer.json",
    "tokenizer-assets/tokenizer_config.json": CAMPAIGN / "tokenizer-assets" / "tokenizer_config.json",
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
    print(f"payload written to {out_dir} ({total / 1e6:.1f} MB, "
          f"{len(manifest)} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "tmp" / "ds5-kernel-payload")
    args = parser.parse_args()
    main(args.out)
