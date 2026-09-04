#!/usr/bin/env python3
"""Kaggle microbench runner for the isolated FP4-decode prototype.

Codex: run on 2x T4 WITHOUT touching the live campaign checkout. From a fresh
notebook cell (or a clean clone), copy this file + fp4_decode_vec.cu beside
each other and run:

    python3 kaggle_microbench_fp4_decode.py --workdir /tmp/t4kn --iters 200

The script: verifies SM75 via nvidia-smi, compiles the self-contained .cu
with nvcc (no dee build needed), optionally extracts ONE real official expert
fixture (packed w1/w3/w2 + scales, 13,369,344 bytes) from a mounted
DeepSeek-V4-Flash-0731 shard for the w1 [2048,4096] case, runs the trusted
scalar port (production algorithm) vs the vectorized candidate, requires
bitwise-identical output (exit nonzero otherwise), warms up, benchmarks, and
writes machine-readable JSON. Bounded memory (~120 MiB device). No
full-model generation. Never fabricates numbers: without a T4 it refuses.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CU_FILE = HERE / "fp4_decode_vec.cu"
OFFICIAL_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

# One routed expert, w1-style [out=2048, in=4096] packed FP4 + e8m0 scales.
W1_OUT, W1_IN = 2048, 4096
W1_PACKED = W1_OUT * (W1_IN // 2)
W1_SCALE = W1_OUT * (W1_IN // 32)


def run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_sm75() -> dict:
    info: dict = {"nvidia_smi": False, "gpus": []}
    try:
        proc = run(["nvidia-smi", "--query-gpu=index,name,compute_cap",
                    "--format=csv,noheader"], timeout=60)
    except FileNotFoundError:
        raise SystemExit("REFUSE: nvidia-smi not found; no T4 here.")
    if proc.returncode != 0:
        raise SystemExit(f"REFUSE: nvidia-smi failed: {proc.stderr[:300]}")
    info["nvidia_smi"] = True
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            info["gpus"].append(
                {"index": parts[0], "name": parts[1], "compute_cap": parts[2]})
    if not info["gpus"]:
        raise SystemExit("REFUSE: no GPUs enumerated.")
    for gpu in info["gpus"]:
        if gpu["compute_cap"] != "7.5":
            raise SystemExit(
                f"REFUSE: GPU {gpu['index']} ({gpu['name']}) is SM{gpu['compute_cap']}, "
                "not SM75. This package is T4-only.")
    return info


def check_nvcc(workdir: Path) -> str:
    proc = run(["nvcc", "--version"], timeout=60)
    if proc.returncode != 0:
        raise SystemExit("REFUSE: nvcc not available on this host.")
    first = (proc.stdout.strip().splitlines() or [""])[0]
    (workdir / "nvcc_version.txt").write_text(proc.stdout, encoding="utf-8")
    return first


def extract_real_fixture(shard: Path, workdir: Path) -> dict:
    """Pull layers.6.ffn.experts.0.w1 packed+scale from one official shard.

    Parses only the safetensors header (8-byte LE length + JSON), then reads
    the two raw byte ranges. Verifies sizes against the FP4 layout. No torch
    dependency.
    """
    sten: dict = {"attempted": str(shard), "status": "skipped", "reason": ""}
    if not shard.is_file():
        sten["reason"] = "shard file not present; synthetic fixture will be used"
        return sten
    try:
        with open(shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        names = [f"layers.6.ffn.experts.0.w1.{s}" for s in ("weight", "scale")]
        with open(shard, "rb") as fh:
            blobs = {}
            for name in names:
                entry = header.get(name)
                if not entry:
                    sten["reason"] = f"tensor {name} not in this shard"
                    return sten
                begin, end = entry["data_offsets"]
                fh.seek(data_start + begin)
                blobs[name] = fh.read(end - begin)
        weight, scale = blobs[names[0]], blobs[names[1]]
        if len(weight) != W1_PACKED or len(scale) != W1_SCALE:
            sten["reason"] = (f"unexpected sizes weight={len(weight)} scale={len(scale)}; "
                              "synthetic fixture will be used")
            return sten
        (workdir / "w1_real.packed.bin").write_bytes(weight)
        (workdir / "w1_real.scale.bin").write_bytes(scale)
        sten.update({"status": "extracted",
                     "weight_bytes": len(weight), "scale_bytes": len(scale),
                     "model": OFFICIAL_MODEL, "revision": OFFICIAL_REV})
    except Exception as exc:  # noqa: BLE001
        sten["reason"] = f"parse failed ({exc}); synthetic fixture will be used"
    return sten


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/t4kn"))
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--shard", type=Path, default=None,
                        help="optional official shard for one real w1 fixture")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.time()
    args.workdir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "package": "t4_kernel_next/fp4_decode_vec",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": OFFICIAL_MODEL, "revision": OFFICIAL_REV,
    }
    report["gpu"] = check_sm75()
    report["nvcc"] = check_nvcc(args.workdir)

    cu_dest = args.workdir / "fp4_decode_vec.cu"
    cu_dest.write_bytes(CU_FILE.read_bytes())
    binary = args.workdir / "fp4_decode_vec"
    build = run(["nvcc", "-O3", "-arch=sm_75", "-o", str(binary), str(cu_dest)],
                timeout=900)
    report["build"] = {"exit": build.returncode,
                       "stderr_tail": build.stderr[-2000:]}
    if build.returncode != 0:
        raise SystemExit(f"BUILD FAILED:\n{build.stderr[-4000:]}")

    if args.shard is not None:
        report["real_fixture"] = extract_real_fixture(args.shard, args.workdir)
    else:
        report["real_fixture"] = {"status": "skipped",
                                  "reason": "no --shard given; synthetic fixture"}

    runs = []
    # Case 1: synthetic [2048,4096] (always).
    runs.append({"name": "synthetic-w1-shape",
                 "args": ["--out", "2048", "--in", "4096"]})
    # Case 2: real w1 bytes if extracted.
    if report["real_fixture"].get("status") == "extracted":
        runs.append({"name": "real-w1-bytes",
                     "args": ["--out", "2048", "--in", "4096",
                              "--packed-file", str(args.workdir / "w1_real.packed.bin"),
                              "--scale-file", str(args.workdir / "w1_real.scale.bin")]})
    # Case 3: synthetic down-projection shape [4096,2048].
    runs.append({"name": "synthetic-down-shape",
                 "args": ["--out", "4096", "--in", "2048"]})

    results = []
    failed = False
    for spec in runs:
        cmd = [str(binary), "--iters", str(args.iters)] + spec["args"]
        proc = run(cmd, timeout=900)
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            payload = {"status": "unparseable", "stdout": proc.stdout[-2000:],
                       "stderr": proc.stderr[-2000:]}
        payload["case"] = spec["name"]
        payload["exit"] = proc.returncode
        results.append(payload)
        if proc.returncode != 0 or payload.get("bitwise_identical") is not True:
            failed = True
    report["results"] = results
    report["elapsed_s"] = time.time() - t0
    report["verdict"] = ("PASS: candidate bitwise-identical in all cases; "
                         "see ref_ms/vec_ms (kernel time only, not TPS)."
                         if not failed else
                         "FAIL: correctness gate did not pass; no timing is admissible.")

    out_path = args.out_json or (args.workdir / "fp4_decode_vec_report.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nreport: {out_path}", file=sys.stderr)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
