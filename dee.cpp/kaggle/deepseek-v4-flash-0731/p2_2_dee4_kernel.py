#!/usr/bin/env python3
"""P2.2 Kaggle kernel: repack DeepSeek-V4-Flash experts into DEE4 format,
then benchmark contiguous reads vs safetensors random gathers.

Strategy: repack layers 0-2 (768 experts, ~9.8 GiB) to fit within
/kaggle/working's 20 GiB limit.  This is enough to prove the approach.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Clone the repo to get the repack tool
REPO_URL = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"

print("=== P2.2 DEE4 Repack ===", flush=True)

# ── 1. Clone repo ───────────────────────────────────────────────────
repo_dir = Path("/kaggle/working/dee-repo")
if repo_dir.exists():
    shutil.rmtree(repo_dir)
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(repo_dir)],
    check=True,
)
sys.path.insert(0, str(repo_dir / "dee.cpp" / "kaggle" / "deepseek-v4-flash-0731"))
sys.path.insert(0, str(repo_dir / "dee.cpp" / "scripts"))

# ── 2. Find safetensors shards ──────────────────────────────────────
DATASET_DIR = Path("/kaggle/input/deepseek-v4-flash-0731-shards")
shards = sorted(DATASET_DIR.glob("model-*.safetensors"))
index = DATASET_DIR / "model.safetensors.index.json"
print(f"Shards: {len(shards)}, index exists: {index.is_file()}", flush=True)

# ── 3. Run repack ────────────────────────────────────────────────────
from repack_to_dee4 import repack, benchmark_dee4_read, EXPERT_TOTAL_BYTES, EXPERTS_PER_LAYER

OUTPUT_DIR = Path("/kaggle/working/dee4-test")

t0 = time.monotonic()
report = repack(
    DATASET_DIR,
    OUTPUT_DIR,
    start_layer=0,
    end_layer=3,   # layers 0,1,2 = 768 experts
    dry_run=False,
)
repack_elapsed = time.monotonic() - t0

# ── 4. Benchmark contiguous DEE4 reads ───────────────────────────────
bench = benchmark_dee4_read(OUTPUT_DIR, n_experts=64)

# ── 5. Compare: safetensors random gather (6 pieces per expert) ──────
# Simulate the current access pattern: 6 reads per expert, scattered
import hashlib
import struct

def time_safetensors_gather(n_experts: int = 64) -> dict:
    """Time the current safetensors access pattern."""
    # Read index
    idx = json.loads(index.read_text("utf-8"))
    wm = idx["weight_map"]

    # Cache shard headers
    shard_headers = {}
    shard_paths = {}
    for shard_name in sorted(set(wm.values())):
        p = DATASET_DIR / shard_name
        shard_paths[shard_name] = p
        with open(p, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            shard_headers[shard_name] = json.loads(f.read(hl))

    t0 = time.monotonic()
    total_bytes = 0
    read_count = 0

    for layer in range(3):
        for expert_id in range(min(n_experts // 3, 256)):
            for proj in ["w1", "w2", "w3"]:
                for kind in ["weight", "scale"]:
                    name = f"layers.{layer}.ffn.experts.{expert_id}.{proj}.{kind}"
                    if name not in wm:
                        continue
                    shard_name = wm[name]
                    header = shard_headers[shard_name]
                    offsets = header[name]["data_offsets"]
                    length = offsets[1] - offsets[0]
                    with open(shard_paths[shard_name], "rb") as f:
                        f.seek(8 + offsets[0])
                        data = f.read(length)
                    total_bytes += len(data)
                    read_count += 1
                    if read_count >= n_experts * 6:
                        break
                if read_count >= n_experts * 6:
                    break
            if read_count >= n_experts * 6:
                break
        if read_count >= n_experts * 6:
            break

    elapsed = time.monotonic() - t0
    return {
        "mode": "safetensors-random-gather",
        "n_experts": read_count // 6,
        "n_reads": read_count,
        "total_mib": total_bytes / (1 << 20),
        "elapsed_s": elapsed,
        "mbps": total_bytes / max(elapsed, 0.001) / (1 << 20),
        "reads_per_second": read_count / max(elapsed, 0.001),
    }

st_result = time_safetensors_gather(n_experts=64)

# ── 6. Report ────────────────────────────────────────────────────────
result = {
    "schema": "p2.2-v1",
    "repack": {
        "elapsed_s": repack_elapsed,
        "total_experts": report["total_experts_repacked"],
        "total_gib": report["total_bytes_repacked"] / (1 << 30),
        "throughput_mibps": report["throughput_mibps"],
        "errors": len(report["errors"]),
    },
    "dee4_read_benchmark": bench,
    "safetensors_benchmark": st_result,
    "speedup": {
        "read_mbps_ratio": round(bench["aggregate_mbps"] / max(st_result["mbps"], 0.01), 1),
        "io_count_reduction": f"{st_result['n_reads']} random -> {len(bench['tests'])} sequential",
    },
}

print(json.dumps(result, indent=2), flush=True)

# Write evidence
(Path("/kaggle/working") / "p2.2-dee4-result.json").write_text(
    json.dumps(result, indent=2), "utf-8"
)

print("\n✓ P2.2 complete", flush=True)