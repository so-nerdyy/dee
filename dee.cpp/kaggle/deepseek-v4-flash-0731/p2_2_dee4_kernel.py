#!/usr/bin/env python3
"""P2.2 v3: inspect dataset structure, then repack."""
import json, os, shutil, struct, subprocess, sys, time, urllib.request
from pathlib import Path

REPO_URL = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

print("=== P2.2 DEE4 Repack v3 ===", flush=True)

# ── 0. Inspect dataset ─────────────────────────────────────────────
DATASET_DIR = Path("/kaggle/input/deepseek-v4-flash-0731-shards")
WORKING = Path("/kaggle/working")
print(f"DATASET_DIR: {DATASET_DIR}")
print(f"exists: {DATASET_DIR.exists()}, is_dir: {DATASET_DIR.is_dir()}")
for p in sorted(DATASET_DIR.iterdir()):
    kind = "DIR" if p.is_dir() else "FILE"
    size = ""
    if p.is_file():
        size = f" ({p.stat().st_size/(1<<30):.2f}GiB)"
    print(f"  {kind}: {p.name}{size}")
SAFETENSORS = sorted(list(DATASET_DIR.glob("**/*.safetensors")))
print(f"Found {len(SAFETENSORS)} safetensors")
for s in SAFETENSORS[:5]:
    print(f"  {s.relative_to(DATASET_DIR)} ({s.stat().st_size/(1<<30):.2f}GiB)")

# ── 1. Clone repo ─────────────────────────────────────────────────
repo_dir = WORKING / "dee-repo"
if repo_dir.exists(): shutil.rmtree(repo_dir)
subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(repo_dir)], check=True)
sys.path.insert(0, str(repo_dir / "dee.cpp" / "kaggle" / "deepseek-v4-flash-0731"))
sys.path.insert(0, str(repo_dir / "dee.cpp" / "scripts"))

# ── 2. Download index ─────────────────────────────────────────────
INDEX_PATH = WORKING / "model.safetensors.index.json"
if not INDEX_PATH.is_file():
    url = f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/{REV}/model.safetensors.index.json"
    print(f"Downloading index: {url}", flush=True)
    data = urllib.request.urlopen(url, timeout=300).read()
    INDEX_PATH.write_bytes(data)
    print(f"Index downloaded: {len(data)} bytes", flush=True)

from repack_to_dee4 import repack, benchmark_dee4_read

if len(SAFETENSORS) == 0:
    print("NO SAFETENSORS FOUND — aborting")
    sys.exit(0)

# ── 3. Run repack ─────────────────────────────────────────────────
OUTPUT_DIR = WORKING / "dee4-test"
t0 = time.monotonic()
report = repack(DATASET_DIR, OUTPUT_DIR, index_path=INDEX_PATH, start_layer=0, end_layer=3, dry_run=False)
repack_elapsed = time.monotonic() - t0

bench = benchmark_dee4_read(OUTPUT_DIR, n_experts=64)

# ── 4. Compare safetensors random gather ──────────────────────────
def time_safetensors_gather(n_experts=64):
    idx = json.loads(INDEX_PATH.read_text("utf-8"))
    wm = idx["weight_map"]
    shard_headers, shard_paths = {}, {}
    for sn in sorted(set(wm.values())):
        p = DATASET_DIR / sn
        if not p.is_file():
            # try glob
            candidates = list(DATASET_DIR.glob(f"**/{sn}"))
            if candidates: p = candidates[0]
            else: continue
        shard_paths[sn] = p
        with open(p, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            shard_headers[sn] = json.loads(f.read(hl))
    t0 = time.monotonic(); total_bytes = 0; read_count = 0
    for layer in range(3):
        for eid in range(min(n_experts // 3, 256)):
            for proj in ["w1","w2","w3"]:
                for kind in ["weight","scale"]:
                    name = f"layers.{layer}.ffn.experts.{eid}.{proj}.{kind}"
                    if name not in wm: continue
                    sn = wm[name]
                    if sn not in shard_paths: continue
                    hdr = shard_headers[sn]
                    off = hdr[name]["data_offsets"]; length = off[1] - off[0]
                    with open(shard_paths[sn], "rb") as f:
                        f.seek(8 + off[0]); data = f.read(length)
                    total_bytes += len(data); read_count += 1
            if read_count >= n_experts * 6: break
        if read_count >= n_experts * 6: break
    elapsed = time.monotonic() - t0
    return {"mode":"safetensors-random-gather","n_experts":read_count//6,"n_reads":read_count,
            "total_mib":total_bytes/(1<<20),"elapsed_s":elapsed,
            "mbps":total_bytes/max(elapsed,0.001)/(1<<20),
            "reads_per_second":read_count/max(elapsed,0.001)}

st_result = time_safetensors_gather(64)
result = {"schema":"p2.2-v3",
    "repack":{"elapsed_s":repack_elapsed,"total_experts":report["total_experts_repacked"],
              "total_gib":report["total_bytes_repacked"]/(1<<30),
              "throughput_mibps":report["throughput_mibps"],"errors":len(report["errors"])},
    "dee4_read_benchmark":bench,"safetensors_benchmark":st_result,
    "speedup":{"read_mbps_ratio":round(bench["aggregate_mbps"]/max(st_result["mbps"],0.01),1),
               "io_count_reduction":f"{st_result['n_reads']} random -> {len(bench['tests'])} sequential"}}
print(json.dumps(result, indent=2), flush=True)
(WORKING / "p2.2-dee4-result.json").write_text(json.dumps(result, indent=2), "utf-8")
print("\nP2.2 complete", flush=True)
