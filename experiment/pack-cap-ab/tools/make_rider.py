#!/usr/bin/env python3
"""Build the rider-only pread kernel package (kernels/rider/).

The rider runs AFTER the completed pack-cap A/B (never between arms):
  1. mount the canonical shards dataset,
  2. clone + checkout pinned engine commit 217a3335 via the embedded
     repair bundle (GitHub history was force-pushed; the bundle carries
     the bit-identical commit keyed to the current remote tip),
  3. rebuild the v50 trace bank byte-identically (repack_to_dee4 CLI,
     hash-gated to the sealed v50 journal),
  4. validate bank identity (data_sha256 must equal the value observed in
     every pack-cap arm), then
  5. run bench_expert_pread.py (embedded b64+sha256) at depths 1..16,
     patterns seq+dispersed, repeat=2 (coldish + warm passes), with the
     CORRECTED argument shape: --store points at experts.dee4 (the
     multi-record file), --journal-meta at metadata.json.

No model, no decode, no A/B rerun. Storage-class match is preserved by
dispatching on a T4 VM (same VM class as the A/B arms) even though the
rider itself is storage-only.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPROOT = HERE.parent
BUNDLE = EXPROOT / "bundle" / "repair-217a3335-prereq-7b137846.bundle"
BENCH = Path("/c/Users/carth/Downloads/dee-staging/tools/bench_expert_pread.py")
if not BENCH.is_file():
    BENCH = HERE.parents[3] / "dee-staging" / "tools" / "bench_expert_pread.py"

ENGINE_COMMIT = "217a33359b06a0453444a698ec52e4078b77e388"
MODEL_REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
V50_JOURNAL_REL = ("benchmark_reports/deepseek-v4-flash-0731-t4/"
                   "v50-evidence-20260829T195940Z/routed_experts.jsonl")
V50_JOURNAL_SHA256 = "665aac3e8db570237c6dc6acaf08dc39f2af890e8a04e400ce7154f1a858dae1"
V50_FINAL_CHAIN_SHA256 = "086f8ca83b6a3c467cdf950096141fa9bc3e55a285d7d1fed8a0ad9913e3eb3d"
EXPECTED_BANK_DATA_SHA256 = "c83462ba50d2154a241cabf7d2b8d084e46f2eaa54416a96c90770ca7172d6e8"
DATASET_DIR = "/kaggle/input/datasets/nivind/deepseek-v4-flash-0731-shards"
N_SHARDS = 48

# The kernel script uses @@TOKEN@@ substitution only (never str.format) so its
# own braces and f-strings stay literal.
KERNEL_TEMPLATE = r'''#!/usr/bin/env python3
"""Rider-only concurrent-pread measurement (post pack-cap A/B).

Storage-only: no model, no decode, no A/B rerun. Runs the sealed
bench_expert_pread.py against a byte-identical rebuild of the v50 trace
bank. All cache states labeled honestly (coldish = fadvise DONTNEED,
page-cache ground truth unknown; warm = immediate repeat).
"""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ENGINE_COMMIT = "@@ENGINE_COMMIT@@"
MODEL_REV = "@@MODEL_REV@@"
V50_JOURNAL_REL = "@@V50_JOURNAL_REL@@"
V50_JOURNAL_SHA256 = "@@V50_JOURNAL_SHA256@@"
V50_FINAL_CHAIN_SHA256 = "@@V50_FINAL_CHAIN_SHA256@@"
EXPECTED_BANK_DATA_SHA256 = "@@BANK_SHA256@@"
DATASET_DIR = Path("@@DATASET_DIR@@")
N_SHARDS = @@N_SHARDS@@
ROOT = Path("/tmp/dsv4-native-src")
BANK = Path("/tmp/dsv4-dee4-v3-trace")
WORK = Path("/kaggle/working")

BUNDLE_B64 = "@@BUNDLE_B64@@"
BUNDLE_SHA256 = "@@BUNDLE_SHA256@@"
BENCH_B64 = "@@BENCH_B64@@"
BENCH_SHA256 = "@@BENCH_SHA256@@"

LOG_LINES = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def embed_verify(b64, expected, dest, what):
    raw = base64.b64decode(b64)
    got = hashlib.sha256(raw).hexdigest()
    if got != expected:
        raise RuntimeError(f"{what} sha256 mismatch: {got} != {expected}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    log(f"{what} embedded OK sha256={got[:16]}")
    return dest


def preflight():
    if not DATASET_DIR.is_dir():
        raise RuntimeError(f"dataset mount absent: {DATASET_DIR}")
    shards = sorted(DATASET_DIR.glob("model-*.safetensors"))
    if len(shards) != N_SHARDS:
        raise RuntimeError(f"expected {N_SHARDS} shards, found {len(shards)}")
    mt = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable"):
            mt[k] = round(int(v.strip().split()[0]) / 2**20, 2)
    log(f"preflight OK shards={len(shards)} mem={mt}")
    return shards


def clone_and_checkout(bundle_path):
    if ROOT.exists():
        shutil.rmtree(ROOT)
    subprocess.run(["git", "clone", "--branch", "freebuff/deepseek-v4-flash-0731-t4",
                    "--single-branch", "https://github.com/so-nerdyy/dee.git", str(ROOT)],
                   check=True)
    subprocess.run(["git", "-C", str(ROOT), "config", "lfs.process.allowincomplete", "true"],
                   check=False)
    subprocess.run(["git", "-C", str(ROOT), "fetch", str(bundle_path),
                    "refs/heads/codex/dee4-bounded-fill-storage"], check=True)
    subprocess.run(["git", "-C", str(ROOT), "checkout", "--quiet", ENGINE_COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != ENGINE_COMMIT:
        raise RuntimeError(f"checkout mismatch: {head}")
    log(f"engine checkout verified {head}")


def build_bank():
    if (BANK / "metadata.json").is_file():
        log("bank exists; validating in place")
        return
    if BANK.exists() and any(BANK.iterdir()):
        raise RuntimeError(f"non-empty bank dir without metadata; refusing: {BANK}")
    idx = DATASET_DIR / "model.safetensors.index.json"
    if not idx.is_file():
        idx = WORK / "model.safetensors.index.json"
        url = (f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
               f"resolve/{MODEL_REV}/model.safetensors.index.json")
        log(f"downloading index from HF: {url}")
        idx.write_bytes(urllib.request.urlopen(url, timeout=300).read())
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / "dee.cpp" / "kaggle" / "deepseek-v4-flash-0731" / "repack_to_dee4.py"),
           "--safetensors-dir", str(DATASET_DIR),
           "--output-dir", str(BANK),
           "--index-path", str(idx),
           "--trace-journal", str(ROOT / "dee.cpp" / V50_JOURNAL_REL),
           "--expected-trace-journal-sha256", V50_JOURNAL_SHA256,
           "--expected-trace-final-chain-sha256", V50_FINAL_CHAIN_SHA256,
           "--validate-samples", "12"]
    log("bank build: " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    (WORK / "rider-repack-stdout.log").write_text(
        (r.stdout or "")[-20000:] + "\n=== STDERR ===\n" + (r.stderr or "")[-20000:])
    log(f"bank build exit={r.returncode} wall={time.time() - t0:.1f}s")
    if r.returncode != 0:
        raise RuntimeError("repack_to_dee4 failed; see rider-repack-stdout.log")


def validate_bank():
    meta = json.loads((BANK / "metadata.json").read_text("utf-8"))
    checks = {
        "record_bytes": meta.get("record_bytes") == 13369344,
        "total_experts": meta.get("total_experts") == 2364,
        "data_sha256": meta.get("data_sha256") == EXPECTED_BANK_DATA_SHA256,
    }
    out = {"schema": "pack-cap-ab/rider-bank-validation-v1", "checks": checks,
           "observed": {k: meta.get(k) for k in
                        ("record_bytes", "total_experts", "data_sha256",
                         "trace_journal_sha256")},
           "expected_data_sha256": EXPECTED_BANK_DATA_SHA256,
           "status": "PASS" if all(checks.values()) else "FAIL"}
    (WORK / "rider-bank-validation.json").write_text(json.dumps(out, indent=2))
    log("bank validation: " + json.dumps(checks))
    if not all(checks.values()):
        raise RuntimeError("bank identity mismatch")


def run_bench(bench_path):
    cmd = [sys.executable, str(bench_path),
           "--store", str(BANK / "experts.dee4"),
           "--journal-meta", str(BANK / "metadata.json"),
           "--journal", str(ROOT / "dee.cpp" / V50_JOURNAL_REL),
           "--records", "96",
           "--depths", "1", "2", "3", "4", "6", "8", "12", "16",
           "--patterns", "seq", "dispersed",
           "--repeat", "2",
           "--label", "pack-cap-ab-rider-20260907",
           "--out", str(WORK / "pread-rider.json")]
    log("bench: " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    (WORK / "rider-bench-stdout.log").write_text(
        (r.stdout or "")[-20000:] + "\n=== STDERR ===\n" + (r.stderr or "")[-20000:])
    log(f"bench exit={r.returncode} wall={time.time() - t0:.1f}s")
    if r.returncode != 0:
        raise RuntimeError("bench failed; see rider-bench-stdout.log")


def main():
    log("=== pack-cap-ab pread rider (storage-only) ===")
    shards = preflight()
    bundle = embed_verify(BUNDLE_B64, BUNDLE_SHA256, Path("/tmp/repair.bundle"), "repair bundle")
    bench = embed_verify(BENCH_B64, BENCH_SHA256,
                         Path("/tmp/pread-rider/bench_expert_pread.py"), "bench")
    clone_and_checkout(bundle)
    build_bank()
    validate_bank()
    run_bench(bench)
    (WORK / "rider-kernel-log.txt").write_text("\n".join(LOG_LINES))
    log("=== rider COMPLETE ===")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=EXPROOT / "kernels" / "rider")
    ap.add_argument("--slug", default="dee-cpp-dsv4-pread-rider-20260907")
    args = ap.parse_args()

    bundle_raw = BUNDLE.read_bytes()
    bench_raw = BENCH.read_bytes()
    tokens = {
        "@@ENGINE_COMMIT@@": ENGINE_COMMIT,
        "@@MODEL_REV@@": MODEL_REV,
        "@@V50_JOURNAL_REL@@": V50_JOURNAL_REL,
        "@@V50_JOURNAL_SHA256@@": V50_JOURNAL_SHA256,
        "@@V50_FINAL_CHAIN_SHA256@@": V50_FINAL_CHAIN_SHA256,
        "@@BANK_SHA256@@": EXPECTED_BANK_DATA_SHA256,
        "@@DATASET_DIR@@": DATASET_DIR,
        "@@N_SHARDS@@": str(N_SHARDS),
        "@@BUNDLE_SHA256@@": hashlib.sha256(bundle_raw).hexdigest(),
        "@@BENCH_SHA256@@": hashlib.sha256(bench_raw).hexdigest(),
        "@@BUNDLE_B64@@": base64.b64encode(bundle_raw).decode("ascii"),
        "@@BENCH_B64@@": base64.b64encode(bench_raw).decode("ascii"),
    }
    script = KERNEL_TEMPLATE
    for k, v in tokens.items():
        script = script.replace(k, v)
    assert "@@" not in script, "unsubstituted placeholder remains"

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.slug}.py").write_text(script, encoding="utf-8")
    meta = {
        "id": f"nivind/{args.slug}",
        "title": "DSV4 pack-cap pread rider (storage-only, post A/B)",
        "code_file": f"{args.slug}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "dataset_sources": ["nivind/deepseek-v4-flash-0731-shards"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (args.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"rider package written to {args.out}")
    print(f"  script bytes: {len(script)}")
    print(f"  bundle sha: {hashlib.sha256(bundle_raw).hexdigest()[:16]}")
    print(f"  bench sha: {hashlib.sha256(bench_raw).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
