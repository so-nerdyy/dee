#!/usr/bin/env python3
"""P100 torch probe: verify an sm_60-capable torch wheel works on a Kaggle P100.

The preinstalled Kaggle torch (2.4+, cu124) has no sm_60 kernels -> any P100
allocation dies with cudaErrorNoKernelImageForDevice.  torch 2.3.1+cu118 is
the last line with P100 (sm_60) support.  This probe proves the repair path
before we wire it into the full generate harness.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open("/kaggle/working/probe.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass

def main() -> int:
    log("=== P100 torch probe start ===")
    # 1) What GPU did we get?
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        log("GPU: " + out.strip().replace("\n", " | "))
    except Exception as e:
        log(f"nvidia-smi -L failed: {e}")
    try:
        cc = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True).strip()
        log(f"compute_cap: {cc}")
    except Exception as e:
        log(f"compute_cap query failed: {e}")

    # 2) Show the preinstalled torch
    pre = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
        text=True, stderr=subprocess.STDOUT).strip()
    log(f"preinstalled torch: {pre}")

    # 3) If compute cap < 7.0, repair the torch install
    if cc and float(cc.split(",")[0].strip()) < 7.0:
        log("P100 detected -> installing torch 2.3.1+cu118 (last sm_60 line)")
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "torch==2.3.1+cu118", "--index-url",
             "https://download.pytorch.org/whl/cu118"],
            capture_output=True, text=True, timeout=600)
        log(f"pip install rc={r.returncode} in {time.time()-t0:.0f}s")
        if r.returncode != 0:
            log("PIP FAILED: " + r.stderr[-2000:])
            return 1
        post = subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
            text=True, stderr=subprocess.STDOUT).strip()
        log(f"repaired torch: {post}")

    # 4) Exercise CUDA on the GPU
    test = r'''
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
dev = torch.device("cuda:0")
print("device name:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
y = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
z = (x @ y).float()
print("matmul ok, sum =", z.sum().item())
a = torch.randn(512, 2048, device=dev, dtype=torch.float16)
w = torch.randn(2048, 4096, device=dev, dtype=torch.float16)
b = (a @ w).float()
print("big matmul ok, sum =", b.sum().item())
torch.cuda.synchronize()
print("PASS")
'''
    r = subprocess.run([sys.executable, "-c", test],
                       capture_output=True, text=True, timeout=300)
    log("--- torch test stdout ---")
    for ln in r.stdout.strip().splitlines():
        log("  " + ln)
    if r.returncode != 0:
        log("--- torch test stderr (tail) ---")
        for ln in r.stderr.strip().splitlines()[-15:]:
            log("  " + ln)
        return 1
    log("PROBE PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
