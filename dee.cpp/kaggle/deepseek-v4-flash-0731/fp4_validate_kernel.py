"""Single-GPU P2.3 validation: build dee_core + FP4 regression tests on a T4.

The dual-T4 generation kernel keeps getting P100/single-GPU allocations
(v18-v22), which cannot load the sm_75 cubins.  This kernel runs on whatever
single GPU Kaggle provides and validates that the P2.3 packed-FP4 cache code
compiles for sm_75 AND that the FP4 expert test (now including the
packed-cache dtype) passes.  It does not need the checkpoint dataset.

Usage: kaggle kernels push -p <dir with this file + kernel-metadata.json>
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
COMMIT = os.environ.get("NATIVE_COMMIT", "")

ROOT = Path("/tmp/dsv4-fp4-validate-src")
DEE = ROOT / "dee.cpp"
BUILD = DEE / "build-kaggle"


def run(args, **kw):
    print("+", " ".join(str(a) for a in args), flush=True)
    return subprocess.run(args, **kw)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    log("=== GPU identity ===")
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True,
                                      stderr=subprocess.STDOUT)
        log(out.strip())
    except Exception as e:
        log(f"nvidia-smi failed: {e}")
        return 2
    if "T4" not in out:
        # Not a T4: the sm_75 cubins cannot load.  Report and exit - this is
        # the same infra rejection the generate kernel now catches early.
        log("NOT_A_T4 - sm_75 cubins cannot load on this GPU; exiting early")
        return 3

    log("=== clone + checkout ===")
    if ROOT.exists():
        run(["rm", "-rf", str(ROOT)])
    run(["git", "clone", "--branch", BRANCH, "--single-branch",
         REPO, str(ROOT)])
    if COMMIT:
        run(["git", "-C", str(ROOT), "checkout", "--quiet", COMMIT])
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    log(f"pinned commit {head}")

    log("=== build dee_core + FP4 regression tests ===")
    run(["cmake", "-S", str(DEE), "-B", str(BUILD),
         "-DCMAKE_CUDA_ARCHITECTURES=75", "-DDEE_CUDA=ON",
         "-DDEE_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=Release"])
    r = run(["cmake", "--build", str(BUILD), "--target", "dee_core",
             "test_deepseek_v4_fp4_cuda", "test_deepseek_v4_fp4_expert",
             "-j", str(os.cpu_count() or 4)])
    if r.returncode != 0:
        log("BUILD_FAILED")
        return 1

    failures = 0
    for target in ("test_deepseek_v4_fp4_cuda", "test_deepseek_v4_fp4_expert"):
        r = run([str(BUILD / target)], cwd=str(DEE))
        if r.returncode != 0:
            failures += 1
            log(f"TEST_FAILED: {target}")
        else:
            log(f"TEST_PASSED: {target}")

    log("FP4_VALIDATE_RESULT " + ("PASS" if failures == 0 else f"FAIL({failures})"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
