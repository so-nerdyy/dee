#!/usr/bin/env python3
"""P5c cohort-shape isolation driver (Kaggle GPU job).

Direct evidence for the Phase-5 numerical-equivalence contract.  The v2
cohort campaign (fixed build) showed the corruption defect gone — c1's
warm units bit-stable, c0 anchored — yet 22/24 ``row == c1`` gates still
failed with a new signature: ULP-scale expert-set diffs at prefill,
chaotic amplification across decode, one prompt (p5) bit-exact at every
K.  The hypothesis: batched GEMMs (m=K*L* vs m=L*) select different
cuBLAS kernels/reduction orders -> last-ulp differences -> top-6 flips
at near-tied router boundaries.

This driver runs ONE arm (``mN``) through the real model + engine in a
single process so the comparison carries no cross-run state:

    unit c0: prompt 0 at K=1          (singleton reference)
    unit c1: prompt 5 at K=1          (the v2 always-exact prompt)
    unit c2: prompts 0-7 at K=8       (member0 = identical padded p0)
    unit c3: prompts 0-7 at K=8       (same-shape bitwise determinism)

``NATIVE_ISO_DUMP=1`` wires per-layer capture dicts through
``generate_cohort`` and the route hook dumps the actual tensors
(``iso-c{u}_s{step}_l{layer}_{key}.npy``): router_scores / expert_ids /
routing_weights for all 43 layers at steps 0-7, plus the residual-chain
hidden tensors for layers 0-3 at step 0.  Offline analysis then:

  * finds the first elementwise hidden-state divergence (member0 of
    c2 vs the c0 singleton),
  * measures the rank-6/rank-7 router-score margin at every expert-set
    flip site and compares it to the measured tensor perturbation,
  * verifies member5 (p5) reproduces the c1 singleton bitwise, and
  * verifies c2 == c3 member-for-member (same-shape determinism).

The kernel script is self-configuring: it clones the branch, then
imports the proven phase5b machinery (build + store assembly + run_arm
collection + report) and injects only this arm — no env-var dependence.

--selfcheck validates the resolved arm/env without launching anything.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/so-nerdyy/dee.git"
BRANCH = "research/phase4-cache-hierarchy"
COMMIT = os.environ.get("P5_PINNED_COMMIT", "")
ROOT = Path(os.environ.get("P5_SOURCE_ROOT", "/tmp/dee-p5c-src"))
KDIR = ROOT / "dee.cpp" / "kaggle" / "deepseek-v4-flash-0731"

# The p5b driver module lives in the cloned tree; its module-level path
# constants read P5_SOURCE_ROOT at import time, so the clone + env set
# must happen BEFORE the import.
if not (ROOT / "dee.cpp" / "CMakeLists.txt").is_file():
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch",
         REPO, str(ROOT)], check=True, timeout=900)
if COMMIT:
    subprocess.run(["git", "-C", str(ROOT), "checkout", "--quiet", COMMIT],
                   check=True, timeout=120)
os.environ["P5_SOURCE_ROOT"] = str(ROOT)
sys.path.insert(0, str(KDIR))

import phase5b_mechanism_driver as p5b  # noqa: E402

# ---- 8-prompt workload (identical to the Phase-5 campaign) ----------
_B = ("<" + chr(0xFF5C) + "begin" + chr(0x2581) + "of" + chr(0x2581)
      + "sentence" + chr(0xFF5C) + ">")
_A = "<" + chr(0xFF5C) + "Assistant" + chr(0xFF5C) + ">"
_TEXTS = [
    "Explain how mRNA vaccines work, from injection to immune memory.",
    "Write a Python function that finds the longest common subsequence "
    "of two strings, and explain its time and space complexity.",
    "A fair coin is flipped until two consecutive heads appear. What is "
    "the expected number of flips? Show your reasoning step by step.",
    "Write the opening paragraph of a hard science fiction novel about "
    "a generation ship whose crew discovers the laws of physics are "
    "slightly different two light-years from Earth.",
    "Explain in one sentence why the sky is blue.",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three causes of the French Revolution.",
]
PROMPTS = [_B + t + _A for t in _TEXTS]
PROMPTS.append(PROMPTS[0])  # q7 = q0 verbatim (determinism probe)

PACK_C8H = int(9.75 * (1 << 30))   # c8h host pack/GPU from the campaign
LRU_CAP_C8H_GIB = 19.5             # c8h host-LRU cap from the campaign

ISO_ARM = {
    # mN: cohort-shape isolation.  groups order matters for analysis:
    # c0/c1 are the K=1 references (prompts 0 and 5), c2/c3 are the
    # identical K=8 cohorts.  pad_to="max" pads every unit to the
    # workload-global L* so the K=1 rows are byte-identical inputs to
    # the corresponding cohort members.
    **p5b._BASE,
    "arm_id": "mN",
    "host_pack_gpu0_bytes": PACK_C8H,
    "host_pack_gpu1_bytes": PACK_C8H,
    "lru_total_cap_gib": LRU_CAP_C8H_GIB,
    "prompts_json": PROMPTS,
    "cohort": {"groups": [[0], [5],
                         [0, 1, 2, 3, 4, 5, 6, 7],
                         [0, 1, 2, 3, 4, 5, 6, 7]],
               "pad_to": "max"},
    "route_weight_journal": "1",
    "extra_env": {
        "NATIVE_ISO_DUMP": "1",
        "NATIVE_ISO_H_LAYERS": "0,1,2,3",
        "NATIVE_ISO_STEPS": "0,1,2,3,4,5,6,7",
        "NATIVE_ISO_H_STEPS": "0",
        # The isolation design repeats prompt indices across units by
        # construction (p0/p5 at K=1 and inside the K=8 cohorts).
        "NATIVE_COHORT_ALLOW_DUPES": "1",
    },
}

p5b.ARMS = [ISO_ARM]
# Four units in one runner process: two K=1 (~20 min each) + two K=8
# (~40 min each) against a cold-reset store — generous bound.
p5b.ARM_TIMEOUT_S = 10800


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        sys.exit(0 if p5b.selfcheck() else 1)
    p5b.main()
    sys.exit(0)
