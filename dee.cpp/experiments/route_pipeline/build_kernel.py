#!/usr/bin/env python3
"""Build the host-sync profile Kaggle kernel directory.

Regenerates the profiler patch from this branch, embeds it (base64 +
sha256) plus the audited variant RULES into the session-driver template,
and writes kernel-metadata.json. The generated kernel dir is committed for
audit so the exact pushed bytes are reviewable.

Usage:
    python3 build_kernel.py --out dee.cpp/experiments/route_pipeline/kernel_host_sync
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
KERNEL_ID = "nivind/dee-cpp-host-sync-profile-20260905"
FILL_KERNEL_ID = "nivind/dee-cpp-fill-measure-20260909"
DATASET = "nivind/deepseek-v4-flash-0731-shards"

PROFILER_FILES = (
    "dee.cpp/src/engine.cpp",
    "dee.cpp/include/dee/engine.h",
    "dee.cpp/include/dee/profiling.h",
    "dee.cpp/src/profiling.cpp",
    "dee.cpp/pydee/pydee.cpp",
    "dee.cpp/scripts/deepseek_v4_layer_candidate.py",
)

# Fill flavor adds the committed fill-extra patch (hand-ported onto 217a333)
# on top of the generated profiler patch. See patches/fill-extra-217a333.patch.


def git_base() -> str:
    proc = subprocess.run(["git", "merge-base", "HEAD", "217a333"],
                          capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError("cannot determine merge-base with 217a333")
    return proc.stdout.strip()


def fill_patch() -> tuple[bytes, list[str]]:
    """Fill flavor: profile patch (generated, same as profile flavor) PLUS
    the committed fill-extra patch (hand-ported onto 217a333, verified to
    apply sequentially in tests). The committed file — not a worktree diff —
    is embedded so the bytes are reviewable and stable."""
    prof = profiler_patch()
    extra = (HERE / "patches" / "fill-extra-217a333.patch").read_bytes()
    files = (list(PROFILER_FILES) + [
        "dee.cpp/include/dee/host_pack_cache.h",
        "dee.cpp/src/host_pack_cache.cpp",
        "dee.cpp/include/dee/expert_store.h",
        "dee.cpp/src/expert_store.cpp",
        "dee.cpp/CMakeLists.txt",
        "dee.cpp/tools/fill_replay.cpp",
    ])
    return prof + extra, files


def profiler_patch() -> bytes:
    base = git_base()
    proc = subprocess.run(["git", "diff", base, "--", *PROFILER_FILES],
                          capture_output=True, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError("git diff failed")
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flavor", choices=("profile", "fill"),
                        default="profile")
    args = parser.parse_args()

    sys.path.insert(0, str(HERE))
    from make_profile_variant import RULES  # noqa: E402

    if args.flavor == "fill":
        patch, files = fill_patch()
        template_name = "kernel_fill_driver_template.py"
        kernel_id = FILL_KERNEL_ID
        title = "dee-cpp-fill-measure-20260909"
        patch_prefix = "FILL_PATCH"
    else:
        patch, files = profiler_patch(), PROFILER_FILES
        template_name = "kernel_session_driver_template.py"
        kernel_id = KERNEL_ID
        title = "dee-cpp-host-sync-profile-20260905"
        patch_prefix = "PROFILER_PATCH"
    if not patch.strip():
        raise RuntimeError("empty patch; nothing to embed")
    patch_sha = hashlib.sha256(patch).hexdigest()
    rules_json = json.dumps([{"id": r[0], "old": r[1], "new": r[2], "why": r[3]}
                             for r in RULES], indent=2).encode("utf-8")
    rules_sha = hashlib.sha256(rules_json).hexdigest()
    head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                          capture_output=True, text=True,
                          cwd=str(ROOT)).stdout.strip()
    import time as _time
    build_id = f"{head}-{_time.strftime('%Y%m%dT%H%M%SZ', _time.gmtime())}"
    template = (HERE / template_name).read_text(encoding="utf-8")
    for placeholder in (f"@@{patch_prefix}_B64@@",
                        f"@@{patch_prefix}_SHA256@@",
                        "@@VARIANT_RULES_B64@@", "@@VARIANT_RULES_SHA256@@",
                        "@@DRIVER_BUILD_ID@@"):
        if placeholder not in template:
            raise RuntimeError(f"template missing {placeholder}")
    driver = template.replace(
        f"@@{patch_prefix}_B64@@",
        base64.b64encode(patch).decode("ascii")).replace(
        f"@@{patch_prefix}_SHA256@@", patch_sha).replace(
        "@@VARIANT_RULES_B64@@",
        base64.b64encode(rules_json).decode("ascii")).replace(
        "@@VARIANT_RULES_SHA256@@", rules_sha).replace(
        "@@DRIVER_BUILD_ID@@", build_id)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "session-driver.py").write_text(driver, encoding="utf-8")
    import re as _re
    # Unified-diff hunk headers (@@ -a,b +c,d @@) contain lowercase/digits
    # and never match; any ALL-CAPS @@TOKEN@@ left is an unfilled
    # placeholder and fails the build.
    leftover = sorted(set(_re.findall(r"@@[A-Z_]+@@", driver)))
    if leftover:
        raise RuntimeError(f"unfilled placeholders: {leftover}")
    (args.out / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id,
        "title": title,
        "code_file": "session-driver.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "enable_tpu": "false",
    }, indent=2), encoding="utf-8")
    driver_path = args.out / "session-driver.py"
    driver_sha = (hashlib.sha256(driver_path.read_bytes()).hexdigest()
                  if driver_path.is_file() else None)
    (args.out / "BUILD-PROOF.json").write_text(json.dumps({
        "profiler_patch_sha256": patch_sha,
        "profiler_patch_base": git_base(),
        "profiler_files": list(files),
        "flavor": args.flavor,
        "rule_ids": [r[0] for r in RULES],
        "rules_sha256": rules_sha,
        "driver_build_id": build_id,
        "driver_sha256": driver_sha,
        "kernel_id": kernel_id,
    }, indent=2), encoding="utf-8")
    print(f"kernel dir: {args.out}")
    print(f"patch sha: {patch_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
