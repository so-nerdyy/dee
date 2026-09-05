#!/usr/bin/env python3
"""Build the two counterbalanced session kernel packages for pack-cap-ab.

Arm A = byte-original base harness (cap 17.0).
Arm B = base harness with EXACTLY ONE change: LRU_TOTAL_CAP_GIB 17.0 -> 20.0.

Each session kernel = ONE self-contained driver script (session_driver.py
with both arm harnesses embedded base64+sha256) + kernel-metadata.json.
Session 1 order A,B; session 2 order B,A (pre-registered, fixed). The pread
rider is embedded and enabled ONLY in session 2 (contract: rider runs only
after all A/B arms are terminal; session 2 tail is the last arm).

The base harness is the seal-era executing artifact pulled from the
canonical kernel (dee-cpp-dsv4-host-reuse-217-ab-20260904). Its sha256 is
pinned in experiment-contract.json and re-verified here; any drift aborts.

Launch (matches the seal-era dispatch recipe recorded in
dee.cpp/tmp/host-reuse-217-ab-20260904/run_control.py):
    kaggle kernels push <dir> --timeout 14400 --acc NvidiaTeslaT4
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
BASE = PKG / "base" / "dee-cpp-dsv4-host-reuse-217-ab-20260904.py"
CONTRACT = PKG / "experiment-contract.json"
DRIVER = HERE / "session_driver.py"
PREAD_SRC = Path("C:/Users/carth/Downloads/dee-staging/tools/bench_expert_pread.py")
REPAIR_BUNDLE = PKG / "bundle" / "repair-217a3335-prereq-7b137846.bundle"
CAP_LINE_OLD = "    LRU_TOTAL_CAP_GIB = 17.0"
CAP_LINE_NEW = "    LRU_TOTAL_CAP_GIB = 20.0"
# GitHub force-push repair (2026-09-05): the remote branch now tips at the old
# baseline 7b137846, so commit 217a3335 no longer exists on GitHub and the
# seal-era incremental bundle (ac2bac46, prereq: then-tip) can't apply. The
# repair bundle carries the BIT-IDENTICAL commit 217a3335 with prerequisite
# = the CURRENT remote tip 7b137846 (which every fresh clone provides).
# Only the two embedded source constants change (same bundle head name, so
# the harness fetch refspec is untouched); checkout verifies the same pin.
REPAIR_PREREQ = "7b137846893c46b3aed2c7e322f0dbbd8d3ce0ec"
BUNDLE_SHA_OLD = "ac2bac46c471dcfd977835712d1d351c4e06d3195fc54727e8660adebe7d4341"
BUNDLE_SHA_NEW = "76e1b437c6bf521d0c38d8797adf4d354a07c836fa1c92eae9328e38a11c353b"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_base() -> str:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    want = contract["canonical_baseline"]["base_harness_sha256"]
    got = sha256(BASE)
    if got != want:
        raise SystemExit(
            f"base harness sha256 drift: {got} != pinned {want}; refusing")
    return got


def patch_source_integrity(text: str) -> str:
    """Swap BOTH embedded source-integrity constants for the repair bundle.

    GitHub was force-pushed (branch tip 7b137846), so the seal-era incremental
    bundle no longer applies on a fresh clone: its prerequisite commit is gone.
    The repair bundle carries the BIT-IDENTICAL commit 217a3335 with
    prerequisite = the CURRENT remote tip (which every fresh clone provides).
    The fetch refspec and checkout pin are NOT modified. Exactly two
    assignments change; anything else aborts.
    """
    if not REPAIR_BUNDLE.is_file():
        raise SystemExit(f"repair bundle missing: {REPAIR_BUNDLE}")
    bundle_bytes = REPAIR_BUNDLE.read_bytes()
    got = hashlib.sha256(bundle_bytes).hexdigest()
    if got != BUNDLE_SHA_NEW:
        raise SystemExit(f"repair bundle sha drift: {got} != {BUNDLE_SHA_NEW}")
    new_b64 = base64.b64encode(bundle_bytes).decode()
    # Single-line SOURCE_BUNDLE_B64 = "..." assignment (seal-era layout).
    pat = re.compile(r'SOURCE_BUNDLE_B64 = "[A-Za-z0-9+/=]+"')
    if len(pat.findall(text)) != 1:
        raise SystemExit("expected exactly 1 SOURCE_BUNDLE_B64 assignment")
    text = pat.sub('SOURCE_BUNDLE_B64 = "' + new_b64 + '"', text)
    old_sha = f'SOURCE_BUNDLE_SHA256 = "{BUNDLE_SHA_OLD}"'
    if text.count(old_sha) != 1:
        raise SystemExit("expected exactly 1 old SOURCE_BUNDLE_SHA256 line")
    text = text.replace(old_sha, f'SOURCE_BUNDLE_SHA256 = "{BUNDLE_SHA_NEW}"')
    return text


def make_arm_a() -> tuple[Path, str]:
    """Arm A = byte-original base harness + GitHub-force-push repair bundle.

    The ONLY changes vs the committed seal-era artifact are the two embedded
    source-integrity constants (see patch_source_integrity). Everything else -
    including the checkout pin 217a3335 and the cap constant 17.0 - is
    byte-identical. The repair is recorded in harness-provenance.json.
    """
    src = BASE.read_text(encoding="utf-8")
    src = patch_source_integrity(src)
    out = PKG / "harness_arm_a_repaired.py"
    out.write_text(src, encoding="utf-8")
    return out, sha256(out)


def make_arm_b() -> tuple[Path, str]:
    src = BASE.read_text(encoding="utf-8")
    src = patch_source_integrity(src)
    n = src.count(CAP_LINE_OLD)
    if n != 1:
        raise SystemExit(f"expected exactly 1 cap line, found {n}; refusing")
    patched = src.replace(CAP_LINE_OLD, CAP_LINE_NEW)
    a_text = BASE.read_text(encoding="utf-8")
    # exactly-three-line-diff proof vs the ORIGINAL (sha line, B64 line, cap):
    diffs = [(x, y) for x, y in zip(a_text.splitlines(), patched.splitlines())
             if x != y]
    if len(diffs) != 3:
        raise SystemExit(f"arm B must differ by exactly 3 lines "
                         f"(bundle sha + bundle b64 + cap); got {len(diffs)}; "
                         f"refusing")
    if not any("LRU_TOTAL_CAP_GIB = 17.0" in x and
               "LRU_TOTAL_CAP_GIB = 20.0" in y for x, y in diffs):
        raise SystemExit("cap line not among the diffs; refusing")
    if not any("SOURCE_BUNDLE_SHA256" in x for x, y in diffs):
        raise SystemExit("bundle sha line not among the diffs; refusing")
    out = PKG / "harness_arm_b_cap20.py"
    out.write_text(patched, encoding="utf-8")
    return out, sha256(out)


def emit_diff_record(a: Path, b: Path) -> str:
    """Diff the two ACTUAL arm harnesses (arm B vs arm A): the only line that
    differs must be the cap constant (both share the repair bundle)."""
    r = subprocess.run(["git", "diff", "--no-index", "--", str(a), str(b)],
                       capture_output=True, text=True)
    return r.stdout


def build_driver(session: str, order: str, arm_a: bytes, arm_a_sha: str,
                 arm_b: bytes, arm_b_sha: str, pread: bytes | None,
                 pread_sha: str) -> str:
    text = DRIVER.read_text(encoding="utf-8")
    subs = {
        "@@ARM_A_B64@@": base64.b64encode(arm_a).decode(),
        "@@ARM_A_SHA256@@": arm_a_sha,
        "@@ARM_B_B64@@": base64.b64encode(arm_b).decode(),
        "@@ARM_B_SHA256@@": arm_b_sha,
        "@@SESSION_ID@@": session,
        "@@ARM_ORDER@@": order,
        "@@PREAD_B64@@": base64.b64encode(pread).decode() if pread else "",
        "@@PREAD_SHA256@@": pread_sha or "",
        "@@PREAD_ENABLE@@": "1" if (pread and session == "session2") else "0",
    }
    for k, v in subs.items():
        if k not in text:
            raise SystemExit(f"placeholder {k} missing from driver template")
        text = text.replace(k, v)
    # NOTE: do NOT test for a bare "@@" here — the driver legitimately
    # contains the literal "@@" inside its rider sanity check.
    for k in subs:
        if k in text:
            raise SystemExit(f"unsubstituted placeholder {k} remains; refusing")
    return text


def make_session(slug: str, title: str, session: str, order: str,
                 driver_text: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "session-driver.py").write_text(driver_text, encoding="utf-8")
    meta = {
        "id": slug,
        "title": title,
        "code_file": "session-driver.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [
            "nivind/deepseek-v4-flash-0731-shards",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "enable_tpu": "false",
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"session package: {out_dir} (order {order})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1-dir", type=Path, default=PKG / "kernels" / "session1")
    ap.add_argument("--s2-dir", type=Path, default=PKG / "kernels" / "session2")
    ap.add_argument("--pread", type=Path, default=PREAD_SRC)
    args = ap.parse_args()

    base_sha = verify_base()
    arm_a, arm_a_sha = make_arm_a()
    arm_b, arm_b_sha = make_arm_b()
    diff = emit_diff_record(arm_a, arm_b)
    prov = {
        "schema": "pack-cap-ab/harness-provenance-v2",
        "github_force_push_repair": {
            "discovered": "2026-09-05: first dispatch failed at checkout "
                          "(217a3335 no longer on GitHub; branch force-pushed "
                          "to 7b137846)",
            "repair_bundle": str(REPAIR_BUNDLE.relative_to(PKG)),
            "repair_bundle_sha256": BUNDLE_SHA_NEW,
            "carries_commit": "217a33359b06a0453444a698ec52e4078b77e388 "
                              "(bit-identical, verified by git cat-file after "
                              "fresh clone + fetch)",
            "prerequisite": REPAIR_PREREQ + " (current remote tip, "
                            "provided by every fresh clone)",
            "harness_lines_changed": [
                "SOURCE_BUNDLE_SHA256 (ac2bac46... -> 76e1b437...)",
                "SOURCE_BUNDLE_B64 (payload swap, same head refspec)",
                "+ LRU_TOTAL_CAP_GIB on arm B ONLY",
            ],
            "checkout_pin_unchanged": True,
            "fetch_refspec_unchanged": True,
            "validation": "fresh clone of force-pushed branch + repair bundle "
                          "-> checkout 217a3335 succeeded locally",
        },
        "arm_A": {"file": str(arm_a.relative_to(PKG)), "sha256": arm_a_sha,
                  "base_file": str(BASE.relative_to(PKG)),
                  "base_sha256": base_sha,
                  "modifications": [
                      "SOURCE_BUNDLE_SHA256/B64 -> repair bundle (identical "
                      "engine commit; GitHub force-push repair)"]},
        "arm_B": {"file": str(arm_b.relative_to(PKG)), "sha256": arm_b_sha,
                  "modifications": [
                      "SOURCE_BUNDLE_SHA256/B64 -> repair bundle (identical "
                      "engine commit; GitHub force-push repair)",
                      "LRU_TOTAL_CAP_GIB 17.0 -> 20.0 (single line)"]},
        "unified_diff": diff,
    }
    (PKG / "harness-provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8")

    arm_a_bytes = arm_a.read_bytes()
    arm_b_bytes = arm_b.read_bytes()
    pread_bytes = None
    pread_sha = ""
    if args.pread.is_file():
        pread_bytes = args.pread.read_bytes()
        pread_sha = hashlib.sha256(pread_bytes).hexdigest()
    else:
        print(f"NOTE: pread bench not found at {args.pread}; rider disabled")

    d1 = build_driver("session1", "A,B", arm_a_bytes, arm_a_sha,
                      arm_b_bytes, arm_b_sha, pread_bytes, pread_sha)
    d2 = build_driver("session2", "B,A", arm_a_bytes, arm_a_sha,
                      arm_b_bytes, arm_b_sha, pread_bytes, pread_sha)
    make_session("nivind/dee-cpp-dsv4-pack-cap-s1r-20260905",
                 "dee-cpp-dsv4-pack-cap-s1r-20260905",
                 "session1", "A,B", d1, args.s1_dir)
    make_session("nivind/dee-cpp-dsv4-pack-cap-s2r-20260905",
                 "dee-cpp-dsv4-pack-cap-s2r-20260905",
                 "session2", "B,A", d2, args.s2_dir)
    print(json.dumps({"base_sha256": base_sha,
                      "arm_a_repaired_sha256": arm_a_sha,
                      "arm_b_sha256": arm_b_sha,
                      "repair_bundle_sha256": BUNDLE_SHA_NEW,
                      "pread_sha256": pread_sha,
                      "driver_s1_sha256": hashlib.sha256(d1.encode()).hexdigest(),
                      "driver_s2_sha256": hashlib.sha256(d2.encode()).hexdigest(),
                      "diff_lines": len(diff.splitlines())}, indent=1))


if __name__ == "__main__":
    main()
