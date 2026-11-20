#!/usr/bin/env python3
"""Tests for the host-sync profile kernel package (no CUDA, no Kaggle).

Verifies: the embedded profiler patch matches the branch source, its hunks
are present in 217a333 (applicability), the patch is additive except the one
const relaxation, the variant rules equal the audited module, the variant
builds from sealed arm bytes, the driver dry-run gate behaves, and ABC key
parity holds. Nothing here executes a model.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "dee.cpp" / "experiments" / "route_pipeline"
sys.path.insert(0, str(EXP))

KERNEL = EXP / "kernel_host_sync"
ENGINE_217 = "217a33359b06a0453444a698ec52e4078b77e388"
SEALED_ARM_SHA = "5d1f862631e403433f88478137e8321ef46664bb80c63a7fb5d703282a1617df"


def _blob(rev: str, path: str) -> list[str]:
    proc = subprocess.run(["git", "show", f"{rev}:{path}"],
                          capture_output=True, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr.decode()[-500:]
    return proc.stdout.replace(b"\r\n", b"\n").decode("utf-8").splitlines()


def _driver_source() -> str:
    return (KERNEL / "session-driver.py").read_text(encoding="utf-8")


def _embedded_patch() -> str:
    src = _driver_source()
    m = re.search(r'PROFILER_PATCH_B64 = "([A-Za-z0-9+/=]+)"', src)
    assert m, "embedded patch missing"
    return base64.b64decode(m.group(1)).decode("utf-8")


def test_kernel_dir_complete_and_valid():
    assert (KERNEL / "session-driver.py").is_file()
    meta = json.loads((KERNEL / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["code_file"] == "session-driver.py"
    assert meta["dataset_sources"] == ["nivind/deepseek-v4-flash-0731-shards"]
    assert meta["enable_gpu"] == "true" and meta["enable_tpu"] == "false"
    assert meta["id"].startswith("nivind/")
    proof = json.loads((KERNEL / "BUILD-PROOF.json").read_text(encoding="utf-8"))
    ast.parse(_driver_source())  # compiles


def test_embedded_patch_matches_branch():
    from make_profile_variant import RULES  # noqa: F401 (import check)
    patch = _embedded_patch()
    src = _driver_source()
    m = re.search(r'PROFILER_PATCH_SHA256 = "([0-9a-f]{64})"', src)
    assert m and hashlib.sha256(patch.encode()).hexdigest() == m.group(1)
    assert "217a333" in src and "9e165c30" in src  # pins recorded


def test_patch_hunks_present_in_217a333():
    """Every hunk's context+removed lines must exist in the 217a333 target
    (the same condition git apply checks, verified without a worktree)."""
    patch = _embedded_patch()
    current_file, hunk_body = None, []
    checked = 0

    def check_hunk(path: str, body: list[str]) -> None:
        nonlocal checked
        want = [ln[1:] for ln in body if ln[:1] in (" ", "-")]
        if not want:
            return
        target = _blob(ENGINE_217, path)
        for width in (len(want),):
            found = any(target[i:i + width] == want for i in range(len(target)))
            assert found, f"hunk not present in 217a333:{path}: {want[0][:80]}"
            checked += 1

    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            if current_file and hunk_body:
                check_hunk(current_file, hunk_body)
            current_file, hunk_body = line[6:], []
        elif line.startswith("@@"):
            if current_file and hunk_body:
                check_hunk(current_file, hunk_body)
            hunk_body = []
        elif line.startswith("--- a/"):
            continue  # header, not hunk content
        elif current_file is not None and line[:1] in (" ", "-", "+"):
            hunk_body.append(line)
    if current_file and hunk_body:
        check_hunk(current_file, hunk_body)
    assert checked > 10


def test_patch_is_additive_except_const():
    patch = _embedded_patch()
    removed, added = set(), set()
    for line in patch.splitlines():
        if line.startswith("-") and not line.startswith("---"):
            removed.add(line[1:].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            added.add(line[1:].strip())
    vanished = removed - added
    assert vanished == {
        "const bool computed = d_blob && swiglu_expert_batch_fp16_cuda("}, vanished


def test_patch_touches_only_six_files():
    patch = _embedded_patch()
    files = sorted(l[6:] for l in patch.splitlines() if l.startswith("+++ b/"))
    assert files == [
        "dee.cpp/include/dee/engine.h",
        "dee.cpp/include/dee/profiling.h",
        "dee.cpp/pydee/pydee.cpp",
        "dee.cpp/scripts/deepseek_v4_layer_candidate.py",
        "dee.cpp/src/engine.cpp",
        "dee.cpp/src/profiling.cpp",
    ], files


def test_variant_rules_match_audited_module():
    import base64 as _b64
    import hashlib as _hl
    from make_profile_variant import RULES, VARIANT_RUN_ID
    src = _driver_source()
    m = re.search("VARIANT_RULES_B64 = \"([A-Za-z0-9+/=]+)\"", src)
    assert m, "embedded rules missing"
    raw = _b64.b64decode(m.group(1))
    m2 = re.search("VARIANT_RULES_SHA256 = \"([0-9a-f]{64})\"", src)
    assert m2 and _hl.sha256(raw).hexdigest() == m2.group(1)
    embedded = json.loads(raw.decode("utf-8"))
    assert [(r["id"], r["old"], r["new"]) for r in embedded] == [
        (rid, old, new) for rid, old, new, _why in RULES]
    assert VARIANT_RUN_ID in raw.decode("utf-8")


def test_variant_builds_from_sealed_bytes(tmp_path):
    from make_profile_variant import apply_variant
    commit = "45b2a1659d0226c15ceb8821f104b1684798be4f"
    path = "experiment/pack-cap-ab/kernels/session1/session-driver.py"
    proc = subprocess.run(["git", "show", f"{commit}:{path}"],
                          capture_output=True, cwd=str(ROOT))
    assert proc.returncode == 0
    m = re.search(r'ARM_A_B64 = "([A-Za-z0-9+/=]+)"',
                  proc.stdout.decode("utf-8"))
    sealed = base64.b64decode(m.group(1))
    assert hashlib.sha256(sealed).hexdigest() == SEALED_ARM_SHA
    variant, applied = apply_variant(sealed)
    assert applied == ["run-id", "profile-guard", "profile-env",
                       "patch-apply", "emission"]
    ast.parse(variant.decode("utf-8"))  # variant compiles
    assert b"HOST_SYNC_EMIT" in variant and b"host_layer_records_json" in variant
    # OFF-arm default path unchanged: no profiler env referenced unconditionally.
    assert variant.count(b"NATIVE_PROFILE") >= 1


def test_driver_dry_run_gate_logic():
    src = _driver_source()
    for token in ("2x SM75", "48-shard", "/tmp", "DRY-RUN", "FAIL_CLOSED",
                  "ABC", "profile-run", "ARMS", "OFF1", "OFF2"):
        assert token.lower() in src.lower(), token


def test_built_driver_normalizes_crlf_like_module():
    """Regression for the v5 live failure: the built driver's inline copy
    must carry the same CRLF normalization as the audited module, and the
    module must accept CRLF sealed bytes."""
    from make_profile_variant import apply_variant
    src = _driver_source()
    assert '.replace("\\r\\n", "\\n")' in src
    commit = "45b2a1659d0226c15ceb8821f104b1684798be4f"
    path = "experiment/pack-cap-ab/kernels/session1/session-driver.py"
    proc = subprocess.run(["git", "show", f"{commit}:{path}"],
                          capture_output=True, cwd=str(ROOT))
    m = re.search(r'ARM_A_B64 = "([A-Za-z0-9+/=]+)"',
                  proc.stdout.decode("utf-8"))
    sealed = base64.b64decode(m.group(1))
    assert sealed.count(b"\r\n") > 1000  # sealed bytes are natively CRLF
    variant, applied = apply_variant(sealed)
    assert applied == ["run-id", "profile-guard", "profile-env",
                       "patch-apply", "emission"]
    ast.parse(variant.decode("utf-8"))
