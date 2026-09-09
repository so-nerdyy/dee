#!/usr/bin/env python3
"""Build the host-sync profile arm variant from sealed arm-A bytes.

Applies a minimal set of EXACT-match replacements (each anchor must occur
exactly once or the build fails). The variant differs from sealed arm A
only in: run ID, profile-guard allowance (env-gated), and an env-gated
host-record emission block. No math, routing, cache, or sync change.

Usage:
    python3 make_profile_variant.py --sealed armA_sealed.py --out variant.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

VARIANT_RUN_ID = "host-sync-profile-17g-20260905-v1"

EMISSION_BLOCK = '''
        if os.environ.get("HOST_SYNC_EMIT", "0") == "1":
            try:
                _hsr = {"cuda0": json.loads(eng0.host_layer_records_json())}
                if not SINGLE_GPU:
                    _hsr["cuda1"] = json.loads(eng1.host_layer_records_json())
                (WORK / "host-sync-records.json").write_text(
                    json.dumps(_hsr, indent=2))
                _hpy = 0
                for _li in range(int(cfg.n_layers)):
                    try:
                        _fn = model.layer(int(_li)).ffn_fn
                    except Exception:
                        continue
                    _dump = getattr(_fn, "dump_host_profile", None)
                    if _dump is None:
                        continue
                    _dump(str(WORK / f"host-profile-rows-l{_li:02d}.json"))
                    _hpy += 1
                result["host_sync_emission"] = {
                    "engines": True, "py_layers": _hpy}
            except Exception as _exc:  # never fail the run over emission
                log(f"host-sync emission failed: {_exc}")
                result["host_sync_emission"] = {"error": repr(_exc)}
'''

# (rule id, exact old, exact new, why). All olds must occur exactly once.
RULES = [
    ("run-id",
     '"run_id": "host-reuse-217-candidate-q6l3-20260904-v2"',
     f'"run_id": "{VARIANT_RUN_ID}"',
     "evidence separation; IDs/text contract untouched"),
    ("profile-guard",
     "or EXPERT_STORE_BACKEND != \"dee4_trace\" or PROFILE_STAGES or DIAGNOSTICS",
     "or EXPERT_STORE_BACKEND != \"dee4_trace\" or (PROFILE_STAGES "
     "and os.environ.get(\"HOST_SYNC_PROFILE_ARM\", \"0\") != \"1\") "
     "or DIAGNOSTICS",
     "allow profiler flags only when the profile arm env is set"),
    ("profile-env",
     "    if not os.environ.get(\"NATIVE_PROFILE\"):\n"
     "        PROFILE_STAGES = bool(cfg.get(\"profile_stages\", PROFILE_STAGES))",
     "    if not os.environ.get(\"NATIVE_PROFILE\"):\n"
     "        PROFILE_STAGES = bool(cfg.get(\"profile_stages\", PROFILE_STAGES))\n"
     "    if os.environ.get(\"HOST_SYNC_PROFILE_ARM\", \"0\") == \"1\":\n"
     "        PROFILE_STAGES = True",
     "enable profiler flags without any NATIVE_* override (the sealed "
     "preflight forbids NATIVE_* env)"),
    ("patch-apply",
     "    log(f\"pinned commit {head}\")",
     "    log(f\"pinned commit {head}\")\n"
     "    _hspp = os.environ.get(\"HOST_SYNC_PATCH_PATH\", \"\")\n"
     "    _hsha = os.environ.get(\"HOST_SYNC_PATCH_SHA256\", \"\")\n"
     "    if _hspp:\n"
     "        _praw = Path(_hspp).read_bytes()\n"
     "        if hashlib.sha256(_praw).hexdigest() != _hsha:\n"
     "            raise RuntimeError(\"profiler patch sha mismatch\")\n"
     "        _chk = subprocess.run([\"git\", \"-C\", str(ROOT), \"apply\", \"--check\",\n"
     "                               str(_hspp)], capture_output=True, text=True)\n"
     "        if _chk.returncode != 0:\n"
     "            raise RuntimeError(\"profiler patch does not apply: \"\n"
     "                               + _chk.stderr[-2000:])\n"
     "        run([\"git\", \"-C\", str(ROOT), \"apply\", str(_hspp)])\n"
     "        _stat = subprocess.run([\"git\", \"-C\", str(ROOT), \"diff\", \"--name-only\"],\n"
     "                               capture_output=True, text=True).stdout\n"
     "        _untracked = subprocess.run([\"git\", \"-C\", str(ROOT), \"ls-files\",\n"
     "                                     \"--others\", \"--exclude-standard\"],\n"
     "                               capture_output=True, text=True).stdout\n"
     "        _files = sorted(_stat.split())\n"
     "        _new = sorted(_untracked.split())\n"
     "        _expect = os.environ.get(\"HOST_SYNC_PATCH_FILES\", \"\").split(\",\")\n"
     "        _expect = sorted(e for e in _expect if e) or sorted([\"dee.cpp/include/dee/engine.h\",\n"
     "                             \"dee.cpp/include/dee/profiling.h\",\n"
     "                             \"dee.cpp/pydee/pydee.cpp\",\n"
     "                             \"dee.cpp/scripts/deepseek_v4_layer_candidate.py\",\n"
     "                             \"dee.cpp/src/engine.cpp\",\n"
     "                             \"dee.cpp/src/profiling.cpp\"])\n"
     "        _expect_tracked = sorted(e for e in _expect\n"
     "                                 if e != \"dee.cpp/tools/fill_replay.cpp\")\n"
     "        _expect_new = sorted(e for e in _expect\n"
     "                             if e == \"dee.cpp/tools/fill_replay.cpp\")\n"
     "        if _files != _expect_tracked or _new != _expect_new:\n"
     "            raise RuntimeError(\"profiler patch touched unexpected files: \"\n"
     "                               + repr((_files, _new)))\n"
     "        log(\"profiler patch applied (see source-identity-proof)\")\n"
     "        (WORK / \"source-identity-proof.txt\").write_text(\n"
     "            f\"base=217a333 patch_sha={_hsha}\\n\" + \"\\n\".join(_files) + \"\\n\")",
     "apply the sha-pinned profiler patch to the verified clone (ON arm "
     "only; OFF arms have no HOST_SYNC_PATCH_PATH and stay pristine)"),
    ("emission",
     '        result["model_cuda_stage_profile"] = model.cuda_stage_profile()',
     '        result["model_cuda_stage_profile"] = model.cuda_stage_profile()\n'
     + EMISSION_BLOCK.rstrip("\n"),
     "env-gated host-record emission beside the existing stage dump"),
]


def apply_variant(arm_bytes: bytes) -> tuple[bytes, list[str]]:
    # Sealed bytes carry CRLF; normalize to LF (Linux execution target).
    # Anchors are single- or multi-line exact matches after normalization.
    text = arm_bytes.decode("utf-8").replace("\r\n", "\n")
    applied = []
    for rule_id, old, new, _why in RULES:
        count = text.count(old)
        if count != 1:
            raise ValueError(
                f"rule {rule_id}: anchor occurs {count}x (need exactly 1)")
        text = text.replace(old, new)
        applied.append(rule_id)
    return text.encode("utf-8"), applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    variant, applied = apply_variant(args.sealed.read_bytes())
    args.out.write_bytes(variant)
    print(f"variant: {hashlib.sha256(variant).hexdigest()}")
    print(f"rules applied: {applied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
