#!/usr/bin/env python3
"""Phase 2 skeleton — seven-variant cap-32 lifecycle matrix.

Gated strictly on Phase 1 SEAL.json result == PASS. This skeleton enforces
the gate, lists the seven cap-32 variants that will run on Kaggle after
Phase 1 seals, and does NOT invoke the engine until a launch is explicitly
authorized. Lives in a separate file so Phase 1 and Phase 2 are not mixed.

Variant names mirror the M3 v17 lineage. Their naming is the contract —
the orchestrator (`run_milestone4_capacity_sweep.py`) does not yet
recognize every one of these run_ids; that extension is Phase-2 work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Seven variants on cap-32; mirrors the M3 v17 lifecycle line-up.
SEVEN_VARIANTS = (
    "capacity-32-control",
    "capacity-32-profiled",
    "capacity-32-cold-primary",
    "capacity-32-cache-capacity-4-override",
    "capacity-32-cache-disabled",
    "capacity-32-long-prompt",
    "capacity-32-reference-present",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seal-dir",
        required=True,
        type=Path,
        help="Phase 1 seal directory containing SEAL.json",
    )
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--launch-engine",
        action="store_true",
        help=(
            "If set, the skeleton will attempt to fire the orchestrator for "
            "all seven variants once the seal gate passes. Without this flag, "
            "the script only validates the contract."
        ),
    )
    return p.parse_args()


def gate(seal_dir: Path) -> dict[str, object]:
    seal_path = seal_dir / "SEAL.json"
    if not seal_path.is_file():
        raise SystemExit(f"Phase 2 abort: SEAL.json not found at {seal_path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("result") != "PASS":
        reason = seal.get("terminal_reason", "<unspecified>")
        raise SystemExit(f"Phase 2 abort: Phase 1 SEAL is not PASS ({reason})")
    return seal


def main() -> int:
    args = parse_args()
    seal = gate(args.seal_dir)
    payload = {
        "result": "SKELETON_VALIDATED",
        "phase1_seal_run_id": seal.get("identity", {}).get("run_id"),
        "phase1_seal_terminal_timestamp": seal.get("terminal_timestamp_utc"),
        "phase1_pass": True,
        "would_launch": list(SEVEN_VARIANTS),
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "would_invoke_engine": bool(args.launch_engine),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
