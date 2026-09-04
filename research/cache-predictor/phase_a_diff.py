#!/usr/bin/env python3
"""Check whether candidate DSV4 traces differ in routing CONTENT (read-only)."""
from __future__ import annotations

import json
import os
from pathlib import Path

TRACE_ROOT = Path(os.environ.get(
    "DEE_TRACE_ROOT",
    str(Path(__file__).resolve().parents[2].parent
        / "dynamic_expert_eviction" / "dee.cpp" / "tmp"),
))

DIRS = [
    "v65-terminal-fetch-20260903T0202Z",
    "v64-terminal-fetch-20260903T0120Z",
    "v63-terminal-fetch-20260903T0048Z",
    "v62-live-20260902T003709Z",
    "v61-live-20260902T000809Z",
    "v60-live-20260901T040935Z",
    "v58-live-20260831T052427Z",
    "v57-live-20260831T044713Z",
    "v56-live-20260831T041432Z",
    "v55-live-20260831T030956Z",
    "v54-live-20260831T025719Z",
    "v52-live-20260830T133749",
    "v51-evidence-20260830T132128Z",
    "v50-evidence-20260829T195940",
    "kt-input-capture-v3-output",
]

def content(path: Path):
    routes = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = json.loads(line)
            routes.append((rec.get("forward_step"), rec.get("layer"),
                           rec.get("phase"), rec.get("token_rows"),
                           json.dumps(rec.get("expert_ids_rank_order"))))
    return routes

base = content(TRACE_ROOT / DIRS[0] / "routed_experts.jsonl")
for d in DIRS[1:]:
    c = content(TRACE_ROOT / d / "routed_experts.jsonl")
    same = c == base
    diffs = sum(1 for a, b in zip(base, c) if a != b) if len(c) == len(base) else -1
    print(f"{d:44s} same_content={same} diff_records={diffs}")

# Also inspect metadata of the newest run
meta = json.loads((TRACE_ROOT / DIRS[0] / "dee4-metadata.json").read_text())
print("\ndee4-metadata keys:", sorted(meta.keys())[:30])
