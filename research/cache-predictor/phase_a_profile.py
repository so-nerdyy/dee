#!/usr/bin/env python3
"""Phase A: profile all candidate routed_experts.jsonl traces (READ-ONLY).

Never mutates trace files. Computes per-trace statistics, hash
provenance, and journal-chain integrity so the evidence set for the
cache-policy tournament can be chosen defensibly.

Usage:  python research/cache-predictor/phase_a_profile.py
Output: research/cache-predictor/results/phase_a_trace_inventory.json
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Traces live in the main checkout (sealed evidence is not in this worktree).
# Override with DEE_TRACE_ROOT env var if the checkout moves.
import os
TRACE_ROOT = Path(os.environ.get(
    "DEE_TRACE_ROOT",
    str(REPO.parent / "dynamic_expert_eviction" / "dee.cpp" / "tmp"),
))
OUT = Path(__file__).resolve().parent / "results"

# Newest first; every known routed_experts.jsonl under dee.cpp/tmp.
CANDIDATE_DIRS = [
    "v65-terminal-fetch-20260903T0202Z",
    "v65-terminal-fetch-20260903T020220Z",
    "v64-terminal-fetch-20260903T0120Z",
    "v63-terminal-fetch-20260903T0048Z",
    "v62-live-20260902T003709Z",
    "v61-live-20260902T000809Z",
    "v60-live-20260901T040935Z",
    "v59-live-20260901T034302Z",
    "v58-live-20260901T025855Z",
    "v58-live-20260831T052427Z",
    "v57-live-20260831T044713Z",
    "v56-live-20260831T041432Z",
    "v56-live-20260831T035018Z",
    "v55-live-20260831T030956Z",
    "v54-live-20260831T025719Z",
    "v52-live-20260830T133749",
    "v51-evidence-20260830T132128Z",
    "v50-evidence-20260829T195940",
    "kt-input-capture-v3-output",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sidecars(d: Path) -> dict:
    """Pull provenance fields from result.json / integrity.json if present."""
    out: dict = {}
    for name in ("result.json", "integrity.json", "dee4-metadata.json"):
        p = d / name
        if not p.is_file():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            out[name] = {"error": f"unparseable: {exc}"}
            continue
        info: dict = {}
        rj = doc.get("route_journal") if isinstance(doc, dict) else None
        if isinstance(rj, dict):
            info["route_journal"] = {
                k: rj.get(k)
                for k in (
                    "schema_version",
                    "n_layers",
                    "topk",
                    "record_count",
                    "completed_forwards",
                    "route_journal_complete",
                )
                if k in rj
            }
        acc = doc.get("acceptance") if isinstance(doc, dict) else None
        if isinstance(acc, dict):
            info["acceptance_keys"] = sorted(acc.keys())[:12]
        res = doc.get("result") if isinstance(doc, dict) else None
        if isinstance(res, str):
            info["result"] = res
        model = doc.get("model") if isinstance(doc, dict) else None
        if isinstance(model, dict):
            info["model_keys"] = sorted(model.keys())[:12]
        out[name] = info or {"keys": sorted(doc.keys())[:20] if isinstance(doc, dict) else "?"}
    return out


def profile_trace(path: Path) -> dict:
    n_records = 0
    malformed = 0
    layers: set = set()
    forwards: set = set()
    unique_experts: set = set()
    selections = 0
    token_rows_hist: Counter = Counter()
    phases: Counter = Counter()
    run_ids: set = set()
    schemas: set = set()
    topks: set = set()
    chain_links_ok = 0
    chain_links_bad = 0
    prev_chain: str | None = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            ids = rec.get("expert_ids_rank_order")
            if not isinstance(ids, list) or not ids:
                malformed += 1
                continue
            n_records += 1
            schemas.add(rec.get("schema_version"))
            topks.add(rec.get("topk"))
            layers.add(rec.get("layer"))
            forwards.add(rec.get("forward_step"))
            run_ids.add(rec.get("run_id"))
            phases[str(rec.get("phase", "?"))] += 1
            token_rows_hist[str(rec.get("token_rows"))] += 1
            for row in ids:
                selections += len(row)
                unique_experts.update(row)
            c = rec.get("chain_sha256")
            p = rec.get("previous_chain_sha256")
            if isinstance(c, str) and isinstance(p, str):
                if prev_chain is None or p == prev_chain:
                    chain_links_ok += 1
                else:
                    chain_links_bad += 1
            if isinstance(c, str):
                prev_chain = c

    return {
        "file": str(path.relative_to(TRACE_ROOT.parent.parent)),
        "bytes": path.stat().st_size,
        "sha256_first_record_chain": prev_chain,
        "records": n_records,
        "malformed_lines": malformed,
        "forwards": len(forwards),
        "layers_seen": len(layers),
        "unique_experts": len(unique_experts),
        "total_selections": selections,
        "phases_records": dict(phases),
        "token_rows_hist": dict(token_rows_hist),
        "run_ids": sorted(str(r) for r in run_ids),
        "schema_versions": sorted(str(s) for s in schemas),
        "topk_values": sorted(str(t) for t in topks),
        "chain_links_ok": chain_links_ok,
        "chain_links_bad": chain_links_bad,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name in CANDIDATE_DIRS:
        d = TRACE_ROOT / name
        trace = d / "routed_experts.jsonl"
        if not trace.is_file():
            inventory.append({"dir": name, "present": False})
            continue
        entry: dict = {"dir": name, "present": True}
        entry.update(profile_trace(trace))
        entry["file_sha256"] = sha256_file(trace)
        entry["sidecars"] = read_sidecars(d)
        inventory.append(entry)

    doc = {
        "purpose": "Phase A trace discovery/validation (read-only)",
        "scan_root": str(TRACE_ROOT),
        "traces": inventory,
    }
    out_path = OUT / "phase_a_trace_inventory.json"
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    # Console summary table
    print(f"{'dir':44s} {'recs':>5s} {'fwd':>4s} {'lay':>4s} {'unq':>4s} "
          f"{'sel':>7s} {'bad':>4s} {'link_bad':>8s} rows")
    for e in inventory:
        if not e.get("present"):
            print(f"{e['dir']:44s} MISSING")
            continue
        print(f"{e['dir']:44s} {e['records']:5d} {e['forwards']:4d} "
              f"{e['layers_seen']:4d} {e['unique_experts']:4d} "
              f"{e['total_selections']:7d} {e['malformed_lines']:4d} "
              f"{e['chain_links_bad']:8d} {e['token_rows_hist']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
