#!/usr/bin/env python3
"""Analyze fetched live arms with the shared evidence pipeline (no CUDA).

Reads raw fetched arm dirs (native-generate-result.json +
host-sync-records.json + host-profile-rows-l*.json), converts to the
canonical profile-run/ layout (flattened rows, token attribution), then runs
the IDENTICAL functions the live pack/run_evidence path uses: pair
validation, merge, closure, ABC ingestion, ranking, next-A/B.

Usage:
    python3 analyze_live.py --fetch DIR --out OUTDIR --prompt-hash H \\
        --abc fetched/abc-mechanics.json

No timings are invented: every number is parsed from fetched files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evidence import (  # noqa: E402
    attribute_tokens,
    build_next_ab,
    ingest_abc,
    load_profile_run,
    merge_records,
    rank_from_evidence,
    records_to_csv,
    rollup_per_token,
    validate_pair,
)
from host_profiler import compute_closure  # noqa: E402


def convert_arm(arm_dir: Path, out_dir: Path, prompt_hash: str,
                profiling: bool) -> dict:
    res = json.loads((arm_dir / "native-generate-result.json").read_text(encoding="utf-8"))
    hs_path = arm_dir / "host-sync-records.json"
    cpp_records = []
    if hs_path.is_file():
        hs_data = json.loads(hs_path.read_text(encoding="utf-8"))
        if isinstance(hs_data, dict):
            for gpu_recs in hs_data.values():
                if isinstance(gpu_recs, dict) and isinstance(gpu_recs.get("records"), list):
                    cpp_records.extend(gpu_recs["records"])
                elif isinstance(gpu_recs, list):
                    cpp_records.extend(gpu_recs)
    py_rows = []
    for path in sorted(arm_dir.glob("host-profile-rows-l*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            py_rows.extend(data["records"])
        elif isinstance(data, list):
            py_rows.extend(data)
    tokens = res.get("decode_tokens") or len(res.get("generated_token_ids", [])) or None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps({
        "status": "ok" if res.get("classification") == "ACCEPT_CORRECTNESS" else "arm-invalid",
        "tokens": tokens,
        "metrics": {"decode_wall_s": res.get("decode_wall_s"), "tokens": tokens},
        "generated_ids": res.get("generated_token_ids"),
        "decoded_text": res.get("decoded_text"),
        "source_sha": res.get("commit"),
        "prompt_hash": prompt_hash,
        "config": {"cache_dtype": "fp4", "expert_store": "dee4_trace",
                   "lanes": 3, "qdepth": 6, "cap_gib_total": 17.0,
                   "profile_stages": profiling,
                   "DEE_HOST_PROFILE": profiling},
        "hardware": {"count": 2, "names": ["Tesla T4", "Tesla T4"]},
    }, indent=2), encoding="utf-8")
    (out_dir / "host-profile.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in py_rows), encoding="utf-8")
    (out_dir / "stage-profile.json").write_text(json.dumps(
        {"host_layer_records": {"records": cpp_records}}, indent=2), encoding="utf-8")
    (out_dir / "correctness.json").write_text(json.dumps(
        {"classification": res.get("classification"),
         "gates": {"accept_correctness":
                   res.get("classification") == "ACCEPT_CORRECTNESS"}}, indent=2),
        encoding="utf-8")
    return {"cpp": len(cpp_records), "py": len(py_rows),
            "classification": res.get("classification"),
            "decode_wall_s": res.get("decode_wall_s")}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-hash", required=True)
    parser.add_argument("--abc", type=Path, required=True)
    args = parser.parse_args(argv)

    converted = {}
    for arm, key, profiling in (("profile-off1", "off", False),
                                ("profile-on", "on", True),
                                ("profile-off2", "off2", False)):
        converted[key] = convert_arm(args.fetch / arm, args.out / f"profile-run-{key}",
                                     args.prompt_hash, profiling)
    print(json.dumps({"converted": converted}, indent=1))

    off = load_profile_run(args.out / "profile-run-off")
    on = load_profile_run(args.out / "profile-run-on")
    pair = validate_pair(off, on)
    print(json.dumps({"pair": pair}, indent=1))

    cpp = on["stage"]["host_layer_records"]["records"]
    attr_notes = attribute_tokens(cpp, on["py_rows"])
    records, merge_notes = merge_records(cpp, on["py_rows"])
    # Closure reconciles DECODE tokens against decode_wall: token 0 is the
    # multi-row prefill call (its native wall dwarfs decode rows) and is
    # reported separately, never mixed into decode closure.
    decode_records = [r for r in records if r["token"] != 0]
    prefill_rows = [r for r in records if r["token"] == 0]
    prefill_native = round(sum(r.get("native_call_wall_ms") or 0.0
                               for r in prefill_rows), 3)
    write_json(args.out / "host-sync-attribution.json",
               {"records": records,
                "merge_notes": merge_notes,
                "attribution_notes": attr_notes,
                "prefill_note": {
                    "rows": len(prefill_rows),
                    "native_call_wall_ms": prefill_native,
                    "excluded_from_decode_closure": True}})
    (args.out / "per-layer.csv").write_text(records_to_csv(records), encoding="utf-8")
    write_json(args.out / "per-token.json",
               {"tokens": rollup_per_token(records)})
    wall_s = pair["on_wall_s"]
    closure = compute_closure(decode_records, decode_wall_ms=wall_s * 1000.0)
    write_json(args.out / "closure.json", closure)
    mechanics = ingest_abc(json.loads(args.abc.read_text(encoding="utf-8")))
    write_json(args.out / "abc-mechanics.json", mechanics)
    ranking = rank_from_evidence(closure, mechanics)
    write_json(args.out / "candidate-ranking.json", ranking)
    next_ab = build_next_ab(ranking)
    write_json(args.out / "next-ab.json", next_ab)
    bundle = {"classification": "LIVE_ANALYSIS_OK",
              "pair": pair, "records": len(records),
              "closure_status": ranking.get("closure_status"),
              "ranking": {k: v.get("rank") for k, v in
                          ranking.get("candidates", {}).items()},
              "next": next_ab.get("classification")}
    write_json(args.out / "bundle.json", bundle)
    print(json.dumps(bundle, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
