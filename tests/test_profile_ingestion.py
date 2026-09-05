#!/usr/bin/env python3
"""Unit tests for evidence.py ingestion (Phase B/D/E/F logic).

Covers: C++/Python normalization, merge precedence without duplicate
accounting, CSV/token rollups, pair validation, ABC ingestion, ranking from
parsed evidence, and single-candidate next-A/B derivation. No CUDA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parent.parent / "dee.cpp" / "experiments" / "route_pipeline"
sys.path.insert(0, str(EXP))

from evidence import (  # noqa: E402
    EvidenceError,
    PairError,
    build_next_ab,
    ingest_abc,
    load_profile_run,
    merge_records,
    rank_from_evidence,
    records_to_csv,
    rollup_per_token,
    validate_pair,
)


def _cpp(token=0, layer=0, device=0, native=10.0, sync=2.0):
    return {"token": token, "layer": layer, "device": device,
            "spans": {
                "native_call_wall": {"ms": native, "provenance": "HOST_WALL"},
                "source_lookup_wait": {"ms": 1.0, "provenance": "HOST_WALL"},
                "native_output_sync": {"ms": sync, "provenance": "HOST_WALL"}},
            "counters": {"ids_bytes": 24}}


def _py(token=0, layer=0, device=0, native=10.0):
    return {"token": token, "layer": layer, "device": device,
            "route_d2h_host_wait_ms": 0.5,
            "native_call_wall_ms": native,  # duplicate of C++: must be dropped
            "combine_ms": 0.3,
            "shared_host_wall_ms": 1.0,
            "ids_bytes": 24}


def test_merge_prefers_cpp_native_and_drops_duplicate():
    records, notes = merge_records([_cpp()], [_py()])
    assert len(records) == 1
    rec = records[0]
    assert rec["native_call_wall_ms"] == 10.0
    assert rec["route_d2h_host_wait_ms"] == 0.5
    assert rec["combine_ms"] == 0.3
    assert rec["shared_expert_ms"] == 1.0
    assert any("dropped" in note for note in notes)


def test_merge_keeps_python_only_records():
    records, notes = merge_records([], [_py(layer=3)])
    assert len(records) == 1
    assert records[0]["native_call_wall_ms"] == 10.0
    assert any("python-only" in note for note in notes)


def test_merge_rejects_duplicates_and_unknown_spans():
    with pytest.raises(EvidenceError):
        merge_records([_cpp(), _cpp()], [])
    bad = _cpp()
    bad["spans"]["mystery"] = {"ms": 1.0}
    with pytest.raises(EvidenceError):
        merge_records([bad], [])
    with pytest.raises(EvidenceError):
        merge_records([{"token": 0}], [])


def test_csv_and_token_rollup():
    records, _ = merge_records([_cpp(0, 0, 0), _cpp(0, 1, 0)], [_py(0, 0, 0)])
    csv = records_to_csv(records)
    lines = csv.strip().splitlines()
    assert lines[0].startswith("token,layer,device,")
    assert len(lines) == 3
    rolled = rollup_per_token(records)
    assert len(rolled) == 1 and rolled[0]["layers"] == 2
    assert rolled[0]["by_field_ms"]["native_call_wall_ms"] == 20.0
    assert "decode_ms" not in rolled[0]["by_field_ms"]  # nested: never rolled up


def _result(wall=1.0, **over):
    base = {"status": "ok", "metrics": {"decode_wall_s": wall, "tokens": 2},
            "generated_ids": [[1]], "decoded_text": "t", "source_sha": "s",
            "prompt_hash": "p", "tokens": 2,
            "config": {"model": "m", "profile_stages": False},
            "hardware": {"count": 2}}
    base.update(over)
    return base


def _correct():
    return {"gates": {"a": True, "b": True}}


def test_validate_pair_ok_and_perturbation():
    off = {"result": _result(wall=10.0), "correctness": _correct()}
    on = {"result": _result(wall=11.0), "correctness": _correct()}
    out = validate_pair(off, on)
    assert out["pair"] == "VALID"
    assert out["perturbation_s"] == 1.0
    assert out["perturbation_fraction"] == 0.1


def test_validate_pair_rejects_mismatches():
    good = {"result": _result(), "correctness": _correct()}
    bad_ids = {"result": _result(generated_ids=[[9]]), "correctness": _correct()}
    with pytest.raises(PairError):
        validate_pair(good, bad_ids)
    bad_cfg = {"result": _result(), "correctness": _correct()}
    bad_cfg["result"]["config"]["topk"] = 8
    with pytest.raises(PairError):
        validate_pair(good, bad_cfg)
    bad_gate = {"result": _result(), "correctness": {"gates": {"a": False}}}
    with pytest.raises(PairError):
        validate_pair(good, bad_gate)
    no_wall = {"result": _result(), "correctness": _correct()}
    del no_wall["result"]["metrics"]
    with pytest.raises(PairError):
        validate_pair(good, no_wall)


def test_ingest_abc_parses_fields_not_stdout():
    payload = {"gpu": "t4", "status": "ok",
               "A_hash_staging": {"status": "ok",
                                  "serial_ms": {"p50": 12.0},
                                  "batched_ms": {"p50": 7.0}},
               "B_contention": {"status": "ok", "efficiency_c": 0.9},
               "C_barrier": {"status": "failing"}}
    out = ingest_abc(payload)
    assert out["cases"]["A_hash_staging"]["metrics"]["hidden_ms"] == 5.0
    assert out["cases"]["B_contention"]["metrics"]["efficiency_c"] == 0.9
    assert out["cases"]["C_barrier"]["status"] == "UNKNOWN"
    with pytest.raises(EvidenceError):
        ingest_abc("not-a-dict")


def test_ranking_from_parsed_evidence():
    closure = {"wall_ms": 100.0, "closure_fraction": 0.95, "closure_ok": True,
               "by_field_ms": {"native_call_wall_ms": 80.0,
                               "route_d2h_host_wait_ms": 1.0,
                               "shared_expert_ms": 1.0,
                               "combine_ms": 1.0},
               "by_field_nested_ms": {"native_output_sync_wait_ms": 8.0,
                                      "decode_ms": 1.0,
                                      "expert_compute_ms": 1.0}}
    mechanics = {"cases": {
        "A_hash_staging": {"status": "ok", "metrics": {"hidden_ms": 0.2}},
        "B_contention": {"status": "ok", "metrics": {"efficiency_c": 0.9}},
        "C_barrier": {"status": "ok", "metrics": {}}}}
    ranking = rank_from_evidence(closure, mechanics)
    assert ranking["candidates"]["EVENT_HANDOFF"]["rank"] == "PROMOTE"
    assert ranking["candidates"]["SHARED_OVERLAP"]["rank"] == "HOLD"
    assert ranking["candidates"]["HASH_EARLY_STAGING"]["rank"] == "HOLD"
    assert ranking["candidates"]["C1_C2"]["rank"] == "HOLD"
    assert ranking["closure_status"] == "OK"


def test_next_ab_single_and_empty():
    ranking = {"closure_status": "OK", "candidates": {
        "EVENT_HANDOFF": {"rank": "PROMOTE", "reason": "x"},
        "SHARED_OVERLAP": {"rank": "PROMOTE", "reason": "y"},
        "ROUTE_D2H_NARROW": {"rank": "HOLD", "reason": "z"},
        "HASH_EARLY_STAGING": {"rank": "HOLD", "reason": "w"},
        "C1_C2": {"rank": "HOLD", "reason": "v"}}}
    nxt = build_next_ab(ranking)
    assert nxt["candidate"] == "EVENT_HANDOFF"  # priority order, deterministic
    assert nxt["runner_up"] == ["SHARED_OVERLAP"]
    assert "ONLY" in nxt["ab_shape"]["candidate"]
    empty = {"closure_status": "PROFILE_INCOMPLETE", "candidates": {
        name: {"rank": "HOLD", "reason": "r"} for name in
        ("EVENT_HANDOFF", "SHARED_OVERLAP", "ROUTE_D2H_NARROW",
         "HASH_EARLY_STAGING", "C1_C2")}}
    nxt2 = build_next_ab(empty)
    assert nxt2["classification"] == "NO_OVERLAP_OPTIMIZATION_JUSTIFIED"
    assert "PROFILE_INCOMPLETE" in nxt2["reason"]


def test_load_profile_run_rejects_missing(tmp_path):
    with pytest.raises(EvidenceError):
        load_profile_run(tmp_path)
