#!/usr/bin/env python3
"""Tests for the pack-cap forensics tooling and the pread-rider fix.

Self-contained unit tests (synthetic journal, synthetic LRU streams) always
run. Evidence-grounded tests run against results/live when present and skip
otherwise. No sealed evidence is modified.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXPROOT = HERE.parent
sys.path.insert(0, str(EXPROOT / "tools"))
import forensics  # noqa: E402

LIVE = EXPROOT / "results" / "live"
EVIDENCE_PRESENT = (LIVE / "s1" / "session1-A" / "result.json").is_file()


# ---------------------------------------------------------------------------
# Synthetic replay correctness
# ---------------------------------------------------------------------------

def _journal_record(fs, layer, device, phase, rows):
    return {"forward_step": fs, "layer": layer, "device": device,
            "phase": phase, "expert_ids_rank_order": rows,
            "start_pos": fs, "topk": 6, "token_rows": len(rows)}


def _write_journal(path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_larger_lru_has_no_new_misses_synthetic(tmp_path):
    # 6 distinct records; alternate touch order so budget 4 gets hits that
    # budget 2 cannot (a sequential re-scan defeats LRU at any budget).
    recs = [_journal_record(0, 0, "cuda:0", "prefill", [[0, 1, 2, 3, 4, 5]])]
    for fs in range(1, 6):
        keys = list(range(6)) if fs % 2 else list(range(5, -1, -1))
        recs.append(_journal_record(fs, 0, "cuda:0", "decode", [keys]))
    jp = tmp_path / "journal.jsonl"
    _write_journal(jp, recs)
    phase, stream = forensics.load_stream(jp)
    pd = {fs: phase[("cuda:0", fs)] for fs in stream["cuda:0"]}
    last_a: dict = {}
    am, ah, ae, adm, _ = forensics.replay_device(stream["cuda:0"], pd, 2, last_a)
    last_b: dict = {}
    bm, bh, be, bdm, _ = forensics.replay_device(stream["cuda:0"], pd, 4, last_b)
    keys_a = {(m["forward_step"], m["layer"], m["expert"]) for m in am}
    keys_b = {(m["forward_step"], m["layer"], m["expert"]) for m in bm}
    assert keys_b <= keys_a, "larger LRU must have a subset of misses"
    assert len(keys_a - keys_b) > 0
    # eliminated entries carry recency + bytes metadata
    eliminated = [m for m in am if (m["forward_step"], m["layer"], m["expert"]) in keys_a - keys_b]
    assert all(m["storage_bytes"] == forensics.RECORD_BYTES for m in eliminated)
    assert all(m["compulsory"] or m["distinct_since_prev_use"] is not None for m in eliminated)


def test_recency_ruler_distances(tmp_path):
    # record 0 used at step 0, again at step 5 -> finite distance; record 9 first use at step 5.
    recs = [_journal_record(0, 0, "cuda:0", "prefill", [[0, 1, 2, 3, 4, 5]])]
    recs.append(_journal_record(5, 0, "cuda:0", "decode", [[0, 9, 9, 9, 9, 9]]))
    jp = tmp_path / "j.jsonl"
    _write_journal(jp, recs)
    phase, stream = forensics.load_stream(jp)
    pd = {fs: phase[("cuda:0", fs)] for fs in stream["cuda:0"]}
    last: dict = {}
    misses, *_ = forensics.replay_device(stream["cuda:0"], pd, 2, last)
    by_expert = {m["expert"]: m for m in misses if m["phase"] == "decode"}
    assert by_expert[0]["compulsory"] is False
    assert by_expert[0]["distinct_since_prev_use"] is not None
    assert by_expert[9]["compulsory"] is True


def test_replay_device_matches_validated_semantics():
    # evict-on-insert at budget 2: resident set == last 2 distinct keys
    stream = defaultdict(lambda: defaultdict(list))
    stream["cuda:0"][0] = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 0), (0, 3)]
    phase = defaultdict(lambda: {0: "decode"})
    last: dict = {}
    misses, hits, evict, _, _ = forensics.replay_device(stream["cuda:0"], phase[None], 2, last)
    # 0,1,2,3 miss; re-request of 0 misses again (evicted), re-request of 3 hits
    assert (len(misses), hits, evict) == (5, 1, 3)


# ---------------------------------------------------------------------------
# Evidence-grounded tests (skip without the live evidence)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not EVIDENCE_PRESENT, reason="live arm evidence absent")
def test_ledger_matches_measured_delta():
    ledger = forensics.build_ledger(LIVE)
    c = ledger["counts"]
    assert c["A_misses_total_replay"] in (2480, 2481)
    assert c["B_misses_total_replay"] == 2429
    assert c["eliminated_replay"] in (51, 52)
    assert c["new_misses_in_B_replay"] == 0
    elim = ledger["eliminated"]
    assert all(m["compulsory"] is False for m in elim), \
        "every eliminated miss must be a capacity re-read"
    total_bytes = sum(m["storage_bytes"] for m in elim)
    assert 650e6 < total_bytes < 720e6  # ~681.8 MB replay / 695 MB live


@pytest.mark.skipif(not EVIDENCE_PRESENT, reason="live arm evidence absent")
def test_criticality_has_no_position_robust_cap_effect():
    ledger = forensics.build_ledger(LIVE)
    metrics = forensics.arm_metrics(LIVE)
    crit = forensics.criticality(ledger, metrics)
    total = sum(crit["eliminated_misses_by_classification"].values())
    assert total == ledger["counts"]["eliminated_replay"]
    counts = crit["eliminated_misses_by_classification"]
    # the sign-flip class is the LARGEST single class; no same-sign class
    # dominates, and every same-sign delta is sub-noise (<= 0.26 s)
    flips = counts.get("NOT_DEMAND_BLOCKING_AT_TOKEN_GRANULARITY", 0)
    assert flips == max(counts.values())
    assert flips > counts.get("CRITICAL_WAIT_CANDIDATE", 0)
    assert flips > counts.get("CANDIDATE_REGRESSION", 0)
    for e in crit["per_token"]:
        if e["token_level_criticality"] in ("CRITICAL_WAIT_CANDIDATE", "CANDIDATE_REGRESSION"):
            assert abs(e["delta_B_minus_A_session1_s"]) <= 0.5
            assert abs(e["delta_B_minus_A_session2_s"]) <= 0.5


@pytest.mark.skipif(not EVIDENCE_PRESENT, reason="live arm evidence absent")
def test_order_decomposition_sums():
    metrics = forensics.arm_metrics(LIVE)
    order = forensics.order_effect(metrics)
    assert order["walls_s"] == {"s1A": 70.646, "s1B": 68.483,
                                "s2B": 72.918, "s2A": 70.692}
    assert order["position_sensitivity_s"]["A"] == pytest.approx(0.046, abs=0.005)
    assert order["position_sensitivity_s"]["B"] == pytest.approx(-4.435, abs=0.005)
    s = order["position_effect_sums"]
    # session effect on tokens 1-11 (both session-2 arms slower)
    assert s["tokens_1_11"]["A_second_minus_A_first_s"] > 2.5
    assert s["tokens_1_11"]["B_second_minus_B_first_s"] < -1.0
    # position effect on tokens 12-14 (second arm faster, both arms)
    assert s["tokens_12_14"]["A_second_minus_A_first_s"] < 0
    assert s["tokens_12_14"]["B_second_minus_B_first_s"] < 0


@pytest.mark.skipif(not EVIDENCE_PRESENT, reason="live arm evidence absent")
def test_model_comparison_prefers_read_batch_wall_and_rejects_old_model():
    metrics = forensics.arm_metrics(LIVE)
    model = forensics.model_comparison(metrics)
    p = model["predictors"]
    assert p["measured_read_batch_wall_s"]["pearson_r"] > p["miss_count"]["pearson_r"]
    assert model["old_model_check"]["predicted_wall_gain_from_52_fewer_misses_s"] == \
        pytest.approx(2.53, abs=0.01)
    assert "REJECTED" in model["old_model_check"]["verdict"]


# ---------------------------------------------------------------------------
# Rider argument-shape fix (Part E regression)
# ---------------------------------------------------------------------------

def test_session_driver_rider_targets_experts_dee4():
    src = (EXPROOT / "tools" / "session_driver.py").read_text(encoding="utf-8")
    assert '"--store", str(store / "experts.dee4")' in src, \
        "rider --store must point at the multi-record file"
    assert '"--journal-meta", str(store / "metadata.json")' in src
    assert '"--store", str(store / "metadata.json")' not in src, \
        "the fcc-era bug: metadata.json is smaller than one record"


def test_make_rider_package_invariants(tmp_path):
    import make_rider  # noqa: E402  (EXPROOT/tools on sys.path)
    out = tmp_path / "rider"
    argv = sys.argv
    sys.argv = ["make_rider.py", "--out", str(out), "--slug", "test-rider"]
    try:
        make_rider.main()
    finally:
        sys.argv = argv
    script = (out / "test-rider.py").read_text(encoding="utf-8")
    assert "@@" not in script
    assert '--store", str(BANK / "experts.dee4")' in script
    assert "rev-parse" in script and "checkout mismatch" in script
    compile(script, "test-rider.py", "exec")
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["nivind/deepseek-v4-flash-0731-shards"]
    assert meta["enable_gpu"] == "true"
    # embedded payload hashes are self-consistent
    import base64
    b64 = script.split('BENCH_B64 = "')[1].split('"')[0]
    sha = script.split('BENCH_SHA256 = "')[1].split('"')[0]
    assert hashlib.sha256(base64.b64decode(b64)).hexdigest() == sha
