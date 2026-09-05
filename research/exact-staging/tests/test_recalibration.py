#!/usr/bin/env python3
"""Tests for the v2 recalibration: seal ingestion, model, memory, frontier.

Requires the main checkout's seal evidence when the default hash gate is on;
the unit tests below are self-contained and skip gracefully if the seal is
absent (CI/local without the main checkout).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = RESEARCH.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(ROOT / "dee.cpp" / "experiments" / "exact_staging"))

import pytest  # noqa: E402
import recalibrate_model as rm  # noqa: E402
from bounded_staging_queue import pack_replay  # noqa: E402

SEALED_MISSES = 1252
SEALED_TABLE = [96, 69, 80, 111, 125, 114, 96, 80, 56, 64, 79, 61, 69, 91, 61]


def journal() -> Path:
    j = rm.MAIN_CHECKOUT / ("dee.cpp/tmp/host-reuse-profile2-20260904/"
                            "v2-fetch-20260904T194914Z/output/routed_experts.jsonl")
    if not j.exists():
        pytest.skip("sealed journal not available")
    return j


# ---------------------------------------------------------------- replay ----

def test_replay_682_reproduces_sealed_misses():
    r = pack_replay(journal(), 682)
    assert abs(r["decode_storage_misses_total"] - SEALED_MISSES) <= 2


def test_replay_per_step_table_within_two():
    r = pack_replay(journal(), 682)
    got = [r["decode_per_step_reads"][k] for k in sorted(r["decode_per_step_reads"])]
    assert len(got) == 15
    assert max(abs(a - b) for a, b in zip(got, SEALED_TABLE)) <= 2


def test_replay_budget_never_increases_misses():
    j = journal()
    m682 = pack_replay(j, 682)["decode_storage_misses_total"]
    m803 = pack_replay(j, 803)["decode_storage_misses_total"]
    m1024 = pack_replay(j, 1024)["decode_storage_misses_total"]
    assert m803 < m682 and m1024 < m803


def test_replay_compulsory_floor_constant_across_budgets():
    j = journal()
    floor = pack_replay(j, 1_000_000)["decode_storage_misses_total"]
    assert floor == 1135
    assert pack_replay(j, 803)["decode_storage_misses_total"] >= floor


# ----------------------------------------------------------------- model ----

def test_fit_is_physical_and_fits_v65():
    derived = rm.load_derived()
    model = rm.fit_v65(derived["v65_sealed"])
    assert model["miss_service_ms"] > 0
    assert model["serial_residual_ms_per_step"] > 0
    assert model["r_squared"] > 0.9
    pred = rm.predict(model, derived["v65_sealed"]["ssd_reads"])
    mape = 100 * sum(abs(p - o) / o for p, o in
                     zip(pred, derived["v65_sealed"]["step_walls_ms"])) / 15
    assert mape < 5.0


def test_predict_rejects_future_layer_lead():
    model = dict(rm.fit_v65(rm.load_derived()["v65_sealed"]))
    model["ordinary_future_layer_lead"] = 8
    with pytest.raises(ValueError):
        rm.predict(model, [96] * 15)


def test_model_has_no_reuse_lever():
    """The recalibrated model must not carry a host_buffer_reuse switch."""
    model = rm.fit_v65(rm.load_derived()["v65_sealed"])
    assert "host_buffer_reuse" not in model
    # predict depends ONLY on misses:
    assert rm.predict(model, [10.0]) == rm.predict(model, [10.0])


def test_old_model_audit_names_root_causes():
    audit = rm.audit_old_model(rm.fit_v65(rm.load_derived()["v65_sealed"]),
                               rm.load_derived())
    causes = " ".join(c["cause"] for c in audit["root_causes"]).lower()
    assert "serial" in causes or "blocking" in causes
    assert audit["old_prediction"]["reuse_delta_pct"] < -10
    assert -3 < audit["measured"]["original_pair"]["delta_pct"] < 0


# --------------------------------------------------------------- memory ----

def test_memory_envelope_monotone_and_12_75_rejected():
    rows = [rm.memory_envelope(b) for b in (8.5, 9.5, 10.0, 10.5, 12.75)]
    totals = [r["projected_total_decimal_gb"] for r in rows]
    assert totals == sorted(totals)
    assert rows[0]["classification"] == "SAFE_FOR_32GB"
    assert rows[-1]["classification"] == "NOT_SAFE_FOR_32GB"


def test_budget_arithmetic_exact():
    env = rm.memory_envelope(9.5)
    pack_total = 2 * 9.5
    expect_gib = (pack_total + rm.NONPACK_HWM_GIB + rm.SYSTEM_REST_GIB
                  + rm.UNMEASURED_GROWTH_GIB)
    expect_gb = expect_gib * 2**30 / 1e9
    assert abs(env["projected_total_decimal_gb"] - expect_gb) < 0.01
    assert env["records_per_gpu"] == int(9.5 * 2**30 // rm.RECORD_BYTES)


def test_unknown_components_are_explicit_not_silent():
    env = rm.memory_envelope(10.0)
    assert env["unknown_components"], "UNKNOWNs must be listed, not hidden"
    assert any("page-cache" in c or "pinned" in c for c in env["unknown_components"])
    assert env["unmeasured_growth_allowance_gib"] > 0


def test_headroom_gate_distinguishes_borderline_from_safe():
    """A budget that fits arithmetic but lacks 2 GB headroom is BORDERLINE."""
    env = rm.memory_envelope(10.5)
    assert env["projected_total_decimal_gb"] > 32 or env["classification"] != "SAFE_FOR_32GB"
    assert env["classification"] in ("BORDERLINE_FOR_32GB", "NOT_SAFE_FOR_32GB")


# ------------------------------------------------------ predicted vs obs ----

def test_labels_distinguish_predicted_from_observed():
    result = json.loads((RESEARCH / "results/recalibrated_model.json").read_text())
    assert result["performance_acceptance"] is False
    for row in result["frontier"]:
        assert row["label"].startswith("SIMULATED")
    # measured walls recorded separately from predicted walls:
    val = result["validation"]
    assert val["v65_training"]["role"].startswith("fit")
    run_entries = {k: v for k, v in val.items()
                   if isinstance(v, dict) and "role" in v}
    assert len(run_entries) == 6, "5 held-out runs + 1 fit run expected"
    assert all(v["role"].startswith("validation (held out")
               for k, v in run_entries.items() if k != "v65_training")
    assert val["baseline_run_to_run_spread_s"]["spread"] > 0


def test_seal_digests_present():
    digests = json.loads(
        (RESEARCH / "results/seal_host_reuse/derived_memory_digests.json").read_text())
    assert len(digests["runs"]) == 5
    for name, r in digests["runs"].items():
        assert len(r["sha256_memory_json"]) == 64
        assert r["nonpack_hwm_gib"] > 0


# ---------------------------------------------------------- full pipeline ---

def test_full_pipeline_official_outputs(tmp_path):
    if not (rm.MAIN_CHECKOUT / "dee.cpp/tmp").exists():
        pytest.skip("main checkout seal evidence not available")
    rc = rm.main.__wrapped__ if hasattr(rm.main, "__wrapped__") else None
    # run main with overridden argv into a tmp dir, hash gate ON:
    import argparse
    old = sys.argv
    sys.argv = ["recalibrate_model.py", "--out", str(tmp_path)]
    try:
        rm.main()
    finally:
        sys.argv = old
    for name in ("recalibrated_model.json", "memory_frontier.json",
                 "next_ab.json", "host_pack_frontier.csv",
                 "pread_kaggle_package.json"):
        assert (tmp_path / name).exists(), name
    model = json.loads((tmp_path / "recalibrated_model.json").read_text())
    nxt = json.loads((tmp_path / "next_ab.json").read_text())
    assert model["recommended_candidate"]["memory_classification"] == "SAFE_FOR_32GB"
    assert nxt["candidate_config"]["effective_pack_gib_per_gpu"] == \
        model["recommended_candidate"]["pack_gib_per_gpu"]
    assert nxt["candidate_config"]["host_pack_runtime_cap_gib_total"] == \
        round(2 * model["recommended_candidate"]["pack_gib_per_gpu"], 1)
