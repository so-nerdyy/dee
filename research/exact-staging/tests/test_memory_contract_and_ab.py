#!/usr/bin/env python3
"""Tests for the memory-unit contract and the A/B statistics package.

The centerpiece is test_memory_budget_table_matches_json: it regenerates the
sentinel-marked markdown tables from the committed results JSON and compares
row-by-row — the exact regression that would have caught the fcc8ca2
MEMORY_BUDGET.md vs memory_frontier.json discrepancy (31.36 GB vs 29.23 GB
at 10.0 GiB/GPU).

Also covers: unit-name linting of every memory field, the dual-system
classification rules, the headroom identity, and the A/B statistics tools
(synthetic data only — no measurements are fabricated).
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent                # research/exact-staging
ROOT = RESEARCH.parent.parent         # worktree root
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(ROOT / "tools"))

import pytest  # noqa: E402
import recalibrate_model as rm  # noqa: E402

RESULTS = RESEARCH / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------- unit contract / lint -----

MEMORY_FIELD_RE = re.compile(
    r"(gib|_decimal_gb|_bytes|bytes_.*|.*_bytes)$")


def memory_fields(obj, path=""):
    """Yield (key_path, value) for keys that look memory-related."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{path}.{k}" if path else k
            if re.search(r"(gib|gb|bytes|memory|memtotal|available)", k,
                         re.IGNORECASE) and isinstance(v, (int, float)):
                yield kp, k, v
            yield from memory_fields(v, kp)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from memory_fields(v, f"{path}[{i}]")


def test_no_unqualified_gb_fields_in_memory_frontier():
    mf = _load("memory_frontier.json")
    bad = [kp for kp, k, _ in memory_fields(mf)
           if re.search(r"\bgb\b", k, re.IGNORECASE)
           and not k.endswith("_decimal_gb")]
    assert not bad, f"unqualified 'GB' memory fields banned by contract: {bad}"


def test_unit_contract_block_present_and_consistent():
    mf = _load("memory_frontier.json")
    uc = mf["unit_contract"]
    assert uc["authoritative_for_host_execution"] == "MEASURED_HOST_MEMTOTAL"
    systems = uc["systems"]
    assert systems["MEASURED_HOST_MEMTOTAL"]["limit_gib"] == 31.35
    assert systems["STRICT_32_DECIMAL_GB"]["limit_decimal_gb"] == 32.0
    assert mf["limit_gib"] == pytest.approx(32.0 * 1e9 / 2**30, abs=0.01)


def test_dual_headroom_identity_and_both_systems_reported():
    """headroom to MemTotal == projected min-available (by construction)."""
    for r in _load("memory_frontier.json")["rows"]:
        assert r["headroom_gib_to_measured_memtotal"] == pytest.approx(
            r["projected_min_MemAvailable_gib"], abs=0.011)
        assert r["headroom_gib_to_measured_memtotal"] == pytest.approx(
            31.35 - r["projected_total_gib"], abs=0.011)
        assert r["headroom_decimal_gb_to_strict_contract"] == pytest.approx(
            32.0 - r["projected_total_decimal_gb"], abs=0.011)
        for f in ("classification_measured_host_memtotal",
                  "classification_strict_32_decimal_gb", "classification"):
            assert r[f] in ("SAFE_FOR_32GB", "BORDERLINE_FOR_32GB",
                            "NOT_SAFE_FOR_32GB")


def test_combined_class_is_conservative_intersection():
    order = {"SAFE_FOR_32GB": 0, "BORDERLINE_FOR_32GB": 1, "NOT_SAFE_FOR_32GB": 2}
    for r in _load("memory_frontier.json")["rows"]:
        worst = max((r["classification_measured_host_memtotal"],
                     r["classification_strict_32_decimal_gb"]),
                    key=lambda c: order[c])
        assert r["classification"] == worst


def test_gib_to_decimal_gb_conversion_correct():
    GIB, GB = 2**30, 1e9
    for r in _load("memory_frontier.json")["rows"]:
        assert r["projected_total_decimal_gb"] == pytest.approx(
            r["projected_total_gib"] * GIB / GB, abs=0.011)


# --------------------------------------- markdown == JSON (fcc8ca2 bug) -----

def _parse_md_table(md: str, after_marker: str) -> list[list[str]]:
    begin = f"<!-- BEGIN GENERATED:{after_marker} -->"
    end = f"<!-- END GENERATED:{after_marker} -->"
    block = md.split(begin)[1].split(end)[0]
    rows = []
    for line in block.strip().splitlines():
        line = line.strip()
        if line.startswith("|") and not re.match(r"^\|[\s\-|]+\|$", line):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def test_memory_budget_table_matches_json():
    """THE regression test for the fcc8ca2 discrepancy: every rendered row
    must equal the JSON row (unit-explicit columns, both systems)."""
    md = (RESEARCH / "MEMORY_BUDGET.md").read_text(encoding="utf-8")
    rows = _parse_md_table(md, "memory-table")
    js = _load("memory_frontier.json")["rows"]
    assert len(rows) == len(js) + 1, "header + one row per budget (separator excluded)"
    hdr = rows[0]
    col = {name: i for i, name in enumerate(hdr)}
    for jr, row in zip(js, rows[1:]):
        assert float(row[col["Budget GiB/GPU"]]) == jr["pack_gib_per_gpu"]
        assert float(row[col["Pack total GiB"]]) == jr["pack_total_gib"]
        assert float(row[col["Projected total GiB"]]) == jr["projected_total_gib"]
        assert float(row[col["Projected total decimal GB"]]) == \
            jr["projected_total_decimal_gb"]
        assert float(row[col["Headroom GiB to measured MemTotal (31.35)"]]) == \
            jr["headroom_gib_to_measured_memtotal"]
        assert float(row[col["Headroom decimal GB to strict 32 GB contract"]]) == \
            jr["headroom_decimal_gb_to_strict_contract"]
        assert row[col["Class (measured MemTotal)"]] == \
            rm._short_cls(jr["classification_measured_host_memtotal"])
        assert row[col["Class (strict 32 decimal GB)"]] == \
            rm._short_cls(jr["classification_strict_32_decimal_gb"])
        assert row[col["Combined class"]] == rm._short_cls(jr["classification"])


def test_frontier_table_matches_json():
    md = (RESEARCH / "HOST_PACK_FRONTIER.md").read_text(encoding="utf-8")
    rows = _parse_md_table(md, "frontier-table")
    js = _load("recalibrated_model.json")["frontier"]
    assert len(rows) == len(js) + 1
    hdr = rows[0]
    col = {name: i for i, name in enumerate(hdr)}
    for jr, row in zip(js, rows[1:]):
        assert float(row[col["GiB/GPU"]]) == jr["pack_gib_per_gpu"]
        assert int(row[col["records"]]) == jr["records_per_gpu"]
        assert int(row[col["misses"]]) == jr["decode_misses_replayed"]
        assert int(row[col["capacity"]]) == jr["decode_capacity_misses"]
        assert float(row[col["sim wall (s, SIMULATED)"]]) == \
            jr["simulated_decode_wall_s"]
        assert row[col["class (strict contract)"]] == \
            rm._short_cls(jr["memory_class_strict_contract"])


def test_old_buggy_values_are_gone():
    """The exact numbers from the fcc8ca2 hand-typed table must not reappear
    in the generated memory table (10.0 -> 31.36 GB / 0.64 GB)."""
    md = (RESEARCH / "MEMORY_BUDGET.md").read_text(encoding="utf-8")
    begin = "<!-- BEGIN GENERATED:memory-table -->"
    end = "<!-- END GENERATED:memory-table -->"
    block = md.split(begin)[1].split(end)[0]
    assert "31.36" not in block and "0.64 |" not in block
    # and the corrected values ARE there:
    assert "| 10.0 | 20.0 | 27.227 | 29.23 |" in block


# ------------------------------------------------ dual-system specifics -----

def test_strict_contract_is_more_conservative_than_memtotal_somewhere():
    """The finer grid must expose rows where MemTotal=SAFE but strict=BORDERLINE
    (31.35 GiB host > 29.80 GiB strict limit in GiB terms)."""
    rows = _load("memory_frontier.json")["rows"]
    split = [r for r in rows
             if r["classification_measured_host_memtotal"] == "SAFE_FOR_32GB"
             and r["classification_strict_32_decimal_gb"] != "SAFE_FOR_32GB"]
    assert split, "expected the 10.5-11.0 GiB/GPU band to differ by system"
    assert all(10.0 < r["pack_gib_per_gpu"] <= 11.0 for r in split)


def test_recommendation_safe_under_both_systems():
    mf = _load("memory_frontier.json")
    rec = mf["recommended"]
    row = next(r for r in mf["rows"] if r["pack_gib_per_gpu"] == rec)
    assert row["classification_measured_host_memtotal"] == "SAFE_FOR_32GB"
    assert row["classification_strict_32_decimal_gb"] == "SAFE_FOR_32GB"


def test_frontier_grid_covers_required_budgets():
    got = [r["pack_gib_per_gpu"] for r in _load("memory_frontier.json")["rows"]]
    for b in (8.5, 9.0, 9.5, 10.0, 10.5, 10.75, 11.0, 11.25, 11.5, 12.0,
              12.25, 12.75):
        assert b in got


# --------------------------------------------------- A/B statistics ---------

@pytest.fixture(scope="module")
def ab():
    return _load("ab_noise.json")


@pytest.fixture(scope="module")
def plan():
    return _load("ab_power_plan.json")


def test_ab_noise_labels_and_direction(ab):
    eff = ab["paired_ab_effect"]
    assert eff["n_pairs"] == 2
    assert eff["direction_consistent"] is True
    for p in eff["pairs"]:
        assert p["delta_s"] == pytest.approx(
            p["candidate_s"] - p["baseline_s"], abs=0.0011)
    # measured deltas match the seal exactly (no fabrication):
    deltas = sorted(p["delta_s"] for p in eff["pairs"])
    assert deltas == pytest.approx([-1.929, -0.671], abs=0.0011)


def test_ab_noise_baseline_variance(ab):
    bv = ab["baseline_variance"]
    walls = bv["walls_s"]
    assert sorted(walls) == pytest.approx([71.362, 72.267, 73.213], abs=0.0011)
    assert bv["sd_s"] < bv["spread_s"]
    assert bv["within_notebook_sd_s"].startswith("UNKNOWN")


def test_bootstrap_labeled_exploratory(ab):
    b = ab["bootstrap_ci"]
    assert b["label"] == "EXPLORATORY_STATISTIC"
    assert "degenerate" in b["caveat"]
    lo, hi = b["ci95_mean_delta_s"]
    assert lo <= ab["paired_ab_effect"]["mean_delta_s"] <= hi


def test_per_token_dependence_caveat_present(ab):
    for pair, d in ab["per_token_paired"].items():
        assert "dependent" in d["dependence_caveat"]
        assert d["n_tokens"] == 15
        assert d["label"] == "EXPLORATORY_STATISTIC"


def test_lane_era_runs_excluded_from_baseline(ab):
    assert "lane" in ab["lane_era_context"]["note"].lower()
    for n, r in ab["runs"].items():
        if n in ("v63_3lane", "v64_3lane"):
            assert "lane" in r["group"] and "6lane" not in r["group"]


def test_power_table_monotone_and_honest(plan):
    by_sigma = {}
    for t in plan["pairs_required_table"]:
        by_sigma.setdefault(t["sigma_source"], []).append(t)
    for _, ts in by_sigma.items():
        ts.sort(key=lambda t: t["effect_s"])
        mean_rule = [t["pairs_mean_rule_z2"] for t in ts]
        assert mean_rule == sorted(mean_rule, reverse=True), \
            "bigger effect must never need more pairs"


def test_power_direction_rule_unattainable_for_1pct(plan):
    """phi(E/sigma)=0.79 < 0.90, and p^n decreases in n -> never attainable."""
    for t in plan["pairs_required_table"]:
        if t["effect"] == "1_pct":
            assert isinstance(t["pairs_direction_rule_p90"], str)
            assert "UNATTAINABLE" in t["pairs_direction_rule_p90"]


def test_direction_rule_math():
    pab = _import(ROOT / "tools" / "plan_ab_power.py")
    sigma = 0.926
    # 2% effect (1.446 s): phi(1.5616)=0.9408; p^2=0.885<0.9, so n=1:
    assert pab.pairs_direction_rule(1.446, sigma) == 1
    # 1% effect: p=0.783, p^1 < 0.9 -> None:
    assert pab.pairs_direction_rule(0.723, sigma) is None
    # tiny sigma -> attainable:
    assert pab.pairs_direction_rule(0.723, 0.2) is not None


def test_effective_sample_size_correction():
    pab = _import(ROOT / "tools" / "plan_ab_power.py")
    assert pab.effective_sample_size(15, 0.0) == 15
    n = pab.effective_sample_size(15, 0.49)
    assert 4 <= n <= 7  # (1-rho)/(1+rho) ~ 0.342 * 15 ~ 5


def test_pack_plan_consistency(plan, ab):
    pp = plan["pack_cap_experiment_plan"]
    assert pp["pairs_budgeted_upfront"] == 2
    assert pp["is_15_tokens_enough"]["answer"] is True
    assert "longer" in pp["would_longer_decode_help"]["answer"].lower() or \
        pp["would_longer_decode_help"]["answer"].startswith("not")
    model = _load("recalibrated_model.json")
    expected = (model["recommended_candidate"]["miss_reduction_vs_8gib5"]
                * model["model"]["miss_service_ms"] / 1000)
    assert pp["predicted_effect_s"] == pytest.approx(expected, abs=0.01)


def test_next_ab_references_preregistered_rule():
    nxt = _load("next_ab.json")
    rule = nxt["analysis_plan"]["stopping_rule_preregistered"]
    assert "futility" in rule and "no_early_accept" in rule
    assert nxt["analysis_plan"]["n_pairs_budgeted_upfront"] == 2
    assert nxt["candidate_config"]["host_pack_runtime_cap_gib_total"] == 20.0
    assert nxt["candidate_config"]["effective_pack_gib_per_gpu"] == 10.0


def test_all_tool_outputs_carry_labels(ab, plan):
    def has_label(o):
        if isinstance(o, dict):
            if "label" in o or "labels" in o:
                return True
            return any(has_label(v) for v in o.values())
        if isinstance(o, list):
            return any(has_label(v) for v in o)
        return False
    assert has_label(ab) and has_label(plan)
