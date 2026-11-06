"""Phase-5 W3: verifier unit tests — journal-row extraction + chain audit.

Fabricates c1-style singleton journals and a K=2 cohort journal with the
real RoutedExpertJournal so the chain hashing is authentic, then checks
extract_row_stream / streams_equal / audit_chain.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle" / "deepseek-v4-flash-0731"))

from deepseek_v4_native_generate import RoutedExpertJournal  # noqa: E402
from tools.phase5.p5_verify import (  # noqa: E402
    Checker, audit_chain, extract_row_stream, load_journal, streams_equal)

N_LAYERS, TOPK, LSTAR, N_TOK = 2, 3, 3, 4


def _make_journal(path: Path, rows_by_step: dict[int, list[list[int]]],
                  lstar: int) -> None:
    """Write a journal: step0 = prefill (matrix rows_by_step[0]), steps
    1..N_TOK-1 = decode records with the given per-forward matrices."""
    j = RoutedExpertJournal(path, run_id="test", n_layers=N_LAYERS,
                            topk=TOPK)
    for step in range(N_TOK):
        start = 0 if step == 0 else lstar + step - 1
        for layer in range(N_LAYERS):
            j.append_layer(step=step, start_pos=start, layer=layer,
                           device="cpu",
                           expert_ids=rows_by_step[(step, layer)]
                           if (step, layer) in rows_by_step
                           else rows_by_step[step])
    j.close()


def _fab(tmp: Path):
    """Build a c1 singleton journal (prompt p1 routes) + a K=2 cohort
    journal whose row 1 equals it.  Returns (singleton_recs,
    cohort_recs)."""
    # Singleton journal for "prompt 1": distinct routes per step/layer.
    single = {}
    for step in range(N_TOK):
        for layer in range(N_LAYERS):
            if step == 0:
                single[(step, layer)] = [
                    [10 * layer + 3 * s + e for e in range(TOPK)]
                    for s in range(LSTAR)]
            else:
                single[(step, layer)] = [[layer * 10 + step * 3 + e
                                          for e in range(TOPK)]]
    _make_journal(tmp / "routed_experts-c1.jsonl", single, LSTAR)

    # Cohort K=2 over [p0, p1]: row 1 = p1's routes; row 0 arbitrary.
    cohort = {}
    for step in range(N_TOK):
        for layer in range(N_LAYERS):
            if step == 0:
                row0 = [[90 + layer + s + e for e in range(TOPK)]
                        for s in range(LSTAR)]
                cohort[(step, layer)] = row0 + single[(step, layer)]
            else:
                cohort[(step, layer)] = [
                    [90 + layer + step + e for e in range(TOPK)],
                    single[(step, layer)][0]]
    _make_journal(tmp / "routed_experts-c0.jsonl", cohort, LSTAR)
    return (load_journal(tmp / "routed_experts-c1.jsonl"),
            load_journal(tmp / "routed_experts-c0.jsonl"))


def test_row_extraction_matches_singleton(tmp_path: Path) -> None:
    single, cohort = _fab(tmp_path)
    ref = extract_row_stream(single, row=0, k=1, lstar=LSTAR)
    got = extract_row_stream(cohort, row=1, k=2, lstar=LSTAR)
    same, detail = streams_equal(got, ref)
    assert same, detail


def test_row0_is_independent(tmp_path: Path) -> None:
    single, cohort = _fab(tmp_path)
    row0 = extract_row_stream(cohort, row=0, k=2, lstar=LSTAR)
    ref = extract_row_stream(single, row=0, k=1, lstar=LSTAR)
    same, _ = streams_equal(row0, ref)
    assert not same


def test_chain_audit_passes_fabricated(tmp_path: Path) -> None:
    single, cohort = _fab(tmp_path)
    ck = Checker()
    audit_chain(ck, "single", single)
    audit_chain(ck, "cohort", cohort)
    assert not ck.fails


def test_chain_audit_detects_tamper(tmp_path: Path) -> None:
    single, _ = _fab(tmp_path)
    tampered = [dict(r) for r in single]
    tampered[2] = dict(tampered[2],
                       expert_ids_rank_order=[[0] * TOPK])
    ck = Checker()
    audit_chain(ck, "tampered", tampered)
    assert ck.fails


def test_extraction_rejects_wrong_rows(tmp_path: Path) -> None:
    single, cohort = _fab(tmp_path)
    # Pretend K=3 — prefill matrix (6 rows) != 3*3.
    with pytest.raises(ValueError):
        extract_row_stream(cohort, row=0, k=3, lstar=LSTAR)
