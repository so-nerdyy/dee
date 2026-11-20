"""Anchor reproduction gate — the shared target for every Phase-7 instrument.

This module extracts the sealed anchor run's ground-truth counters
(``experiments/route_pipeline/fill-live-t4x2-20260909/result.json`` +
``routed_experts.jsonl``) into ``data/anchor_gate.json`` and
``data/anchor_tokens.csv``, and **declares the gate tolerances here, before
any simulator exists** — the tolerances are pre-registered by the scaffold
commit so the digital twin (``theory.sim``) cannot tune its gate to itself.

Ground truth extracted (per forward step, pooled over both GPUs):
    storage_requests, host_pack_hits, host_pack_misses, resident_hits,
    cold_loads, h2d_copies, evictions, wall_ms  + per-GPU engine/host_pack/
    expert_store totals for per-GPU replay checks.

Definitional note carried in the output (disagreement is data): the Phase-7
brief's gate band "host_pack hit share (23-25%)" does NOT match the pooled
dedup definition computed from ``result.json`` (2,618 / 5,099 = 51.3%); only
``cuda0 host_pack hits / pooled requests`` (1,223 / 5,099 = 24.0%) lands in
that band.  Every defensible definition is computed below and flagged; the
sim reproduces the per-token integer arrays (unambiguous) and reports each
share definition rather than adopting one silently.

No fitting happens here: everything is extracted or declared.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from . import paths, sources
from .constants import C
from .util import write_csv

# --------------------------------------------------------------------------
# Pre-registered gate tolerances (declared with this scaffold, before any
# simulator implementation existed).  theory.sim may state a TIGHTER
# tolerance but may not loosen these; if the gate fails, the brief's rule
# applies: say so and stop — no predictions from an unanchored sim.
# --------------------------------------------------------------------------
GATE_TOLERANCE = C(
    "anchor_gate_tolerance", {
        "storage_requests_per_decode_token": {
            "mean_tol_rel": 0.10,        # mean of 15 decode tokens within +-10%
            "per_token_tol_rel": 0.15,   # each token within +-15%
            "min_tokens_in_tol": 12},    # >= 12/15 tokens inside per-token tol
        "host_pack_hit_share": {
            "definition": "pooled_dedup = sum(host_pack_hits) / "
                          "sum(host_pack_hits + host_pack_misses)",
            "tol_pp": 3.0},              # within +-3 percentage points
        "pooled_counters": {
            "cold_loads_tol_rel": 0.05,
            "resident_hits_tol_rel": 0.20,
            "h2d_copies_tol_rel": 0.05},
        "decode_wall_shape": {
            "mean_tol_rel": 0.30,
            "per_token_tol_rel": 0.50,
            "min_tokens_in_tol": 12,
            "spearman_min": 0.5},
        "calibration_policy": (
            "any sim parameter fitted to anchor data must be declared "
            "CALIBRATED with fitted quantity + residual, and the gate then "
            "holds only over quantities NOT used in the fit (the fit/holdout "
            "split must be stated in SIM_VS_THEORY.md)")},
    "ASSUMPTION",
    "dee.cpp/theory/theory/anchor_gate.py (Phase-7 scaffold; declared before "
    "any simulator existed)",
    "gate floor for theory.sim REPLAY MODE; method choice, not a measurement",
    "mixed")

# Measured share of host-tier absorption under every defensible definition.
SHARE_DEFINITIONS = C(
    "anchor_host_share_definitions", "computed in data/anchor_gate.json",
    "DERIVED",
    "dee.cpp/experiments/route_pipeline/fill-live-t4x2-20260909/result.json "
    "(per_token_accounting + host_pack + engine_stats)",
    "host_pack hit share is definition-dependent: pooled_dedup 51.3% vs the "
    "Phase-7 brief's claimed 23-25% (only cuda0_hits/pooled_requests = 24.0% "
    "lands in that band); THEORY.md L327 is internally inconsistent "
    "(cites 2,618/5,099 for a 23-25% claim = 51.3%)",
    "fraction")


def _shares(res: dict) -> Dict[str, Any]:
    pt = res["per_token_accounting"]
    dec = [r for r in pt if r["phase"] == "decode"]
    hits = sum(r["host_pack_hits"] for r in pt)
    misses = sum(r["host_pack_misses"] for r in pt)
    dedup = hits + misses
    raw_slots = 5676          # 946 token-rows x topk 6 (recomputed in run())
    res_hits = sum(r["resident_hits"] for r in pt)
    fills = sum(r["cold_loads"] for r in pt)
    hp = res["host_pack"]
    es = res["engine_stats"]

    def gpu_hits(dev: str) -> int:
        return int(hp[dev]["hits"])

    def gpu_lookups(dev: str) -> int:
        return int(hp[dev]["hits"]) + int(hp[dev]["misses"])

    defs = {
        "pooled_dedup": {
            "formula": "sum(host_pack_hits) / sum(host_pack_hits+misses)",
            "value": hits / dedup, "hits": hits, "denominator": dedup},
        "per_gpu.cuda0": {
            "formula": "host_pack.cuda0.hits / (hits+misses) [cuda0]",
            "value": gpu_hits("cuda0") / gpu_lookups("cuda0"),
            "hits": gpu_hits("cuda0"), "denominator": gpu_lookups("cuda0")},
        "per_gpu.cuda1": {
            "formula": "host_pack.cuda1.hits / (hits+misses) [cuda1]",
            "value": gpu_hits("cuda1") / gpu_lookups("cuda1"),
            "hits": gpu_hits("cuda1"), "denominator": gpu_lookups("cuda1")},
        "cuda0_hits_over_pooled_requests": {
            "formula": "host_pack.cuda0.hits / pooled dedup requests",
            "value": gpu_hits("cuda0") / dedup,
            "hits": gpu_hits("cuda0"), "denominator": dedup},
        "cuda1_hits_over_pooled_requests": {
            "formula": "host_pack.cuda1.hits / pooled dedup requests",
            "value": gpu_hits("cuda1") / dedup,
            "hits": gpu_hits("cuda1"), "denominator": dedup},
        "decode_only_dedup": {
            "formula": "decode rows: sum(hits) / sum(hits+misses)",
            "value": (sum(r["host_pack_hits"] for r in dec)
                      / sum(r["host_pack_hits"] + r["host_pack_misses"]
                            for r in dec)),
            "hits": sum(r["host_pack_hits"] for r in dec),
            "denominator": sum(r["host_pack_hits"] + r["host_pack_misses"]
                               for r in dec)},
        "hits_over_device_fills": {
            "formula": "sum(host_pack_hits) / sum(cold_loads) "
                       "(fill-source reading; ambiguous: host lookup may "
                       "also run on VRAM-resident hits)",
            "value": hits / fills, "hits": hits, "denominator": fills},
        "hits_over_raw_slots": {
            "formula": "sum(host_pack_hits) / raw rank slots",
            "value": hits / raw_slots, "hits": hits,
            "denominator": raw_slots},
    }
    for d in defs.values():
        d["in_brief_band_23_25pct"] = bool(0.23 <= d["value"] <= 0.25)

    # Brief band vs measured decode band (definitional mismatch, reported).
    dec_sr = [r["storage_requests"] for r in dec]
    bands = {
        "brief_claim_host_share": [0.23, 0.25],
        "brief_claim_storage_requests": [83, 155],
        "measured_decode_storage_requests": {
            "mean": float(np.mean(dec_sr)), "min": min(dec_sr),
            "max": max(dec_sr)},
        "measured_whole_gen_storage_requests_per_emitted_token":
            res["byte_accounting"]["storage_requests_per_generated_token"],
        "note": ("brief bands are aggregate/loose: 83 = mean decode "
                 "storage_requests (measured 56-125 per token), 155 = whole-"
                 "generation per-emitted-token incl. prefill amortization"),
    }
    # engine/host_pack per-GPU totals for per-GPU replay checks
    gpu_totals = {}
    for dev in ("cuda0", "cuda1"):
        e, h, s = es[dev], hp[dev], res["expert_store"][dev]
        gpu_totals[dev] = {
            "requests": int(h["hits"]) + int(h["misses"]),
            "host_pack_hits": int(h["hits"]),
            "host_pack_misses": int(h["misses"]),
            "host_pack_evictions": int(h["evictions"]),
            "host_slots": int(h["entries"]),
            "host_budget_bytes": int(h["budget_bytes"]),
            "resident_hits": int(e["resident_hits"]),
            "cold_loads": int(e["cold_loads"]),
            "device_evictions": int(e["evictions"]),
            "resident_experts": int(e["resident_experts"]),
            "h2d_copies": int(e["h2d_copies"]),
            "h2d_bytes": int(e["h2d_bytes"]),
            "source_reads": int(s["source_reads"]),
            "source_read_bytes": int(s["pread_bytes"]),
            "mean_read_ms": float(s["average_read_ms"]),
            "p50_read_ms": float(s["p50_read_ms"]),
            "p95_read_ms": float(s["p95_read_ms"]),
            "fill_lanes": int(s["max_source_read_lanes"]),
            "max_fill_queue_depth": int(s["max_source_read_queue_depth"]),
        }
    return {"definitions": defs, "bands": bands, "gpu_totals": gpu_totals,
            "resident_hits_total": res_hits, "device_fills_total": fills}


def _stream_summary() -> Dict[str, Any]:
    from collections import Counter

    st = sources.load_sealed_stream()
    raw_slots = st.n_slots
    s = st.stream()
    counts = Counter(s)
    uniq = len(counts)
    repeats = sum(1 for v in counts.values() if v >= 2)
    return {
        "token_rows": sum(len(v) for v in st.raw.values()),
        "layer_calls": st.n_calls,
        "raw_rank_slots": raw_slots,
        "dedup_cache_requests": len(s),
        "unique_records": uniq,
        "records_used_by_ge2_layer_calls": repeats,
        "layers": st.n_layers, "experts_per_layer": st.experts_per_layer,
        "topk": st.topk,
    }


def run(fast: bool = False, **kw) -> Dict[str, Any]:
    paths.ensure_outputs()
    res = sources.load_sealed_result()
    pt = res["per_token_accounting"]

    rows: List[dict] = []
    for r in pt:
        rows.append({
            "step": r["step"], "phase": r["phase"],
            "wall_ms": r["wall_ms"],
            "storage_requests": r["storage_requests"],
            "host_pack_hits": r["host_pack_hits"],
            "host_pack_misses": r["host_pack_misses"],
            "resident_hits": r["resident_hits"],
            "cold_loads": r["cold_loads"],
            "h2d_copies": r["h2d_copies"],
            "h2d_bytes": r["h2d_bytes"],
            "evictions": r["evictions"],
        })
    write_csv("anchor_tokens.csv", rows, list(rows[0].keys()))

    dec = [r for r in pt if r["phase"] == "decode"]
    out = {
        "source": {
            "result_json": paths.rel(paths.SEALED_RESULT),
            "routed_experts": paths.rel(paths.SEALED_ROUTED),
            "run_id": res["run_id"], "commit": res["commit"]},
        "config": {
            "vram_slots_per_gpu": int(res["engine_stats"]["cuda0"]
                                      ["resident_experts"]),
            "vram_budget_bytes_per_gpu": int(res["engine_stats"]["cuda0"]
                                             ["device_expert_cache_reserved_bytes"]),
            "host_slots_per_gpu": int(res["host_pack"]["cuda0"]["entries"]),
            "host_budget_bytes_per_gpu": int(res["host_pack"]["cuda0"]
                                             ["budget_bytes"]),
            "record_bytes": res["expert_store"]["cuda0"]
                              ["average_request_bytes"],
            "layer_count_executed": res["layer_count_executed"],
            "fill_lanes": res["expert_store"]["cuda0"]["max_source_read_lanes"],
            "max_fill_queue_depth": res["expert_store"]["cuda0"]
                                     ["max_source_read_queue_depth"]},
        "per_token": rows,
        "decode_wall_ms": [r["wall_ms"] for r in dec],
        "decode_storage_requests": [r["storage_requests"] for r in dec],
        "decode_wall_shape_stats": {
            "mean_ms": float(np.mean([r["wall_ms"] for r in dec])),
            "std_ms": float(np.std([r["wall_ms"] for r in dec])),
            "min_ms": float(np.min([r["wall_ms"] for r in dec])),
            "max_ms": float(np.max([r["wall_ms"] for r in dec]))},
        "decode_tok_s": res["decode_tok_s"],
        "decode_wall_s": res["decode_wall_s"],
        "whole_gen": {
            "storage_requests_per_generated_token":
                res["byte_accounting"]["storage_requests_per_generated_token"],
            "storage_bytes_per_generated_token":
                res["byte_accounting"]["storage_bytes_per_generated_token"],
            "storage_requests_total":
                res["byte_accounting"]["storage_requests_total"],
            "storage_bytes_total":
                res["byte_accounting"]["storage_bytes_total"],
            "expert_h2d_bytes_total":
                res["byte_accounting"]["expert_h2d_bytes_total"]},
        "host_share": _shares(res),
        "stream": _stream_summary(),
        "gate_tolerance": GATE_TOLERANCE.value,
    }
    p = paths.DATA_DIR / "anchor_gate.json"
    p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return {"anchor_gate": out, "gate_json": str(p)}
