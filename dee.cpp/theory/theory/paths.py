"""In-repo artifact paths (relative to ``dee.cpp/``) and output locations.

All input paths are READ-ONLY for this package; all writes land in
``theory/data/`` and ``theory/figs/``.
"""
from __future__ import annotations

import os
from pathlib import Path

# theory/ package root (this file lives in theory/theory/)
THEORY_DIR = Path(__file__).resolve().parent.parent
# dee.cpp/ root
DEE_ROOT = THEORY_DIR.parent

DATA_DIR = THEORY_DIR / "data"
FIGS_DIR = THEORY_DIR / "figs"

# --------------------------------------------------------------------------
# Input artifacts (all in-repo, all read-only for us)
# --------------------------------------------------------------------------
# Primary 50k-event real router trace (Ornith milestone-2.5, 4 MoE layers).
TRACE_JSONL_GZ = DEE_ROOT / (
    "benchmark_reports/milestone-2.5/kaggle-forensics-latest-output/"
    "ornith-milestone25-evidence/expert-trace.jsonl.gz")
MILESTONE_DIR = TRACE_JSONL_GZ.parent
EXPERT_CACHE_ANALYSIS = MILESTONE_DIR / "expert-cache-analysis.json"
TRANSFER_ANALYSIS = MILESTONE_DIR / "transfer-analysis.json"
LAYER_TIMING = MILESTONE_DIR / "layer-timing.json"
BOTTLENECK_RANKING = MILESTONE_DIR / "bottleneck-ranking.json"
MATRIX_SUMMARY = MILESTONE_DIR / "matrix-summary.json"

# DSv4-Flash-0731 sealed 16-token router journal (43 layers, top-6), the
# trace the phase-2 working-set sim was run over.
SEALED_DIR = DEE_ROOT / "experiments/route_pipeline/fill-live-t4x2-20260909"
SEALED_ROUTED = SEALED_DIR / "routed_experts.jsonl"
SEALED_RESULT = SEALED_DIR / "result.json"
SEALED_PROFILE = SEALED_DIR / "profile.json"
SEALED_RUN_CONFIG = SEALED_DIR / "run_config.json"

# Trace-replay cache simulator whose measured/verdict numbers model B must
# reproduce (LRU, Belady MIN, reuse-distance coverage).
P23_SIM = DEE_ROOT / (
    "benchmark_reports/deepseek-v4-flash-0731-t4/p2_3_trace_replay_simulator.py")
CACHE1_ANALYSIS = DEE_ROOT / (
    "benchmark_reports/deepseek-v4-flash-0731-t4/CACHE1_ANALYSIS.md")

# Phase-1 storage verdict (0.29-0.37 GiB/s bank ceiling, feed-side verdict).
STORAGE_VERDICT = DEE_ROOT.parent / "research/route-pipeline/STORAGE_VERDICT.md"

# Phase-3 store specs: record geometry for the portability models.
SPECS_DIR = DEE_ROOT / "tools/phase3/specs"
SPECS = {
    "mimo_v2_flash": SPECS_DIR / "mimo_v2_flash.json",
    "mimo_v2_pro": SPECS_DIR / "mimo_v2_pro.json",
    "minimax_m3": SPECS_DIR / "minimax_m3.json",
}

# Measured decode stats used as the empirical anchor.
KAGGLE_RUN_CONFIG = DEE_ROOT / "kaggle/deepseek-v4-flash-0731/run_config.json"


def ensure_outputs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)


def rel(p: Path) -> str:
    """Path as repo-relative forward-slash string, for citation."""
    try:
        return str(p.relative_to(DEE_ROOT.parent)).replace(os.sep, "/")
    except ValueError:
        return str(p).replace(os.sep, "/")
