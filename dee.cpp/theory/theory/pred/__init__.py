"""B. The predictor lab: train and evaluate REAL next-layer / next-step expert
predictors on the real traces (Phase-7 component B).

The ONLY legal prediction surface (AGENTS.md exactness contract): a prediction
may drive prefetch HINTS only.  A hint may stage a record early; the native
router stays authoritative and a wrong hint only wastes bandwidth.  Every
number produced here is a HINT-quality number (recall / precision of announced
candidate records), never an execution claim.

Prediction tasks (both lead = 1 layer-call / 1 decode step):

  * ``next_layer``: predict S_{t,l+1} from S_{t,l} (and history) within one
    token -- the surface of THEORY.md section 7 / FALSIFICATION P9-P10;
  * ``next_step``: predict S_{t+1,l} from S_{t,l} (and history) at the same
    layer across consecutive forward steps.

NO FUTURE LEAKAGE: time-ordered train/test splits fixed BEFORE any test
number was computed (see ``data/pred_splits.json`` for the exact definition).
Nothing about the test period -- vocabulary, features, hyperparameters -- is
fit on test data.  The expert-id vocabulary is fixed by model geometry
(0..255), not learned from data.

Arms (all emit a ranked candidate list per source context so m can be
swept):

  1. popularity      -- per target-layer empirical prior (cheap baseline);
  2. cond_pooled     -- empirical-conditional baseline P(j | S_l) exactly as
                        in theory/prefetch.py + THEORY.md section 7;
  3. cond_perlayer   -- the same estimator keyed per layer-pair;
  4. logreg_pairwise -- hashed-feature logistic regression (scikit-learn),
                        shared across layers, per (source, candidate) pair;
  5. mlp_set         -- small CPU torch MLP, source features -> 256-way
                        multi-label set prediction;
  6. xlayer_bag      -- within-token cross-layer predictor: learned set
                        embeddings of earlier layers' sets in the same token.

New constants are registered HERE (never in theory/constants.py) via
``from ..constants import C`` so provenance.csv picks them up automatically.
"""
from __future__ import annotations

from ..constants import C

# --------------------------------------------------------------------------
# Reproducibility / evaluation configuration (fixed a priori)
# --------------------------------------------------------------------------
PRED_SEED = C("pred_seed", 20260915, "CALIBRATED",
              "theory/pred/__init__.py (fixed a priori for deterministic regeneration)",
              "numpy / scikit-learn / torch CPU seeds", "seed")
PRED_M_REQUIRED = C("pred_m_required", [4, 8, 16], "CALIBRATED",
                    "AGENT_BRIEF_PHASE7.md deliverable B (metrics at m in {4,8,16})",
                    "required candidate-list sizes per source context", "records")
PRED_M_SWEEP = C("pred_m_sweep", [1, 2, 4, 8, 16, 32], "CALIBRATED",
                 "theory/pred/__init__.py (superset sweep so m can be traded off)",
                 "swept candidate-list sizes per source context", "records")
PRED_MIN_PAIR_SUPPORT = C("pred_min_pair_support", 5, "CALIBRATED",
                          "theory/temporal.py MIN_PAIR_SUPPORT (same convention for the "
                          "empirical-conditional arms)", "min source-expert observations "
                          "before a P(j|i) pair is trusted", "observations")
PRED_BOOT = C("pred_bootstrap_resamples", 400, "CALIBRATED",
              "theory/pred/metrics.py (percentile bootstrap, cluster resampling)",
              "cluster bootstrap resamples (0 under --fast); clusters are tokens "
              "(T2) or forward steps (T1)", "resamples")

# --------------------------------------------------------------------------
# Record geometry for the byte-value accounting
# --------------------------------------------------------------------------
PRED_T2_RECORD_BYTES = C("pred_ornith_record_bytes", 6291456, "MEASURED",
                         "dee.cpp/benchmark_reports/milestone-2.5/kaggle-forensics-latest-output/"
                         "ornith-milestone25-evidence/expert-trace.jsonl.gz "
                         "(expert_request.expert_bytes = 6291456 on all 8011 rows)",
                         "DISAGREEMENT with the task brief's carried 13,369,344 B "
                         "'Ornith-class' figure: the Ornith trace records 6 MiB/expert. "
                         "Both are reported in data/pred_byte_value.csv; the "
                         "saved/consumed ratio is invariant to record size", "bytes")
PRED_VRAM_SCENARIOS = C("pred_vram_scenarios_slots", [0, 281, 1024], "ASSUMPTION",
                        "281 = theory/constants.py ANCHOR_VRAM_SLOTS (MEASURED anchor "
                        "device residency); 0 = always-cold bound; 1024 = generous "
                        "residency ceiling (ASSUMPTION, no measured tier holds it)",
                        "would-have-been-cold is evaluated against LRU residency at "
                        "these capacities (slots)", "slots", uncertainty=(0, 1024))

# --------------------------------------------------------------------------
# Split definition -- FIXED BEFORE ANY TEST NUMBER WAS COMPUTED (pre-registered
# in this module and recorded with counts in data/pred_splits.json).
# --------------------------------------------------------------------------
PRED_SPLIT_T2 = C("pred_split_t2", {
    "train_stream_groups": [
        ["dual-cold-primary", "dual-warm-profiled", "dual-warm-reference-present"],
        ["dual-cache-disabled", "dual-cache-capacity-4", "single-t4-warm"],
    ],
    "test_stream_groups": [["dual-long-prompt"]],
    "dropped_stream_groups": [["dual-one-token"]],
}, "DERIVED",
    "dee.cpp/benchmark_reports/milestone-2.5/kaggle-forensics-latest-output/"
    "ornith-milestone25-evidence/expert-trace.jsonl.gz (stream-identity analysis: "
    "byte-identical route_selection streams across runs, sha compared per run)",
    "TRAIN = routing-stream groups 1-2 (240 layer-calls after dedup), TEST = "
    "dual-long-prompt (440 layer-calls, disjoint run_id, strictly later in trace "
    "time). Grouping is REQUIRED: runs 1-3 carry byte-identical routing streams and "
    "runs 4/5/8 are one stream, so any run-level split would leak test routing into "
    "training. dual-one-token is a strict prefix of group 1 and is dropped.")
PRED_SPLIT_T1 = C("pred_split_t1", {"train_forward_steps": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    "test_forward_steps": [11, 12, 13, 14, 15]}, "DERIVED",
                  "dee.cpp/experiments/route_pipeline/fill-live-t4x2-20260909/routed_experts.jsonl "
                  "(16 forward_steps x 43 layers; time-ordered cut)",
                  "T1 has a single stream, so the split is a strict time cut: "
                  "train = steps 0-10 (473 layer-calls), test = steps 11-15 (215). "
                  "All test calls are decode-phase.")

# --------------------------------------------------------------------------
# Model hyperparameters -- fixed a priori, never tuned on test data
# --------------------------------------------------------------------------
PRED_LR = C("pred_logreg", {"hash_dim": 1024, "loss": "log_loss", "alpha": 1e-4,
                            "max_iter": 50, "classes": 256}, "ASSUMPTION",
            "theory/pred/arms.py (scikit-learn SGDClassifier on crc32-hashed features)",
            "hashed-feature logistic regression, shared across layers; fixed a "
            "priori, no hyperparameter search on any evaluation data")
PRED_MLP = C("pred_mlp", {"emb_dim": 8, "hidden": 32, "lr": 0.01,
                          "steps": 300, "device": "cpu", "threads": 1}, "ASSUMPTION",
             "theory/pred/arms.py (torch CPU, small multi-label MLP, Adam lr=0.01)",
             "source features -> 256-way BCE set predictor; fixed a priori")
PRED_XLAYER = C("pred_xlayer", {"emb_dim": 8, "history_layers": 3, "hidden": 32, "lr": 0.01,
                                "steps": 300, "device": "cpu", "threads": 1}, "ASSUMPTION",
                "theory/pred/arms.py (torch CPU, learned set embeddings, Adam lr=0.01)",
                "within-token cross-layer arm: mean set embeddings of layers l,l-1,l-2 "
                "-> 256-way BCE; fixed a priori")

PRED_T1_RECORD_NOTE = C("pred_dsv4_record_bytes_handle", 13369344, "MEASURED",
                        "theory/constants.py D4_RECORD (fill-live-t4x2-20260909/result.json)",
                        "T1 (sealed DSv4) packed expert record size used for T1 byte "
                        "accounting; handle registered here for the pred ledger", "bytes")


def run(fast: bool = False):
    """Run the predictor lab stage (see theory/pred/lab.py)."""
    from .lab import run as _run
    return _run(fast=fast)
