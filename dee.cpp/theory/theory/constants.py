"""Every constant used by the theory layer, with provenance.

``sources.py`` re-reads the in-repo MEASURED values at run time and
regenerates ``data/provenance.csv``; this module is the human-readable
declaration and the DERIVED/ASSUMPTION arithmetic.  Nothing in the package
may import a number from anywhere else.

Provenance tags: MEASURED | DERIVED | CALIBRATED | ASSUMPTION.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

MiB = 1 << 20
GiB = 1 << 30
GiB_F = float(GiB)


@dataclass(frozen=True)
class Const:
    name: str
    value: Any
    tag: str                      # MEASURED | DERIVED | CALIBRATED | ASSUMPTION
    source: str                   # artifact path or "ASSUMPTION"
    note: str = ""
    unit: str = ""
    uncertainty: Optional[Tuple[float, float]] = None   # (low, high) where known


CONSTS: Dict[str, Const] = {}


def C(name, value, tag, source, note="", unit="", uncertainty=None) -> Const:
    c = Const(name, value, tag, source, note, unit, uncertainty)
    CONSTS[name] = c
    return c


# ==========================================================================
# 1. Model geometry — DeepSeek-V4-Flash-0731 (canonical torture model)
# ==========================================================================
SEAL = "dee.cpp/experiments/route_pipeline/fill-live-t4x2-20260909"
BR = "dee.cpp/theory/AGENT_BRIEF.md"

D4_LAYERS = C("d4_layers", 43, "MEASURED",
              f"{SEAL}/result.json (layer_count_executed=43, engine_config.num_layers=43)",
              "43 routed MoE layers + 3 MTP buckets in the full universe", "layers")
D4_EXPERTS = C("d4_experts_per_layer", 256, "MEASURED",
               f"{SEAL}/result.json (engine_config.num_experts=256)", "routed experts per layer")
D4_TOPK = C("d4_topk", 6, "MEASURED",
            f"{SEAL}/result.json (engine_config.topk=6) + routed_experts.jsonl topk field",
            "authoritative routed top-k; 1 shared expert handled off the cached path", "experts")
D4_HIDDEN = C("d4_hidden", 4096, "MEASURED", f"{SEAL}/result.json (engine_config.hidden=4096)")
D4_INTER = C("d4_inter", 2048, "MEASURED", f"{SEAL}/result.json (engine_config.inter=2048)")
D4_RECORD = C("d4_record_bytes", 13369344, "MEASURED",
              f"{SEAL}/result.json (expert_store.average_request_bytes=13369344.0; "
              "same value in tools/phase3/p3_builder.py DEE4 packing)",
              "packed FP4 record: 12.75 MiB = 12.00 MiB e2m1 payload + 0.75 MiB e8m0 scales", "bytes")

# Full universe (Phase-3 corrected): 43 main + 3 MTP/DSpark buckets = 46 buckets.
D4_UNIVERSE_RECORDS = C("d4_universe_records", 11776, "MEASURED",
                        "dee.cpp/tools/phase3/p3_bench.py (P3_BENCH_UNIVERSE=11776 = 46x256)",
                        "46 buckets x 256 experts incl. mtp.{0,1,2}", "records")
D4_UNIVERSE_BYTES = C("d4_universe_bytes", 157_437_394_944, "MEASURED",
                      "AGENTS.md CPU-4/10 ledger (157,437,394,944 B = 146.625 GiB published)",
                      "complete published packed store", "bytes")

# ==========================================================================
# 2. Trace-derived constants (sealed DSv4 journal) — recomputed by sources.py
# ==========================================================================
# NOTE: the landmark "935/2364 records ever repeat" is the count of records
# used by >=2 *distinct layer-calls* over the per-layer-call deduplicated
# stream (the engine stages each unique expert once per layer call).  Raw
# rank-slot counting gives 1098/2364.  Both are emitted in data/popularity.
D4_TRACE_RECORDS = C("d4_sealed_unique_records", 2364, "MEASURED",
                     f"{SEAL}/routed_experts.jsonl (unique (layer,expert) over 5676 rank slots)", "records")
D4_TRACE_REPEAT = C("d4_sealed_repeat_records", 935, "MEASURED",
                    f"{SEAL}/routed_experts.jsonl recomputed; matches AGENTS.md 935/2364",
                    "records used by >=2 distinct layer-calls", "records")
D4_TRACE_ACCESSES_RAW = C("d4_sealed_accesses_raw", 5676, "MEASURED",
                          f"{SEAL}/routed_experts.jsonl (946 token-rows x topk 6)", "rank slots")
D4_TRACE_ACCESSES = C("d4_sealed_accesses_dedup", 5099, "MEASURED",
                      f"{SEAL}/routed_experts.jsonl per-layer-call deduplicated stream", "cache requests")

# Empirical anchor (the number the model must predict within ~2x).
ANCHOR_TPS = C("anchor_decode_tok_s", 0.21, "MEASURED",
               f"{SEAL}/result.json (decode_tok_s=0.21; decode 15 tok / 71.479 s)", "tok/s",
               uncertainty=(0.20, 0.22))
ANCHOR_TPS_RAW = C("anchor_decode_tok_s_raw", 15 / 71.479, "DERIVED",
                   f"{SEAL}/result.json (decode_tokens=15, decode_wall_s=71.479)", "tok/s")
ANCHOR_BSSD_LO = C("anchor_bank_B_ssd_lo", 0.29 * GiB_F, "MEASURED",
                   "research/route-pipeline/STORAGE_VERDICT.md (0.29 GB/s production)", "B/s")
ANCHOR_BSSD_HI = C("anchor_bank_B_ssd_hi", 0.37 * GiB_F, "MEASURED",
                   "research/route-pipeline/STORAGE_VERDICT.md (0.37-0.51 GB/s rider rand; 0.37 qd1)", "B/s")

# Per-token byte accounting of the anchor run (whole generation, 16 emitted).
ANCHOR_STORE_BYTES_PER_TOK = C("anchor_storage_bytes_per_tok", 2073083904.0, "MEASURED",
                               f"{SEAL}/result.json (measured_roofline.storage.bytes_per_emitted_token)",
                               "33.17 GB over 16 emitted tokens incl. prefill", "bytes/tok")
ANCHOR_H2D_BYTES_TOTAL = C("anchor_h2d_bytes_total", 59413364736, "MEASURED",
                           f"{SEAL}/result.json (byte_accounting.expert_h2d_bytes_total)", "bytes")
ANCHOR_STORE_REQ_PER_TOK = C("anchor_storage_requests_per_tok", 155.0625, "MEASURED",
                             f"{SEAL}/result.json (byte_accounting.storage_requests_per_generated_token)", "req/tok")
ANCHOR_SOURCE_READ_BW = C("anchor_source_read_Bps", 129053541.84288082, "MEASURED",
                          f"{SEAL}/result.json (measured_roofline.storage.observed_source_read_bytes_per_second)",
                          "effective aggregate source-read rate of the run (lane-sum read time / wall)", "B/s")
ANCHOR_READ_MS = C("anchor_mean_read_ms", 103.59, "MEASURED",
                   f"{SEAL}/result.json (expert_store.cuda0.average_read_ms=103.5997)",
                   "single-record pread service time ~= device busy time", "ms")
ANCHOR_VRAM_SLOTS = C("anchor_vram_slots", 281, "MEASURED",
                      f"{SEAL}/result.json (engine_stats.cuda0.resident_experts=281 @ 3.5 GiB budget)",
                      "device resident packed records per GPU", "slots")
ANCHOR_HOST_SLOTS = C("anchor_host_slots", 682, "MEASURED",
                      f"{SEAL}/result.json (host_pack.cuda0.entries=682 @ 8.5 GiB per GPU)",
                      "host-tier packed records per GPU", "slots")
ANCHOR_VRAM_BUDGET = C("anchor_vram_budget", 3758096384, "MEASURED",
                       f"{SEAL}/run_config.json (cache_budget_bytes_per_gpu=3758096384 = 3.5 GiB)", "bytes")
ANCHOR_HOST_BUDGET = C("anchor_host_budget", 9126805504, "MEASURED",
                       f"{SEAL}/result.json (engine_config.host_pack_cache_bytes=9126805504 = 8.5 GiB/GPU)", "bytes")

# ==========================================================================
# 3. Working-set sim landmarks (phase2-ws-policy, causal LRU on sealed trace)
#    AGENT_BRIEF.md records these as the numbers model B must reproduce.
#    Independent recomputation on routed_experts.jsonl reproduces them exactly
#    (data/cache_sim_check.csv).
# ==========================================================================
WS_KNEE = C("ws_lru_knee_gib", 16.0, "MEASURED",
            f"{BR} L33-34 (research/phase2-ws-policy @ 95dfe0d, sealed-trace pooled LRU)",
            "causal pooled LRU RAM knee", "GiB")
WS_LRU_MINUS_MIN = C("ws_lru_minus_min_pp", 2.8, "MEASURED",
                     f"{BR} L33-34 (LRU ~2.8pp below offline MIN at the knee)", "pp")
WS_MIN_AT = C("ws_reaches_min_gib", 32.0, "MEASURED", f"{BR} L34 (LRU reaches MIN by ~32 GiB)", "GiB")
WS_VRAM_REPAIR = C("ws_vram_repair_gb_per_response", 14.48, "MEASURED",
                   f"{BR} L35 (candidate `pure last_used`, ~281-slot VRAM cap)", "GB/response")

# ==========================================================================
# 4. Non-DSv4 model specs (Phase-3 tooling, record geometry only)
# ==========================================================================
MODELS_EXTRA_NOTE = ("tools/phase3/specs/*.json; record_bytes/experts_per_layer read directly; "
                     "layer counts n_layers + n_mtp_buckets as declared")

# ==========================================================================
# 5. Hardware (bandwidth, throughput, price)
# ==========================================================================
HW = "hardware cell"
# Measured anchor platform.
T4_H2D_B = C("t4_h2d_Bps", 11436367189.722427, "MEASURED",
             f"{SEAL}/result.json (measured_roofline.pcie_h2d.observed_bytes_per_second=11.44 GB/s agg.)",
             "aggregate observed H2D copy-engine rate, 2xT4", "B/s")
T4_FLOP_PER_TOUCH = C("d4_flop_per_record_touch", 50331648, "DERIVED",
                      "2 x (2048x4096 + 2048x4096 + 4096x2048) MACs = 2 x 25,165,824, "
                      "shapes from result.json engine_config hidden=4096 inter=2048 "
                      "+ tools/phase3 mimo spec expected tensor shapes",
                      "routed-FFN FLOPs for one token through one expert record", "FLOP")
T4_ROUTED_MS_PER_TOK = C("t4_routed_ms_per_tok", 101.571578, "MEASURED",
                         f"{SEAL}/result.json (measured_roofline.routed_compute.measured_gpu_ms_per_emitted_token)",
                         "GPU routed-expert time per emitted decode token (b=1)", "ms")
T4_FLOPS_UTIL = C("t4_eff_flops_per_s", 6 * 43 * 50331648 / 0.101571578, "DERIVED",
                  f"topk 6 x 43 layers x {50331648} FLOP / 0.101571578 s "
                  f"(T4_ROUTED_MS_PER_TOK, T4_FLOP_PER_TOUCH)",
                  "effective routed-expert arithmetic rate at b=1 (launch-bound)", "FLOP/s")
T4_TOUCH_US = C("t4_touch_us", 1e6 * 0.101571578 / 258, "DERIVED",
                "T4_ROUTED_MS_PER_TOK / (topk 6 x 43 layers)",
                "mean per-record-touch cost at b=1", "us")
T4_TOUCH_OVERHEAD_US = C("t4_touch_overhead_us",
                         1e6 * (0.101571578 - 6 * 43 * 50331648 / 8.1e12) / 258, "DERIVED",
                         "(T4_ROUTED_MS_PER_TOK - W_tok/8.1e12 peak) / 258 touches",
                         "per-touch cost beyond arithmetic (dispatch + gather + setup) in "
                         "the anchor run (use_batched_experts=false): ~387 us/touch; "
                         "carried to all cells as CALIBRATED implementation cost (range "
                         "50-400 us in FALSIFICATION); uses T4 dense-FP16 peak 8.1 TFLOP/s "
                         "(ASSUMPTION)", "us")
T4_DENSE_MS_PER_TOK = C("t4_dense_ms_per_tok", 180.0, "CALIBRATED",
                        f"{SEAL}/host-profile-rows-l*.json (shared_device_ms ~3.86 ms/layer-call at b=1 "
                        "+ combine_ms ~0.2 ms) x 43 layers, attention/router from the same rows",
                        "dense path (attention + router + shared expert + combine) per decode token", "ms")
CPU_EXEC_MS_PER_EXPERT = C("cpu_exec_ms_per_expert", 2750.0, "MEASURED",
                           "AGENTS.md CPU 6/10 (portable-torch 2,750 ms/expert real-geometry bench)", "ms")

# Cell-wide compute model (roofline.py):
#   t_touch(b, cell) = t0 + b * topk * w_touch / F_peak_eff
# t0 = T4_TOUCH_OVERHEAD_US (ASSUMPTION carried across GPUs: CPU-side launch +
# gather dominates at b=1), F_peak_eff = eff * dense-tensor peak (ASSUMPTION
# eff = 0.6).  On the T4 cell the model is calibrated to reproduce the
# MEASURED 101.57 ms exactly at b=1 by construction.
COMPUTE_EFF = C("compute_eff_assumption", 0.6, "ASSUMPTION",
                "fraction of vendor dense-tensor peak realized by batched FP4->FP16 "
                "dequant GEMMs at decode batch sizes; no in-repo microbenchmark exists (data needed)",
                "fraction", uncertainty=(0.3, 0.8))

# Candidate Phase-6 cells.  T4 numbers are MEASURED; the rest are ASSUMPTION
# with the stated provenance class ("vendor PCIe/HBM spec", "Modal listing").
@dataclass(frozen=True)
class Cell:
    name: str
    n_gpu: int
    gpu_label: str
    B_ssd: float          # cold-store read bandwidth, B/s (aggregate)
    B_h2d: float          # host->device aggregate, B/s
    F_peak: float         # vendor dense-tensor peak FLOP/s (aggregate); eff in COMPUTE_EFF
    vram_bytes: float     # expert-cache budget per GPU, bytes
    price_gpu_s: float    # $/s per GPU
    price_cpu_core_s: float
    n_cpu_cores: int
    price_ram_gib_s: float
    provenance: str
    B_ssd_tag: str = "ASSUMPTION"
    B_h2d_tag: str = "ASSUMPTION"
    F_tag: str = "ASSUMPTION"
    price_tag: str = "ASSUMPTION"


GB = 1e9
TCELLS = [
    Cell("1xL4", 1, "L4", 2.5 * GB, 25.0 * GB, 121e12, 22 * GiB_F,
         0.000222, 0.0000131, 8, 0.00000222,
         "Modal price schedule in AGENT_BRIEF (ASSUMPTION); L4: 25 GB/s PCIe4 x16 H2D, 121 TFLOPS dense-FP8-tensor nominal"),
    Cell("2xL4", 2, "L4", 5.0 * GB, 50.0 * GB, 242e12, 22 * GiB_F,
         0.000222, 0.0000131, 16, 0.00000222, "same as 1xL4, two devices"),
    Cell("1xA10", 1, "A10", 3.0 * GB, 25.0 * GB, 125e12, 22 * GiB_F,
         0.000306, 0.0000131, 8, 0.00000222,
         "Modal price schedule (ASSUMPTION); A10: 25 GB/s PCIe4 x16, 125 TFLOPS dense-FP8-tensor nominal"),
    Cell("1xL40S", 1, "L40S", 6.0 * GB, 50.0 * GB, 362e12, 48 * GiB_F,
         0.000542, 0.0000131, 16, 0.00000222,
         "Modal price schedule (ASSUMPTION); L40S: 50 GB/s PCIe4 x16, 362 TFLOPS dense-FP8-tensor nominal"),
    Cell("1xRTXPRO6000", 1, "RTX PRO 6000", 8.0 * GB, 75.0 * GB, 700e12, 96 * GiB_F,
         0.000842, 0.0000131, 16, 0.00000222,
         "Modal price schedule (ASSUMPTION); Blackwell WS: 75 GB/s PCIe5 x16, 700 TFLOPS dense-FP4-tensor nominal"),
    Cell("CPU-only", 0, "none", 3.0 * GB, 0.0, 0.0, 0.0,
         0.0, 0.0000131, 32, 0.00000222,
         "no GPU; expert execution cost from CPU_EXEC_MS_PER_EXPERT (MEASURED portable-torch)"),
]
# The anchor cell (2xT4): B_ssd / B_h2d / F all MEASURED or DERIVED from
# fill-live-t4x2-20260909 (see ANCHOR_* constants).
T4_CELL = Cell("2xT4", 2, "T4", 0.33 * GB, T4_H2D_B.value, 8.1e12,
               3.5 * GiB_F, 0.0, 0.0000131, 8, 0.00000222,
               "MEASURED anchor cell (fill-live-t4x2-20260909)",
               B_ssd_tag="MEASURED", B_h2d_tag="MEASURED",
               F_tag="DERIVED(calibrated to MEASURED routed ms/tok)", price_tag="ASSUMPTION")

ALL_CELLS = [T4_CELL] + TCELLS

# Modal price schedule (AGENT_BRIEF L38-40) — explicitly ASSUMPTION.
MODAL_PRICES = C("modal_price_schedule", "see Cell entries", "ASSUMPTION", BR,
                 "L4 $0.000222/s, A10 $0.000306/s, L40S $0.000542/s, "
                 "RTX PRO 6000 $0.000842/s, CPU $0.0000131/core/s, RAM $0.00000222/GiB/s, "
                 "volume $0.09/GiB/mo, first 1 TiB free")

# Dense-residency baseline for the break-even analysis (conservative).
DENSE_BASELINE = C("dense_baseline", {
    "resident_bytes": 568 * GiB_F,        # DSv4-Flash bf16 routed+dense resident
    "gpu": "8xH100-80GB-class",
    "price_gpu_s": 0.003223,              # ASSUMPTION, conservative H100 cloud $/GPU/s
    "n_gpu": 8,
    "achieved_tok_s": 200.0,              # ASSUMPTION, conservative achieved decode for 13B-active MoE
}, "ASSUMPTION", "AGENT_BRIEF L110 + conservative cloud pricing",
    "DSv4-Flash bf16 ~568 GiB resident on H100-class hardware; $3.223/GPU/h; "
    "200 tok/s achieved decode is deliberately conservative for a 13B-active model")

# ==========================================================================
# 6. Serving / economics assumptions
# ==========================================================================
SERVE_ASSUMPTIONS = C("serve_assumptions", {
    "arrival": "Poisson(lambda)",
    "service": "decode batch-of-b, exponential-ish durations (M/M/1 per replica)",
    "request_tokens": 512,
    "prefill_positions": 512,
    "cross_request_popularity": "mixture of per-request popularity over K~Poisson(lambda*W) concurrent",
}, "ASSUMPTION", "AGENT_BRIEF L102-106 (structure); numbers chosen here",
    "request length 512 decode + 512 prefill positions; W = mean response time")

VOLUME_PRICE_GIB_MO = C("volume_price_gib_month", 0.09, "ASSUMPTION", BR,
                        "first 1 TiB free", "$/GiB/mo")

# ==========================================================================
# 7. Fitted popularity parameters (CALIBRATED) — populated by popularity.py,
#    declared here so downstream modules have typed handles.  The values are
#    regenerated deterministically; see data/popularity_fits.csv.
# ==========================================================================
POPPY_FIT_NOTE = C("popularity_fit", "generated", "CALIBRATED",
                   "theory/data/popularity_fits.csv (regenerated by theory.popularity)",
                   "Zipf-Mandelbrot (s,q), lognormal (mu,sigma), uniform per layer + pooled")

# Model transfer assumptions for the non-traced models (B): popularity shape
# s is carried from DSv4-Flash per-layer pooled fits; this is an ASSUMPTION.
S_TRANSFER = C("s_transfer_assumption", "DSv4-Flash pooled s applied to MiMo/MiniMax",
               "ASSUMPTION", "AGENT_BRIEF L82 ('H(M) curves for all four store specs')",
               "no router traces exist for MiMo-V2.6 / MiniMax-M3 in-repo (data needed, "
               "see FALSIFICATION.md); popularity shape is transferred from DSv4-Flash, "
               "NOT asserted to be universal")
