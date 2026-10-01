"""Sim-layer constants (Phase-7 component A) — every number tagged + sourced.

Registered through ``theory.constants.C`` so ``data/provenance.csv`` picks them
up automatically on the next ``run_all provenance``.  Nothing here is invented:
MEASURED values cite the artifact line, DERIVED shows its arithmetic, CALIBRATED
states its residual, ASSUMPTION states its range.
"""
from __future__ import annotations

from ..constants import C, GiB_F, MiB

SEAL = "dee.cpp/experiments/route_pipeline/fill-live-t4x2-20260909"
ENG = "dee.cpp/src/engine.cpp + include/dee/{engine,vram_cache,host_pack_cache}.h"

# --------------------------------------------------------------------------
# 1. Pipeline mechanics transcribed from the artifacts (MEASURED / DERIVED)
# --------------------------------------------------------------------------
SIM_FILL_LANES = C(
    "sim_fill_lanes", 3, "MEASURED",
    f"{SEAL}/result.json (expert_store.cuda0.max_source_read_lanes=3; "
    "engine_config.source_read_lanes=3)",
    "concurrent pread lanes per GPU host-pack fill (HostPackCache::set_fill_lanes)",
    "lanes")

SIM_FILL_QD = C(
    "sim_fill_queue_depth", 6, "MEASURED",
    f"{SEAL}/result.json (engine_config.source_read_queue_depth=6); "
    "engine.cpp:3142 loops get_batch in sub-batches of queue_depth and blocks "
    "per sub-batch (HostSpanGuard FillWait)",
    "prepare_fp4_experts issues fills in blocking sub-batches of 6 requests",
    "requests")

SIM_LAYER_SPLIT = C(
    "sim_layer_split", 22, "MEASURED",
    f"{SEAL}/result.json model_cuda_stage_profile.layers[]: layers 0-21 -> "
    "cuda:0, layers 22-42 -> cuda:1 (device field of every row)",
    "static model-parallel layer->GPU assignment; the routed_experts.jsonl "
    "device column confirms it for all 688 records", "layer")

SIM_STAGING_ORDER = C(
    "sim_staging_order", "ascending expert id", "DERIVED",
    f"{ENG} engine.cpp:565-567 (active_experts built by iterating expert "
    "0..num_experts-1 over non-empty groups) + :592 stage_expert(..., K - i)",
    "per layer-call the engine stages the DISTINCT experts in ascending id "
    "order with priority = K - i (position within the call); this ordering "
    "is what makes the priority term bite and is confirmed by an exact "
    "counter replay (see SIM_VS_THEORY.md)", "policy")

SIM_HOST_CONSULT = C(
    "sim_host_consult", "eager whole-call", "DERIVED",
    f"{ENG} engine.cpp:3120-3253 (prepare_fp4_experts calls "
    "HostPackCache::get_batch on every chunk member BEFORE any VRAM staging) "
    "+ :3267 (get_staging_fp4 re-consults the pack on both hit and miss paths)",
    "the host pack is consulted and, on a miss, FILLED for every requested "
    "record even when the record is VRAM-resident.  This is why "
    "per_token_accounting.storage_requests == host_pack_misses token-for-token "
    "and why decode wall correlates with storage_requests (r=0.987) but not "
    "with cold_loads (r=-0.07)", "policy")

SIM_HOST_POLICY = C(
    "sim_host_policy", "LRU, batch-protected", "MEASURED",
    f"{ENG} src/host_pack_cache.cpp:224-314 (get_batch: in-batch duplicate -> "
    "hit, in-batch keys protected from eviction, unique misses admitted at "
    "MRU front in request order)",
    "682 slots/GPU at 8.5 GiB", "policy")

SIM_VRAM_POLICY_ANCHOR = C(
    "sim_vram_policy_anchor", "RankPriority (last_used + priority*2^20)",
    "MEASURED",
    f"{ENG} include/dee/vram_cache.h:244-247 (PRIORITY_WEIGHT = 1<<20, "
    "EvictionPolicy::RankPriority is the default the anchor ran with)",
    "Recency (pure last_used) is the Phase-2 recommended repair and is "
    "simulated as the counterfactual arm", "policy")

SIM_FILL_COALESCING = C(
    "sim_fill_coalescing", "in-flight + in-batch", "MEASURED",
    f"{ENG} src/async_prefetcher.cpp:455-473 (in-batch duplicate counted, "
    "find_inflight coalesces a request whose H2D is still in flight) + "
    "src/host_pack_cache.cpp:245-256 (in-batch duplicate -> counted hit, "
    "one fill serves it)",
    "anchor run reports engine_stats duplicate_requests=0 and inflight_hits=0, "
    "so both mechanisms are inert there and only bind on longer streams",
    "policy")

# --------------------------------------------------------------------------
# 2. Service-time model (MEASURED quantiles -> DERIVED quantile function)
# --------------------------------------------------------------------------
SIM_READ_MS_MEAN = C(
    "sim_read_ms_mean", 103.5997, "MEASURED",
    f"{SEAL}/result.json (expert_store.cuda0.average_read_ms=103.599670928777; "
    "cuda1 103.58979490284142)",
    "per-record pread service time for one 13,369,344 B record, POOLED over "
    "prefill + decode (= 257,020 ms fill_worker_ms / 2,481 reads)", "ms")

SIM_READ_MS_DECODE = C(
    "sim_read_ms_decode", 104111.523 / 1252, "DERIVED",
    f"{SEAL}/result.json per_token_accounting (decode rows): "
    "sum(source_read_wall_ms)=104,111.5 ms / sum(storage_requests)=1,252 reads "
    "= 83.156 ms/read",
    "decode-phase per-read service is 20% FASTER than the pooled mean; the "
    "phase gap (83.2 vs 124.4 ms) is unexplained by the artifacts and is "
    "reported as an open measurement gap in SIM_VS_THEORY.md", "ms")

SIM_READ_MS_PREFILL = C(
    "sim_read_ms_prefill", 152908.536 / 1229, "DERIVED",
    f"{SEAL}/result.json per_token_accounting step 0: "
    "source_read_wall_ms=152,908.536 ms / storage_requests=1,229 reads "
    "= 124.417 ms/read",
    "prefill-phase per-read service; the pooled 103.5997 = "
    "(1229*124.417 + 1252*83.156)/2481 checks exactly", "ms")

SIM_READ_MS_P50 = C(
    "sim_read_ms_p50", 104.49955, "MEASURED",
    f"{SEAL}/result.json (expert_store.cuda0.p50_read_ms=104.49955)", "ms")

SIM_READ_MS_P95 = C(
    "sim_read_ms_p95", 163.912439, "MEASURED",
    f"{SEAL}/result.json (expert_store.cuda0.p95_read_ms=163.912439)", "ms")

SIM_READ_MS_MAX = C(
    "sim_read_ms_max", 168.568131, "MEASURED",
    f"{SEAL}/result.json (expert_store.cuda0.max_read_ms=168.568131)", "ms")

SIM_READ_STREAM_MIB_S = C(
    "sim_read_stream_mib_s", 123.0699, "MEASURED",
    f"{SEAL}/result.json (expert_store.cuda0.read_bandwidth_mib_s=123.0698889; "
    "cuda1 123.0816222)",
    "per-lane positional-read bandwidth; 3 lanes x 123.07 MiB/s = 369.2 MiB/s "
    "= 0.361 GiB/s, consistent with the Phase-1 0.29-0.37 GiB/s bank ceiling",
    "MiB/s")

SIM_READ_MS_FLOOR = C(
    "sim_read_ms_floor", 86.9, "DERIVED",
    "solved so the piecewise-linear quantile function through the MEASURED "
    "(p50=104.500, p95=163.912, max=168.568) has mean 103.5997: "
    "0.25*t_min + 81.877 = 103.5997 -> t_min = 86.89 ms",
    "implied fastest read; matches the mincore page-cache-resident tail "
    "(expert_store.mincore_resident_bytes 504,909,824 of 18,583,388,160 B)",
    "ms")

SIM_READ_QUANTILE_SHAPE = C(
    "sim_read_quantile_shape", "piecewise-linear quantile function",
    "ASSUMPTION",
    "interpolation between the three MEASURED quantiles (p50, p95, max) and "
    "the DERIVED floor; the service-time SHAPE between quantiles is not "
    "measured in-repo",
    "the simulated read-time distribution matches the measured mean, p50, p95 "
    "and max exactly; only the within-quantile shape is assumed (range: a "
    "two-point mixture over [86.9, 104.5] U [150, 168.6] would shift the token "
    "wall by < 3% because the makespan over 3 lanes averages >= 2 draws)",
    "shape")

# --------------------------------------------------------------------------
# 3. Compute / dense stage costs (DERIVED from MEASURED)
# --------------------------------------------------------------------------
SIM_TOUCH_MS = C(
    "sim_touch_ms", 1e3 * 0.101571578 / 258, "DERIVED",
    "T4_ROUTED_MS_PER_TOK 101.571578 ms / (topk 6 x 43 layers) = 0.39369 ms; "
    "the same value as constants.T4_TOUCH_US",
    "per expert-record compute touch at b=1 on 2xT4 (dequant + 3 GEMMs + "
    "dispatch)", "ms")

SIM_DENSE_CALL_MS = C(
    "sim_dense_call_ms", 13.272, "DERIVED",
    f"{SEAL}/result.json model_cuda_stage_profile.per_start_pos_ms/{{7..21}}: "
    "(attention_prep + attention_state + ffn_prep + ffn_hc_post + output_cast "
    "+ shared_expert + router) / 43 layer-calls, averaged over the 15 decode "
    "tokens (per-token values 12.64-15.67 ms/call)",
    "non-routed per layer-call cost (attention + shared expert + combine + "
    "prep) at b=1", "ms/call")

SIM_DENSE_PREFILL_CALL_MS = C(
    "sim_dense_prefill_call_ms", 20933.9 / 43.0, "DERIVED",
    f"{SEAL}/result.json model_cuda_stage_profile.per_start_pos_ms/0: "
    "same non-routed sum 20,933.9 ms / 43 calls = 486.835 ms/call (the shared "
    "expert alone is 19,377.8 ms of it)",
    "prefill (token_rows=7) non-routed per layer-call cost", "ms/call")

SIM_TOUCH_CALL_MS = C(
    "sim_touch_call_ms", 0.101571578e3 / 43.0, "DERIVED",
    f"{SEAL}/result.json measured_roofline.routed_compute."
    "measured_gpu_ms_per_emitted_token = 101.571578 ms / 43 layer-calls "
    "= 2.36213 ms per call = topk 6 x 0.39369 ms per record touch",
    "routed-expert GPU cost of one layer-call at b=1", "ms/call")

# --------------------------------------------------------------------------
# 4. Timing CALIBRATION — fit/holdout split is declared here and in the report.
#    FIT: exactly one scalar (SIM_ORCH_MS, per layer-call orchestration slack).
#         Fitted on the SINGLE pooled mean decode wall of the anchor run.
#    HOLDOUT (gate holds over these): per-token wall shape (per-token +-50%,
#         Spearman), every cache counter (storage_requests, host share,
#         cold_loads, resident_hits, h2d_copies), the prefill wall, and the
#         whole-generation storage_requests_per_emitted_token (155.06).
#    The cache half of the gate is a pure REPLAY: no cache parameter is fitted.
# --------------------------------------------------------------------------
SIM_ORCH_MS = C(
    "sim_orch_ms_per_call", 0.0, "CALIBRATED",
    "fitted scalar: per layer-call orchestration slack (host scheduling, route "
    "D2H, python/native boundary), fitted ONCE to the pooled mean decode wall "
    "of the anchor run and then frozen",
    "residual after the fit is reported per token in data/sim_anchor_replay.json "
    "(target residual: |mean decode wall| error 0% by construction; per-token "
    "residual is a HOLDOUT and is reported unadjusted)",
    "ms/call")

# --------------------------------------------------------------------------
# 5. Cross-check tolerances — DECLARED BEFORE the cross-check was run.
# --------------------------------------------------------------------------
SIM_XCHECK_TOL = C(
    "sim_crosscheck_tolerance", {
        "hit_rate_abs_pp": 5.0,
        "cold_records_per_tok_rel": 0.15,
        "token_time_rel": 0.25,
        "grid_pass_fraction_min": 0.80,
    }, "ASSUMPTION",
    "declared in theory/sim/constants.py before theory.sim.crosscheck was run "
    "(method choice, not a measurement)",
    "sim vs theory.cache / theory.roofline agreement bands on identical "
    "synthetic streams; a grid point passes only if all three quantities are "
    "inside their band", "mixed")

# --------------------------------------------------------------------------
# 6. Synth-mode generator (fitted popularity + temporal structure)
# --------------------------------------------------------------------------
SIM_SYNTH_SEED = C("sim_synth_seed", 20260916, "ASSUMPTION",
                   "fixed numpy Generator seed for every synth stream",
                   "any fixed seed; determinism is the requirement", "seed")

SIM_SYNTH_REQUESTS = C("sim_synth_requests_full", 100_000, "ASSUMPTION",
                       "Phase-7 brief requirement (>= 100,000-request streams)",
                       "--fast shrinks this to 10,000 without changing conclusions",
                       "requests")

SIM_SYNTH_STICKY = C(
    "sim_synth_sticky", 0.375, "DERIVED",
    "CACHE1_ANALYSIS.md 2 (measured consecutive-token expert overlap 37.5%) "
    "used as the within-layer temporal reuse probability of the generator",
    "probability that a layer-call repeats a member of the previous call at "
    "the same layer (decode-step self-correlation); the cross-layer lift "
    "(12.2x over marginal, theory.temporal) is reproduced by a second, "
    "within-token channel", "probability")

SIM_SYNTH_CROSSLAYER = C(
    "sim_synth_crosslayer_lift", 12.2, "MEASURED",
    "theory/data/temporal_crosslayer.csv (mean lift over the marginal, T1)",
    "P(j in S_{l+1} | i in S_l) mean lift; used to size the generator's "
    "within-token cross-layer coupling", "x marginal")
