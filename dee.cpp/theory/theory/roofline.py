"""D. Throughput bound: the pipeline roofline of the NVMe->RAM->VRAM hierarchy.

    T_tok(b, M_host, M_vram) = max( cold_bytes/B_SSD,
                                    spill_bytes/B_H2D,
                                    expert_compute(b)/F,
                                    dense_path(b) )

with

    cold_bytes  = P_rec * L * U(b) * (1 - H_v(M_v)) * (1 - H_h(M_h))
    spill_bytes = P_rec * L * U(b) * (1 - H_v(M_v))

    U(b)        = E[distinct records requested by one layer-call]
                = sum_i (1 - (1 - p_i)^(b*k))          (batch-overlap correction)

so the same expert missed once serves the whole batch — never k*(1-H).

H_v / H_h are Che TTL hit rates at the VRAM and host capacities.  The
two-level product (1-H_v)(1-H_h) assumes conditional independence of the
two levels given a miss (stated approximation; error quantified against the
replay in cache.py).

T_tok is a lower bound on latency: it is the pipeline-max (perfect overlap of
the four resources).  The anchor run does not achieve perfect overlap; the
achieved fraction eta = T_bound / T_measured is reported, not hidden, and is
carried as a CALIBRATED range into the other cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cache import che_hit_curve, che_tau
from .popularity import fit_zipf_mandelbrot, profile_from_stream, zipf_mandelbrot_p
from .util import dlog, figure, savefig, write_csv, write_json

GIB = float(1 << 30)
GIB_PER_S = GIB          # bandwidth units: bytes/s; 0.29-0.37 GiB/s as bytes/s


@dataclass
class ModelGeom:
    name: str
    label: str
    layers: int
    experts_per_layer: int
    topk: int
    record_bytes: int
    flops_per_touch: float     # FLOPs for one token through one record
    source: str

    @property
    def touches_per_token(self) -> int:
        return self.layers * self.topk

    @property
    def universe_records(self) -> int:
        return self.layers * self.experts_per_layer

    @property
    def universe_bytes(self) -> int:
        return self.universe_records * self.record_bytes


def load_geoms() -> Dict[str, ModelGeom]:
    from .constants import T4_FLOP_PER_TOUCH
    from .sources import load_stores

    out: Dict[str, ModelGeom] = {}
    for key, s in load_stores().items():
        if key == "dsv4_flash":
            flops = float(T4_FLOP_PER_TOUCH.value)
        elif key == "minimax_m3":
            flops = float(2 * (3072 * 6144 + 3072 * 6144 + 6144 * 3072))
        elif key == "mimo_v2_flash":
            flops = float(2 * (2048 * 2048 + 2048 * 2048 + 4096 * 1024))
        else:  # mimo_v2_pro
            flops = float(2 * (2048 * 3072 + 2048 * 3072 + 6144 * 1024))
        out[key] = ModelGeom(key, s.label, s.layers, s.experts_per_layer, s.topk,
                             s.record_bytes, flops, s.source)
    return out


@dataclass
class PopModel:
    """Record inclusion probabilities for one layer-call.

    The router's top-k picks are *distinct* experts (sampling without
    replacement), so the correct per-record quantity is the inclusion
    probability pi_i = P(record i is requested by one layer-call), with
    sum_i pi_i = k per layer.  For a batch of b rows a record is requested
    with probability u_i(b) = 1 - (1 - pi_i)^b — this is the batch-overlap
    correction: the same expert missed once serves the whole batch.
    """
    pi: np.ndarray          # (layers, experts_per_layer) inclusion probability
    s: float
    q: float
    source: str

    def u(self, b: int) -> np.ndarray:
        return 1.0 - (1.0 - self.pi) ** b


def build_pop(geom: ModelGeom, stream=None, s_override=None, q_override=None) -> PopModel:
    """Popularity model for a geometry.

    Measured for dsv4_flash (sealed trace: per-layer-call inclusion
    frequencies).  For the other stores the DSv4-Flash (s,q) is transferred —
    an explicit ASSUMPTION (constants.S_TRANSFER) — and pi_i = k * p_i with
    p_i the Zipf-Mandelbrot rank curve within layer.  No router trace exists
    for MiMo-V2.6 / MiniMax-M3 in-repo (see FALSIFICATION.md, data needed).
    """
    if geom.name != "dsv4_flash":
        n_cat = geom.universe_records
        s = s_override if s_override is not None else 0.6
        q = q_override if q_override is not None else 1.0
        p_full = zipf_mandelbrot_p(s, q, n_cat)
        p = p_full.reshape(geom.layers, geom.experts_per_layer)
        pi = np.minimum(1.0, geom.topk * p)
        src = "zipf_mandelbrot_transferred (ASSUMPTION)"
    else:
        from .sources import load_sealed_stream

        sealed = load_sealed_stream() if stream is None else stream
        n_calls = {}
        present = {}
        for (tok, layer) in sealed.keys:
            n_calls[layer] = n_calls.get(layer, 0) + 1
            for e in sealed.sets[(tok, layer)]:
                present[(layer, e)] = present.get((layer, e), 0) + 1
        pi = np.zeros((geom.layers, geom.experts_per_layer), dtype=float)
        for (layer, e), v in present.items():
            pi[layer, e] = v / max(1, n_calls[layer])
        prof = profile_from_stream(sealed.stream(), geom.layers,
                                   geom.experts_per_layer, "dsv4", "sealed")
        z = fit_zipf_mandelbrot(prof)
        s, q = (s_override if s_override is not None else z["s"],
                q_override if q_override is not None else z["q"])
        src = "sealed_dsv4_empirical_inclusion"
    return PopModel(pi, s, q, src)


def dense_ms_for(cell, geom: ModelGeom) -> float:
    """Dense-path ms/token scaled from the T4 measurement.

    Scales linearly with MoE layer count (43 on DSv4-Flash) and inversely
    with GPU arithmetic rate relative to the T4 (ASSUMPTION: the dense path
    is bandwidth/launch bound and tracks peak rate within a factor ~2).
    The CPU-only cell keeps the T4 rate (dense ops run on CPU there, which is
    if anything slower).
    """
    from .constants import T4_DENSE_MS_PER_TOK

    base = float(T4_DENSE_MS_PER_TOK.value) * (geom.layers / 43.0)
    if cell.n_gpu == 0:
        return base
    return base * (8.1e12 / max(1.0, cell.F_peak))


def per_call_distinct(pop: PopModel, b: int, k: int) -> float:
    """E[distinct records requested per token (all layers)] with batch b."""
    return float(pop.u(b).sum())


_CHE_CACHE: Dict[Tuple[int, int, int], "object"] = {}


def _request_weights(pop: PopModel, geom: ModelGeom, b: int):
    """Per-record request probabilities u_i(b) per layer-call, flattened."""
    u = pop.u(b).reshape(-1)
    tot = u.sum()
    return u, (u / max(1e-30, tot))


def _che_table(pop: PopModel, geom: ModelGeom, b: int):
    """Che TTL table cached per (popularity, geometry, batch)."""
    key = (id(pop), id(geom), b)
    tab = _CHE_CACHE.get(key)
    if tab is None:
        from .cache import CheTable

        _u, w = _request_weights(pop, geom, b)
        tab = CheTable(w)
        _CHE_CACHE[key] = tab
    return tab


def hit_rates(pop: PopModel, geom: ModelGeom, m_v: float, m_h: float,
              b: int = 1) -> Dict[str, float]:
    """Two-level Che TTL hit rates at the VRAM and host capacities.

    Coupled form (derived in THEORY.md section 3): the VRAM level sees the
    full request stream with per-record rate u_i; its steady-state per-record
    hit probability is h_v,i = 1 - exp(-w_i tau_v) at its TTL tau_v.  The host
    level sees the *miss stream* u_i (1 - h_v,i), so its TTL is solved on
    those weights and its per-record hit probability is conditional on a
    VRAM miss.  The miss-through per record is (1 - h_v,i)(1 - h_h,i).
    """
    from .cache import CheTable

    u, w = _request_weights(pop, geom, b)
    tab_v = _che_table(pop, geom, b)
    tau_v = tab_v.tau_at_capacity(float(max(1, int(m_v)))) if m_v > 0 else 0.0
    h_v = 1.0 - np.exp(-w * tau_v)
    u_h = u * (1.0 - h_v)
    tot_h = u_h.sum()
    if tot_h <= 0 or m_h <= 0:
        h_h = np.zeros_like(h_v)
    else:
        w_h = u_h / tot_h
        tab_h = CheTable(w_h)
        tau_h = tab_h.tau_at_capacity(float(max(1, int(m_h))))
        h_h = 1.0 - np.exp(-w_h * tau_h)
    tot = u.sum()
    H_v = float((u * h_v).sum() / max(1e-30, tot))
    miss_v = float((u * (1.0 - h_v)).sum())
    miss_h = float((u * (1.0 - h_v) * (1.0 - h_h)).sum())
    H_h = 1.0 - miss_h / max(1e-30, miss_v)          # conditional on a VRAM miss
    return {"H_v": H_v, "H_h": float(H_h),
            "U_total_per_token": float(tot),
            "miss_through_fraction": miss_h / max(1e-30, tot)}


def finite_hit_rates(pop: PopModel, geom: ModelGeom, m_v: float, m_h: float,
                     batches: Sequence[int]) -> Dict[str, float]:
    """Finite-window Che hit rates over a generation with per-token batches.

    ``batches`` is the batch size of each token in the horizon (the anchor
    generation is [7] prefill + [1]*15 decode).  Record j is requested with
    probability u_j(b_t) at token t, so its request count over the horizon is
    sum_t u_j(b_t); compulsory first touches are misses at both levels and
    the host level sees exactly the device-level misses.
    """
    from .cache import CheTable

    counts = np.zeros_like(pop.pi, dtype=float)
    for bt in batches:
        counts += pop.u(int(bt))
    counts = counts.reshape(-1)
    _u, w = _request_weights(pop, geom, 1)
    tab = _che_table(pop, geom, 1)
    _hits, _tot, _miss, nv, hv = tab.level_counts(counts, m_v)
    miss_v = np.maximum(nv - hv, 0.0)
    hits_h, tot_h, miss_h, nh, _hh = tab.level_counts(miss_v, m_h)
    return {
        "H_v": float(hv.sum() / max(1.0, nv.sum())),
        "H_h": float(hits_h / max(1.0, tot_h)),
        "n_v": float(nv.sum()), "n_h": float(tot_h),
        "miss_v": float(miss_v.sum()), "miss_h": float(miss_h.sum()),
    }


@dataclass
class RoofPoint:
    cell: str
    model: str
    batch: int
    host_gib: float
    vram_slots: int
    cold_bytes_per_tok: float
    spill_bytes_per_tok: float
    cold_records_per_tok: float
    t_storage: float
    t_h2d: float
    t_compute: float
    t_dense: float
    t_bound: float
    t_pred: float
    tps_bound: float
    tps_pred: float
    limiter: str
    H_v: float
    H_h: float
    U_per_layer: float


def roofline_point(geom: ModelGeom, pop: PopModel, cell, b: int,
                   host_records_per_gpu: float,
                   dense_ms_per_tok: float, t0_us: float, flops_eff: float,
                   vram_slots_per_gpu: Optional[int] = None,
                   dense_batch_fixed_frac: float = 0.35,
                   horizon_tokens: int = 512,
                   finite: bool = False,
                   batches: Optional[Sequence[int]] = None,
                   eta_storage: float = 0.225,
                   eta_h2d: float = 0.5) -> RoofPoint:
    """Token-latency bound and overlap-corrected prediction.

    ``t_bound`` is the pipeline max over the four resources — a hard lower
    bound on latency assuming perfect overlap of storage fill, H2D, expert
    arithmetic and the dense path.  ``t_pred`` adds the *unhidden* fill
    service back sequentially:

        t_pred = t_dense + t_compute
                 + (1 - eta_storage) * n_cold * t_service
                 + (1 - eta_h2d) * t_h2d

    because record fills are demand-gated by the layer dependency chain
    (<= k legally-known experts per layer call: research/route-pipeline/
    STORAGE_VERDICT.md), so their service time cannot be pipelined away the
    way a bulk transfer could.  eta_storage is CALIBRATED on the anchor run
    (0.225: 22.5% of fill service hidden behind compute/dense there).

    Accounting per emitted token at batch b:
      D(b)  = sum_i [1 - (1 - pi_i)^b]   expected distinct records requested
      M_v   = D(b) (1 - H_v)             device fills (host-tier lookups)
      M_h   = M_v (1 - H_h)              cold SSD record reads
      spill = P_rec * M_v                 H2D bytes
    (Never k*(1-H): one miss serves the whole batch.)
    """
    L, k, wrec = geom.layers, geom.topk, geom.record_bytes
    n_gpu = max(1, cell.n_gpu)
    v_slots = (vram_slots_per_gpu if vram_slots_per_gpu is not None
               else int(cell.vram_bytes / wrec))
    h_slots = host_records_per_gpu

    D = per_call_distinct(pop, b, k)               # distinct records / token
    U = D / max(1, L)                              # per-layer average
    if finite:
        bl = list(batches) if batches else [b] * int(horizon_tokens)
        hr = finite_hit_rates(pop, geom, n_gpu * v_slots, n_gpu * h_slots, bl)
        n_tok = len(bl)
        misses_v = hr["miss_v"] / max(1, n_tok)    # device fills per token
        cold_recs = hr["miss_h"] / max(1, n_tok)   # SSD record reads per token
    else:
        hr = hit_rates(pop, geom, n_gpu * v_slots, n_gpu * h_slots, b)
        misses_v = D * (1.0 - hr["H_v"])           # host fills per token
        cold_recs = D * hr.get("miss_through_fraction",
                               (1.0 - hr["H_v"]) * (1.0 - hr["H_h"]))
    cold = wrec * cold_recs
    spill = wrec * misses_v

    # Storage feed model.  Reads are demand-gated by the layer dependency
    # chain (storage verdict: <= k legal reads per layer call, 96% device
    # busy, pread ~= service).  One record's service time is the device busy
    # time for its bytes, so the demand path is (records/token) x service,
    # independent of lane count.  cold_bytes/B is only a ceiling.
    t_service = wrec / max(1.0, cell.B_ssd)        # one record read (device busy)
    t_demand = cold_recs * t_service
    t_bandwidth = cold / max(1.0, cell.B_ssd)
    t_storage = max(t_demand, t_bandwidth)

    t_h2d = spill / max(1.0, cell.B_h2d) if cell.B_h2d > 0 else 0.0
    if cell.n_gpu == 0:
        t_compute = L * k * 2750.0e-3 / max(1, b)      # every touch executed on CPU
    else:
        launch = (D * (t0_us * 1e-6)) / max(1, b)      # one launch per distinct record
        arith = L * k * (geom.flops_per_touch / max(1.0, flops_eff))
        t_compute = launch + arith
    t_dense = (dense_ms_per_tok * 1e-3) * (
        dense_batch_fixed_frac + (1.0 - dense_batch_fixed_frac) * b) / max(1, b)

    terms = {"storage": t_storage, "h2d": t_h2d, "compute": t_compute,
             "dense": t_dense}
    limiter = max(terms, key=terms.get)
    t_bound = max(terms.values())
    t_pred = (t_dense + t_compute
              + (1.0 - eta_storage) * t_demand
              + (1.0 - eta_h2d) * t_h2d)
    return RoofPoint(
        cell=cell.name, model=geom.name, batch=b,
        host_gib=host_records_per_gpu * wrec / GIB,
        vram_slots=v_slots, cold_bytes_per_tok=cold, spill_bytes_per_tok=spill,
        cold_records_per_tok=cold_recs,
        t_storage=t_storage, t_h2d=t_h2d, t_compute=t_compute, t_dense=t_dense,
        t_bound=t_bound, t_pred=t_pred,
        tps_bound=1.0 / t_bound if t_bound > 0 else float("inf"),
        tps_pred=1.0 / t_pred if t_pred > 0 else float("inf"),
        limiter=limiter, H_v=hr["H_v"], H_h=hr["H_h"], U_per_layer=U)


# ==========================================================================
# Anchor check
# ==========================================================================
def anchor_check() -> Dict[str, object]:
    """Predict the measured anchor (~0.21 tok/s at 0.29-0.37 GiB/s on 2xT4).

    Two miss models are scored and both are reported:
      * steady-state Che TTL (the closed form of section B) — invalidly
        assumes a long stationary window;
      * finite-window Che (compulsory first touches counted as misses) —
        the correct model for a 16-token generation.
    The required overlap factor eta_storage is calibrated once on the
    0.33 GiB/s case and then applied unchanged to the other two bandwidth
    cases, which is what makes the check a prediction rather than a fit.
    """
    from .constants import (ANCHOR_BSSD_HI, ANCHOR_BSSD_LO, ANCHOR_HOST_SLOTS,
                            ANCHOR_TPS, ANCHOR_TPS_RAW, ANCHOR_VRAM_SLOTS,
                            COMPUTE_EFF, T4_CELL, T4_DENSE_MS_PER_TOK,
                            T4_TOUCH_OVERHEAD_US, Cell)

    geoms = load_geoms()
    g = geoms["dsv4_flash"]
    pop = build_pop(g)
    flops_eff = COMPUTE_EFF.value * T4_CELL.F_peak
    b = 1
    v_slots = int(ANCHOR_VRAM_SLOTS.value)          # per GPU
    h_slots = int(ANCHOR_HOST_SLOTS.value)          # per GPU
    horizon = 16

    cases = [("0.29 GiB/s", ANCHOR_BSSD_LO.value),
             ("0.37 GiB/s", ANCHOR_BSSD_HI.value),
             ("0.33 GiB/s mid", 0.33 * GIB)]

    def mk_cell(b_ssd):
        c = T4_CELL
        return Cell(c.name, c.n_gpu, c.gpu_label, b_ssd, c.B_h2d, c.F_peak,
                    c.vram_bytes, c.price_gpu_s, c.price_cpu_core_s,
                    c.n_cpu_cores, c.price_ram_gib_s, c.provenance,
                    c.B_ssd_tag, c.B_h2d_tag, c.F_tag, c.price_tag)

    t_measured = 71.479 / 15

    # --- measured accounting split prefill vs decode (MEASURED)
    from .sources import load_sealed_result

    res = load_sealed_result()
    pta = res.get("per_token_accounting", [])
    dec = [r for r in pta if r.get("phase") == "decode"]
    pre = [r for r in pta if r.get("phase") == "prefill"]
    meas_cold_decode = (sum(r["storage_requests"] for r in dec) / max(1, len(dec)))
    meas_cold_whole = (sum(r["storage_requests"] for r in pta) / max(1, len(pta)))
    meas_fill_decode = (sum(r["cold_loads"] for r in dec) / max(1, len(dec)))

    # --- miss-model comparison at the anchor configuration (mid bandwidth)
    mid = mk_cell(0.33 * GIB)
    p_steady = roofline_point(g, pop, mid, b, h_slots,
                              T4_DENSE_MS_PER_TOK.value,
                              T4_TOUCH_OVERHEAD_US.value, flops_eff,
                              vram_slots_per_gpu=v_slots, finite=False)
    p_finite = roofline_point(g, pop, mid, b, h_slots,
                              T4_DENSE_MS_PER_TOK.value,
                              T4_TOUCH_OVERHEAD_US.value, flops_eff,
                              vram_slots_per_gpu=v_slots, finite=True,
                              batches=[1] * 15)
    from .constants import ANCHOR_READ_MS

    miss_model_rows = [
        {"miss_model": "steady_state_che",
         "cold_records_per_tok_pred": p_steady.cold_records_per_tok,
         "cold_bytes_per_tok_pred": p_steady.cold_bytes_per_tok,
         "cold_records_per_tok_measured_decode": meas_cold_decode,
         "rel_err_records": (p_steady.cold_records_per_tok - meas_cold_decode)
                            / meas_cold_decode,
         "H_v": p_steady.H_v, "H_h": p_steady.H_h},
        {"miss_model": "finite_window_che_15decode",
         "cold_records_per_tok_pred": p_finite.cold_records_per_tok,
         "cold_bytes_per_tok_pred": p_finite.cold_bytes_per_tok,
         "cold_records_per_tok_measured_decode": meas_cold_decode,
         "rel_err_records": (p_finite.cold_records_per_tok - meas_cold_decode)
                            / meas_cold_decode,
         "H_v": p_finite.H_v, "H_h": p_finite.H_h},
    ]

    # --- eta_storage calibration on the mid case (finite model), then
    #     PREDICTION at 0.29 and 0.37 (same eta, different B_ssd).
    p_cal = p_finite
    t_compute_dense = p_cal.t_compute + p_cal.t_dense
    wrec = g.record_bytes
    t_service_mid = wrec / (0.33 * GIB)
    demand_mid = p_cal.cold_records_per_tok * t_service_mid
    t_h2d_mid = p_cal.t_h2d
    eta_h2d = 0.5                                # ASSUMPTION (staging overlap)
    unhidden_needed = t_measured - t_compute_dense - (1 - eta_h2d) * t_h2d_mid
    eta_storage = 1.0 - unhidden_needed / max(1e-12, demand_mid)
    eta_storage = float(min(0.95, max(0.0, eta_storage)))

    rows = []
    for tag, b_ssd in cases:
        c2 = mk_cell(b_ssd)
        for finite, mname in ((False, "steady_state_che"),
                              (True, "finite_window_che_15decode")):
            p = roofline_point(g, pop, c2, b, h_slots,
                               T4_DENSE_MS_PER_TOK.value,
                               T4_TOUCH_OVERHEAD_US.value, flops_eff,
                               vram_slots_per_gpu=v_slots, finite=finite,
                               batches=[1] * 15,
                               eta_storage=eta_storage, eta_h2d=eta_h2d)
            rows.append({
                "B_ssd_case": tag, "miss_model": mname,
                "cold_records_per_tok_pred": p.cold_records_per_tok,
                "cold_bytes_per_tok_pred": p.cold_bytes_per_tok,
                "spill_bytes_per_tok_pred": p.spill_bytes_per_tok,
                "t_storage": p.t_storage, "t_h2d": p.t_h2d,
                "t_compute": p.t_compute, "t_dense": p.t_dense,
                "t_bound": p.t_bound, "t_pred": p.t_pred,
                "tps_bound": p.tps_bound, "tps_pred": p.tps_pred,
                "t_measured": t_measured,
                "ratio_measured_over_pred": t_measured and
                    (1.0 / t_measured) / p.tps_pred,
                "limiter": p.limiter, "H_v": p.H_v, "H_h": p.H_h,
                "eta_storage_used": eta_storage,
            })
    write_csv("anchor_check.csv", rows)

    tps_meas = float(ANCHOR_TPS.value)
    pred_finite = [r["tps_pred"] for r in rows
                   if r["miss_model"] == "finite_window_che_15decode"]
    ratios = [tps_meas / v for v in pred_finite]
    from .constants import ANCHOR_SOURCE_READ_BW, ANCHOR_STORE_BYTES_PER_TOK

    t_whole = ANCHOR_STORE_BYTES_PER_TOK.value / ANCHOR_SOURCE_READ_BW.value
    result = {
        "measured_tps": tps_meas,
        "measured_tps_raw": ANCHOR_TPS_RAW.value,
        "measured_wall_s_per_tok": t_measured,
        "miss_model_comparison": miss_model_rows,
        "measured_cold_records_per_tok_decode": meas_cold_decode,
        "measured_cold_records_per_tok_whole_gen": meas_cold_whole,
        "measured_device_fills_per_tok_decode": meas_fill_decode,
        "calibrated_eta_storage": eta_storage,
        "calibrated_at_B_ssd": "0.33 GiB/s mid (finite model)",
        "predicted_tps_pred": pred_finite,
        "ratio_measured_over_pred": ratios,
        "within_2x": bool(all(0.5 <= r <= 2.0 for r in ratios)),
        "rows": rows,
        "prefill_inclusive_check": {
            "storage_bytes_per_emitted_tok": ANCHOR_STORE_BYTES_PER_TOK.value,
            "source_read_Bps": ANCHOR_SOURCE_READ_BW.value,
            "t_whole_gen_storage": t_whole,
            "tps_whole_gen_bound": 1.0 / t_whole,
            "measured_whole_gen_tps": 16 / (91.938 + 71.479),
            "ratio": (16 / (91.938 + 71.479)) / (1.0 / t_whole),
            "note": "whole-generation (prefill 91.9 s + decode 71.5 s) vs the "
                    "storage-only bound: ratio 1.25x — the run is ~80% "
                    "storage-service bound once prefill is included",
        },
        "eta_overlap": {
            "t_bound_mid": p_cal.t_bound,
            "t_measured": t_measured,
            "eta": p_cal.t_bound / t_measured,
            "note": "pipeline-max assumes perfect overlap; eta<1 = unoverlapped "
                    "orchestration/sync (stage profile everything_else bucket)",
        },
    }
    write_json("anchor_check.json", result)
    dlog("   anchor: measured", tps_meas, "tok/s vs pred",
         [round(v, 3) for v in pred_finite], "ratio", [round(r, 2) for r in ratios])
    return result


# ==========================================================================
# Curves vs host budget, per (model, cell)
# ==========================================================================
def run(cache_result=None) -> Dict[str, object]:
    from .constants import (ALL_CELLS, COMPUTE_EFF, T4_CELL,
                            T4_DENSE_MS_PER_TOK, T4_TOUCH_OVERHEAD_US)

    dlog("D. throughput roofline")
    anchor = anchor_check()

    geoms = load_geoms()
    rows: List[dict] = []
    t0 = float(T4_TOUCH_OVERHEAD_US.value)
    eta_storage = float(anchor["calibrated_eta_storage"])
    for gname, g in geoms.items():
        pop = build_pop(g)
        for cell in ALL_CELLS:
            flops_eff = COMPUTE_EFF.value * cell.F_peak if cell.n_gpu else 0.0
            dense = dense_ms_for(cell, g)
            for b in (1, 4, 8, 16):
                for host_gib in (2, 4, 8, 16, 32, 64, 128, 256):
                    h_slots = host_gib * GIB / g.record_bytes / max(1, cell.n_gpu)
                    for horizon, finite, hname in (
                            (512, False, "steady_state"),
                            (16, True, "finite_16tok")):
                        p = roofline_point(g, pop, cell, b, h_slots, dense, t0,
                                           flops_eff, horizon_tokens=horizon,
                                           finite=finite,
                                           batches=([7] + [1] * 15) if finite else None,
                                           eta_storage=eta_storage)
                        rows.append({
                            "cell": cell.name, "model": gname, "batch": b,
                            "horizon": hname,
                            "host_gib_total": host_gib,
                            "host_records_per_gpu": h_slots,
                            "vram_slots_per_gpu": p.vram_slots,
                            "cold_bytes_per_tok": p.cold_bytes_per_tok,
                            "spill_bytes_per_tok": p.spill_bytes_per_tok,
                            "cold_mib_per_tok": p.cold_bytes_per_tok / (1 << 20),
                            "cold_records_per_tok": p.cold_records_per_tok,
                            "t_storage": p.t_storage, "t_h2d": p.t_h2d,
                            "t_compute": p.t_compute, "t_dense": p.t_dense,
                            "t_bound": p.t_bound, "tps_bound": p.tps_bound,
                            "t_pred": p.t_pred, "tps_pred": p.tps_pred,
                            "limiter": p.limiter, "H_v": p.H_v, "H_h": p.H_h,
                            "U_per_layer": p.U_per_layer,
                            "popularity_source": pop.source,
                            "B_ssd": cell.B_ssd, "B_h2d": cell.B_h2d,
                            "price_gpu_s": cell.price_gpu_s,
                        })
    write_csv("roofline_curves.csv", rows)
    write_json("roofline_anchor.json", anchor)
    _plots(rows, geoms)
    dlog("   roofline grid:", len(rows), "points")
    return {"rows": rows, "anchor": anchor, "geoms": geoms}


def _plots(rows, geoms) -> None:
    fig, ax = figure("roofline_tps_vs_host", (7.8, 5.0))
    cells = sorted({r["cell"] for r in rows})
    for cell in cells:
        rs = [r for r in rows if r["cell"] == cell and r["model"] == "dsv4_flash"
              and r["batch"] == 1 and r["horizon"] == "steady_state"]
        rs.sort(key=lambda r: r["host_gib_total"])
        ax.plot([r["host_gib_total"] for r in rs], [r["tps_bound"] for r in rs],
                "-", lw=1.0, alpha=0.5, color=None)
        ax.plot([r["host_gib_total"] for r in rs], [r["tps_pred"] for r in rs],
                "o-", ms=3, label=cell)
    ax.axhline(0.21, color="r", ls="--", lw=1.2)
    ax.annotate("measured anchor 0.21 tok/s (2xT4, bank)", (2.2, 0.22), color="r", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("host RAM budget (GiB total)")
    ax.set_ylabel("TPS (tok/s)")
    ax.set_title("Decode TPS prediction (solid) and pipeline bound (faded) vs host budget — DeepSeek-V4-Flash, b=1")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "roofline_tps_vs_host.png")

    fig, ax = figure("roofline_batch_scaling", (7.6, 4.6))
    for cell in ("2xT4", "1xL40S", "1xRTXPRO6000", "CPU-only"):
        rs = [r for r in rows if r["cell"] == cell and r["model"] == "dsv4_flash"
              and r["host_gib_total"] == 32 and r["horizon"] == "steady_state"]
        rs.sort(key=lambda r: r["batch"])
        if rs:
            ax.plot([r["batch"] for r in rs], [r["tps_pred"] for r in rs],
                    "o-", ms=4, label=cell)
            ax.plot([r["batch"] for r in rs], [r["tps_bound"] for r in rs],
                    ":", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch b")
    ax.set_ylabel("TPS (tok/s, per replica)")
    ax.set_title("Batch scaling of the TPS prediction (host 32 GiB; dotted = pipeline bound)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "roofline_batch_scaling.png")

    fig, ax = figure("roofline_bytes_per_token", (7.6, 4.6))
    for cell in ("2xT4", "1xL4", "1xL40S"):
        for b, style in ((1, "-"), (8, "--")):
            rs = [r for r in rows if r["cell"] == cell and r["model"] == "dsv4_flash"
                  and r["batch"] == b and r["horizon"] == "steady_state"]
            rs.sort(key=lambda r: r["host_gib_total"])
            ax.plot([r["host_gib_total"] for r in rs],
                    [r["cold_mib_per_tok"] for r in rs], style, ms=3,
                    label=f"{cell} b={b} (SSD)")
            ax.plot([r["host_gib_total"] for r in rs],
                    [r["spill_bytes_per_tok"] / (1 << 20) for r in rs], ":",
                    lw=1.0, label=f"{cell} b={b} (H2D)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("host RAM budget (GiB total)")
    ax.set_ylabel("bytes per generated token (MiB)")
    ax.set_title("Slow-tier bytes per token (roofline = slow-tier bytes/token)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "roofline_bytes_per_token.png")
