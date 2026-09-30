"""B. Cache hit-rate model: closed form (Che TTL approximation) vs replay.

Closed form (common-TTL residence model under an Independent Reference
Model, derived in THEORY.md section 3):

    h_j(tau)  = 1 - exp(-w_j * tau)              per-record hit probability
    M(tau)    = sum_j (1 - exp(-w_j * tau))      expected resident records
    H(M)      = sum_j w_j * h_j(tau(M))          aggregate hit rate

with w_j the record popularity (per cache request) and tau the residence
budget measured in cache-request steps.  tau(M) is the unique positive root
of M(tau) = M (monotone increasing), found by bisection.

Empirical side: pooled LRU and offline Belady(MIN) replay over the real
per-layer-call deduplicated stream, plus a shuffled control that isolates
the temporal-correlation error of the IRM assumption.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .util import dlog, figure, savefig, write_csv, write_json

INF = float("inf")


# ==========================================================================
# Closed form
# ==========================================================================
class CheTable:
    """Che TTL curves tabulated on a tau grid (exact within interpolation).

    For a popularity vector w the pair

        M(tau) = sum_j (1 - exp(-w_j tau))        expected resident records
        H(tau) = sum_j w_j (1 - exp(-w_j tau))    aggregate hit rate

    is monotone in tau, so both are tabulated once on a log grid and
    inverted by interpolation.  This is O(N * n_tau) once instead of an
    O(N) bisection per query — the roofline needs thousands of queries.
    """

    def __init__(self, w: np.ndarray, n_tau: int = 400,
                 tau_max_mult: float = 1e6):
        self.w_full = np.asarray(w, dtype=float)
        self.mask = self.w_full > 0
        self.w = self.w_full[self.mask]
        if self.w.size == 0:
            self.tau = np.array([0.0])
            self.M = np.array([0.0])
            self.H = np.array([0.0])
            return
        t_lo = 1e-6 / float(self.w.max())
        t_hi = tau_max_mult / float(self.w.max())
        tau = np.geomspace(t_lo, t_hi, n_tau)
        e = 1.0 - np.exp(-np.outer(self.w, tau))     # (N, n_tau)
        self.tau = tau
        self.M = e.sum(axis=0)
        self.H = (self.w[:, None] * e).sum(axis=0)

    def occupancy(self, tau: float) -> float:
        return float(np.interp(tau, self.tau, self.M))

    def hit_at_capacity(self, M: float) -> float:
        """H at resident capacity M (record units) — the Che TTL closed form."""
        if M <= 0 or self.w.size == 0:
            return 0.0
        if M >= self.w.size:
            return float(self.H[-1])
        return float(np.interp(M, self.M, self.H))

    def tau_at_capacity(self, M: float) -> float:
        if M <= 0 or self.w.size == 0:
            return 0.0
        return float(np.interp(M, self.M, self.tau))

    def curve(self, capacities) -> Dict[str, np.ndarray]:
        caps = np.asarray(capacities, dtype=float)
        return {"M": caps,
                "H": np.array([self.hit_at_capacity(c) for c in caps]),
                "tau": np.array([self.tau_at_capacity(c) for c in caps])}

    def hit_finite(self, M: float, counts: np.ndarray) -> float:
        """Finite-window Che: compulsory first touches counted as misses."""
        hits, tot = self.level_counts(counts, M)[:2]
        return float(np.clip(hits / max(1.0, tot), 0.0, 1.0))

    def level_counts(self, counts: np.ndarray, M: float):
        """Per-record (hits, total, misses) at capacity M over a finite window.

        In a window with n_j requests to record j the TTL residence model
        predicts one compulsory (first-touch) miss plus n_j - 1 reuses, of
        which a fraction 1 - exp(-w_j tau) find the record resident:

            hits_j   = (n_j - 1) (1 - exp(-w_j tau))
            misses_j = n_j - hits_j
            M(tau)   = sum_j (1 - exp(-w_j tau))      (residence occupancy)

        This is the correction that dominates on the sealed trace: the window
        is far too short for the steady-state limit (R = 5,099 requests over
        U = 2,364 records, so at most (R-U)/R = 53.6% of accesses can ever
        hit regardless of capacity).  As n_j -> infinity the per-record hit
        probability converges to 1 - exp(-w_j tau), the steady-state Che form.
        """
        counts = np.asarray(counts, dtype=float)
        tau = self.tau_at_capacity(float(M))
        if counts.size == self.mask.size:
            n = counts[self.mask]                  # full category space -> keep alignment
        elif counts.size >= self.w.size:
            n = counts[: self.w.size]
        else:
            n = np.zeros(self.w.size)
        hits_j = np.maximum(n - 1.0, 0.0) * (1.0 - np.exp(-self.w * tau))
        total = float(n.sum())
        return float(hits_j.sum()), total, n - hits_j, n, hits_j


def che_tau(w: np.ndarray, M: float, lo: float = 0.0, hi: float = 1e12,
            iters: int = 120) -> float:
    """Solve sum_j (1 - exp(-w_j tau)) = M for tau (bisection, monotone)."""
    return _che_tau_bisect(w, M, lo, hi, iters)


def _che_tau_bisect(w: np.ndarray, M: float, lo: float, hi: float,
                    iters: int) -> float:
    w = np.asarray(w, dtype=float)
    w = w[w > 0]
    if M <= 0 or w.size == 0:
        return 0.0
    if M >= w.size:
        return hi

    def occ(tau: float) -> float:
        return float(np.sum(1.0 - np.exp(-w * tau)))

    t_hi = hi
    while occ(t_hi) < M and t_hi < 1e18:
        t_hi *= 4.0
    t_lo = lo
    for _ in range(iters):
        mid = 0.5 * (t_lo + t_hi)
        if occ(mid) < M:
            t_lo = mid
        else:
            t_hi = mid
    return 0.5 * (t_lo + t_hi)


def che_hit_curve(w: np.ndarray, capacities: Sequence[int]) -> Dict[str, np.ndarray]:
    """H(M) closed form over a sweep of resident-record capacities M."""
    tab = CheTable(w)
    return tab.curve(capacities)


def distinct_per_call(p: np.ndarray, k: int, b: int) -> float:
    """E[distinct experts requested by one layer-call] for batch size b.

    Each of the b rows selects k experts; with-replacement across rows and
    rank slots the request count for expert i is Binomial(b*k, p_i), so

        E[distinct] = sum_i (1 - (1 - p_i)^(b*k)).

    This is the batch-overlap correction: the same expert missed once serves
    the whole batch.
    """
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(np.sum(1.0 - (1.0 - p) ** (k * b)))


def batch_misses_per_call(w_layer: np.ndarray, k: int, b: int,
                          h: np.ndarray) -> float:
    """E[misses in one layer-call] with per-record hit probabilities h.

        E[misses] = sum_i [1 - (1-p_i)^(b k)] * (1 - h_i)
    """
    p = np.asarray(w_layer, dtype=float)
    h = np.asarray(h, dtype=float)
    m = p > 0
    return float(np.sum((1.0 - (1.0 - p[m]) ** (k * b)) * (1.0 - h[m])))


# ==========================================================================
# Empirical replay: LRU, Belady MIN, stack-distance coverage
# ==========================================================================
def lru_replay(stream: Sequence[Tuple[int, int]], cap: int) -> Dict[str, float]:
    cache: "OrderedDict[Tuple[int, int], None]" = OrderedDict()
    hits = evict = 0
    for key in stream:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            if len(cache) >= cap and cap > 0:
                cache.popitem(last=False)
                evict += 1
            cache[key] = None
    n = len(stream)
    return {"hits": hits, "misses": n - hits, "evictions": evict,
            "hit_rate": hits / n if n else 0.0, "total": n, "cap": cap}


def min_replay(stream: Sequence[Tuple[int, int]], cap: int) -> Dict[str, float]:
    """Belady MIN offline optimum (evict farthest next use)."""
    n = len(stream)
    future: Dict[Tuple[int, int], deque] = {}
    for i, key in enumerate(stream):
        future.setdefault(key, deque()).append(i)
    resident: Dict[Tuple[int, int], int] = {}   # key -> next use index (or INF)
    hits = 0
    for i, key in enumerate(stream):
        fu = future[key]
        fu.popleft()
        nxt = fu[0] if fu else INF
        if key in resident:
            hits += 1
            resident[key] = nxt
            continue
        if cap > 0 and len(resident) >= cap:
            victim = max(resident, key=lambda kk: resident[kk])
            del resident[victim]
        resident[key] = nxt
    return {"hits": hits, "misses": n - hits,
            "hit_rate": hits / n if n else 0.0, "total": n, "cap": cap}


def stack_distance_hist(stream: Sequence[Tuple[int, int]]) -> Dict[int, int]:
    """LRU stack (reuse) distance histogram in distinct-key units."""
    seen: "OrderedDict[Tuple[int, int], None]" = OrderedDict()
    hist: Dict[int, int] = {}
    first = 0
    for key in stream:
        if key in seen:
            ordered = list(seen.keys())
            d = len(ordered) - 1 - ordered.index(key)
            hist[d] = hist.get(d, 0) + 1
            seen.move_to_end(key)
        else:
            first += 1
            seen[key] = None
    hist[INF] = first        # never reused
    return hist


def coverage_from_hist(hist: Dict[int, int], cap: int) -> float:
    """LRU-capturable reuse: P(stack distance < cap)."""
    tot = sum(v for k, v in hist.items() if k != INF) + hist.get(INF, 0)
    cov = sum(v for k, v in hist.items() if k != INF and k < cap)
    return cov / tot if tot else 0.0


def shuffled_stream(stream: Sequence[Tuple[int, int]], seed: int = 20260915):
    """IRM control: same empirical popularity, temporal correlation destroyed."""
    idx = np.random.default_rng(seed).permutation(len(stream))
    return [stream[i] for i in idx]


def popularity_vector(stream: Sequence[Tuple[int, int]], n_layers: int,
                      experts_per_layer: int) -> np.ndarray:
    from collections import Counter

    cnt = Counter(stream)
    w = np.zeros(n_layers * experts_per_layer, dtype=float)
    for (layer, e), v in cnt.items():
        if 0 <= layer < n_layers and 0 <= e < experts_per_layer:
            w[layer * experts_per_layer + e] = v
    s = w.sum()
    return w / s if s > 0 else w


# ==========================================================================
# Driver
# ==========================================================================
GIB = float(1 << 30)

BUDGETS_GIB = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 96, 147]


def run(popularity_result=None) -> Dict[str, object]:
    from .constants import (D4_RECORD, WS_KNEE, WS_LRU_MINUS_MIN, WS_MIN_AT)
    from .sources import load_sealed_stream, load_stores

    dlog("B. cache model")
    stores = load_stores()
    sealed = load_sealed_stream()
    stream = sealed.stream()
    rec = int(D4_RECORD.value)
    w = popularity_vector(stream, 43, 256)
    che_tab = CheTable(w)

    # ---------- empirical sweep on the sealed trace ---------------------
    rows = []
    hist = stack_distance_hist(stream)
    n_req = len(stream)
    n_uniq = len(set(stream))
    for gib in BUDGETS_GIB:
        cap = max(1, int(gib * GIB // rec))
        lru = lru_replay(stream, cap)
        mn = min_replay(stream, cap)
        sh = lru_replay(shuffled_stream(stream), cap)
        che_steady = float(che_hit_curve(w, [cap])["H"][0])
        counts = np.zeros(43 * 256, dtype=float)
        from collections import Counter as _Counter

        for (layer, e), v in _Counter(stream).items():
            counts[layer * 256 + e] = v
        che_finite = che_tab.hit_finite(cap, counts)
        rows.append({
            "trace": "sealed_dsv4", "budget_gib": gib, "capacity_records": cap,
            "lru_hit": lru["hit_rate"], "min_hit": mn["hit_rate"],
            "lru_minus_min_pp": 100.0 * (lru["hit_rate"] - mn["hit_rate"]),
            "lru_evictions": lru["evictions"],
            "che_hit": che_steady,
            "che_finite_hit": che_finite,
            "che_finite_error_pp": 100.0 * (che_finite - lru["hit_rate"]),
            "shuffled_lru_hit": sh["hit_rate"],
            "coverage_stack": coverage_from_hist(hist, cap),
            "temporal_correlation_gain_pp": 100.0 * (lru["hit_rate"] - sh["hit_rate"]),
            "che_vs_shuffled_pp": 100.0 * (che_steady - sh["hit_rate"]),
        })

    # knee diagnostics (reproduce the phase-2 ws-policy landmarks)
    knee = _knee(records_capacity_per_gib=GIB / rec, rows=rows, rec=rec, stream=stream)
    h_max = max(r["lru_hit"] for r in rows)
    frac_knees = {}
    for frac in (0.5, 0.8, 0.95):
        key = f"lru_at_{int(frac*100)}pct_of_max"
        row = next((r for r in rows if r["lru_hit"] >= frac * h_max), None)
        frac_knees[key] = {
            "measured_records": row["capacity_records"] if row else None,
            "measured_gib": row["budget_gib"] if row else None,
        }
    landmarks = {
        "measured_knee_gib": WS_KNEE.value,
        "recomputed_knee_gib": knee["knee_gib"],
        "measured_lru_minus_min_pp": WS_LRU_MINUS_MIN.value,
        "recomputed_lru_minus_min_pp_at_knee": knee["lru_minus_min_pp_at_knee"],
        "measured_reaches_min_gib": WS_MIN_AT.value,
        "recomputed_reaches_min_gib": knee["reaches_min_gib"],
        "measured_repeat_records": 935,
        "recomputed_repeat_records": knee["repeat_records"],
        "unique_records": knee["unique_records"],
        "lru_hit_max": h_max,
        "finite_window_hit_ceiling": (knee["repeat_records"] * 2
                                      - knee["unique_records"])
        / max(1, sum(1 for _ in stream)) if stream else 0.0,
        "frac_knee_records_measured": frac_knees,
    }

    # ---------- closed form for all four stores -------------------------
    from .popularity import fit_zipf_mandelbrot, profile_from_stream, zipf_mandelbrot_p

    store_rows = []
    curve_rows = []
    for sname, store in stores.items():
        # popularity source: sealed DSv4-Flash pooled profile for all four.
        # For the three non-DSv4 stores this transfer is an ASSUMPTION
        # (constants.S_TRANSFER) — no router trace exists for them in-repo.
        prof = profile_from_stream(stream, 43, 256, sname, "sealed_dsv4")
        z = fit_zipf_mandelbrot(prof)
        p = zipf_mandelbrot_p(z["s"], z["q"], store.universe_records)
        caps = _cap_grid(store)
        curve = che_hit_curve(p, caps)
        for c, h, t in zip(caps, curve["H"], curve["tau"]):
            curve_rows.append({
                "store": sname, "capacity_records": int(c),
                "capacity_bytes": int(c) * store.record_bytes,
                "capacity_gib": c * store.record_bytes / GIB,
                "che_hit": h, "tau_requests": t,
                "s_used": z["s"], "q_used": z["q"],
                "popularity_source": "sealed_dsv4_transfer" if sname != "dsv4_flash"
                                     else "sealed_dsv4_measured",
                "transfer_assumption": sname != "dsv4_flash",
            })
        m50 = _capacity_at(curve["H"], caps, 0.50)
        m80 = _capacity_at(curve["H"], caps, 0.80)
        m95 = _capacity_at(curve["H"], caps, 0.95)
        store_rows.append({
            "store": sname, "label": store.label,
            "universe_records": store.universe_records,
            "universe_gib": store.universe_bytes / GIB,
            "record_bytes": store.record_bytes,
            "s": z["s"], "q": z["q"],
            "geometric_decay_per_rank_pct": z.get("geometric_decay_per_rank_pct"),
            "degenerate_to_geometric": z.get("degenerate_to_geometric"),
            "cap_at_H50_records": m50, "cap_at_H50_gib": m50 * store.record_bytes / GIB,
            "cap_at_H80_records": m80, "cap_at_H80_gib": m80 * store.record_bytes / GIB,
            "cap_at_H95_records": m95, "cap_at_H95_gib": m95 * store.record_bytes / GIB,
            "popularity_source": "sealed_dsv4_transfer" if sname != "dsv4_flash"
                                 else "sealed_dsv4_measured",
        })

    write_csv("cache_sim_check.csv", rows)
    write_csv("cache_sim_landmarks.json.csv", [{"metric": k, "value": v}
                                               for k, v in landmarks.items()],
              ["metric", "value"])
    write_json("cache_sim_landmarks.json", landmarks)
    write_csv("cache_che_curves.csv", curve_rows)
    write_csv("cache_store_knees.csv", store_rows)

    _plots(rows, curve_rows, store_rows)

    summary = {
        "landmarks": landmarks,
        "rows": rows,
        "store_knees": store_rows,
        "irm_delta_at_knee": _irm_delta(rows, knee["knee_gib"]),
    }
    write_json("cache_summary.json", {
        "landmarks": landmarks,
        "store_knees": store_rows,
        "irm_delta_at_knee": summary["irm_delta_at_knee"],
    })
    dlog("   knee recomputed", landmarks["recomputed_knee_gib"], "GiB vs measured",
         landmarks["measured_knee_gib"], "GiB")
    return summary


# ==========================================================================
# helpers
# ==========================================================================
def _cap_grid(store) -> np.ndarray:
    n = store.universe_records
    frac = np.unique(np.concatenate([
        np.linspace(0.002, 0.05, 12),
        np.linspace(0.05, 0.5, 18),
        np.linspace(0.5, 1.0, 8),
    ]))
    caps = np.unique(np.maximum(1, (frac * n).astype(int)))
    return caps


def _capacity_at(H: np.ndarray, caps, target: float) -> int:
    for h, c in zip(H, caps):
        if h >= target:
            return int(c)
    return int(caps[-1])


def _knee(records_capacity_per_gib: float, rows, rec: int, stream) -> Dict[str, float]:
    """Locate the LRU knee (max curvature of the hit-rate vs capacity curve),
    the budget where LRU is within 2.8pp of MIN (or the pp gap at the knee),
    and where LRU equals MIN."""
    from collections import Counter

    cnt = Counter(stream)
    uniq = len(cnt)
    repeats = sum(1 for v in cnt.values() if v > 1)

    xs = np.array([r["capacity_records"] for r in rows], dtype=float)
    hs = np.array([r["lru_hit"] for r in rows], dtype=float)
    ms = np.array([r["min_hit"] for r in rows], dtype=float)
    gib = np.array([r["budget_gib"] for r in rows], dtype=float)
    # discrete curvature in log-capacity
    d1 = np.gradient(hs, np.log(xs))
    d2 = np.gradient(d1, np.log(xs))
    i = int(np.argmin(d2))         # most negative curvature = shoulder
    knee_gib = float(gib[i])
    gap_at_knee = float(100.0 * (hs[i] - ms[i]))
    reaches = None
    for r in rows:
        if r["lru_minus_min_pp"] >= -0.5:
            reaches = r["budget_gib"]
            break
    return {
        "knee_gib": knee_gib,
        "lru_minus_min_pp_at_knee": gap_at_knee,
        "reaches_min_gib": float(reaches if reaches is not None else float("nan")),
        "unique_records": uniq,
        "repeat_records": repeats,
    }


def _irm_delta(rows, at_gib: float) -> Dict[str, float]:
    r = min(rows, key=lambda x: abs(x["budget_gib"] - at_gib))
    return {
        "budget_gib": r["budget_gib"],
        "real_lru_hit": r["lru_hit"],
        "shuffled_lru_hit": r["shuffled_lru_hit"],
        "che_steady_hit": r["che_hit"],
        "che_finite_hit": r["che_finite_hit"],
        "temporal_correlation_gain_pp": 100.0 * (r["lru_hit"] - r["shuffled_lru_hit"]),
        "che_steady_vs_shuffled_pp": 100.0 * (r["che_hit"] - r["shuffled_lru_hit"]),
        "che_finite_vs_shuffled_pp": 100.0 * (r["che_finite_hit"] - r["shuffled_lru_hit"]),
    }


def _plots(rows, curve_rows, store_rows) -> None:
    fig, ax = figure("cache_hitrate", (7.6, 4.8))
    gib = [r["budget_gib"] for r in rows]
    ax.plot(gib, [100 * r["lru_hit"] for r in rows], "o-", label="LRU replay (real stream)", color="#1f77b4")
    ax.plot(gib, [100 * r["min_hit"] for r in rows], "s--", label="Belady MIN replay", color="#2ca02c")
    ax.plot(gib, [100 * r["shuffled_lru_hit"] for r in rows], "^:", label="LRU on shuffled (IRM control)", color="#ff7f0e")
    ax.plot(gib, [100 * r["che_hit"] for r in rows], "-", label="Che TTL closed form (steady state)", color="#9467bd")
    ax.plot(gib, [100 * r["che_finite_hit"] for r in rows], "-", lw=1.2,
            label="Che TTL + finite-window correction", color="#8c564b")
    ax.plot(gib, [100 * r["coverage_stack"] for r in rows], ".", label="stack-distance coverage", color="gray")
    ax.axvline(16, color="k", lw=0.8, alpha=0.6)
    ax.annotate("measured knee 16 GiB", (16, 5), fontsize=8, rotation=90, alpha=0.8)
    ax.axvline(32, color="k", lw=0.8, alpha=0.4)
    ax.annotate("LRU=MIN 32 GiB", (32, 5), fontsize=8, rotation=90, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("host budget (GiB, 13,369,344 B records)")
    ax.set_ylabel("hit rate (%)")
    ax.set_title("Host-tier hit rate vs budget — sealed DSv4-Flash trace (5,099 requests, 2,364 records)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "cache_hitrate.png")

    fig, ax = figure("cache_che_four_stores", (7.6, 4.8))
    by_store: Dict[str, List[dict]] = {}
    for r in curve_rows:
        by_store.setdefault(r["store"], []).append(r)
    for sname, rs in by_store.items():
        rs = sorted(rs, key=lambda r: r["capacity_gib"])
        ax.plot([r["capacity_gib"] for r in rs], [100 * r["che_hit"] for r in rs],
                "-", lw=1.5, label=f"{sname} ({rs[0]['universe_records'] if 'universe_records' in rs[0] else ''})")
    ax.set_xscale("log")
    ax.set_xlabel("resident budget (GiB)")
    ax.set_ylabel("closed-form hit rate H(M) (%)")
    ax.set_title("Che TTL hit-rate model, four store specs (DSv4 popularity transferred where noted)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "cache_che_four_stores.png")

    fig, ax = figure("cache_error_decomposition", (7.2, 4.4))
    ax.plot(gib, [r["temporal_correlation_gain_pp"] for r in rows], "o-",
            label="temporal correlation gain (real - shuffled)")
    ax.plot(gib, [r["che_vs_shuffled_pp"] for r in rows], "s--",
            label="Che approx. error (che - shuffled)")
    ax.plot(gib, [r["lru_minus_min_pp"] for r in rows], "^:",
            label="LRU - MIN (policy gap)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("host budget (GiB)")
    ax.set_ylabel("hit rate difference (pp)")
    ax.set_title("IRM error decomposition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, "cache_error_decomposition.png")
