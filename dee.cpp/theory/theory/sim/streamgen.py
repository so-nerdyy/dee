"""Synthetic request-stream generation + the closed-form models it is checked
against (``theory.cache`` Che steady / finite-window, ``theory.roofline`` T_pred).

The generator is built from the FITTED popularity / temporal models, not from a
free-form toy distribution:

  * popularity — ``theory.popularity``'s families fitted on the sealed stream.
    The Zipf-Mandelbrot MLE is *geometric-degenerate* (THEORY.md 2.1), so the
    generator uses its honest limit, an exponential-in-rank (geometric) decay
    ``p_(r) proportional exp(-d * r)``, plus the lognormal-rank family as an
    alternative arm.  Rank->expert labels are exchangeable under every family
    (theory.popularity docstring), so each layer's rank curve is attached to a
    SEEDED random permutation of its expert ids.
  * temporal — two measured channels from ``theory.temporal`` / CACHE1_ANALYSIS:
    (1) decode-step self-correlation: a layer-call reuses members of the
        previous token's set at the same layer with probability
        ``SIM_SYNTH_STICKY`` = 0.375 (the measured consecutive-token overlap);
    (2) within-token cross-layer correlation: layer ``l+1`` inherits
        ``phi_l(i)`` for a parent expert ``i`` of layer ``l``, reproducing the
        measured 12.2x conditional lift (``SIM_SYNTH_CROSSLAYER``).

Every generated stream is validated against its own targets and the achieved
statistics are emitted — a generator that misses its target is reported, not
quietly accepted.
"""
from __future__ import annotations

import heapq
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (SIM_SYNTH_CROSSLAYER, SIM_SYNTH_STICKY)
from .engine import Key, RouterCall, Stream

INF = float("inf")


# ==========================================================================
# Popularity families (the fitted models, re-expressed for sampling)
# ==========================================================================
def geometric_rank_p(decay_per_rank: float, n: int) -> np.ndarray:
    """Zipf-Mandelbrot geometric-degenerate limit: p_(r) ~ exp(-d*r)."""
    r = np.arange(1, n + 1, dtype=float)
    lp = -decay_per_rank * r
    lp -= lp.max()
    w = np.exp(lp)
    return w / w.sum()


def lognormal_rank_p(mu: float, sigma: float, n: int) -> np.ndarray:
    """Lognormal rank curve: p_(r) proportional exp(mu + sigma*z_r)."""
    from scipy import stats

    u = 1.0 - (np.arange(1, n + 1) - 0.5) / n
    z = stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    lp = sigma * z
    lp -= lp.max()
    w = np.exp(lp)
    return w / w.sum()


def fit_decay_from_sealed(stream_keys: Sequence[Key], n_layers: int,
                          experts_per_layer: int) -> Dict[str, float]:
    """Fit the generator's popularity parameters on the sealed dedup stream.

    Uses ``theory.popularity``'s own fitting functions so the generator draws
    from THE fitted models rather than a parallel re-fit.  The per-layer decay
    is the value whose induced per-layer entropy matches the measured per-layer
    entropy of the sealed stream (theory.popularity entropy of per-layer
    profiles), because the pooled decay (0.146 %/rank) is a *pooled* quantity
    and would flatten every layer to near-uniform 8 bits.
    """
    from ..popularity import (Profile, fit_lognormal_rank, fit_zipf_mandelbrot,
                              profile_per_layer)

    profs = profile_per_layer(list(stream_keys), experts_per_layer,
                              n_layers, "sealed")
    z = fit_zipf_mandelbrot(
        Profile("pooled", np.bincount(
            [l * experts_per_layer + e for l, e in stream_keys],
            minlength=n_layers * experts_per_layer),
            n_layers * experts_per_layer, len(stream_keys), "sealed"))
    ln = fit_lognorm_rank = fit_lognormal_rank(
        Profile("pooled", np.bincount(
            [l * experts_per_layer + e for l, e in stream_keys],
            minlength=n_layers * experts_per_layer),
            n_layers * experts_per_layer, len(stream_keys), "sealed"))
    # per-layer geometric decay solved so the induced entropy matches the
    # measured per-layer entropy (closed form: binary search on d).
    ent = []
    for layer, p in profs.items():
        h = float(-(p.emp_p[p.emp_p > 0] * np.log2(p.emp_p[p.emp_p > 0])).sum())
        ent.append(h)
    target_h = float(np.mean(ent)) if ent else 5.24
    d = _decay_for_entropy(target_h, experts_per_layer)
    return {
        "pooled_zm_s": z["s"], "pooled_zm_q": z["q"],
        "geometric_decay_per_rank_pooled":
            z.get("geometric_decay_per_rank_pct", float("nan")),
        "lognormal_mu": ln["mu"], "lognormal_sigma": ln["sigma"],
        "per_layer_entropy_target_bits": target_h,
        "per_layer_decay_per_rank": d,
    }


def _decay_for_entropy(target_h_bits: float, n: int) -> float:
    """Binary search the geometric decay whose entropy is target_h_bits."""
    def entropy(d: float) -> float:
        p = geometric_rank_p(d, n)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    lo, hi = 1e-6, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if entropy(mid) > target_h_bits:
            lo = mid                       # flatter -> higher entropy
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ==========================================================================
# The generator
# ==========================================================================
@dataclass
class StreamGenConfig:
    n_tokens: int
    n_layers: int
    experts_per_layer: int
    topk: int
    batch: int = 1
    decay_per_rank: float = 0.026
    family: str = "geometric"            # 'geometric' | 'lognormal'
    lognormal_mu: float = 1.46
    lognormal_sigma: float = 1.84
    sticky: float = 0.375                # decode-step self-correlation
    cross_prob: float = 0.55             # within-token cross-layer inheritance
    prefill_rows: int = 7
    prefill_tokens: int = 1
    seed: int = 20260916
    label: str = "synth"


class StreamGenerator:
    """Long synthetic request streams from the fitted popularity/temporal model.

    One forward step = one token (decode) or one prompt row-group (prefill);
    each of ``n_layers`` layer-calls requests ``topk`` experts per batch row and
    is deduplicated exactly the way the engine deduplicates (distinct experts,
    staged in ascending id order).
    """

    def __init__(self, cfg: StreamGenConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        n = cfg.experts_per_layer
        # rank curve -> expert id: a seeded permutation per layer (labels are
        # exchangeable under the fitted families)
        if cfg.family == "lognormal":
            base = lognormal_rank_p(cfg.lognormal_mu, cfg.lognormal_sigma, n)
        else:
            base = geometric_rank_p(cfg.decay_per_rank, n)
        self.p = np.zeros((cfg.n_layers, n))
        for l in range(cfg.n_layers):
            perm = self.rng.permutation(n)
            self.p[l] = base[np.argsort(perm)]      # rank r -> expert perm[r]
        self.cum = np.cumsum(self.p, axis=1)
        self.cum[:, -1] = 1.0
        # cross-layer coupling maps phi_l (expert id at layer l -> id at l+1)
        self.phi = [self.rng.permutation(n) for _ in range(cfg.n_layers - 1)]

    def _draw_rank(self, l: int, size: int) -> np.ndarray:
        u = self.rng.random(size)
        return np.searchsorted(self.cum[l], u, side="right")

    def _layer_set(self, l: int, prev_same: List[int], prev_layer: List[int],
                   rows: int) -> List[int]:
        """Deduplicated expert set for one layer-call at batch ``rows``."""
        cfg = self.cfg
        want = cfg.topk * rows
        out: List[int] = []
        seen = set()
        for _ in range(want):
            u = self.rng.random()
            if l > 0 and prev_layer and u < cfg.cross_prob:
                parent = prev_layer[self.rng.integers(len(prev_layer))]
                cand = int(self.phi[l - 1][parent])
            elif prev_same and u < cfg.cross_prob + (1.0 - cfg.cross_prob) * cfg.sticky:
                cand = prev_same[self.rng.integers(len(prev_same))]
            else:
                cand = int(self._draw_rank(l, 1)[0])
            # de-duplicate: re-draw from the marginal on a collision
            tries = 0
            while cand in seen and tries < 4:
                cand = int(self._draw_rank(l, 1)[0])
                tries += 1
            if cand in seen:
                continue
            seen.add(cand)
            out.append(cand)
        return sorted(out)

    def generate(self) -> Stream:
        cfg = self.cfg
        devmap = {l: ("cuda:0" if l < cfg.n_layers // 2 + cfg.n_layers % 2
                      else "cuda:1") for l in range(cfg.n_layers)}
        calls: List[RouterCall] = []
        prev_by_layer: Dict[int, List[int]] = {}
        token = 0
        for _ in range(cfg.prefill_tokens):
            for l in range(cfg.n_layers):
                prev_layer = prev_by_layer.get(l - 1, [])
                s = self._layer_set(l, prev_by_layer.get(l, []), prev_layer,
                                    cfg.prefill_rows)
                calls.append(RouterCall(token, l, devmap[l], tuple(s),
                                        phase="prefill",
                                        token_rows=cfg.prefill_rows))
                prev_by_layer[l] = s
            token += 1
        for _ in range(cfg.n_tokens):
            for l in range(cfg.n_layers):
                prev_layer = prev_by_layer.get(l - 1, [])
                s = self._layer_set(l, prev_by_layer.get(l, []), prev_layer,
                                    cfg.batch)
                calls.append(RouterCall(token, l, devmap[l], tuple(s),
                                        phase="decode", token_rows=cfg.batch))
                prev_by_layer[l] = s
            token += 1
        return Stream(calls, cfg.n_layers, cfg.experts_per_layer, cfg.topk,
                      cfg.label)

    def generated_stats(self, stream: Stream) -> Dict[str, float]:
        """Achieved generator statistics vs their measured targets."""
        # consecutive-token same-layer overlap (target 0.375)
        by_tl: Dict[Tuple[int, int], set] = {}
        for c in stream.calls:
            by_tl[(c.token, c.layer)] = set(c.experts)
        toks = stream.token_ids()
        inter = same = 0
        for a, b in zip(toks, toks[1:]):
            for l in range(stream.n_layers):
                sa, sb = by_tl.get((a, l), set()), by_tl.get((b, l), set())
                inter += len(sa & sb)
                same += len(sa)
        overlap = inter / same if same else 0.0
        # cross-layer conditional lift (target 12.2x)
        marg: Counter = Counter()
        pair: Counter = Counter()
        n_pairs = 0
        for t in toks:
            for l in range(stream.n_layers - 1):
                sa = by_tl.get((t, l), set())
                sb = by_tl.get((t, l + 1), set())
                for j in sb:
                    marg[j] += 1
                for i in sa:
                    for j in sb:
                        pair[(i, j)] += 1
                n_pairs += 1
        total = sum(marg.values()) or 1
        lifts = []
        for (i, j), c in pair.items():
            p_joint = c / max(1, n_pairs)
            p_j = marg[j] / total
            if p_j > 0 and c >= 5:
                lifts.append(p_joint / p_j)
        lift = float(np.mean(lifts)) if lifts else 0.0
        return {
            "requests": stream.n_requests,
            "tokens": len(toks),
            "unique_records": len(set(stream.keys())),
            "consecutive_token_overlap": overlap,
            "overlap_target": float(SIM_SYNTH_STICKY.value),
            "crosslayer_lift": lift,
            "crosslayer_lift_target": float(SIM_SYNTH_CROSSLAYER.value),
        }


# ==========================================================================
# Closed-form wrappers (the models the sim is cross-checked against)
# ==========================================================================
def che_two_level(counts: np.ndarray, w: np.ndarray, m_v: int, m_h: int,
                  finite: bool) -> Dict[str, float]:
    """``theory.cache`` Che two-level form evaluated on a stream's counts.

    ``counts`` = observed request count per record over the window,
    ``w``      = the record's per-request probability (the IRM the closed form
                 assumes).  ``finite=True`` applies THEORY.md 3.1's compulsory
                 first-touch correction; ``finite=False`` is the steady form.
    """
    from ..cache import CheTable

    tab = CheTable(w)
    if finite:
        hits_v, tot, miss_v, nv, hv = tab.level_counts(counts, m_v)
        hits_h, tot_h, miss_h, _nh, _hh = tab.level_counts(miss_v, m_h)
        n_req = float(tot)
        return {
            "H_v": float(hv.sum() / max(1.0, nv.sum())),
            "H_h_conditional": float(hits_h / max(1.0, tot_h)),
            "cold_records_total": float(miss_h.sum()),
            "fills_total": float(miss_v.sum()),
            "requests": n_req,
            "host_share_eager": float(hv.sum() / max(1.0, nv.sum()) * 0 + 1.0)
                               if n_req == 0 else float("nan"),
        }
    tau_v = tab.tau_at_capacity(float(m_v))
    h_v = 1.0 - np.exp(-w * tau_v)
    u_h = w * (1.0 - h_v)
    tot_h = u_h.sum()
    if tot_h <= 0 or m_h <= 0:
        h_h = np.zeros_like(h_v)
    else:
        from ..cache import CheTable as _T
        th = _T(u_h / tot_h)
        tau_h = th.tau_at_capacity(float(m_h))
        h_h = 1.0 - np.exp(-(u_h / tot_h) * tau_h)
    H_v = float(w.dot(h_v))
    miss_v_rate = float(w.dot(1.0 - h_v))
    miss_h_rate = float(w.dot((1.0 - h_v) * (1.0 - h_h)))
    n_req = float(counts.sum())
    return {
        "H_v": H_v,
        "H_h_conditional": 1.0 - miss_h_rate / max(1e-30, miss_v_rate),
        "cold_records_total": miss_h_rate * n_req,
        "fills_total": miss_v_rate * n_req,
        "requests": n_req,
        "host_share_eager": float("nan"),
    }


def counts_and_weights(stream: Stream) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical per-record counts and IRM request probabilities."""
    n = stream.n_layers * stream.experts_per_layer
    cnt = np.zeros(n, dtype=float)
    for l, e in stream.keys():
        cnt[l * stream.experts_per_layer + e] += 1
    tot = cnt.sum()
    return cnt, (cnt / tot if tot > 0 else cnt)


# ==========================================================================
# Fast cache replays (MIN with a heap; the theory layer's MIN is O(n*cap))
# ==========================================================================
def lru_replay_counts(stream: Stream, cap: int,
                      host_shared: bool = False) -> Dict[str, float]:
    """LRU replay over the eager whole-call consult stream (host tier view)."""
    caches = {}
    hits = misses = evict = 0
    for c in stream.calls:
        cache = caches.setdefault("shared" if host_shared else c.device,
                                  OrderedDict())
        for k in c.key_set:
                if k in cache:
                    cache.move_to_end(k)
                    hits += 1
                else:
                    misses += 1
                    if len(cache) >= cap:
                        cache.popitem(last=False)
                        evict += 1
                    cache[k] = None
    n = hits + misses
    return {"hits": hits, "misses": misses, "hit_rate": hits / n if n else 0.0,
            "evictions": evict, "requests": n}


def min_replay_counts(keys: Sequence[Key], cap: int) -> Dict[str, float]:
    """Belady MIN (offline optimum) with a heap — O(n log cap)."""
    n = len(keys)
    nxt: Dict[Key, List[int]] = {}
    for i, k in enumerate(keys):
        nxt.setdefault(k, []).append(i)
    pos = {k: 0 for k in nxt}
    heap: List[Tuple[int, int, Key]] = []
    resident: Dict[Key, int] = {}
    serial = 0
    hits = 0
    for i, k in enumerate(keys):
        lst = nxt[k]
        pos[k] += 1
        future = lst[pos[k]] if pos[k] < len(lst) else INF
        if k in resident:
            hits += 1
            resident[k] = future
            serial += 1
            heapq.heappush(heap, (future, serial, k))
            continue
        if cap > 0 and len(resident) >= cap:
            while heap:
                f, s, vk = heapq.heappop(heap)
                if vk in resident and resident[vk] == f:
                    del resident[vk]
                    break
        resident[k] = future
        serial += 1
        heapq.heappush(heap, (future, serial, k))
    return {"hits": hits, "misses": n - hits,
            "hit_rate": hits / n if n else 0.0, "requests": n}
