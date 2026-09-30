"""A. Expert popularity law: fits, goodness-of-fit, entropy, coverage.

Families compared on the *rank-frequency* profile of expert access counts
over the full record universe (seen + never-seen categories):

  1. Uniform baseline        p_(i) = 1/N
  2. Pure Zipf (power law)   p_(i) = i^(-s) / Z(s,N)
  3. Zipf-Mandelbrot         p_(i) = (i+q)^(-s) / Z(s,q,N)
  4. Lognormal rank curve    p_(i) proportional to exp(mu + sigma*z_i),
                             z_i = Phi^-1(1 - (i-0.5)/N)   (lognormal quantile
                             rate over the category space; the "lognormal on
                             frequency" family, ranked)
  5. Poisson-lognormal abundance (separate sampling model: c_i ~ Poisson(L f_i),
     f_i iid LogNormal) reported on its own likelihood scale.

Likelihoods use the *sorted (label-marginalized) assignment*: category labels
are exchangeable under every candidate family, so the maximized multinomial
likelihood assigns the largest p_(i) to the largest observed count.  The
multinomial coefficient is identical across families and cancels in the
comparison: log-likelihoods are comparable ACROSS families but are not
absolute generative scores.  KS-style statistic D = max_m |C_emp(m) -
C_model(m)| over rank prefixes of the coverage curve (tail-relevant).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, special, stats

from .util import bootstrap_ci, dlog, figure, rng, savefig, write_csv, write_json


# ==========================================================================
# Core objects
# ==========================================================================
@dataclass
class Profile:
    name: str
    counts: np.ndarray            # counts per category (any order)
    n_categories: int             # full category space (incl. never-seen)
    n: int                        # total observed accesses
    source: str

    def __post_init__(self):
        self.counts = np.asarray(self.counts, dtype=np.int64)
        self.n = int(self.counts.sum())
        self.sorted_counts = np.sort(self.counts)[::-1]
        self.emp_p = self.sorted_counts / max(1, self.n)
        self.n_seen = int((self.counts > 0).sum())
        self.zeros = self.n_categories - self.n_seen
        # standard-normal quantile grid for rank-curve families
        u = 1.0 - (np.arange(1, self.n_categories + 1) - 0.5) / self.n_categories
        self.z_grid = stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))

    def coverage(self) -> np.ndarray:
        """C(m) = cumulative empirical mass over the top-m observed categories."""
        c = np.cumsum(self.emp_p)
        if self.n_categories > c.size:
            c = np.concatenate([c, np.full(self.n_categories - c.size, 1.0)])
        return c


def profile_from_stream(pairs: Sequence[Tuple[int, int]], n_layers: int,
                        experts_per_layer: int, name: str, source: str) -> Profile:
    cnt = Counter(pairs)
    n_categories = n_layers * experts_per_layer
    counts = np.zeros(n_categories, dtype=np.int64)
    for (layer, e), v in cnt.items():
        if 0 <= layer < n_layers and 0 <= e < experts_per_layer:
            counts[layer * experts_per_layer + e] = v
    return Profile(name, counts, n_categories, int(counts.sum()), source)


def profile_per_layer(pairs: Sequence[Tuple[int, int]], experts_per_layer: int,
                      n_layers: int, source: str) -> Dict[int, Profile]:
    by_layer: Dict[int, Counter] = {}
    for (layer, e) in pairs:
        by_layer.setdefault(layer, Counter())[e] += 1
    out = {}
    for layer, cnt in by_layer.items():
        counts = np.zeros(experts_per_layer, dtype=np.int64)
        for e, v in cnt.items():
            if 0 <= e < experts_per_layer:
                counts[e] = v
        out[layer] = Profile(f"layer{layer:02d}", counts, experts_per_layer,
                             int(counts.sum()), source)
    return out


# ==========================================================================
# Shared scoring
# ==========================================================================
def _loglik_sorted(model_p: np.ndarray, sorted_counts: np.ndarray) -> float:
    """Sorted-assignment multinomial log-likelihood (constant dropped)."""
    m = min(model_p.size, sorted_counts.size)
    p = np.clip(model_p[:m], 1e-300, 1.0)
    return float(np.sum(sorted_counts[:m] * np.log(p)))


def _ks(model_p: np.ndarray, prof: Profile) -> float:
    m = min(model_p.size, prof.emp_p.size)
    return float(np.max(np.abs(np.cumsum(prof.emp_p[:m]) - np.cumsum(model_p[:m]))))


def zipf_mandelbrot_p(s: float, q: float, n_categories: int) -> np.ndarray:
    """p_(i) = (i+q)^(-s) / Z, computed in log space (no underflow)."""
    i = np.arange(1, n_categories + 1, dtype=float)
    lp = -s * np.log(i + q)
    lp -= lp.max()
    w = np.exp(lp)
    return w / w.sum()


# ==========================================================================
# Families
# ==========================================================================
def fit_uniform(prof: Profile) -> Dict[str, float]:
    p = np.full(prof.n_categories, 1.0 / prof.n_categories)
    return {"family": "uniform", "loglik": _loglik_sorted(p, prof.sorted_counts),
            "ks_D": _ks(p, prof), "n_params": 0, "degenerate_to_geometric": False}


def fit_zipf_pure(prof: Profile) -> Dict[str, float]:
    """Pure power law (q = 0)."""
    n_cat = prof.n_categories
    cnt = prof.sorted_counts
    log_i = np.log(np.arange(1, n_cat + 1, dtype=float))

    def nll(theta):
        s = np.exp(theta[0])
        lp = -s * log_i
        lp -= lp.max()
        p = np.exp(lp)
        return -_loglik_sorted(p / p.sum(), cnt)

    best = None
    for s0 in (0.3, 0.8, 1.5):
        r = optimize.minimize(nll, [np.log(s0)], method="Nelder-Mead",
                              options={"maxiter": 600, "xatol": 1e-5, "fatol": 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    s = float(np.exp(best.x[0]))
    p = zipf_mandelbrot_p(s, 0.0, n_cat)
    return {"family": "zipf_pure", "s": s, "q": 0.0, "loglik": float(-best.fun),
            "ks_D": _ks(p, prof), "n_params": 1, "degenerate_to_geometric": False}


def fit_zipf_mandelbrot(prof: Profile, x0=None, n_starts: int = 3,
                        maxiter: int = 800) -> Dict[str, float]:
    """Zipf-Mandelbrot MLE on the sorted-assignment multinomial.

    The free-q MLE is known to degenerate to the *geometric* limit
    (q -> large with s/q = const, i.e. exponential decay over ranks, NOT a
    regularly-varying power law) whenever the observed rank curve has an
    exponential tail.  ``degenerate_to_geometric`` reports that outcome.
    """
    n_cat = prof.n_categories
    cnt = prof.sorted_counts

    # Parameterize by the DECAY RATE over ranks, r = s/q:  s = r*q with
    # q = exp(u).  In these coordinates the geometric limit (q -> large,
    # s/q = r fixed) is a bounded interior point instead of a runaway, so
    # the MLE converges to a well-conditioned optimum.
    def nll(theta):
        r_ = np.exp(theta[0])
        q_ = np.exp(theta[1])
        return -_loglik_sorted(zipf_mandelbrot_p(r_ * q_, q_, n_cat), cnt)

    starts = []
    if x0 is not None:
        starts.append((np.log(max(x0[0] / max(x0[1], 1e-9), 1e-6)), np.log(max(x0[1], 1.0))))
    for r0, q0 in ((1e-3, 5.0), (0.05, 30.0), (0.3, 5.0), (0.01, 200.0)):
        starts.append((np.log(r0), np.log(q0)))
    best = None
    for x_start in starts[:max(2, n_starts)]:
        r = optimize.minimize(nll, list(x_start), method="Nelder-Mead",
                              options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    decay = float(np.exp(best.x[0]))                 # s/q, geometric decay rate
    q = float(np.exp(best.x[1]))
    s = decay * q
    p = zipf_mandelbrot_p(s, q, n_cat)
    curvature = float(decay ** 2 / max(s, 1e-12))    # 1/s, scaled shape of the bend
    bend_rank = float(1.0 / curvature) if curvature > 0 else float("inf")
    degenerate = bool(q > 0.5 * prof.n_seen or bend_rank > prof.n_seen)
    return {"family": "zipf_mandelbrot", "s": s, "q": q, "loglik": float(-best.fun),
            "ks_D": _ks(p, prof), "n_params": 2,
            "geometric_decay_rate": float(decay),
            "geometric_decay_per_rank_pct": float(100.0 * decay),
            "powerlaw_bend_rank": bend_rank,
            "degenerate_to_geometric": degenerate}


def fit_lognormal_rank(prof: Profile) -> Dict[str, float]:
    """Lognormal rate over the category space, ranked (multinomial family).

    Each expert carries a latent rate f_i = exp(mu + sigma*z_i) with z_i the
    normal quantile assigned to rank i; p_(i) = f_i / sum f.  This is the
    "lognormal on frequency" family evaluated as a rank curve over ALL
    categories (the unseen tail carries its own mass), so its likelihood is
    directly comparable to Zipf / uniform.
    """
    n_cat = prof.n_categories
    cnt = prof.sorted_counts
    z = prof.z_grid

    def curve(mu, sg):
        lp = mu + sg * z
        lp -= lp.max()
        w = np.exp(lp)
        return w / w.sum()

    def nll(theta):
        return -_loglik_sorted(curve(theta[0], np.exp(theta[1])), cnt)

    best = None
    for mu0, sg0 in ((0.5, 0.8), (1.0, 1.5), (0.0, 2.5)):
        r = optimize.minimize(nll, [mu0, np.log(sg0)], method="Nelder-Mead",
                              options={"maxiter": 600, "xatol": 1e-5, "fatol": 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    mu, sigma = float(best.x[0]), float(np.exp(best.x[1]))
    p = curve(mu, sigma)
    return {"family": "lognormal_rank", "mu": mu, "sigma": sigma,
            "loglik": float(-best.fun), "ks_D": _ks(p, prof), "n_params": 2,
            "degenerate_to_geometric": False}


def fit_poisson_lognormal(prof: Profile, maxiter: int = 500) -> Dict[str, float]:
    """Poisson-lognormal species-abundance fit (separate sampling model).

    c_i ~ Poisson(L * f_i), f_i iid LogNormal(mu, sigma); latent rate
    marginalized by 40-node Gauss-Hermite quadrature.  NOT a multinomial
    model: its score lives in ``loglik_poissonmix`` and is comparable only to
    other Poisson-mixture families.  ``loglik`` is NaN by construction.
    The never-seen categories share one zero-count contribution (factored by
    ``prof.zeros``), so only the seen counts enter the per-category loop.
    """
    from numpy.polynomial.hermite import hermgauss

    nodes, weights = hermgauss(40)
    logw = np.log(weights) - 0.5 * np.log(np.pi)
    seen = prof.sorted_counts[:prof.n_seen].astype(float)
    lgamma_c = special.gammaln(seen + 1.0) if seen.size else np.zeros(0)
    c = seen[:, None]
    L = float(prof.n)
    n_zero = float(prof.zeros)

    def nll(theta):
        mu, sg = theta[0], np.exp(theta[1])
        log_lam = np.log(L) + mu + sg * nodes
        lam = np.exp(log_lam)
        zero_term = special.logsumexp(logw - lam)
        if seen.size:
            logp = (c * log_lam[None, :] - lam[None, :] - lgamma_c[:, None])
            ll_seen = special.logsumexp(logp + logw[None, :], axis=1).sum()
        else:
            ll_seen = 0.0
        return -(ll_seen + n_zero * zero_term)

    seen_mean = float(seen.mean()) if seen.size else 1.0
    mu0 = math.log(max(seen_mean / max(1.0, L), 1e-6))
    best = None
    for sg0 in (0.8, 1.8):
        r = optimize.minimize(nll, [mu0, np.log(sg0)], method="Nelder-Mead",
                              options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    mu, sigma = float(best.x[0]), float(np.exp(best.x[1]))
    lp = mu + sigma * prof.z_grid
    lp -= lp.max()
    w = np.exp(lp)
    p = w / w.sum()
    # A Poisson mixture is not a multinomial model, so its rank-curve KS is
    # reported for shape reference only (the multinomial score is NaN).
    return {"family": "poisson_lognormal", "mu": mu, "sigma": sigma,
            "loglik": float("nan"), "loglik_poissonmix": float(-best.fun),
            "ks_D": _ks(p, prof), "n_params": 3, "degenerate_to_geometric": False,
            "ks_scale": "shape_reference_only"}


FAMILIES = [
    ("uniform", fit_uniform),
    ("zipf_pure", fit_zipf_pure),
    ("zipf_mandelbrot", fit_zipf_mandelbrot),
    ("lognormal_rank", fit_lognormal_rank),
    ("poisson_lognormal", fit_poisson_lognormal),
]


def entropy_bits(counts: np.ndarray, n_categories: int) -> Tuple[float, float]:
    """Plug-in Shannon entropy (bits) and log2(N) reference."""
    c = counts[counts > 0].astype(float)
    p = c / c.sum()
    h = float(-(p * np.log2(p)).sum())
    return h, float(np.log2(n_categories))


def fit_all(prof: Profile, quick: bool = False) -> List[Dict[str, float]]:
    rows = []
    for name, fn in FAMILIES:
        if quick and name == "zipf_mandelbrot":
            r = fn(prof, n_starts=2, maxiter=350)
        elif quick and name == "poisson_lognormal":
            r = fn(prof, maxiter=250)
        else:
            r = fn(prof)
        r.update({"profile": prof.name, "n_obs": prof.n,
                  "n_categories": prof.n_categories, "n_seen": prof.n_seen,
                  "zero_frac": prof.zeros / prof.n_categories,
                  "source": prof.source})
        k = r.get("n_params", 0)
        ll = r.get("loglik", float("nan"))
        r["aic"] = -2.0 * ll + 2 * k if np.isfinite(ll) else float("nan")
        r["bic"] = (-2.0 * ll + k * np.log(max(2, prof.n))
                    if np.isfinite(ll) else float("nan"))
        rows.append(r)
    return rows


def _model_curve(fit: Dict[str, float], n_categories: int,
                 z_grid: Optional[np.ndarray] = None) -> np.ndarray:
    fam = fit["family"]
    if fam == "uniform":
        return np.full(n_categories, 1.0 / n_categories)
    if fam in ("zipf_pure", "zipf_mandelbrot"):
        return zipf_mandelbrot_p(fit["s"], fit.get("q", 0.0), n_categories)
    if fam in ("lognormal_rank", "poisson_lognormal"):
        z = z_grid if z_grid is not None else stats.norm.ppf(
            np.clip(1.0 - (np.arange(1, n_categories + 1) - 0.5) / n_categories,
                    1e-12, 1 - 1e-12))
        lp = fit["mu"] + fit["sigma"] * z
        lp -= lp.max()
        w = np.exp(lp)
        return w / w.sum()
    raise ValueError(fam)


# ==========================================================================
# Driver
# ==========================================================================
def run(fast: bool = False) -> Dict[str, object]:
    from .sources import (ORNITH_EXPERTS, ORNITH_LAYERS, load_ornith_streams,
                          load_sealed_stream)

    dlog("A. popularity law")
    orn = load_ornith_streams()
    sealed = load_sealed_stream()

    traces = []
    sealed_pairs = sealed.stream()
    traces.append(("sealed_dsv4_pooled", profile_from_stream(
        sealed_pairs, sealed.n_layers, sealed.experts_per_layer,
        "sealed_dsv4_pooled",
        "dee.cpp/experiments/route_pipeline/fill-live-t4x2-20260909/routed_experts.jsonl")))
    sealed_layers = profile_per_layer(sealed_pairs, sealed.experts_per_layer,
                                      sealed.n_layers, "sealed_dsv4")
    orn_pairs = []
    for s in orn.values():
        orn_pairs.extend(s.stream())
    traces.append(("ornith_pooled", profile_from_stream(
        orn_pairs, ORNITH_LAYERS, ORNITH_EXPERTS, "ornith_pooled",
        "benchmark_reports/milestone-2.5/.../expert-trace.jsonl.gz route_selection")))
    orn_runs = {}
    for rid, s in orn.items():
        orn_runs[rid] = profile_from_stream(s.stream(), s.n_layers,
                                           s.experts_per_layer, f"ornith_{rid}",
                                           "expert-trace.jsonl.gz route_selection")

    # ---- fits ----------------------------------------------------------
    fit_rows: List[Dict[str, float]] = []
    layer_fit_rows: List[Dict[str, float]] = []
    for name, prof in traces:
        fit_rows.extend(fit_all(prof))
    for layer, lp in sorted(sealed_layers.items()):
        for r in fit_all(lp, quick=True):
            r["layer"] = layer
            r["scope"] = "layer"
            r["well_supported"] = lp.n >= 50
            layer_fit_rows.append(r)
    for rid, prof in orn_runs.items():
        for r in fit_all(prof, quick=True):
            r["run_id"] = rid
            r["scope"] = "run"
            layer_fit_rows.append(r)

    # ---- parametric bootstrap CIs (access-level resampling) -------------
    boot_rows = []
    n_boot_iter = 0 if fast else 60
    for name, prof in traces:
        base_zm = fit_zipf_mandelbrot(prof)
        base_lr = fit_lognormal_rank(prof)
        g = rng()
        idx = np.arange(prof.n_categories)
        p_acc = np.clip(prof.counts / max(1, prof.n), 0, 1)
        acc = {"zipf_mandelbrot_s": [], "zipf_mandelbrot_q": [],
               "lognormal_rank_mu": [], "lognormal_rank_sigma": []}
        for _ in range(n_boot_iter):
            draw = g.choice(idx, size=prof.n, replace=True, p=p_acc)
            cnt = np.bincount(draw, minlength=prof.n_categories)
            bp = Profile(f"{name}_boot", cnt, prof.n_categories,
                         int(cnt.sum()), "bootstrap")
            rz = fit_zipf_mandelbrot(bp, x0=(base_zm["s"], base_zm["q"]),
                                     n_starts=1, maxiter=250)
            rl = fit_lognormal_rank(bp)
            acc["zipf_mandelbrot_s"].append(rz["s"])
            acc["zipf_mandelbrot_q"].append(rz["q"])
            acc["lognormal_rank_mu"].append(rl["mu"])
            acc["lognormal_rank_sigma"].append(rl["sigma"])
        for key, vals in acc.items():
            ci = bootstrap_ci(vals, n_boot=500) if vals else {
                "estimate": float("nan"), "lo": float("nan"), "hi": float("nan")}
            boot_rows.append({"profile": name, "param": key,
                              "estimate": ci["estimate"], "lo": ci["lo"],
                              "hi": ci["hi"], "n_boot": n_boot_iter})

    # ---- entropy -------------------------------------------------------
    ent_rows = []
    for name, prof in traces:
        h, hmax = entropy_bits(prof.counts, prof.n_categories)
        ent_rows.append({"profile": name, "scope": "pooled", "layer": "",
                         "entropy_bits": h, "log2_N_bits": hmax,
                         "eff_experts": 2 ** h, "n_obs": prof.n,
                         "n_categories": prof.n_categories})
    for layer, lp in sorted(sealed_layers.items()):
        h, hmax = entropy_bits(lp.counts, lp.n_categories)
        ent_rows.append({"profile": "sealed_dsv4", "scope": "layer", "layer": layer,
                         "entropy_bits": h, "log2_N_bits": hmax,
                         "eff_experts": 2 ** h, "n_obs": lp.n,
                         "n_categories": lp.n_categories})

    # ---- coverage curves -----------------------------------------------
    cov_rows = []
    coverage_store = {}
    for name, prof in traces:
        cov = prof.coverage()
        coverage_store[name] = cov
        for m in range(1, prof.n_categories + 1):
            cov_rows.append({"profile": name, "m": m, "C_m": float(cov[m - 1]),
                             "m_frac": m / prof.n_categories})
    for layer, lp in sorted(sealed_layers.items()):
        cov = lp.coverage()
        for m in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            if m <= cov.size:
                cov_rows.append({"profile": f"sealed_dsv4_layer{layer:02d}", "m": m,
                                 "C_m": float(cov[m - 1]),
                                 "m_frac": m / lp.n_categories})

    # ---- tail diagnostics (where the fits fail) ------------------------
    tail_rows = []
    for name, prof in traces:
        fits = {r["family"]: r for r in fit_rows if r["profile"] == name}
        curves = {fam: _model_curve(r, prof.n_categories, prof.z_grid)
                  for fam, r in fits.items() if np.isfinite(r.get("loglik", float("nan")))}
        m = prof.emp_p.size
        regions = [("head_1_5", 1, 5),
                   ("head_5pct", 1, max(5, m // 20)),
                   ("mid_5pct_50pct", max(2, m // 20), m // 2),
                   ("tail_half", max(1, m // 2), m),
                   ("tail_10pct", max(1, int(m * 0.9)), m)]
        for tag, lo, hi in regions:
            hi = min(hi, m)
            if hi <= lo:
                continue
            e_mass = float(prof.emp_p[lo - 1:hi].sum())
            row = {"profile": name, "region": tag, "rank_lo": lo, "rank_hi": hi,
                   "emp_mass": e_mass}
            for fam, p in curves.items():
                model_mass = float(p[lo - 1:hi].sum())
                row[f"{fam}_mass"] = model_mass
                row[f"{fam}_abs_err"] = abs(model_mass - e_mass)
                row[f"{fam}_rel_err"] = ((model_mass - e_mass) / e_mass
                                         if e_mass > 0 else float("nan"))
            tail_rows.append(row)
        tail_rows.append({"profile": name, "region": "zero_tail",
                          "rank_lo": prof.n_categories - prof.zeros,
                          "rank_hi": prof.n_categories, "emp_mass": 0.0})

    write_csv("popularity_fits.csv", fit_rows + layer_fit_rows)
    write_csv("popularity_bootstrap.csv", boot_rows)
    write_csv("popularity_entropy.csv", ent_rows)
    write_csv("popularity_coverage.csv", cov_rows)
    write_csv("popularity_tail.csv", tail_rows)
    write_json("popularity_summary.json", {
        "pooled_fits": fit_rows,
        "bootstrap": boot_rows,
        "entropy": ent_rows,
    })

    _plots(traces, sealed_layers, fit_rows, coverage_store)
    dlog("   fits ->", [(r["profile"], r["family"], round(r["loglik"], 1)
                         if np.isfinite(r.get("loglik", float("nan"))) else None,
                         round(r["ks_D"], 3)) for r in fit_rows])
    return {"fits": fit_rows, "layer_fits": layer_fit_rows,
            "coverage": coverage_store, "sealed_layers": sealed_layers,
            "traces": traces, "tail": tail_rows}


# ==========================================================================
# Figures
# ==========================================================================
def _plots(traces, sealed_layers, fit_rows, coverage_store) -> None:
    fig, ax = figure("popularity_rankfreq", (7.6, 4.8))
    colors = {"sealed_dsv4_pooled": "#1f77b4", "ornith_pooled": "#d62728"}
    styles = {"zipf_mandelbrot": "-", "zipf_pure": "-.", "lognormal_rank": "--"}
    for name, prof in traces:
        emp = prof.emp_p[prof.emp_p > 0]
        ax.loglog(np.arange(1, emp.size + 1), emp, ".", ms=3,
                  color=colors.get(name, "k"),
                  label=f"{name} empirical (n={prof.n}, N={prof.n_categories})")
        for r in fit_rows:
            if r["profile"] != name or r["family"] not in styles:
                continue
            p = _model_curve(r, prof.n_categories, prof.z_grid)
            if r["family"] == "zipf_mandelbrot":
                lab = (f"Zipf-Mandelbrot s={r['s']:.2f} q={r['q']:.0f}"
                       + (" [geometric-degenerate]" if r.get("degenerate_to_geometric") else ""))
            elif r["family"] == "zipf_pure":
                lab = f"Zipf s={r['s']:.2f}"
            else:
                lab = f"lognormal rank mu={r['mu']:.2f} sig={r['sigma']:.2f}"
            ax.loglog(np.arange(1, p.size + 1), p, styles[r["family"]],
                      lw=1.1, color=colors.get(name, "k"), alpha=0.8, label=lab)
    ax.set_xlabel("rank i")
    ax.set_ylabel("access probability $p_{(i)}$")
    ax.set_title("Expert popularity: rank-frequency vs candidate families")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    savefig(fig, "popularity_rankfreq.png")

    fig, ax = figure("popularity_coverage", (7.6, 4.8))
    for name, cov in coverage_store.items():
        m = np.arange(1, cov.size + 1)
        ax.plot(m / cov.size, cov, "-", lw=1.4, label=f"{name} coverage C(m)")
        for m_frac in (0.05, 0.10, 0.25):
            mm = max(1, int(m_frac * cov.size))
            ax.plot([m_frac], [cov[mm - 1]], "o", ms=5)
            ax.annotate(f"{cov[mm-1]*100:.1f}%", (m_frac, cov[mm - 1]),
                        textcoords="offset points", xytext=(4, -8), fontsize=7)
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=1, label="uniform C(m)=m/N")
    ax.set_xlabel("m / N (fraction of record universe)")
    ax.set_ylabel("captured traffic C(m)")
    ax.set_title("Coverage: fraction of accesses captured by the top-m records")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig(fig, "popularity_coverage.png")

    fig, ax = figure("popularity_perlayer", (7.0, 4.4))
    xs, ys, ns, deg = [], [], [], []
    for layer, lp in sorted(sealed_layers.items()):
        z = fit_zipf_mandelbrot(lp, n_starts=1, maxiter=250)
        xs.append(z["s"])
        ys.append(z["q"])
        ns.append(lp.n)
        deg.append(z.get("degenerate_to_geometric", False))
    sc = ax.scatter(xs, ys, c=ns, cmap="viridis", s=28)
    for (layer, lp), x, y, d in zip(sorted(sealed_layers.items()), xs, ys, deg):
        ax.annotate(f"{layer}{'*' if d else ''}", (x, y), fontsize=6, alpha=0.7,
                    textcoords="offset points", xytext=(3, 2))
    fig.colorbar(sc, ax=ax, label="observations in layer")
    ax.set_xlabel("Zipf-Mandelbrot exponent s")
    ax.set_ylabel("Zipf-Mandelbrot offset q")
    ax.set_title("Per-layer Zipf-Mandelbrot fits, sealed trace ('*' = geometric-degenerate)")
    ax.grid(True, alpha=0.3)
    savefig(fig, "popularity_perlayer.png")
