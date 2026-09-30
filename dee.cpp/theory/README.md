# theory — analytical performance model for exact sparse-MoE inference

Deliverables in this directory:

- **[`THEORY.md`](THEORY.md)** — the derivation report: expert popularity law,
  closed-form cache hit-rate model, temporal structure, throughput roofline
  (with the 0.21 tok/s anchor check), serving cost frontier, prefetch bound,
  sensitivity + regime map.
- **[`FALSIFICATION.md`](FALSIFICATION.md)** — numeric, range-bounded
  predictions, each tagged with the measurement that kills it, plus the
  "data needed" gap list and the known model failures.
- **`theory/`** — the reproducible Python package (stdlib + numpy/scipy/
  matplotlib). From this directory:

      python -m theory.run_all              # everything, ~5-7 min, deterministic
      python -m theory.run_all --fast       # skip bootstrap CIs
      python -m theory.run_all cache D G    # stage subsets (A-G or names)

  Stages: `provenance` (constant ledger), `popularity` (A), `cache` (B),
  `temporal` (C), `roofline` (D), `serving` (E), `prefetch` (F),
  `sensitivity` (G).
- **`data/`** — every generated CSV/JSON (see [`data/README.md`](data/README.md));
  `data/provenance.csv` carries the MEASURED / DERIVED / CALIBRATED /
  ASSUMPTION tag and source path for every constant used.
- **`figs/`** — every generated plot (21 figures).

Ground rules honored: every constant cites an in-repo artifact or is tagged
ASSUMPTION with a range; goodness-of-fit is reported including failures; the
empirical anchor (~0.21 tok/s at 0.29-0.37 GiB/s on 2xT4) is predicted within
2x or explained; zero network calls; the package writes only `data/` and
`figs/`.
