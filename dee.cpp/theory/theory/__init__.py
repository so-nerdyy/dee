"""DEE-Theory: analytical performance model for exact sparse-MoE inference.

Reproducible analysis layer over in-repo artifacts.  Every constant is
declared in :mod:`theory.constants` with a provenance tag:

  MEASURED    - read directly from an in-repo artifact (path given)
  DERIVED     - computed from MEASURED values by stated arithmetic
  CALIBRATED  - fitted to in-repo measurements (fit + residual given)
  ASSUMPTION  - not present in-repo; external/conservative estimate, with range

Run ``python -m theory.run_all`` from ``dee.cpp/`` to regenerate every
figure, table and CSV in ``theory/figs/`` and ``theory/data/``.
"""

__version__ = "0.1.0"
