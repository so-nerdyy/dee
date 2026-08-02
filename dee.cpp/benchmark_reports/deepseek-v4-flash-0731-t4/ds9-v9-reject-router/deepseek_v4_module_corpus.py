"""DS8 input corpus for the expert runtime.

Two sources:

1. **Official hidden-state traces** (primary): JSON/npz files capturing real
   hidden states from the pinned official model (DS5 dependency).  The loader
   accepts either a ``.npz`` (array key ``hidden_states``) or a ``.json``
   (list of numbers).  When no trace is present the generator falls back to
   synthetic vectors and records that the official trace was absent.

2. **Deterministic synthetic distributions** (supplementary robustness):
   - normal:        N(0, 1)
   - low_magnitude: N(0, 0.01)
   - high_magnitude: N(0, 20) clipped to the official swiglu range
   - sparse:        90% zeros, 10% N(0, 1)
   - adversarial:   alternating sign pattern (+1/-1 with magnitude 1)
   - repeated:      a single row broadcast to every token
   - near_zero:     N(0, 1e-6)

Every generator is seeded deterministically (per-case seed derived from a
base seed), so the corpus is reproducible across runs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def load_official_trace(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Load a trusted hidden-state trace.  Returns (tensor_or_None, note)."""
    if path is None or not Path(path).is_file():
        return None, "official hidden-state trace absent"
    try:
        if str(path).endswith(".npz"):
            if np is None:
                return None, "numpy unavailable"
            with np.load(path) as data:
                arr = data["hidden_states"]
            tensor = torch.from_numpy(arr).float()
        else:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            tensor = torch.tensor(raw, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        return {"official_trace": tensor}, (
            f"official hidden-state trace loaded from {path}"
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"official trace failed to load: {exc}"


def _seeded(base_seed: int, salt: str) -> Any:
    if torch is None:
        raise RuntimeError("torch required for corpus generation")
    rng = torch.Generator().manual_seed((base_seed * 7919 + len(salt)) & 0x7FFFFFFF)
    return rng


def synthetic_corpus(
    n_tokens: int,
    hidden: int,
    *,
    base_seed: int = 0,
    names: tuple[str, ...] = (
        "normal", "low_magnitude", "high_magnitude", "sparse",
        "adversarial", "repeated", "near_zero",
    ),
) -> list[tuple[str, torch.Tensor]]:
    """Build the deterministic synthetic input corpus."""
    out: list[tuple[str, torch.Tensor]] = []
    for name in names:
        rng = _seeded(base_seed, name)
        if name == "normal":
            x = torch.randn(n_tokens, hidden, generator=rng)
        elif name == "low_magnitude":
            x = torch.randn(n_tokens, hidden, generator=rng) * 0.01
        elif name == "high_magnitude":
            x = torch.randn(n_tokens, hidden, generator=rng) * 20.0
            x = x.clamp(-448.0, 448.0)  # fp8 e4m3 range guard
        elif name == "sparse":
            x = torch.randn(n_tokens, hidden, generator=rng)
            x[torch.rand(n_tokens, hidden, generator=rng) > 0.1] = 0.0
        elif name == "adversarial":
            row = torch.arange(hidden, dtype=torch.float32) % 2 * 2.0 - 1.0
            x = row.unsqueeze(0).expand(n_tokens, hidden).clone()
        elif name == "repeated":
            x = torch.randn(1, hidden, generator=rng).expand(n_tokens, hidden).clone()
        elif name == "near_zero":
            x = torch.randn(n_tokens, hidden, generator=rng) * 1e-6
        else:
            raise ValueError(f"unknown corpus case {name!r}")
        out.append((name, x))
    return out


def build_corpus(
    n_tokens: int,
    hidden: int,
    *,
    base_seed: int = 0,
    official_trace: Path | None = None,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, Any]]:
    """Full DS8 corpus: synthetic cases + optional official trace."""
    cases = synthetic_corpus(n_tokens, hidden, base_seed=base_seed)
    official, note = load_official_trace(official_trace)
    if official is not None:
        cases.append(("official_trace", official["official_trace"]))
    meta = {"n_tokens": n_tokens, "hidden": hidden, "base_seed": base_seed,
            "official_trace_note": note}
    return cases, meta
