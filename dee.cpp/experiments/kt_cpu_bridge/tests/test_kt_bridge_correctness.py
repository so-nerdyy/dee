"""Component correctness: fp32 reference vs KT-emulated CPU path.

Uses dee trusted fixtures when reachable (scripts/deepseek_v4_expert_reference),
else the bridge's own mirrored reference (byte-identical semantics). Synthetic
fixtures use small multiples of 32. A real-expert path runs if
DEE_REAL_EXPERT_DIR provides real-expert-config.json; otherwise pytest skips.
Does NOT weaken existing dee gates.
"""
from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import torch
import pytest

BRIDGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE / "python"))
DEE_CPP = BRIDGE.parents[1]
sys.path.insert(0, str(DEE_CPP))

from kt_cpu_bridge.codec import REALISTIC_SCALE_BYTES  # noqa: E402
from kt_cpu_bridge.reference import (  # noqa: E402
    error_metrics,
    expert_forward_reference,
    kt_emulated_forward,
)

try:
    from scripts import deepseek_v4_expert_reference as dee_ref  # noqa: E402
    HAS_DEE_REF = True
except Exception:
    dee_ref = None
    HAS_DEE_REF = False


def _make_expert(hidden: int, inter: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    def rnd(shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
    def scl(shape):
        span = torch.randint(REALISTIC_SCALE_BYTES[0], REALISTIC_SCALE_BYTES[1],
                             shape, dtype=torch.int64, generator=g).to(torch.uint8)
        return span
    return (rnd((inter, hidden // 2)), scl((inter, hidden // 32)),
            rnd((hidden, inter // 2)), scl((hidden, inter // 32)),
            rnd((inter, hidden // 2)), scl((inter, hidden // 32)))


def test_matches_dee_trusted_reference_when_available():
    if not HAS_DEE_REF:
        pytest.skip("dee trusted reference not importable")
    hidden, inter = 64, 32
    p1, s1, p2, s2, p3, s3 = _make_expert(hidden, inter, 7)
    torch.manual_seed(11)
    x = torch.randn(3, hidden)
    got = expert_forward_reference(x, p1, s1, p2, s2, p3, s3, 1.0, 10.0)
    exp = dee_ref.expert_forward(x, p1, s1, p2, s2, p3, s3, swiglu_limit=10.0)
    assert torch.equal(got, exp), "bridge reference must be bitwise == dee trusted ref"


def test_clamp_asymmetry_and_disable():
    hidden, inter = 64, 32
    p1, s1, p2, s2, p3, s3 = _make_expert(hidden, inter, 3)
    torch.manual_seed(5)
    x = torch.randn(3, hidden) * 3.0
    # manual expected with asymmetric clamp
    import kt_cpu_bridge.reference as R
    w1 = R.dequantize_weight(p1, s1); w2 = R.dequantize_weight(p2, s2); w3 = R.dequantize_weight(p3, s3)
    gate = x @ w1.t(); up = x @ w3.t()
    exp = (torch.nn.functional.silu(torch.clamp(gate, max=10.0)) *
           torch.clamp(up, min=-10.0, max=10.0)) @ w2.t()
    got = expert_forward_reference(x, p1, s1, p2, s2, p3, s3, 1.0, 10.0)
    assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5)
    unclamped = (torch.nn.functional.silu(gate) * up) @ w2.t()
    assert not torch.allclose(got, unclamped, atol=1e-2, rtol=1e-2)
    assert torch.allclose(
        expert_forward_reference(x, p1, s1, p2, s2, p3, s3, 1.0, 0.0),
        unclamped, atol=1e-5, rtol=1e-5)


def test_routing_weight_placement_equivalence():
    """dee (before-w2) vs KT (after-w2) placement agree to fp32 noise."""
    hidden, inter = 64, 32
    p1, s1, p2, s2, p3, s3 = _make_expert(hidden, inter, 9)
    torch.manual_seed(4)
    x = torch.randn(2, hidden)
    w = 0.37
    a = expert_forward_reference(x, p1, s1, p2, s2, p3, s3, w, 10.0)
    b = expert_forward_reference(x, p1, s1, p2, s2, p3, s3, 1.0, 10.0) * w
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_kt_emulated_metrics_and_determinism():
    hidden, inter = 64, 32
    p1, s1, p2, s2, p3, s3 = _make_expert(hidden, inter, 13)
    torch.manual_seed(21)
    x = torch.randn(4, hidden)
    ref = expert_forward_reference(x, p1, s1, p2, s2, p3, s3, 0.8, 10.0)
    got = kt_emulated_forward(x, p1, s1, p2, s2, p3, s3, 0.8, 10.0)
    got2 = kt_emulated_forward(x, p1, s1, p2, s2, p3, s3, 0.8, 10.0)
    assert torch.equal(got, got2), "KT path must be deterministic"
    m = error_metrics(ref, got)
    assert m["finite"] and m["n"] == ref.numel()
    print("KT-emulated vs fp32:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()})
    # Recorded (not gates on upstream CI): bf16 boundaries keep RELATIVE error
    # small; absolute scale follows the (large) synthetic output magnitude, so
    # gate on relative metrics + cosine, not on raw max_abs.
    assert m["cosine"] > 0.999
    assert m["mean_rel"] < 0.05
    assert m["p95_rel"] < 0.15


def test_real_expert_if_available():
    d = os.environ.get("DEE_REAL_EXPERT_DIR")
    if not d:
        pytest.skip("run bench/verify_real_expert.py with a sealed local bundle")
    sys.path.insert(0, str(BRIDGE / "bench"))
    from verify_real_expert import verify
    config = json.loads((Path(d) / "real-expert-config.json").read_text(encoding="utf-8"))
    report = verify(**{k: Path(config[k]) for k in ("shard", "bundle", "seal", "executor")},
                    layer=config.get("layer", 0), expert=config.get("expert", 155))
    assert report["reference_pass"], report
