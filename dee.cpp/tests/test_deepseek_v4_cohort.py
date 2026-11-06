"""Phase-5 W1: cohort (lockstep batched) execution exactness.

The dee-serve v0 contract: a b=K cohort where every member shares the
same prompt length and position produces per-row outputs identical to
running each member through the b=1 path alone.  These tests build a
tiny synthetic DeepseekV4Model on CPU (fp32 direct FFN) and check that
property bitwise at the token level.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deepseek_v4_layer_reference import (  # noqa: E402
    DeepseekV4Layer,
    make_synthetic_layer_weights,
)
from scripts.deepseek_v4_model import (  # noqa: E402
    DeepseekV4Model,
    ModelConfig,
)

CFG = ModelConfig(
    vocab_size=64,
    dim=64,
    moe_inter_dim=128,
    n_layers=2,
    n_hash_layers=0,
    n_heads=4,
    n_routed=16,
    n_shared=1,
    topk=2,
    route_scale=1.5,
    swiglu_limit=10.0,
    q_lora_rank=32,
    head_dim=128,
    rope_head_dim=64,
    o_groups=2,
    o_lora_rank=32,
    window_size=8,
    index_n_heads=2,
    index_head_dim=128,
    index_topk=8,
    hc_mult=2,
    max_seq_len=32,
    compress_ratios=(4, 4),
)


def _build(max_batch: int) -> DeepseekV4Model:
    # The runtime sets bf16 as the default dtype (deepseek_v4_native_generate
    # does the same); quant helpers reject fp32 activations.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return _build_bf16(max_batch)
    finally:
        torch.set_default_dtype(prev_dtype)


class _DirectFFN:
    """Wrap the layer's fp32-direct fallback with the callable protocol the
    model forward expects (last_route / stats attributes)."""
    def __init__(self, layer: DeepseekV4Layer):
        self._fn = layer._ffn_fp32_direct
        self.last_route: dict = {}
        self.stats: dict = {}

    def __call__(self, x, input_ids, capture):
        return self._fn(x, input_ids, capture)


def _build_bf16(max_batch: int) -> DeepseekV4Model:
    g = torch.Generator().manual_seed(7)
    cfg = CFG
    layers = []
    for i in range(cfg.n_layers):
        w, _routed, _shared = make_synthetic_layer_weights(
            cfg.layer_config(i), seed=100 + i, n_experts=8)
        layer = DeepseekV4Layer(
            cfg.layer_config(i), w, device="cpu", max_batch=max_batch,
            layer_id=i)
        layer.ffn_fn = _DirectFFN(layer)
        layers.append(layer)
    mix = cfg.hc_mult  # checkpoint: hc_head_fn [hc_mult, hc_mult*dim]
    return DeepseekV4Model(
        cfg,
        embed=torch.randn(cfg.vocab_size, cfg.dim, generator=g) * 0.1,
        layers0=layers,
        layers1=[],
        hc_head_fn=(torch.randn(mix, cfg.hc_mult * cfg.dim, generator=g) * 0.1).float(),
        hc_head_base=(torch.randn(mix, generator=g) * 0.1).float(),
        hc_head_scale=(torch.randn(1, generator=g).abs() + 0.5).float(),
        norm_w=torch.randn(cfg.dim, generator=g) * 0.1 + 1.0,
        head=torch.randn(cfg.vocab_size, cfg.dim, generator=g) * 0.1,
        device0="cpu", device1="cpu", diagnostics=True)


def _prompts(k: int, length: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(42)
    return torch.randint(0, CFG.vocab_size, (k, length),
                         generator=g, dtype=torch.long)


def _run_bf16(fn, *args, **kwargs):
    """Run a generate/generate_cohort call under the runtime's bf16 default."""
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return fn(*args, **kwargs)
    finally:
        torch.set_default_dtype(prev_dtype)


def test_cohort_b1_matches_generate() -> None:
    """A K=1 cohort is bit-identical to the b=1 generate() path."""
    ids = _prompts(1, 8)
    solo = _build(1)
    solo.reset_state()
    ref = _run_bf16(solo.generate, ids, 6, eos_id=-1)
    co = _build(2)
    co.reset_state()
    got = _run_bf16(co.generate_cohort, ids, 6, eos_id=-1)
    assert len(got) == 1
    assert got[0] == ref


def test_cohort_k2_rows_match_sequential() -> None:
    """Each row of a K=2 cohort equals the b=1 run of the same prompt."""
    ids = _prompts(2, 8)
    refs = []
    for r in range(2):
        m = _build(1)
        m.reset_state()
        refs.append(_run_bf16(m.generate, ids[r:r + 1], 6, eos_id=-1))
    co = _build(2)
    co.reset_state()
    got = _run_bf16(co.generate_cohort, ids, 6, eos_id=-1)
    assert got == refs


def test_cohort_k4_equal_length() -> None:
    ids = _prompts(4, 8)
    refs = []
    for r in range(4):
        m = _build(1)
        m.reset_state()
        refs.append(_run_bf16(m.generate, ids[r:r + 1], 5, eos_id=-1))
    co = _build(4)
    co.reset_state()
    got = _run_bf16(co.generate_cohort, ids, 5, eos_id=-1)
    assert got == refs


def test_cohort_step_hook_sees_all_rows() -> None:
    """post_step_hook fires once per cohort forward with K tokens."""
    ids = _prompts(3, 8)
    seen = []
    co = _build(3)
    co.reset_state()
    _run_bf16(
        co.generate_cohort, ids, 4, eos_id=-1,
        post_step_hook=lambda step, toks: seen.append((step, list(toks))))
    assert [s for s, _ in seen] == [0, 1, 2, 3]
    assert all(len(t) == 3 for _, t in seen)


def test_generate_unchanged_on_wide_model() -> None:
    """b=1 generate() on a max_batch=2 model matches a max_batch=1 model —
    the kv_cache row slicing must not perturb a narrow batch."""
    ids = _prompts(1, 8)
    a = _build(1)
    a.reset_state()
    ref = _run_bf16(a.generate, ids, 6, eos_id=-1)
    b = _build(2)
    b.reset_state()
    assert _run_bf16(b.generate, ids, 6, eos_id=-1) == ref


def test_warm_process_sequential_units_bit_identical() -> None:
    """Sequential units in one warm process must reproduce bit-exactly.

    Regression guard for the Phase-5 Kaggle finding: units >=1 in a warm
    process diverged on GPU.  The Python/model path must hold this
    invariant (verified on CPU); a failure here means model-side state
    leaked through reset_state."""
    ids = _prompts(1, 8)
    m = _build(1)
    outs = []
    for _ in range(3):
        m.reset_state()
        outs.append(_run_bf16(m.generate, ids, 6, eos_id=-1))
    assert outs[0] == outs[1] == outs[2]


def test_warm_process_sequential_cohorts_bit_identical() -> None:
    """Back-to-back generate_cohort calls on one model must reproduce —
    the cohort-path analogue of the warm-process guard."""
    ids = _prompts(2, 8)
    co = _build(2)
    outs = []
    for _ in range(3):
        co.reset_state()
        outs.append(_run_bf16(co.generate_cohort, ids, 5, eos_id=-1))
    assert outs[0] == outs[1] == outs[2]
