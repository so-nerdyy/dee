"""Stage-profile wiring tests for the DeepSeek-V4 layer wrappers.

The remote harness resolves CUDA events only after measured generation, so
these tests exercise the surrounding contract on CPU with stub FFNs:

1. The wrapper propagates ``start_pos`` to a native FFN profiler before the
   FFN records its row.
2. ``reset_state`` chains the profile reset into the FFN.
3. ``stage_profile_snapshot`` merges the FFN's fine-grained phases (router,
   routed_dispatch_and_native, routed_combine, shared_expert, output_cast)
   into the wrapper totals and per-start-position tables.
4. The wrapper snapshot keeps the aggregation keys the model requires
   (``totals_ms`` and ``per_start_pos_ms``) even with no FFN snapshot present
   (reference fp32 backend / cache backend).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.deepseek_v4_layer_candidate import DeepseekV4NativeFfn
from scripts.deepseek_v4_layer_reference import (
    DeepseekV4Layer,
    LayerConfig,
    make_synthetic_layer_weights,
)
from scripts.deepseek_v4_model import DeepseekV4Model

CFG = LayerConfig(
    hidden=64,
    n_heads=4,
    head_dim=128,
    rope_head_dim=64,
    q_lora_rank=32,
    o_lora_rank=32,
    o_groups=2,
    window_size=8,
    compress_ratio=4,
    index_n_heads=2,
    index_head_dim=128,
    index_topk=8,
    n_routed=16,
    topk=2,
    route_scale=1.5,
    swiglu_limit=10.0,
    norm_eps=1e-6,
    hc_mult=2,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_seq_len=32,
)

FFN_PHASES = (
    "router",
    "routed_dispatch_and_native",
    "routed_combine",
    "shared_expert",
    "output_cast",
)


class _StubNativeFfn:
    """Mimics DeepseekV4NativeFfn's profiling surface without CUDA events."""

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self.start_positions: list[int] = []
        self.reset_calls = 0

    def set_profile_start_pos(self, start_pos: int) -> None:
        self.start_positions.append(int(start_pos))

    def reset_stage_profile(self) -> None:
        self.reset_calls += 1

    def stage_profile_snapshot(self) -> dict:
        if not self.enabled:
            return {
                "enabled": False,
                "layer": -1,
                "calls": 0,
                "totals_ms": {},
            }
        return {
            "enabled": True,
            "layer": -1,
            "device": "cpu",
            "calls": 1,
            "event_interval_count": len(FFN_PHASES),
            "totals_ms": {"router": 1.5, "shared_expert": 2.5},
            "per_start_pos_ms": {"0": {"router": 1.5, "shared_expert": 2.5}},
            "calls_detail": [],
        }

    def __call__(self, x: torch.Tensor, input_ids: torch.Tensor,
                 capture) -> torch.Tensor:
        # Reference-shaped FFN contract: fp32 in -> fp32 out, [b, s, d].
        return x.float()


def _make_layer(cfg: LayerConfig, ffn_fn) -> DeepseekV4Layer:
    w, _, _ = make_synthetic_layer_weights(cfg, seed=5, n_experts=8)
    return DeepseekV4Layer(cfg, w, device="cpu", max_batch=1,
                           ffn_fn=ffn_fn, layer_id=3, profile_stages=True)


def test_wrapper_propagates_start_pos_and_merges_ffn_phases() -> None:
    torch.manual_seed(0)
    stub = _StubNativeFfn()
    layer = _make_layer(CFG, stub)
    x = torch.randn(1, 4, CFG.hc_mult, CFG.hidden, dtype=torch.bfloat16)

    out_prefill = layer.forward(x, start_pos=0)
    out_decode = layer.forward(x[:, :1], start_pos=4)

    assert out_prefill.shape == x.shape
    assert out_decode.shape == x[:, :1].shape
    assert stub.start_positions == [0, 4]
    snapshot = layer.stage_profile_snapshot()
    assert snapshot["calls"] == 0  # no CUDA events on CPU; merge-only path
    for name, expected in (("router", 1.5), ("shared_expert", 2.5)):
        assert abs(snapshot["totals_ms"][name] - expected) < 1e-9, name
        assert abs(snapshot["per_start_pos_ms"]["0"][name] - expected) < 1e-9
    # Wrapper coarse phases are still present alongside the FFN fine phases.
    for name in ("attention_prep", "attention_state", "ffn_prep",
                 "routed_and_shared_ffn", "ffn_hc_post"):
        assert name in snapshot["totals_ms"], name
    assert snapshot["coarse_event_interval_count"] == 0
    assert snapshot["ffn_event_interval_count"] == len(FFN_PHASES)
    assert snapshot["event_interval_count"] == len(FFN_PHASES)
    assert snapshot["totals_are_additive"] is False
    assert snapshot["overlapping_total_groups"] == {
        "routed_and_shared_ffn": ["router", "shared_expert"]}


def test_wrapper_reset_state_chains_ffn_profile_reset() -> None:
    torch.manual_seed(1)
    stub = _StubNativeFfn()
    layer = _make_layer(CFG, stub)

    layer.reset_state()
    assert stub.reset_calls == 1


def test_wrapper_snapshot_without_ffn_snapshot_keeps_required_keys() -> None:
    torch.manual_seed(2)
    # Reference fp32 backend: a bound method with no profiler surface.
    layer = _make_layer(CFG, None)
    x = torch.randn(1, 4, CFG.hc_mult, CFG.hidden, dtype=torch.bfloat16)
    layer.forward(x, start_pos=0)
    layer.forward(x[:, :1], start_pos=4)

    snapshot = layer.stage_profile_snapshot()
    assert "totals_ms" in snapshot
    assert "per_start_pos_ms" in snapshot
    assert snapshot["calls"] == 0
    # No FFN phases leaked into the wrapper-only snapshot.
    for name in FFN_PHASES:
        assert name not in snapshot["totals_ms"], name
    assert snapshot["event_interval_count"] == 0
    assert snapshot["totals_are_additive"] is True


def test_disabled_ffn_snapshot_is_not_merged() -> None:
    torch.manual_seed(3)
    stub = _StubNativeFfn(enabled=False)
    layer = _make_layer(CFG, stub)
    x = torch.randn(1, 4, CFG.hc_mult, CFG.hidden, dtype=torch.bfloat16)
    layer.forward(x, start_pos=0)
    layer.forward(x[:, :1], start_pos=4)

    snapshot = layer.stage_profile_snapshot()
    for name in FFN_PHASES:
        assert name not in snapshot["totals_ms"], name


class _FakeCudaEvent:
    def __init__(self) -> None:
        self.recorded_on = None

    def record(self, stream) -> None:
        self.recorded_on = stream


def test_native_ffn_profile_begin_resets_stale_boundary(monkeypatch) -> None:
    """An aborted prior call must not poison the next event sequence."""
    fake_stream = object()
    monkeypatch.setattr(
        torch.cuda, "Event", lambda *, enable_timing: _FakeCudaEvent())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: fake_stream)

    ffn = DeepseekV4NativeFfn.__new__(DeepseekV4NativeFfn)
    ffn.profile_stages = True
    ffn.device = "cuda:0"
    ffn._active_profile_events = [_FakeCudaEvent() for _ in range(6)]
    ffn._active_profile_boundary = 4  # interrupted previous call

    x = SimpleNamespace(is_cuda=True, device="cuda:0")
    ffn._profile_ffn_begin(x)

    assert ffn._active_profile_boundary == 1
    assert ffn._active_profile_events is not None
    assert ffn._active_profile_events[0].recorded_on is fake_stream
    ffn._profile_ffn_mark()
    assert ffn._active_profile_events[1].recorded_on is fake_stream
    assert ffn._active_profile_boundary == 2


def test_model_profile_counts_coarse_and_fine_intervals(monkeypatch) -> None:
    class _ProfiledLayer:
        profile_stages = True
        device = "cuda:0"

        @staticmethod
        def stage_profile_snapshot() -> dict:
            return {
                "layer": 0,
                "device": "cuda:0",
                "calls": 2,
                "coarse_event_interval_count": 10,
                "ffn_event_interval_count": 10,
                "event_interval_count": 20,
                "totals_are_additive": False,
                "totals_ms": {
                    "routed_and_shared_ffn": 7.0,
                    "router": 1.0,
                },
                "per_start_pos_ms": {
                    "0": {"routed_and_shared_ffn": 7.0, "router": 1.0},
                },
            }

    synchronized = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: synchronized.append(device))
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    model.layers0 = [_ProfiledLayer()]
    model.layers1 = []

    snapshot = model.cuda_stage_profile()

    assert synchronized == ["cuda:0"]
    assert snapshot["coarse_event_interval_count"] == 10
    assert snapshot["ffn_event_interval_count"] == 10
    assert snapshot["event_interval_count"] == 20
    assert snapshot["totals_are_additive"] is False
    assert snapshot["overlapping_total_groups"]["routed_and_shared_ffn"]
