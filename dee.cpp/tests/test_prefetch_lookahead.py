"""Router-ahead prefetch hints (DeepseekV4NativeFfn lookahead) on CPU with a fake engine.

The contract under test: a hint is only ever handed to ``engine.prefetch_hint``.  The route a
layer executes, and its output, are identical with the lookahead attached or not.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deepseek_v4_layer_candidate import DeepseekV4NativeFfn  # noqa: E402

HIDDEN, EXPERTS, TOPK = 16, 32, 4


class FakeEngine:
    def __init__(self):
        self.hints = []

    def prefetch_hint(self, layer, experts):
        experts = np.asarray(experts)
        assert experts.dtype == np.int32
        self.hints.append((int(layer), experts.tolist()))
        return len(experts)


def _ffn(engine, layer_id, seed, hash_vocab=0):
    cfg = SimpleNamespace(topk=TOPK, route_scale=1.5)
    ffn = DeepseekV4NativeFfn(engine=engine, layer_id=layer_id, cfg=cfg, device="cpu",
                              diagnostics=False)
    g = torch.Generator().manual_seed(seed)
    ffn.attach_gate(torch.randn(EXPERTS, HIDDEN, generator=g),
                    None if hash_vocab else torch.randn(EXPERTS, generator=g) * 0.1)
    if hash_vocab:
        ffn.attach_hash(torch.randint(0, EXPERTS, (hash_vocab, TOPK), generator=g))
    calls = []

    def run_experts(xf, ids, weights):
        calls.append((ids.clone(), weights.clone()))
        ffn._score_lookahead(ids.to(torch.int32).numpy())
        out = xf * 2.0
        return out, torch.zeros_like(out)

    ffn._run_experts = run_experts
    ffn.calls = calls
    return ffn


def _inputs(rows=3, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, rows, HIDDEN, generator=g), torch.tensor([[1, 5, 9][:rows]])


def test_lookahead_never_changes_the_route_or_the_output():
    x, ids = _inputs()
    plain_engine, hinted_engine = FakeEngine(), FakeEngine()
    plain = _ffn(plain_engine, 3, seed=1)
    hinted = _ffn(hinted_engine, 3, seed=1)
    hinted.attach_lookahead(_ffn(hinted_engine, 4, seed=2))
    out_plain, out_hinted = plain(x, ids, None), hinted(x, ids, None)
    assert torch.equal(out_plain, out_hinted)
    assert torch.equal(plain.calls[0][0], hinted.calls[0][0])
    assert torch.equal(plain.calls[0][1], hinted.calls[0][1])
    assert plain_engine.hints == [] and len(hinted_engine.hints) == 1


def test_score_layer_hint_is_the_next_gate_on_this_input():
    x, ids = _inputs()
    engine = FakeEngine()
    here, there = _ffn(engine, 3, seed=1), _ffn(engine, 4, seed=2)
    here.attach_lookahead(there)
    here(x, ids, None)
    layer, hinted = engine.hints[0]
    # The next layer's own router on the SAME input picks exactly the hinted set.
    there(x, ids, None)
    actual = sorted(set(there.calls[0][0].flatten().tolist()))
    assert layer == 4 and hinted == actual
    stats = here.lookahead_stats
    assert stats["calls"] == 1 and stats["hinted"] == len(actual) == stats["queued"]
    assert there.lookahead_stats["actual"] == there.lookahead_stats["correct"] == len(actual)


def test_width_widens_the_candidate_set():
    x, ids = _inputs()
    engine = FakeEngine()
    here, there = _ffn(engine, 3, seed=1), _ffn(engine, 4, seed=2)
    here.attach_lookahead(there, width=TOPK + 3)
    here(x, ids, None)
    wide = set(engine.hints[0][1])
    there(x, ids, None)
    assert set(there.calls[0][0].flatten().tolist()) <= wide
    assert len(wide) > TOPK


def test_hash_layer_hint_is_exact():
    x, ids = _inputs()
    engine = FakeEngine()
    here, there = _ffn(engine, 0, seed=1, hash_vocab=12), _ffn(engine, 1, seed=2, hash_vocab=12)
    here.attach_lookahead(there)
    here(x, ids, None)
    expected = sorted(set(there.tid2eid[ids.flatten()].flatten().tolist()))
    assert engine.hints == [(1, expected)]
    assert here.lookahead_stats["exact_calls"] == 1
    there(x, ids, None)
    assert there.lookahead_stats["correct"] == there.lookahead_stats["actual"] == len(expected)


def test_a_wrong_hint_is_only_scored():
    x, ids = _inputs()
    engine = FakeEngine()
    there = _ffn(engine, 4, seed=2)
    there._lookahead_pending = {EXPERTS + 1}          # nothing the router will pick
    out = there(x, ids, None)
    assert torch.equal(out, _ffn(FakeEngine(), 4, seed=2)(x, ids, None))
    assert there.lookahead_stats["scored_calls"] == 1
    assert there.lookahead_stats["correct"] == 0 and there.lookahead_stats["actual"] > 0
    assert there._lookahead_pending is None


def test_lookahead_needs_one_engine():
    here, there = _ffn(FakeEngine(), 3, seed=1), _ffn(FakeEngine(), 4, seed=2)
    with pytest.raises(ValueError):
        here.attach_lookahead(there)
    here.attach_lookahead(None)
    assert here.lookahead_next is None


# --- model wiring -----------------------------------------------------------------------

class _ConfigEngine(FakeEngine):
    def __init__(self, lanes):
        super().__init__()
        self.lanes = lanes

    def runtime_config(self):
        return {"prefetch_hint_lanes": self.lanes}


def _model(engines):
    from scripts import deepseek_v4_model as vm
    cls = next(c for c in vars(vm).values()
               if isinstance(c, type) and hasattr(c, "enable_prefetch_lookahead"))
    layers = [SimpleNamespace(ffn_fn=_ffn(engine, i, seed=i)) for i, engine in enumerate(engines)]
    half = len(layers) // 2
    return cls, SimpleNamespace(layers0=layers[:half], layers1=layers[half:])


def test_model_links_consecutive_layers_on_the_same_engine_only():
    e0, e1 = _ConfigEngine(2), _ConfigEngine(2)
    cls, model = _model([e0, e0, e1, e1])
    assert cls.enable_prefetch_lookahead(model, width=0) == 2      # 0->1 and 2->3, not 1->2
    ffns = [layer.ffn_fn for layer in model.layers0 + model.layers1]
    assert ffns[0].lookahead_next is ffns[1] and ffns[2].lookahead_next is ffns[3]
    assert ffns[1].lookahead_next is None and ffns[3].lookahead_next is None
    x, ids = _inputs()
    for ffn in ffns:
        ffn(x, ids, None)
    stats = cls.prefetch_lookahead_stats(model)
    assert stats["enabled"] and stats["calls"] == 2 and stats["scored_calls"] == 2
    assert stats["recall"] == 1.0 and stats["precision"] == 1.0   # same input, same gate


def test_model_refuses_lookahead_without_hint_lanes():
    engine = _ConfigEngine(0)
    cls, model = _model([engine, engine])
    with pytest.raises(RuntimeError):
        cls.enable_prefetch_lookahead(model)
