from types import SimpleNamespace

import pytest

from scripts.run_ornith_m5i_linear_graph_benchmark import (
    CANDIDATE,
    CONTROL,
    LinearAttentionGraph,
    graph_stats,
    select_graph_mode,
)


class FakeModule:
    layer_idx = 7

    def __init__(self):
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append((hidden_states, kwargs))
        return "eager-output"


class FakeCache:
    def __init__(self, previous: bool):
        self.previous = previous

    def has_previous_state(self, layer_idx: int) -> bool:
        assert layer_idx == 7
        return self.previous


def test_control_uses_original_forward_for_cached_decode():
    module = FakeModule()
    controller = LinearAttentionGraph(module, pool=None)
    hidden = SimpleNamespace(shape=(1, 1, 64))

    result = controller(
        hidden,
        cache_params=FakeCache(previous=True),
        attention_mask="mask",
        marker="kept",
    )

    assert result == "eager-output"
    assert controller.decode_calls == 1
    assert controller.eager_decode_calls == 1
    assert controller.graph_decode_calls == 0
    assert module.calls[0][1]["marker"] == "kept"


def test_prefill_bypasses_graph_even_when_enabled():
    module = FakeModule()
    controller = LinearAttentionGraph(module, pool=None)
    controller.enabled = True
    hidden = SimpleNamespace(shape=(1, 3, 64))

    assert controller(hidden, cache_params=FakeCache(previous=False)) == "eager-output"
    assert controller.prefill_bypasses == 1
    assert controller.decode_calls == 0


def test_mode_selection_resets_counts_and_stats():
    controllers = [LinearAttentionGraph(FakeModule(), pool=None) for _ in range(2)]
    controllers[0].decode_calls = 9

    select_graph_mode(controllers, CANDIDATE)
    assert all(controller.enabled for controller in controllers)
    assert all(controller.decode_calls == 0 for controller in controllers)

    controllers[0].graph = object()
    controllers[0].replays = 3
    stats = graph_stats(controllers)
    assert stats["layer_count"] == 2
    assert stats["graph_ready_count"] == 1
    assert stats["aggregate"]["replays"] == 3

    select_graph_mode(controllers, CONTROL)
    assert not any(controller.enabled for controller in controllers)


def test_mode_selection_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown graph mode"):
        select_graph_mode([], "mystery")
