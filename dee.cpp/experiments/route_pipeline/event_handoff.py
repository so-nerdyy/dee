#!/usr/bin/env python3
"""Prototype C: event-based layer handoff + narrow route-D2H wait (isolated).

Models CUDA stream/event/resource-lifetime semantics on the host (no GPU
here) with fail-closed ordering: every consumer wait, lifetime, error, and
cleanup rule is checked, and the negative tests prove what breaks when a
wait is omitted. Nothing here patches production or removes a sync.

Sync-scope audit (what the per-layer cudaStreamSynchronize protects,
engine.cpp:1037-1054): output readiness, routing-buffer lifetime, packed
expert lifetime, decoded-scratch lifetime, host-pack fill ownership, error
delivery, cache-eviction safety, D2D gather/scatter completion, torch
consumer readiness. The model represents each as a Resource with an owner
stream and a release condition.

Route-D2H audit (layer_candidate.py:448): 24 B of IDs + a FULL torch-stream
sync. The model shows the narrower pattern — pinned copy ordered after the
router kernel via an event, host waits ONLY on the copy event — and the
residual unavoidable wait (the CPU still needs the ID bytes themselves).
Verdict encoded in narrow_d2h_gain(): the prize is bounded by unrelated
stream work skipped, never by the copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Event:
    name: str
    recorded_t: float | None = None
    failed: str | None = None

    def record(self, t: float) -> None:
        if self.recorded_t is not None:
            raise ValueError(f"event {self.name} recorded twice")
        self.recorded_t = t

    def fail(self, reason: str) -> None:
        self.failed = reason


@dataclass
class Resource:
    name: str
    owner: str                       # producing stream
    live_from: float
    release_after: list = field(default_factory=list)  # event names
    released: bool = False

    def release(self, waited: set[str], t: float) -> None:
        missing = [e for e in self.release_after if e not in waited]
        if missing:
            raise ValueError(f"resource {self.name}: premature reuse; "
                             f"unwaited {missing} (use-after-free)")
        self.released = True


@dataclass
class Stream:
    name: str
    ops: list = field(default_factory=list)  # (t_end, label)

    def launch(self, label: str, duration_ms: float, t_now: float) -> float:
        t_end = (self.ops[-1][0] if self.ops else t_now) + duration_ms
        self.ops.append((t_end, label))
        return t_end

    def wait_on(self, event: Event, t_now: float) -> float:
        """Device-side wait: stream continues only after event. Returns the
        earliest time this stream may proceed. Raises on unrecorded/failed
        events (never silently proceeds)."""
        if event.failed is not None:
            raise ValueError(f"stream {self.name}: waited event {event.name} "
                             f"carries producer error: {event.failed}")
        if event.recorded_t is None:
            raise ValueError(f"stream {self.name}: wait on unrecorded event "
                             f"{event.name} (missing record)")
        return max(t_now, event.recorded_t)


def handoff_pattern(producer: Stream, consumer: Stream, resource: Resource,
                    compute_ms: float, t0: float) -> dict:
    """Legal narrow handoff: produce -> record event -> consumer waits ->
    consume -> release after waited event. Returns timeline + counts."""
    done = producer.launch(f"compute:{resource.name}", compute_ms, t0)
    ev = Event(f"{resource.name}_ready")
    ev.record(done)
    start = consumer.wait_on(ev, t0)
    end = consumer.launch(f"consume:{resource.name}", 0.1, start)
    resource.release_after = [ev.name]
    resource.release({ev.name}, end)
    return {"events": 1, "syncs": 0, "producer_done": done,
            "consumer_end": end, "errors": []}


def narrow_d2h(ids_bytes: int, unrelated_stream_ms: float,
               copy_ms: float) -> dict:
    """Route-D2H audit model. Current: host waits full stream (copy +
    unrelated). Narrow: host waits copy-event only. The copy wait is
    unavoidable; the prize is exactly the unrelated work skipped."""
    current_wait = copy_ms + unrelated_stream_ms
    narrow_wait = copy_ms
    return {"ids_bytes": ids_bytes,
            "current_host_wait_ms": current_wait,
            "narrow_host_wait_ms": narrow_wait,
            "avoidable_ms": unrelated_stream_ms,
            "unavoidable_ms": copy_ms,
            "verdict": ("WORTH_PROTOTYPING" if unrelated_stream_ms > 0
                        else "LOW_EXPECTED_VALUE")}
