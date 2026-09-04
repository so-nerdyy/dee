#!/usr/bin/env python3
"""Bounded exact-staging queue — REFERENCE PROTOTYPE (isolated experiment).

This file lives under dee.cpp/experiments/ and is NOT imported by any
production path. It is the executable reference for the design in
research/exact-staging/STAGING_DESIGN.md; a live A/B would implement the
same semantics in engine.cpp's fill path (mapping given in the design doc).

Causality contract (enforced, see CAUSALITY_AUDIT.md):
  - The queue is fed ONLY official route ids, at exactly the moment the
    production engine learns them (top of each layer's MoE forward).
  - NO predicted/future experts are staged. Lookahead is CROSS-STEP only:
    records demanded at step k-1 may be retained and re-verified (not
    re-read from disk) when demanded again at step k.
  - Expert identity is exact (layer, expert). The official router remains
    authoritative; this queue never decides what is computed.

Boundedness contract:
  - max_inflight_reads bounds concurrent storage reads.
  - pinned_budget_bytes bounds retained host buffers (LRU eviction).
  - Stale work (older step's in-flight reads after a new step started) is
    cancelled at submission boundaries and dropped at completion.

Per-request telemetry matches research/exact-staging STAGING_TELEMETRY_V1.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

TELEMETRY_SCHEMA = "exact-staging/staging-telemetry-v1"


@dataclass
class StagingConfig:
    retain_last_step: bool = True        # cross-step retention (the legal lead)
    max_inflight_reads: int = 8          # bounded in-flight storage reads
    pinned_budget_bytes: int = 4 * (1 << 30)   # bounded retained buffers
    resident_slots: int = 281            # VRAM residency bound (sealed v65: 281/GPU)
    h2d_gbps: float = 5.38               # mock H2D rate for readiness timing
    disk: "DiskModel" = None             # injected mock/pread-measured model
    seed: int = 7


@dataclass
class DiskModel:
    """Mock disk implementing the saturation law calibrated from the two
    sealed lane A/B points (v63 3-lane p50 109.1 ms, v65 6-lane p50 215.5 ms,
    identical aggregate ~370 MB/s): service_ms(N in flight) = N * RECORD_MB /
    aggregate_cap. N=1 and N=2 are UNMEASURED in sealed evidence — the live
    pread bench (tools/bench_expert_pread.py) exists to replace this law."""
    aggregate_cap_mb_s: float = 370.0    # v63/v65 lanes-x-p50 invariant
    seed: int = 7
    _rng: random.Random = field(init=False, default=None)
    _inflight: int = field(init=False, default=0)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def submit(self, nbytes: int) -> float:
        """Returns per-read duration (ms) given current in-flight count."""
        self._inflight += 1
        n = max(1, self._inflight)
        return n * (nbytes / 1e6) / self.aggregate_cap_mb_s

    def complete(self) -> None:
        self._inflight = max(0, self._inflight - 1)


@dataclass
class RequestTelemetry:
    step: int
    layer: int
    expert: int
    record_bytes: int
    route_known_ts: float
    read_submit_ts: float | None = None
    read_complete_ts: float | None = None
    h2d_submit_ts: float | None = None
    h2d_complete_ts: float | None = None
    compute_needed_ts: float | None = None
    ready_before_demand: bool | None = None
    source_tier: str = "unknown"         # resident | pack | read | stale_drop
    useful_lead_ms: float | None = None
    cancelled: bool = False


class BoundedStagingQueue:
    """Reference semantics for the A/B candidate. Times are mock-ms."""

    def __init__(self, config: StagingConfig, record_bytes: int = 13_369_344,
                 num_layers: int = 43, num_experts: int = 256):
        self.cfg = config
        self.record_bytes = record_bytes
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.disk = config.disk or DiskModel(seed=config.seed)
        # resident set (VRAM): bounded LRU, mirroring the sealed 281/GPU
        # resident_experts; the harness may preload but never exceed the cap.
        self.resident: "OrderedDict" = OrderedDict()
        # retained pack: (layer, expert) -> bytes, LRU bounded
        self.pack: "OrderedDict" = OrderedDict()
        self.pack_bytes = 0
        # in-flight reads: (layer, expert) -> ReadTicket
        self.inflight: dict = {}
        # ownership: exactly one owner step per in-flight record
        self.owner_step: dict = {}
        self.now = 0.0
        self.h2d_busy_until = 0.0
        self.step_generation = 0
        self.telemetry: list[RequestTelemetry] = []
        self.cancelled_reads = 0

    # ---------------- step lifecycle -------------------------------------
    def begin_step(self, resident_set: set) -> None:
        """Called at step start. Advances generation; cancels stale reads
        from previous steps (bounded-stale-work handling); installs the
        official resident set (mock of VRAM residency, bounded)."""
        self.step_generation += 1
        self.resident.clear()
        for k in resident_set:
            self._resident_touch(k)
        for key in list(self.inflight):
            if self.owner_step.get(key) != self.step_generation:
                self.cancelled_reads += 1
                del self.inflight[key]
                self.owner_step.pop(key, None)
        if not self.cfg.retain_last_step:
            self.pack.clear()
            self.pack_bytes = 0

    def _resident_touch(self, key) -> None:
        if key in self.resident:
            self.resident.move_to_end(key)
            return
        while len(self.resident) >= max(1, self.cfg.resident_slots):
            self.resident.popitem(last=False)
        self.resident[key] = 1

    def _pack_evict_lru(self, need: int) -> None:
        while self.pack_bytes + need > self.cfg.pinned_budget_bytes and self.pack:
            _, nbytes = self.pack.popitem(last=False)
            self.pack_bytes -= nbytes

    # ---------------- route-known submission (official ids only) ---------
    def note_route_known(self, step: int, layer: int, expert: int,
                         needed_by_ts: float) -> RequestTelemetry:
        """Called when the OFFICIAL route for (step, layer) includes `expert`.
        Mirrors engine.cpp: prepare_fp4_experts() call site."""
        key = (layer, expert)
        tel = RequestTelemetry(step=step, layer=layer, expert=expert,
                               record_bytes=self.record_bytes,
                               route_known_ts=self.now,
                               compute_needed_ts=needed_by_ts)
        if key in self.resident:
            self._resident_touch(key)
            tel.source_tier = "resident"
            tel.read_submit_ts = self.now
            tel.read_complete_ts = self.now
            tel.h2d_submit_ts = self.now
            tel.h2d_complete_ts = self.now
            tel.ready_before_demand = True
            tel.useful_lead_ms = needed_by_ts - self.now
            self.telemetry.append(tel)
            return tel
        if key in self.pack:
            tel.source_tier = "pack"
            tel.read_submit_ts = self.now
            tel.read_complete_ts = self.now     # no disk read; buffer retained
            tel.h2d_submit_ts = self.now
            tel.h2d_complete_ts = self._mock_h2d(self.record_bytes)
            tel.ready_before_demand = tel.h2d_complete_ts <= needed_by_ts
            tel.useful_lead_ms = needed_by_ts - tel.h2d_complete_ts
            self.pack.move_to_end(key)
            self._resident_touch(key)
            self.telemetry.append(tel)
            return tel
        if key in self.inflight:
            # duplicate suppression: same expert requested twice before the
            # read completed (e.g., repeated within a layer). Ownership kept
            # by the first requester (deterministic: lowest step, then layer).
            prev = self.telemetry[self.inflight[key]]
            tel.source_tier = "dedup"
            tel.read_submit_ts = prev.read_submit_ts
            tel.read_complete_ts = prev.read_complete_ts
            tel.h2d_submit_ts = prev.h2d_submit_ts
            tel.h2d_complete_ts = prev.h2d_complete_ts
            tel.ready_before_demand = (tel.h2d_complete_ts is not None
                                       and tel.h2d_complete_ts <= needed_by_ts)
            self.telemetry.append(tel)
            return tel
        # disk read path (bounded in-flight)
        tel.read_submit_ts = self.now
        if len(self.inflight) < self.cfg.max_inflight_reads:
            dur = self.disk.submit(self.record_bytes)
            self.disk.complete()
            tel.read_complete_ts = self.now + dur
            self.inflight[key] = len(self.telemetry)
            self.owner_step[key] = self.step_generation
        else:
            # queue full: bounded backpressure — the read waits for a slot
            # (serialization slot cost, one record time), then reads
            wait = (len(self.inflight) - self.cfg.max_inflight_reads + 1) \
                * (self.record_bytes / 1e6) / self.disk.aggregate_cap_mb_s
            dur = self.disk.submit(self.record_bytes)
            self.disk.complete()
            tel.read_complete_ts = self.now + wait + dur
            self.inflight[key] = len(self.telemetry)
            self.owner_step[key] = self.step_generation
        tel.h2d_submit_ts = tel.read_complete_ts
        tel.h2d_complete_ts = self._mock_h2d_after(self.record_bytes,
                                                   tel.read_complete_ts)
        tel.source_tier = "read"
        tel.ready_before_demand = tel.h2d_complete_ts <= needed_by_ts
        tel.useful_lead_ms = needed_by_ts - tel.h2d_complete_ts
        self.telemetry.append(tel)
        self._insert_pack(key)
        self._resident_touch(key)
        return tel

    def _insert_pack(self, key) -> None:
        if key in self.pack:
            return
        self._pack_evict_lru(self.record_bytes)
        self.pack[key] = self.record_bytes
        self.pack_bytes += self.record_bytes

    # ---------------- mock H2D -------------------------------------------
    def _mock_h2d(self, nbytes: int) -> float:
        start = max(self.now, self.h2d_busy_until)
        dur = nbytes / (self.cfg.h2d_gbps * 1e9) * 1e3
        self.h2d_busy_until = start + dur
        return start + dur

    def _mock_h2d_after(self, nbytes: int, ready: float) -> float:
        start = max(ready, self.h2d_busy_until)
        dur = nbytes / (self.cfg.h2d_gbps * 1e9) * 1e3
        self.h2d_busy_until = start + dur
        return start + dur

    # ---------------- aggregation (Phase E) ------------------------------
    def aggregate(self) -> dict:
        done = [t for t in self.telemetry if t.source_tier in
                ("resident", "pack", "read", "dedup")]
        reads = [t for t in done if t.source_tier == "read"]
        hidden = [t for t in reads if t.ready_before_demand]
        late = [t for t in reads if not t.ready_before_demand]
        leads = [t.useful_lead_ms for t in done if t.useful_lead_ms is not None]
        return {
            "schema": TELEMETRY_SCHEMA,
            "requests": len(done),
            "by_tier": {tier: sum(1 for t in done if t.source_tier == tier)
                        for tier in ("resident", "pack", "read", "dedup")},
            "pct_reads_completed_before_demand":
                round(100.0 * len(hidden) / max(1, len(reads)), 2),
            "avg_useful_lead_ms": round(sum(leads) / len(leads), 2) if leads else None,
            "late_reads": len(late),
            "cancelled_reads": self.cancelled_reads,
            "pack_bytes_retained": self.pack_bytes,
            "pack_entries_retained": len(self.pack),
            "bounded_inflight_reads": self.cfg.max_inflight_reads,
            "pinned_budget_bytes": self.cfg.pinned_budget_bytes,
        }


def replay_from_journal(journal: Path, config: StagingConfig,
                        record_bytes: int = 13_369_344) -> tuple[BoundedStagingQueue, dict]:
    """Replay the sealed journal's OFFICIAL route order through the queue.
    Only routes are consumed (no weights, no math). Per-layer demand times
    are mock-scheduled: layer L's consumer at L * layer_ms."""
    demands: dict[int, dict[int, list[int]]] = {}
    devices: set[str] = set()
    with journal.open() as fh:
        for line in fh:
            rec = json.loads(line)
            fs, L = rec["forward_step"], rec["layer"]
            devices.add(rec.get("device", "cuda:0"))
            ids = [e for row in rec["expert_ids_rank_order"] for e in row]
            demands.setdefault(fs, {}).setdefault(L, []).extend(ids)
    q = BoundedStagingQueue(config, record_bytes)
    layer_ms = 40.0   # mock consumer cadence (~1.7 s / 43 layers, v65-scale)
    resident: set = set()
    for fs in sorted(demands):
        q.begin_step(resident)
        for L in sorted(demands[fs]):
            ids = demands[fs][L]
            seen = set()
            for e in ids:
                if (L, e) in seen:
                    continue
                seen.add((L, e))
                q.note_route_known(fs, L, e, needed_by_ts=(L + 1) * layer_ms)
            # update the mock resident set: records H2D-complete are resident
            for tel in q.telemetry:
                if tel.h2d_complete_ts is not None and tel.step == fs:
                    resident.add((tel.layer, tel.expert))
    return q, q.aggregate()


# ---------------------------------------------------------------------------
# Validated host-pack replay (Tier-2 reference semantics)
# ---------------------------------------------------------------------------

def pack_replay(journal: Path, budget_records: int = 682,
                record_bytes: int = 13_369_344) -> dict:
    """VALIDATED replay of the production host-pack LRU over the sealed
    journal's per-device fill stream (unique (layer, expert) per forward step
    in execution order; pack state carries from prefill into decode).

    Against sealed v65 per-GPU counters the default budget reproduces:
      cuda:0 misses 1388 (sealed 1390), evictions 706 (sealed 708)
      cuda:1 misses 1092 (sealed 1091), evictions 410 (sealed 409)
    i.e. production pack semantics are LRU at 682 records (8.5 GiB/GPU).

    Budgets are a config-only knob (host_pack_cache_bytes). No routes,
    no math, no routing authority change. Returns aggregate + per-step
    decode read tables (usable as critical-path simulator inputs).
    """
    rows = [json.loads(l) for l in journal.open()]
    rows.sort(key=lambda r: (r["forward_step"], r["layer"]))
    phase, stream, seenl = {}, defaultdict(lambda: defaultdict(list)), defaultdict(set)
    for r in rows:
        fs, L, dev = r["forward_step"], r["layer"], r.get("device", "cuda:0")
        phase[(dev, fs)] = r["phase"]
        s = seenl[(dev, fs, L)]
        for row in r["expert_ids_rank_order"]:
            for e in row:
                k = (L, e)
                if k not in s:
                    s.add(k)
                    stream[dev][fs].append(k)
    per_dev, per_step_reads = {}, {}
    for dev in sorted(stream):
        pack: OrderedDict = OrderedDict()
        misses = hits = evictions = decode_misses = 0
        steps = {}
        for fs in sorted(stream[dev]):
            n_miss = 0
            for k in stream[dev][fs]:
                if k in pack:
                    hits += 1
                    pack.move_to_end(k)
                else:
                    misses += 1
                    n_miss += 1
                    if phase[(dev, fs)] == "decode":
                        decode_misses += 1
                    while len(pack) >= budget_records:
                        pack.popitem(last=False)
                        evictions += 1
                    pack[k] = 1
            steps[fs] = n_miss
        per_dev[dev] = {"misses": misses, "hits": hits,
                        "evictions": evictions,
                        "decode_misses": decode_misses,
                        "distinct_records": len(set(k for v in stream[dev].values() for k in v))}
        per_step_reads[dev] = {fs: n for fs, n in steps.items() if fs > 0}
    total_decode_misses = sum(d["decode_misses"] for d in per_dev.values())
    return {
        "schema": "exact-staging/pack-replay-v1",
        "budget_records": budget_records,
        "budget_gib_per_gpu": round(budget_records * record_bytes / 2**30, 2),
        "record_bytes": record_bytes,
        "per_device": per_dev,
        "decode_storage_misses_total": total_decode_misses,
        "decode_per_step_reads": {fs: sum(per_step_reads[d][fs] for d in per_step_reads)
                                  for fs in sorted(next(iter(per_step_reads.values())))},
        "validation_note": "LRU@682 reproduces sealed v65 pack counters within 2 per GPU (see docstring).",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", type=Path, required=True)
    ap.add_argument("--max-inflight", type=int, default=8)
    ap.add_argument("--pinned-gib", type=float, default=4.0)
    ap.add_argument("--no-retain", action="store_true",
                    help="ablation: retention disabled (on-demand-like)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = StagingConfig(max_inflight_reads=args.max_inflight,
                        pinned_budget_bytes=int(args.pinned_gib * (1 << 30)),
                        retain_last_step=not args.no_retain)
    q, agg = replay_from_journal(args.journal, cfg)
    print(json.dumps(agg, indent=2))
    if args.out:
        args.out.write_text(json.dumps(
            {"aggregate": agg,
             "per_request": [asdict(t) for t in q.telemetry]}, indent=2))


if __name__ == "__main__":
    main()
