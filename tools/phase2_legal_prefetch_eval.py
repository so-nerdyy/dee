#!/usr/bin/env python3
"""Phase-2 legal early-work quantification — what can legally begin before
the router commits, and does any of it beat serial-on-saturated-storage?

ANALYSIS ONLY; no engine code. Replays the sealed v50 canonical route
journal (sha256 665aac3e..., 688 records = 16 forwards x 43 layers,
topk 6) through the validated two-level tier model (per-GPU device
priority-LRU @281 slots -> host plain-LRU @682 slots; host miss == one
real 12.75 MiB pread), then measures how much of the resulting miss
stream each candidate early-work source could cover under a strict
idle-gap admission model:

  * Prefetches may ONLY run inside the per-row disk-idle window (the
    measured gap after a batch's demand fills drain and before the next
    layer's route exists). They never delay a demand read.
  * Every prefetch is a real 12.75 MiB pread charged against the row's
    idle-byte budget, and a real host-LRU insertion (evicts the LRU tail
    => pollution is priced, not assumed away).
  * A prefetch pays iff the record is demanded while still resident.

Sources evaluated (legality per OFFICIAL_LOOKAHEAD.md / AGENTS.md — a
route is official only if computed from the model's own inputs;
predictions may drive prefetch HINTS only):

  hash_L0L2      EXACT: layers 0-2 ids = tid2eid[input_ids] known at token
                 start. Emitted from same-token gaps after rows (t,0) and
                 (t,1), AND from the token-boundary gap after row
                 (t-1,42) toward (t,0..2) — ids exist once sample(t-1)
                 completes, inside that gap (slightly generous: the full
                 gap is charged though ids land partway through it).
  oracle_gap     BOUND ONLY (not legal): perfect future-miss knowledge
                 filling every idle gap; LRU pollution is always priced.
  prevtok        predictor: candidate set = same-layer demand of the
                 previous forward (decode: token t-1; first decode token
                 predicts from the prefill union per layer), ranked by
                 causal activation frequency
  prevtok+freq   same, backfilled with the layer's causal top-frequency
                 records up to the gap's record budget
  mtp / dense / shared: structural, zero bank traffic (documented, not
                 simulated)

Wall model: per-row fill wall = W1(m) measured serial table (Phase-1
fill-live replay) with 35.4 ms/miss linear extrapolation; prefetch can
only REDUCE a row's miss count m. Gap budget per decode row:
gap_ms x bank BW; swept over gap in {23.0 ms derived no-legal-work,
37.6 ms all non-fill time} and BW in {0.29, 0.33, 0.37} GiB/s.

Outputs -> research/phase2-legal-prefetch/results/
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOURNAL = (REPO / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
           "v50-evidence-20260829T195940Z/routed_experts.jsonl")
OUT = REPO / "research/phase2-legal-prefetch/results"

RECORD_BYTES = 13_369_344          # 12.75 MiB sealed DEE4 record
GIB = 1 << 30
MIB = 1 << 20
SPLIT_GPU0 = 22                    # cuda0: layers 0-21, cuda1: 22-42
DEVICE_SLOTS = 281                 # sealed 3.5 GiB arena / 12.75 MiB
HOST_SLOTS = 682                   # sealed 8.5 GiB pack budget / 12.75 MiB
N_LAYERS = 43
N_TOKENS = 16
N_DECODE = 15
HASH_LAYERS = (0, 1, 2)            # tid2eid-routed; ids exact at token start

# Phase-1 measured anchors (fill_matrix_ingest.json / LIVE_PROFILE_RESULTS)
W1 = {0: 0.0, 1: 37.2, 2: 66.1, 3: 105.8, 4: 139.6, 5: 177.2, 6: 221.6}
W1_SLOPE = 35.4                    # ms per extra miss (linear fit of W1)
ROW_WALL_MS = 102.7                # sealed decode row mean (66.233 s / 645)
ROW_FILL_MS = 65.1                 # sealed critical fill per row (42.0/645)
GAP_CONSERVATIVE_MS = 23.0         # derived no-legal-work gap (timeline)
GAP_MAX_MS = ROW_WALL_MS - ROW_FILL_MS   # 37.6 ms = all non-fill time
BWS = (0.29, 0.33, 0.37)           # measured /tmp bank ceiling range


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_batches(path):
    """(step, layer, gpu, [sorted unique experts]) — engine-dedup order,
    same convention as phase2_ws_policy_sim_v4 / phase2_fill_concurrency_sim
    (validated against sealed host-pack counters)."""
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs.sort(key=lambda r: r["record_index"])
    out = []
    for r in recs:
        uniq = sorted({e for row in r["expert_ids_rank_order"] for e in row})
        gpu = 0 if r["layer"] < SPLIT_GPU0 else 1
        out.append({"step": r["forward_step"], "layer": r["layer"],
                    "gpu": gpu, "phase": r["phase"], "experts": uniq})
    return out


class LRU:
    def __init__(self, slots):
        self.slots, self.dq, self.pos, self.evictions = slots, deque(), set(), 0

    def resident(self, key):
        return key in self.pos

    def touch(self, key):
        """Insert/hit as most-recent. Returns True on hit."""
        if key in self.pos:
            self.dq.remove(key)
            self.dq.appendleft(key)
            return True
        if self.slots > 0:
            if len(self.dq) >= self.slots:
                self.pos.discard(self.dq.pop())
                self.evictions += 1
            self.dq.appendleft(key)
            self.pos.add(key)
        return False


class PriorityLRU:
    """Sealed device semantics: score = last_used + prio*2**20."""

    def __init__(self, slots):
        self.slots, self.blocks, self.tick, self.evictions = slots, {}, 0, 0

    def resident(self, key):
        return key in self.blocks

    def touch(self, key, prio):
        self.tick += 1
        if key in self.blocks:
            self.blocks[key] = [self.tick, prio]
            return True
        if self.slots > 0:
            while len(self.blocks) >= self.slots:
                victim = min(self.blocks, key=lambda k: (
                    self.blocks[k][0] + self.blocks[k][1] * (1 << 20)))
                del self.blocks[victim]
                self.evictions += 1
            self.blocks[key] = [self.tick, prio]
        return False


def replay(batches, prefetch=None, gap_ms=0.0, bw_gibps=0.33):
    """Two-level replay with idle-gap prefetch. prefetch(i) -> iterable of
    (gpu, (layer,expert), deadline_row_index) candidates in priority order.
    A candidate needs RECORD_BYTES/bw of disk time, may accumulate it across
    CONSECUTIVE gaps (progress tracked), and must complete during a gap with
    index < its deadline (the row whose demand it serves); an incomplete
    record at its deadline is wasted partial work (counted, never useful).
    Prefetches draw ONLY from the per-row idle gap — never from demand-fill
    time — and each completed prefetch is a real host-LRU insertion
    (pollution priced). Returns per-row counters + prefetch stats."""
    dev = {0: PriorityLRU(DEVICE_SLOTS), 1: PriorityLRU(DEVICE_SLOTS)}
    host = {0: LRU(HOST_SLOTS), 1: LRU(HOST_SLOTS)}
    need_ms = RECORD_BYTES / (bw_gibps * GIB) * 1000.0
    rows = []
    pf_done = pf_used = pf_abandoned = 0
    pf_partial_bytes = 0.0
    pending = {}                   # key -> [gpu, progress_ms, deadline_idx]
    pf_keys = set()
    for i, b in enumerate(batches):
        layer, gpu, K = b["layer"], b["gpu"], len(b["experts"])
        # pending records whose deadline row just arrived unfinished: wasted
        for key in [k for k, v in pending.items() if v[2] <= i]:
            pf_abandoned += 1
            pf_partial_bytes += pending[key][1]
            del pending[key]
        d_hit = h_hit = h_miss = pf_hit = 0
        for j, e in enumerate(b["experts"]):
            key = (layer, e)
            if key in pending:
                # demanded while partially fetched: partial work wasted
                pf_abandoned += 1
                pf_partial_bytes += pending[key][1]
                del pending[key]
            if dev[gpu].touch(key, K - j):
                d_hit += 1
            elif host[gpu].resident(key):
                host[gpu].touch(key)
                h_hit += 1
                if key in pf_keys:
                    pf_hit += 1
            else:
                host[gpu].touch(key)
                h_miss += 1
        pf_used += pf_hit
        rows.append({"step": b["step"], "layer": layer, "gpu": gpu,
                     "requests": K, "device_hits": d_hit,
                     "host_hits": h_hit, "host_misses": h_miss,
                     "prefetch_hits_at_demand": pf_hit,
                     "phase": b["phase"]})
        # ---- idle-gap prefetch admission (never delays demand reads) ----
        # Decode rows only: the prefill pass has no causal history source
        # (cross-request history is the regime-C contract, out of scope).
        if prefetch is None or gap_ms <= 0 or b["phase"] != "decode":
            continue
        budget_ms = gap_ms
        cands = list(prefetch(i))
        # keep already-pending records (earliest deadline first) ahead of
        # new candidates — finishing in-flight work is strictly better
        queue = [k for k, v in
                 sorted(pending.items(), key=lambda kv: kv[1][2])]
        seen = set(queue)
        for g, key, dl in cands:
            if key not in seen:
                pending[key] = [g, 0.0, dl]
                queue.append(key)
                seen.add(key)
        for key in queue:
            if budget_ms <= 0:
                break
            if key not in pending:      # completed/demanded this row
                continue
            g, prog, dl = pending[key]
            if dl <= i + 1 and prog + budget_ms < need_ms - 1e-9:
                continue                 # last gap before demand and cannot
                                         # complete: don't burn the budget
            if dev[g].resident(key) or host[g].resident(key):
                del pending[key]         # became resident anyway: no cost,
                continue                 # no pread charged
            spend = min(need_ms - prog, budget_ms)
            prog += spend
            budget_ms -= spend
            if prog >= need_ms - 1e-9:
                host[g].touch(key)       # completion: real LRU insert
                pf_keys.add(key)
                pf_done += 1
                del pending[key]
            else:
                pending[key][1] = prog
    return rows, {"prefetches_completed": pf_done,
                  "prefetches_abandoned_partial": pf_abandoned,
                  "prefetch_partial_ms_wasted": round(pf_partial_bytes, 1),
                  "prefetch_hits_at_demand": pf_used,
                  "host_evictions": {g: host[g].evictions for g in (0, 1)}}


def demand_map(batches):
    return {(b["step"], b["layer"]): set(b["experts"]) for b in batches}


def build_predictors(batches, base_rows):
    """Return {name: fn(row_index) -> [(gpu,(layer,expert)), ...]}."""
    D = demand_map(batches)
    n = len(batches)
    # causal activation frequency (counts only demands seen so far)
    freq = defaultdict(int)
    freq_snap = []                              # snapshot per row index
    for b in batches:
        freq_snap.append(dict(freq))
        for e in b["experts"]:
            freq[(b["layer"], e)] += 1
    # baseline host-miss keys per row (oracle source; NOT legal, bound only)
    base_miss_keys = []
    dev = {0: PriorityLRU(DEVICE_SLOTS), 1: PriorityLRU(DEVICE_SLOTS)}
    host = {0: LRU(HOST_SLOTS), 1: LRU(HOST_SLOTS)}
    for b in batches:
        layer, gpu, K = b["layer"], b["gpu"], len(b["experts"])
        misses = []
        for j, e in enumerate(b["experts"]):
            key = (layer, e)
            if dev[gpu].touch(key, K - j):
                continue
            if host[gpu].touch(key):
                continue
            misses.append(key)
        base_miss_keys.append(misses)
    # earliest future demand index for each (layer,expert) per row position
    future_use = []
    for i, b in enumerate(batches):
        nxt = {}
        for k in range(i + 1, n):
            nb = batches[k]
            for e in nb["experts"]:
                key = (nb["layer"], e)
                if key not in nxt:
                    nxt[key] = k
        future_use.append(nxt)

    def oracle(i):
        """Perfect knowledge: every actual future baseline host-miss record,
        earliest-demand-first; deadline = the demanding row's index.
        BOUND ONLY — needs future route knowledge, not legal."""
        nxt = future_use[i]
        out = []
        for key in sorted(nxt, key=lambda k: nxt[k]):
            use_i = nxt[key]
            if key not in base_miss_keys[use_i]:
                continue                  # would hit anyway: pointless
            out.append((batches[use_i]["gpu"], key, use_i))
        return out

    def prevtok(i, backfill=0, lookahead_rows=2):
        """Predict near-future rows' demand from the previous forward's
        same-layer sets (decode: token t-1; the first decode token reads
        the prefill union). Emitted for the next `lookahead_rows` rows,
        each ranked by causal activation frequency."""
        out = []
        for j in range(i + 1, min(n, i + 1 + lookahead_rows)):
            tb = batches[j]
            tgt_layer = tb["layer"]
            prev_step = tb["step"] - 1    # latest same-layer observation
            if prev_step < 0:
                prev_step = 0             # first decode: prefill union
            cand = sorted(D.get((prev_step, tgt_layer), ()),
                          key=lambda e: -freq_snap[i].get((tgt_layer, e), 0))
            for e in cand:
                out.append((tb["gpu"], (tgt_layer, e), j))
            if backfill:
                ranked = sorted(range(256),
                                key=lambda e: -freq_snap[i].get(
                                    (tgt_layer, e), 0))
                seen = {e for e in cand}
                added = 0
                for e in ranked:
                    if e not in seen:
                        out.append((tb["gpu"], (tgt_layer, e), j))
                        seen.add(e)
                        added += 1
                    if added >= backfill:
                        break
        return out

    max_step = max(b["step"] for b in batches)

    def hash_exact(i):
        """EXACT source: layers 0-2 ids = tid2eid[input_ids], known at token
        start. From the gap after decode row (t,L) the only early-
        submittable same-token records are hash layers > L (L=0 -> L1/L2
        ids, L=1 -> L2 ids); L0's own ids have no earlier in-token gap.
        Token boundary: ids of token t+1 exist once sample(t) completes —
        inside the gap after row (t,42) — so that gap may also begin
        t+1's L0/L1/L2 records (deadlines = their demand rows)."""
        b = batches[i]
        t, L = b["step"], b["layer"]
        if b["phase"] != "decode":
            return []
        out = []
        if L in (0, 1):
            for tl in range(L + 1, 3):           # remaining hash layers
                j = i + (tl - L)                 # row index of (t, tl)
                for e in sorted(D.get((t, tl), ())):
                    out.append((0, (tl, e), j))  # layers 0-2 all on cuda0
        elif L == N_LAYERS - 1 and t + 1 <= max_step:
            # token-boundary gap -> next token's full hash set
            for tl in HASH_LAYERS:
                j = i + 1 + tl                   # row index of (t+1, tl)
                if j >= n or batches[j]["step"] != t + 1:
                    continue
                for e in sorted(D.get((t + 1, tl), ())):
                    out.append((0, (tl, e), j))
        return out

    return {"oracle_gap": oracle,
            "prevtok": lambda i: prevtok(i, 0),
            "prevtok_freq_backfill": lambda i: prevtok(i, backfill=6),
            "hash_L0L2_exact": hash_exact,
            "base_miss_keys": base_miss_keys}


def fill_wall_ms(m):
    return W1[m] if m in W1 else W1_SLOPE * m


def summarize(rows, label):
    dec = [r for r in rows if r["phase"] == "decode"]
    pre = [r for r in rows if r["phase"] == "prefill"]
    dm = sum(r["host_misses"] for r in dec)
    pm = sum(r["host_misses"] for r in pre)
    dwall = sum(fill_wall_ms(r["host_misses"]) for r in dec)
    pwall = sum(fill_wall_ms(r["host_misses"]) for r in pre)
    return {"label": label, "decode_preads": dm, "prefill_preads": pm,
            "total_preads": dm + pm,
            "decode_fill_wall_s": round(dwall / 1000, 3),
            "prefill_fill_wall_s": round(pwall / 1000, 3),
            "decode_gib": round(dm * RECORD_BYTES / GIB, 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jsha = sha256_of(JOURNAL)
    batches = load_batches(JOURNAL)

    # ---- baseline replay + sealed validation -----------------------------
    base_rows, _ = replay(batches)
    base = summarize(base_rows, "baseline_serial")
    per_gpu_dev_miss = defaultdict(int)
    for r in base_rows:
        per_gpu_dev_miss[r["gpu"]] += (r["requests"] - r["device_hits"])
    # single-level check: every request consults the host pack (legacy
    # prepare-path semantics) -> must reproduce sealed host_pack counters
    single_level_miss = {}
    for g in (0, 1):
        c = LRU(HOST_SLOTS)
        m = 0
        for b in batches:
            if b["gpu"] != g:
                continue
            for e in b["experts"]:
                if not c.touch((b["layer"], e)):
                    m += 1
        single_level_miss[g] = m

    # ---- demand-structure facts ------------------------------------------
    D = demand_map(batches)
    dec_requests = sum(r["requests"] for r in base_rows
                       if r["phase"] == "decode")
    hash_dec_req = sum(r["requests"] for r in base_rows
                       if r["phase"] == "decode" and r["layer"] in HASH_LAYERS)
    hash_dec_miss = sum(r["host_misses"] for r in base_rows
                        if r["phase"] == "decode" and r["layer"] in HASH_LAYERS)
    miss_dist = Counter(r["host_misses"] for r in base_rows
                        if r["phase"] == "decode")

    # prev-token same-layer recall on demand and on the miss stream
    rec_d, rec_m = [], []
    for i, b in enumerate(batches):
        if b["phase"] != "decode" or b["step"] < 2:
            continue
        t, L = b["step"], b["layer"]
        a = D[(t, L)]
        pv = D.get((t - 1, L), set())
        rec_d.append(len(a & pv) / len(a))
    base_miss = build_predictors(batches, base_rows)["base_miss_keys"]
    for i, b in enumerate(batches):
        if b["phase"] != "decode" or b["step"] < 2:
            continue
        t, L = b["step"], b["layer"]
        pv = {(L, e) for e in D.get((t - 1, L), set())}
        mk = set(base_miss[i])
        if mk:
            rec_m.append(len(mk & pv) / len(mk))

    # ---- sweep: gap_ms x BW x source -------------------------------------
    predictors = build_predictors(batches, base_rows)
    sweep_rows = []
    for gap_ms in (GAP_CONSERVATIVE_MS, GAP_MAX_MS):
        for bw in BWS:
            gap_bytes = gap_ms / 1000.0 * bw * GIB
            for name in ("hash_L0L2_exact", "prevtok",
                         "prevtok_freq_backfill", "oracle_gap"):
                rows, st = replay(batches,
                                  prefetch=predictors[name],
                                  gap_ms=gap_ms, bw_gibps=bw)
                s = summarize(rows, name)
                saved_ms = sum(
                    fill_wall_ms(b["host_misses"]) -
                    fill_wall_ms(r["host_misses"])
                    for b, r in zip(base_rows, rows)
                    if r["phase"] == "decode")
                sweep_rows.append({
                    "gap_ms": gap_ms, "bw_gibps": bw, "source": name,
                    "gap_records_per_row": round(gap_bytes / RECORD_BYTES, 3),
                    "decode_preads": s["decode_preads"],
                    "preads_saved": base["decode_preads"] - s["decode_preads"],
                    "prefetches_completed": st["prefetches_completed"],
                    "prefetches_abandoned": st["prefetches_abandoned_partial"],
                    "prefetches_used_at_demand": st["prefetch_hits_at_demand"],
                    "decode_fill_wall_s": s["decode_fill_wall_s"],
                    "wall_saved_s": round(saved_ms / 1000, 3),
                    "evictions0": st["host_evictions"][0],
                    "evictions1": st["host_evictions"][1],
                })
    with open(OUT / "gap_prefetch_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    # ---- per-row miss stream (the workload any source must cover) --------
    with open(OUT / "miss_stream.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base_rows[0].keys()))
        w.writeheader()
        w.writerows(base_rows)

    # ---- breakeven math ---------------------------------------------------
    # Per prefetch record the gap admits ~gap_rec candidates; each converts
    # with probability = predictor precision p; a wrong guess costs one
    # host-LRU eviction (pollution miss probability q = baseline host
    # re-reference rate of the LRU tail). Reported as bounds, not a model.
    summary = {
        "journal_sha256": jsha,
        "journal_records": len(batches),
        "baseline": base,
        "sealed_validation": {
            "sim_singlelevel_host_miss_cuda0": single_level_miss[0],
            "sealed_host_pack_miss_cuda0": 1390,
            "sim_singlelevel_host_miss_cuda1": single_level_miss[1],
            "sealed_host_pack_miss_cuda1": 1091,
            "sim_device_miss_cuda0": per_gpu_dev_miss[0],
            "sealed_device_miss_cuda0": 2285,
            "sim_device_miss_cuda1": per_gpu_dev_miss[1],
            "sealed_device_miss_cuda1": 2159,
            "twolevel_preads_total": base["total_preads"],
            "p2c_sim_preads_total": 2453,
        },
        "demand_structure": {
            "decode_requests": dec_requests,
            "decode_requests_per_row": dec_requests / 645,
            "decode_miss_dist": {str(k): v for k, v in sorted(miss_dist.items())},
            "decode_miss_mean": round(
                sum(k * v for k, v in miss_dist.items()) / 645, 3),
            "hash_decode_requests": hash_dec_req,
            "hash_decode_host_misses": hash_dec_miss,
            "prevtok_samelayer_recall_at6_demand": round(
                sum(rec_d) / len(rec_d), 4),
            "prevtok_samelayer_recall_on_misses": round(
                sum(rec_m) / len(rec_m), 4),
        },
        "constants": {
            "record_bytes": RECORD_BYTES,
            "row_wall_ms": ROW_WALL_MS, "row_fill_ms": ROW_FILL_MS,
            "gap_conservative_ms": GAP_CONSERVATIVE_MS,
            "gap_max_ms": GAP_MAX_MS,
            "w1_table_ms": W1, "w1_slope_ms_per_miss": W1_SLOPE,
            "all_256_per_layer_gib": round(256 * RECORD_BYTES / GIB, 3),
            "device_slots_per_gpu": DEVICE_SLOTS,
            "host_slots_per_gpu": HOST_SLOTS,
        },
    }
    with open(OUT / "legal_prefetch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
