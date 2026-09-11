#!/usr/bin/env python3
"""R4 completeness replay: scan-resistant policies (canonical ARC, LIRS, 2Q,
SLRU, W-TinyLFU) on the sealed v50 route journal, plus scan-pollution and
repeated-request probes.

Why this exists (track R4):
  research/phase2-ws-policy @95dfe0d/8c921ff selected plain causal LRU for the
  host tier on the sealed single-request window, with regime-labeled rows for
  lru/arc/lfu/freq_x_recency/layer_lru/cost_aware/belady and prewarm arms.
  Left unevaluated: the *scan-resistant* family in canonical form (LIRS, 2Q,
  SLRU, W-TinyLFU) and any explicit scan-pollution probe. The sealed stream
  already contains intrinsic scan structure (1,429 of 2,364 unique records
  never repeat), so this replay is a cheap completeness check on the
  "no causal policy beats LRU here" conclusion, NOT a new-policy proposal.

ADMISSION GATE (fail-closed): before reporting any new-policy row, the tool
replays plain LRU + Belady on the reconstructed engine-dedup stream and
requires exact equality with the published sim counters, and within-±1
agreement with the sealed live anchors:

  pooled  (scope full):  642 slots -> 2056/3043/2401  (hits/misses/evictions)
                         963 slots -> 2376/2723/1760
                        1285 slots -> 2592/2507/1222
                        1606 slots -> 2668/2431/825
                        1927 slots -> 2707/2392/465
                        2570 slots -> 2735/2364/0
  per-scope (682 slots): cuda0 layers<22 -> sim 1222/1391/709
                                     (sealed live 1223/1390/708, ±1)
                         cuda1 layers>=22 -> 1395/1091/409 (sealed == sim)
  belady bound:          hits == 2735 at every pooled budget >= 642 slots.

If any gate cell fails the tool writes NOTHING and exits nonzero — the stream
reconstruction or a policy is wrong, and no downstream number is admissible.

Stream reconstruction is byte-for-byte the v4 sim's
(tools/phase2_ws_policy_sim_v4.py @8c921ff): journal records sorted by
record_index; per (forward_step, layer) the engine-dedup batch =
sorted(unique(expert_ids_rank_order)); key = (layer, expert_id).
5,099 requests, 2,364 unique keys, 16 forwards, 43 layers.

Probes (all multi-segment; per-segment counters reported):
  sealed      [sealed]                              -- comparability
  repeat3     [sealed, sealed, sealed]              -- identical-request
             upper bound: cross-request reuse when the next request is
             literally this one (dee-serve best case)
  scan        [sealed, SCAN, sealed]                -- pollution probe:
             SCAN = one sequential pass over the 8,644 untouched
             (layer, expert) records of the 43x256 universe in (layer,eid)
             order (a "foreign request family" worst case), then the sealed
             stream replayed.  Pass-2 misses = re-fetch bytes a scan costs.
             The no-scan reference for the same replay position is the
             second segment of `repeat3`.

Slots: floor(gib * 2^30 / 13,369,344) — identical to the v4 sim and the
tier-replay harness (8 GiB -> 642 slots, 16 -> 1285, ...).

Policies: lru (control), arc (canonical Megiddo-Modha), arc_v4 (the exact
variant shipped in the v4 sim, for cross-implementation parity), lirs, 2q,
slru, wtinylfu_exactfreq, belady (offline MIN bound).

stdlib only. Usage:
  python tools/phase2_scan_resistance_replay.py --out research/prior-art/results/r04_scan_replay.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, OrderedDict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOURNAL_REL = (REPO / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
               "v50-evidence-20260829T195940Z/routed_experts.jsonl")
JOURNAL_FALLBACKS = [
    Path("C:/Users/carth/Downloads/dynamic_expert_eviction/dee.cpp/"
         "benchmark_reports/deepseek-v4-flash-0731-t4/"
         "v50-evidence-20260829T195940Z/routed_experts.jsonl"),
]
JOURNAL_SHA256 = ("665aac3e8db570237c6dc6acaf08dc39f2af890e8a04e400ce7154f"
                  "1a858dae1")

RECORD_BYTES = 13_369_344          # 12.75 MiB DEE4 packed FP4 record
GIB = 1 << 30
N_LAYERS = 43
N_EXPERTS = 256
SPLIT_GPU0 = 22                    # cuda0 layers 0-21, cuda1 layers 22-42

# Published counters this replay must reproduce before anything is reported
# (TIER_REPLAY_VALIDATION.md Result 1; results/validation_v4.json).
GATE_POOLED_LRU = {   # slots -> (hits, misses, evictions)
    642: (2056, 3043, 2401),
    963: (2376, 2723, 1760),
    1285: (2592, 2507, 1222),
    1606: (2668, 2431, 825),
    1927: (2707, 2392, 465),
    2570: (2735, 2364, 0),
}
GATE_SCOPE_LRU_682 = {   # scope -> (sim_hits, sim_miss, sim_evict, sealed)
    "cuda0": (1222, 1391, 709, (1223, 1390, 708)),
    "cuda1": (1395, 1091, 409, (1395, 1091, 409)),
}
GATE_BELADY_HITS = 2735               # pooled, every budget >= 642 slots


def journal_path():
    for p in (JOURNAL_REL, *JOURNAL_FALLBACKS):
        if p.exists():
            return p
    raise FileNotFoundError("v50 routed_experts.jsonl not found")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_batches(path):
    """Identical to phase2_ws_policy_sim_v4.load_batches."""
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs.sort(key=lambda r: r["record_index"])
    batches = []
    for r in recs:
        uniq = sorted({e for row in r["expert_ids_rank_order"] for e in row})
        batches.append((r["forward_step"], r["layer"],
                        [(r["layer"], e) for e in uniq]))
    return batches


def flat_stream(batches):
    return [k for _, _, ks in batches for k in ks]


def slots_for(gib):
    return int(gib * GIB // RECORD_BYTES)


# ---------------------------------------------------------------- policies

class LRU:
    name = "lru"

    def __init__(self, slots, **kw):
        self.slots = slots
        self.od = OrderedDict()
        self.evictions = 0

    def access(self, key):
        if key in self.od:
            self.od.move_to_end(key, last=False)
            return True
        if self.slots <= 0:
            return False
        if len(self.od) >= self.slots:
            self.od.popitem(last=True)
            self.evictions += 1
        self.od[key] = None
        self.od.move_to_end(key, last=False)
        return False


class Belady:
    name = "belady"

    def __init__(self, slots, next_use=None, **kw):
        self.slots = slots
        self.resident = {}
        self.evictions = 0
        self.next_use = next_use if next_use is not None else {}
        self.i = 0

    def access(self, key):
        nu = self.next_use.get((self.i, key), 10 ** 12)
        self.i += 1
        if key in self.resident:
            self.resident[key] = nu
            return True
        if self.slots <= 0:
            return False
        if len(self.resident) >= self.slots:
            far = max(self.resident, key=lambda k: self.resident[k])
            del self.resident[far]
            self.evictions += 1
        self.resident[key] = nu
        return False


class ARCCanonical:
    """Megiddo & Modha ARC (FAST'03). T1/T2 resident LRU lists, B1/B2 ghost
    lists, adaptive target p."""
    name = "arc"

    def __init__(self, slots, **kw):
        self.c = max(1, slots)
        self.t1, self.t2 = OrderedDict(), OrderedDict()
        self.b1, self.b2 = OrderedDict(), OrderedDict()
        self.p = 0
        self.evictions = 0

    def _replace(self, in_b2):
        if self.t1 and (len(self.t1) > self.p
                        or (in_b2 and len(self.t1) == self.p)):
            old, _ = self.t1.popitem(last=True)      # LRU of T1
            self.b1[old] = None
            self.b1.move_to_end(old, last=False)
            self.evictions += 1
        else:
            old, _ = self.t2.popitem(last=True)      # LRU of T2
            self.b2[old] = None
            self.b2.move_to_end(old, last=False)
            self.evictions += 1

    def access(self, key):
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = None
            self.t2.move_to_end(key, last=False)
            return True
        if key in self.t2:
            self.t2.move_to_end(key, last=False)
            return True
        if key in self.b1:                            # ghost hit, B1
            self.p = min(self.c, self.p + max(1, len(self.b2) // len(self.b1)))
            self._replace(False)
            del self.b1[key]
            self.t2[key] = None
            self.t2.move_to_end(key, last=False)
            return False
        if key in self.b2:                            # ghost hit, B2
            self.p = max(0, self.p - max(1, len(self.b1) // len(self.b2)))
            self._replace(True)
            del self.b2[key]
            self.t2[key] = None
            self.t2.move_to_end(key, last=False)
            return False
        l1 = len(self.t1) + len(self.b1)
        l2 = len(self.t2) + len(self.b2)
        if l1 == self.c:
            if len(self.t1) < self.c:
                self.b1.popitem(last=True)
                self._replace(False)
            else:
                self.t1.popitem(last=True)
                self.evictions += 1
        elif l1 < self.c <= l1 + l2:
            if l1 + l2 >= 2 * self.c and self.b2:
                self.b2.popitem(last=True)
            self._replace(False)
        self.t1[key] = None
        self.t1.move_to_end(key, last=False)
        return False


class ARCv4:
    """Byte-faithful port of the v4 sim's ARC variant
    (tools/phase2_ws_policy_sim_v4.py @8c921ff, class ARC). Kept for
    cross-implementation parity against the published 31.30/42.91/50.44/...
    row — differences vs canonical ARC are themselves a finding."""
    name = "arc_v4"

    def __init__(self, slots, **kw):
        self.c = max(1, slots)
        self.t1, self.t2, self.b1, self.b2 = deque(), deque(), deque(), deque()
        self.p = 0
        self.evictions = 0

    def _replace(self, key, in_b2):
        if (self.t1 and
                ((len(self.t1) > self.p) or
                 (in_b2 and len(self.t1) == self.p))):
            self.t1.pop()
            self.evictions += 1
        elif self.t2:
            self.t2.pop()
            self.evictions += 1

    def access(self, key):
        if key in self.t1 or key in self.t2:
            if key in self.t1:
                self.t1.remove(key)
                self.p = min(self.c, self.p + (1 if len(self.b1) >= len(self.b2) else 0))
            else:
                self.t2.remove(key)
            self.t2.appendleft(key)
            return True
        case1 = len(self.t1) + len(self.b1)
        case2 = case1 + len(self.t2) + len(self.b2)
        if case1 == self.c:
            if len(self.t1) < self.c:
                self.b1.pop()
                self._replace(key, False)
            else:
                self.t1.pop()
                self.evictions += 1
        elif case1 < self.c <= case2:
            if case2 >= 2 * self.c and self.b2:
                self.b2.pop()
            self._replace(key, False)
        if key in self.b1:
            self.b1.remove(key)
            self.p = min(self.c, self.p + 1 + len(self.b2) // max(1, len(self.b1)))
        elif key in self.b2:
            self.b2.remove(key)
            self.p = max(0, self.p - (1 + len(self.b1) // max(1, len(self.b2))))
        self.t1.appendleft(key)
        return False


class LIRS:
    """Jiang & Zhang LIRS (SIGMETRICS'02). S = recency stack holding LIR
    blocks + HIR blocks seen recently (resident or ghost); Q = FIFO of
    resident HIR blocks. Bottom of S is always LIR (prune invariant).
    hir_ratio = fraction of slots reserved for resident HIRs (paper: ~1%)."""
    name = "lirs"

    def __init__(self, slots, hir_ratio=0.01, **kw):
        self.c = max(1, slots)
        self.h_cap = max(1, int(round(self.c * hir_ratio)))
        self.l_cap = self.c - self.h_cap
        self.status = {}                    # key -> 'LIR' | 'HIR_R' | 'HIR_NR'
        self.S = OrderedDict()              # MRU first; LIR always present
        self.Q = deque()                    # resident HIRs, LRU at left
        self.qset = set()
        self.n_lir = 0
        self.evictions = 0

    def _prune(self):
        while self.S:
            bottom = next(reversed(self.S))
            if self.status[bottom] == 'LIR':
                break
            del self.S[bottom]              # leaves Q membership untouched

    def _demote_bottom_lir(self):
        bottom = self.S.popitem(last=True)[0]
        self.status[bottom] = 'HIR_R'
        self.Q.append(bottom)
        self.qset.add(bottom)
        self.n_lir -= 1

    def _evict_q_head(self):
        if not self.Q:
            # Cache full of LIRs: demote the bottom LIR then evict it.
            self._demote_bottom_lir()
        victim = self.Q.popleft()
        self.qset.discard(victim)
        self.status[victim] = 'HIR_NR'      # stays in S as ghost until pruned
        self.evictions += 1

    def _resident(self):
        return self.n_lir + len(self.Q)

    def access(self, key):
        st = self.status.get(key)
        if st == 'LIR':
            self.S.move_to_end(key, last=False)
            self._prune()
            return True
        if st == 'HIR_R':
            if key in self.S:
                self.status[key] = 'LIR'
                self.n_lir += 1
                self.S.move_to_end(key, last=False)
                self.Q.remove(key)
                self.qset.discard(key)
                self._demote_bottom_lir()
                self._prune()
            else:
                self.Q.remove(key)
                self.Q.append(key)          # most-recent end of Q
            return True
        # miss (new key or non-resident HIR ghost)
        if st == 'HIR_NR' and key in self.S:  # live ghost in S -> promote
            self._evict_q_head()            # free a resident HIR slot
            self.status[key] = 'LIR'
            self.n_lir += 1
            self.S.move_to_end(key, last=False)
            self._demote_bottom_lir()
            self._prune()
            return False
        # never-seen (or pruned-ghost) miss
        if self._resident() >= self.c:
            self._evict_q_head()
        if self.n_lir < self.l_cap:
            self.status[key] = 'LIR'
            self.n_lir += 1
        else:
            self.status[key] = 'HIR_R'
            self.Q.append(key)
            self.qset.add(key)
        self.S[key] = None
        self.S.move_to_end(key, last=False)
        return False


class TwoQ:
    """Johnson & Shasha 2Q: A1in FIFO (kin=25%c), Am LRU (km=75%c),
    A1out ghost FIFO (kout=50%c). A1in hits do not promote; promotion only
    via A1out ghost hits."""
    name = "2q"

    def __init__(self, slots, **kw):
        self.c = max(1, slots)
        self.kin = max(1, self.c // 4)
        self.kout = max(1, self.c // 2)
        self.km = max(1, self.c - self.kin)
        self.a1in, self.am = deque(), OrderedDict()
        self.a1out = deque()
        self.in_a1in, self.in_a1out = set(), set()
        self.evictions = 0

    def _resident(self):
        return len(self.a1in) + len(self.am)

    def access(self, key):
        if key in self.am:
            self.am.move_to_end(key, last=False)
            return True
        if key in self.in_a1in:               # FIFO hit: no promotion
            return True
        if key in self.in_a1out:              # ghost hit -> promote to Am
            self.in_a1out.discard(key)
            self.a1out.remove(key)
            if len(self.am) >= self.km:
                old, _ = self.am.popitem(last=True)
                self.evictions += 1
            self.am[key] = None
            self.am.move_to_end(key, last=False)
            return False
        # cold miss -> A1in
        if self._resident() >= self.c:
            if self.a1in:
                out = self.a1in.popleft()     # evict A1in head -> ghost
                self.in_a1in.discard(out)
                self.in_a1out.add(out)
                self.a1out.appendleft(out)
                if len(self.a1out) > self.kout:
                    drop = self.a1out.pop()
                    self.in_a1out.discard(drop)
                self.evictions += 1
            elif self.am:
                self.am.popitem(last=True)
                self.evictions += 1
        self.a1in.appendleft(key)
        self.in_a1in.add(key)
        if len(self.a1in) > self.kin:
            out = self.a1in.pop()
            self.in_a1in.discard(out)
            self.in_a1out.add(out)
            self.a1out.appendleft(out)
            if len(self.a1out) > self.kout:
                drop = self.a1out.pop()
                self.in_a1out.discard(drop)
            self.evictions += 1
        return False


class SLRU:
    """Segmented LRU (Karedla et al.): probationary (20%) + protected (80%).
    Miss -> probation MRU; hit in probation -> promote to protected MRU
    (protected overflow demotes its LRU to probation MRU); eviction always
    from probation LRU."""
    name = "slru"

    def __init__(self, slots, **kw):
        self.c = max(1, slots)
        self.pcap = max(1, self.c // 5)
        self.prob, self.prot = OrderedDict(), OrderedDict()
        self.evictions = 0

    def access(self, key):
        if key in self.prot:
            self.prot.move_to_end(key, last=False)
            return True
        if key in self.prob:
            del self.prob[key]
            self.prot[key] = None
            self.prot.move_to_end(key, last=False)
            if len(self.prot) > self.c - self.pcap:
                dem, _ = self.prot.popitem(last=True)
                self.prob[dem] = None
                self.prob.move_to_end(dem, last=False)
                if len(self.prob) > self.pcap:
                    self.prob.popitem(last=True)
                    self.evictions += 1
            return True
        self.prob[key] = None
        self.prob.move_to_end(key, last=False)
        if len(self.prob) > self.pcap:
            self.prob.popitem(last=True)
            self.evictions += 1
        elif self._resident() > self.c:
            self.prob.popitem(last=True)
            self.evictions += 1
        return False

    def _resident(self):
        return len(self.prob) + len(self.prot)


class WTinyLFUExact:
    """W-TinyLFU skeleton (Caffeine): 1% window LRU + main LRU, with
    admission by exact frequency counts (a count-min sketch is a strictly
    weaker estimator, so this is the generous variant — labeled). Counts are
    aged by halving every `aging_interval` insertions (Caffeine-style)."""
    name = "wtinylfu_exactfreq"

    def __init__(self, slots, **kw):
        self.c = max(1, slots)
        self.w = max(1, self.c // 100)
        self.win, self.main = OrderedDict(), OrderedDict()
        self.freq = Counter()
        self.inserts = 0
        self.aging_interval = max(1, 10 * self.c)
        self.evictions = 0
        self.admitted = 0
        self.rejected = 0

    def _age(self):
        for k in list(self.freq):
            self.freq[k] //= 2
            if self.freq[k] == 0:
                del self.freq[k]

    def access(self, key):
        self.freq[key] += 1
        if key in self.win:
            self.win.move_to_end(key, last=False)
            return True
        if key in self.main:
            self.main.move_to_end(key, last=False)
            return True
        # miss: insert into window
        self.win[key] = None
        self.win.move_to_end(key, last=False)
        self.inserts += 1
        if self.inserts % self.aging_interval == 0:
            self._age()
        if len(self.win) > self.w:
            cand, _ = self.win.popitem(last=True)
            if len(self.main) < self.c - self.w:
                self.main[cand] = None
                self.main.move_to_end(cand, last=False)
                self.admitted += 1
            else:
                victim = next(reversed(self.main))
                if self.freq[cand] > self.freq[victim]:
                    del self.main[victim]
                    self.main[cand] = None
                    self.main.move_to_end(cand, last=False)
                    self.evictions += 1
                    self.admitted += 1
                else:
                    self.rejected += 1
                    # candidate dropped: never resident in main
        return False


POLICIES = {
    "lru": LRU, "belady": Belady, "arc": ARCCanonical, "arc_v4": ARCv4,
    "lirs": LIRS, "2q": TwoQ, "slru": SLRU, "wtinylfu_exactfreq": WTinyLFUExact,
}


def run_stream(stream, policy_name, slots, key_positions=None):
    """Play `stream` through one fresh cache; return (hits, misses, evictions)."""
    kw = {}
    if policy_name == "belady":
        # next_use lookup by (index, key): precomputed positions of each
        # key's next access index.
        kw["next_use"] = key_positions or {}
    cache = POLICIES[policy_name](slots, **kw)
    hits = misses = 0
    for k in stream:
        if cache.access(k):
            hits += 1
        else:
            misses += 1
    return hits, misses, cache.evictions


def belady_positions(stream):
    """next_use[(i,key)] = next index > i where key occurs (for Belady)."""
    nxt = {}
    last = {}
    for i in range(len(stream) - 1, -1, -1):
        k = stream[i]
        if k in last:
            nxt[(i, k)] = last[k]
        last[k] = i
    return nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=None)
    ap.add_argument("--budgets", default="4,8,12,16,20,24,32")
    ap.add_argument("--policies",
                    default="lru,belady,arc,arc_v4,lirs,2q,slru,wtinylfu_exactfreq")
    ap.add_argument("--probes", default="sealed,repeat3,scan")
    ap.add_argument("--hir-ratio", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-gate", action="store_true",
                    help="DANGEROUS: run without the sealed-counter gate")
    args = ap.parse_args()

    jp = Path(args.journal) if args.journal else journal_path()
    jsha = sha256_of(jp)
    if jsha != JOURNAL_SHA256:
        print(f"FATAL: journal sha256 {jsha} != sealed {JOURNAL_SHA256}",
              file=sys.stderr)
        sys.exit(2)

    batches = load_batches(jp)
    sealed = flat_stream(batches)
    n_req = len(sealed)
    uniq = sorted(set(sealed))
    n_uniq = len(uniq)

    # sanity on stream anatomy (published: 5,099 requests, 2,364 uniques)
    if n_req != 5099 or n_uniq != 2364:
        print(f"FATAL: stream anatomy {n_req} req / {n_uniq} uniq "
              f"!= sealed 5099 / 2364", file=sys.stderr)
        sys.exit(2)

    budgets = [float(b) for b in args.budgets.split(",")]
    slot_list = [(g, slots_for(g)) for g in budgets]
    policies = args.policies.split(",")
    probes = args.probes.split(",")

    # ---------------- admission gate -------------------------------------
    gate = {"journal_sha256": jsha, "checks": [], "ok": True}
    if not args.skip_gate:
        for slots, want in GATE_POOLED_LRU.items():
            h, m, e = run_stream(sealed, "lru", slots)
            ok = (h, m, e) == want
            gate["checks"].append({"cell": f"pooled lru @{slots}",
                                   "got": [h, m, e], "want": list(want),
                                   "ok": ok})
            gate["ok"] &= ok
        for scope, (sh, sm, se, sealed_want) in GATE_SCOPE_LRU_682.items():
            sub = flat_stream([b for b in batches
                               if (b[1] < SPLIT_GPU0) == (scope == "cuda0")])
            h, m, e = run_stream(sub, "lru", 682)
            ok_sim = (h, m, e) == (sh, sm, se)
            ok_sealed = all(abs(a - b) <= 1 for a, b in
                            zip((h, m, e), sealed_want))
            gate["checks"].append({
                "cell": f"{scope} lru @682", "got": [h, m, e],
                "want_sim": [sh, sm, se], "sealed": list(sealed_want),
                "ok": ok_sim and ok_sealed})
            gate["ok"] &= ok_sim and ok_sealed
        nxt = belady_positions(sealed)
        for g, slots in slot_list:
            if slots < 642:
                continue
            h, m, e = run_stream(sealed, "belady", slots, key_positions=nxt)
            ok = h == GATE_BELADY_HITS
            gate["checks"].append({"cell": f"belady @{slots}",
                                   "got_hits": h,
                                   "want_hits": GATE_BELADY_HITS, "ok": ok})
            gate["ok"] &= ok
        if not gate["ok"]:
            print(json.dumps({"fatal": "ADMISSION GATE FAILED",
                              "gate": gate}, indent=1))
            sys.exit(2)

    # ---------------- probes ----------------------------------------------
    universe = [(l, e) for l in range(N_LAYERS) for e in range(N_EXPERTS)]
    untouched = [k for k in universe if k not in set(uniq)]
    segments = {
        "sealed": [("sealed_p1", sealed)],
        "repeat3": [("sealed_p1", sealed), ("sealed_p2", sealed),
                    ("sealed_p3", sealed)],
        "scan": [("sealed_p1", sealed), ("scan_untouched_8644", untouched),
                 ("sealed_p2_postscan", sealed)],
    }

    results = []
    for probe in probes:
        segs = segments[probe]
        full = [k for _, s in segs for k in s]
        nxt_full = belady_positions(full)
        for g, slots in slot_list:
            for pol in policies:
                kw = {}
                if pol == "lirs":
                    kw["hir_ratio"] = args.hir_ratio
                if pol == "belady":
                    kw["next_use"] = nxt_full
                cache = POLICIES[pol](slots, **kw)
                seg_rows = []
                i = 0
                for sname, sstream in segs:
                    h = m = 0
                    for k in sstream:
                        if pol == "belady":
                            cache.i = i      # Belady indexes the probe stream
                        hit = cache.access(k)
                        h += hit
                        m += not hit
                        i += 1
                    seg_rows.append({"segment": sname, "requests": len(sstream),
                                     "hits": h, "misses": m})
                row = {"probe": probe, "policy": pol, "budget_gib": g,
                       "slots": slots, "segments": seg_rows,
                       "evictions": cache.evictions,
                       "hits": sum(s["hits"] for s in seg_rows),
                       "requests": sum(s["requests"] for s in seg_rows)}
                if pol == "wtinylfu_exactfreq":
                    row["admitted"] = cache.admitted
                    row["rejected"] = cache.rejected
                results.append(row)

    doc = {"tool": Path(__file__).name,
           "journal": str(jp), "journal_sha256": jsha,
           "requests_per_sealed_pass": n_req, "unique_pairs": n_uniq,
           "untouched_universe_scan_len": len(untouched),
           "hir_ratio": args.hir_ratio,
           "gate": gate, "results": results}
    out = json.dumps(doc, indent=1)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out, encoding="utf-8")
        print(f"wrote {p}")
    else:
        print(out)


if __name__ == "__main__":
    main()
