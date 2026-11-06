"""Phase-4 A8: reuse analysis over Phase-3 route journals.

Journal record = one (forward_step, layer) event:
  expert_ids_rank_order[token_row][rank] -> expert id
Engine fetch granularity: unique (layer, expert) per record, one fetch each.
Cache replay = per-device ordered stream of unique-expert requests.

Sanity gate: replay must reproduce GPU-2 measured counters on q0:
  cuda0: 97 hits / 2744 loads @ 74 fp16 slots, host ~1050 hits @ 8.5GiB=668 rec
"""
import json, sys, math
from collections import defaultdict, Counter

BASE = r"C:/Users/carth/Downloads/dynamic_expert_eviction/.freebuff/wt/p4/dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/gpu2-phase3-fullstore"
REC_BYTES = 13_369_344

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def record_stream(rec):
    """Unique experts of one (forward,layer) record, first-seen order."""
    seen, order = set(), []
    for row in rec["expert_ids_rank_order"]:
        for e in row:
            if e not in seen:
                seen.add(e); order.append(e)
    return rec["forward_step"], rec["layer"], rec["device"], order, len(rec["expert_ids_rank_order"])

def build_streams(recs):
    """Per-device ordered access stream of (layer, expert, fwd, n_rows)."""
    by_dev = defaultdict(list)
    for r in sorted(recs, key=lambda x: x["record_index"]):
        fwd, layer, dev, experts, nrows = record_stream(r)
        for e in experts:
            by_dev[dev].append((layer, e, fwd, nrows))
    return by_dev

def lru_replay(events, cap):
    """Plain LRU, uniform-size records. Returns (hits, misses, evictions)."""
    cache, lru_pos, tick = {}, {}, 0
    hits = misses = evictions = 0
    for (layer, e, fwd, nrows) in events:
        k = (layer, e); tick += 1
        if k in cache:
            hits += 1; lru_pos[k] = tick; continue
        misses += 1
        while len(cache) >= cap:
            victim = min(lru_pos, key=lru_pos.get)
            del lru_pos[victim]; del cache[victim]; evictions += 1
        cache[k] = True; lru_pos[k] = tick
    return hits, misses, evictions

def rank_prio_replay(events_with_prio, cap, weight=1 << 20):
    """Buggy live policy: evict min(last_used + priority*W)."""
    score, tick = {}, 0
    hits = misses = evictions = 0
    present = set()
    for (k, prio) in events_with_prio:
        tick += 1
        if k in present:
            hits += 1; score[k] = tick + prio * weight; continue
        misses += 1
        while len(present) >= cap:
            victim = min(present, key=lambda kk: score[kk])
            present.discard(victim); del score[victim]; evictions += 1
        present.add(k); score[k] = tick + prio * weight
    return hits, misses, evictions

def analyze(name, recs):
    print(f"\n================ {name} ================")
    by_dev = build_streams(recs)
    total_events = sum(len(v) for v in by_dev.values())
    all_keys = set()
    for dev, ev in by_dev.items():
        for (l, e, f, n) in ev: all_keys.add((l, e))
    print(f"records={len(recs)}  access-events={total_events}  unique-experts={len(all_keys)}")
    for dev in sorted(by_dev):
        ev = by_dev[dev]
        uniq = set((l, e) for (l, e, f, n) in ev)
        print(f"  {dev}: events={len(ev)}  unique={len(uniq)}  reuse_ratio={len(ev)/len(uniq):.2f}x")
        # reuse distances (event units)
        last = {}; dists = []
        for i, (l, e, f, n) in enumerate(ev):
            k = (l, e)
            if k in last: dists.append(i - last[k])
            last[k] = i
        if dists:
            dists.sort()
            n = len(dists)
            print(f"    reuse-dist p50={dists[n//2]} p90={dists[int(n*0.9)]} p99={dists[int(n*0.99)]} max={dists[-1]} (n={n})")
            for cap in (74, 148, 281, 512, 668):
                within = sum(1 for d in dists if d <= cap) / n * 100
                print(f"      reuse within {cap:>4} events: {within:5.1f}%")
        # hot experts
        cnt = Counter((l, e) for (l, e, f, n) in ev)
        for frac in (0.01, 0.05, 0.10, 0.25):
            topn = max(1, int(len(cnt) * frac))
            cov = sum(c for _, c in cnt.most_common(topn)) / len(ev) * 100
            print(f"    top {frac*100:4.0f}% experts ({topn:4d}) cover {cov:5.1f}% of events")
        # per-layer reuse: avg accesses per unique expert
        per_layer = defaultdict(set); layer_events = Counter()
        for (l, e, f, n) in ev:
            per_layer[l].add(e); layer_events[l] += 1
        lrs = sorted(layer_events[l] / len(per_layer[l]) for l in per_layer)
        print(f"    per-layer reuse ratio p50={lrs[len(lrs)//2]:.2f} min={lrs[0]:.2f} max={lrs[-1]:.2f}")
    return by_dev

def main():
    prompts = {}
    for q in (0, 1, 2):
        recs = load(f"{BASE}/routed_experts-a1-q{q}.jsonl")
        prompts[f"q{q}"] = recs
    streams = {}
    for name, recs in prompts.items():
        streams[name] = analyze(name, recs)

    # --- capacity sweeps per device per prompt ---
    print("\n================ LRU replay: GPU slots ================")
    for name, by_dev in streams.items():
        for dev in sorted(by_dev):
            ev = by_dev[dev]
            row = []
            for cap in (8, 16, 32, 48, 64, 74, 128, 192, 281, 512):
                h, m, x = lru_replay(ev, cap)
                row.append(f"{cap}:{h/len(ev)*100:5.1f}%")
            print(f"  {name} {dev}: " + "  ".join(row))

    print("\n================ LRU replay: host cache (records) ================")
    for name, by_dev in streams.items():
        for dev in sorted(by_dev):
            ev = by_dev[dev]
            row = []
            for cap in (64, 128, 256, 334, 668, 1024, 1336, 2048):
                h, m, x = lru_replay(ev, cap)
                gib = cap * REC_BYTES / (1 << 30)
                row.append(f"{cap}({gib:.1f}GiB):{h/len(ev)*100:5.1f}%")
            print(f"  {name} {dev}: " + "  ".join(row))

    # --- rank-priority (live bug) vs LRU at 74 slots ---
    print("\n======== live rank-priority vs pure LRU @74 slots ========")
    for name, by_dev in streams.items():
        for dev in sorted(by_dev):
            # rebuild stream with per-record unique-order priority: K-k
            ev_p = []
            for r in sorted(prompts[name], key=lambda x: x["record_index"]):
                if r["device"] != dev: continue
                fwd, layer, d, experts, nrows = record_stream(r)
                K = len(experts)
                for k, e in enumerate(experts):
                    ev_p.append(((layer, e), K - k))
            hp, mp, xp = rank_prio_replay(ev_p, 74)
            hl, ml, xl = lru_replay(by_dev[dev], 74)
            print(f"  {name} {dev}: rank-prio {hp/len(ev_p)*100:5.1f}% hits  |  lru {hl/len(by_dev[dev])*100:5.1f}% hits   (n={len(ev_p)})")

    # --- inter-prompt sharing ---
    print("\n================ inter-prompt overlap ================")
    keysets = {}
    for name, by_dev in streams.items():
        ks = set()
        for dev, ev in by_dev.items():
            ks |= set((l, e) for (l, e, f, n) in ev)
        keysets[name] = ks
    names = list(keysets)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            inter = len(keysets[a] & keysets[b])
            print(f"  {a} ∩ {b}: {inter} shared experts ({inter/len(keysets[a])*100:.0f}% of {a}, {inter/len(keysets[b])*100:.0f}% of {b})")

main()
