#!/usr/bin/env python3
"""Probe v3: validate BOTH tiers against sealed counters with tuple keys.

Host tier  (plain LRU, 682 slots/GPU, dedup stream):
  sealed fill-live: cuda0 (hits 1223, misses 1390)  cuda1 (1395, 1091)
VRAM tier  (281 slots/GPU):
  plain LRU        -> sim
  priority LRU     -> sim   (score = last_used + (K-i)*2**20, refreshed on hit)
  sealed v60:       cuda0 (hits 328, cold 2285, ev 2004)
                    cuda1 (hits 327, cold 2159, ev 1878)
"""
import json
from collections import deque

J = ('dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/'
     'v50-evidence-20260829T195940Z/routed_experts.jsonl')
recs = [json.loads(l) for l in open(J) if l.strip()]
recs.sort(key=lambda r: r['record_index'])

BATCHES = []
for r in recs:
    uniq = sorted({e for row in r['expert_ids_rank_order'] for e in row})
    BATCHES.append((r['forward_step'], r['layer'],
                    [(r['layer'], e) for e in uniq]))


def lru_host(split, slots):
    lru, pos = deque(), set()
    ev = cold = hits = 0
    for step, layer, keys in BATCHES:
        if (layer < 22) != (split == 0):
            continue
        for key in keys:
            if key in pos:
                hits += 1
                lru.remove(key)
                lru.appendleft(key)
            else:
                cold += 1
                if len(lru) >= slots:
                    pos.discard(lru.pop())
                    ev += 1
                lru.appendleft(key)
                pos.add(key)
    return hits, cold, ev


def vram(split, slots, mode):
    blocks = {}
    tick = ev = cold = hits = 0
    PW = 1 << 20
    for step, layer, keys in BATCHES:
        if (layer < 22) != (split == 0):
            continue
        K = len(keys)
        for i, key in enumerate(keys):
            tick += 1
            prio = (K - i) if mode == 'priority' else 0
            if key in blocks:
                hits += 1
                blocks[key][0] = tick
                blocks[key][1] = prio
            else:
                cold += 1
                while len(blocks) >= slots:
                    if mode == 'priority':
                        victim = min(blocks, key=lambda k: blocks[k][0] + blocks[k][1] * PW)
                    else:
                        victim = min(blocks, key=lambda k: blocks[k][0])
                    del blocks[victim]
                    ev += 1
                blocks[key] = [tick, prio]
    return hits, cold, ev


if __name__ == '__main__':
    print('== HOST tier, plain LRU, engine order ==')
    for split, name in ((0, 'cuda0'), (1, 'cuda1')):
        h, c, e = lru_host(split, 682)
        print(f'{name} 682 slots: hits={h} misses={c} ev={e}')
    print('sealed fill-live: cuda0 hits=1223 misses=1390 | cuda1 hits=1395 misses=1091')
    print('== VRAM tier, 281 slots ==')
    for mode in ('lru', 'priority'):
        row = []
        for split in (0, 1):
            row.append(vram(split, 281, mode))
        print(f'{mode:9s} cuda0 (hits={row[0][0]}, cold={row[0][1]}, ev={row[0][2]})  '
              f'cuda1 (hits={row[1][0]}, cold={row[1][1]}, ev={row[1][2]})')
    print('sealed v60: cuda0 (hits=328, cold=2285, ev=2004)  cuda1 (hits=327, cold=2159, ev=1878)')
