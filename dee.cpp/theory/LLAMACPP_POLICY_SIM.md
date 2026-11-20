# DEE next-use vs llama.cpp's MoE cache policies (free trace replay)

2026-11-20. No GPU. Question: does DEE's next-use eviction beat llama.cpp's merged
`--moe-cache-mib` policy (ClassLRU, PR #31233) enough to be worth an upstream PR?

Method: `theory/llamacpp_policy_sim.py` replays DSv4-Flash route journals
(`benchmark_reports/deepseek-v4-flash-0731-t4/gpu2-phase4-cache-hierarchy/evidence/a2_fp4/routed_experts-q{0..7}.jsonl`,
8 prompts, ~1,000 decode tokens, concatenated) through one slot pool shared by all 43 layers.
ClassLRU is imported unchanged from llama.cpp `scripts/moe-cache-tune.py` (llama.cpp @ 374d90a).
Hit rate = hits / expert lookups after 64 warm-up tokens, CPU share 0.

## b=1 (one request)

| cache | LRU | ClassLRU | next-use | Belady |
|---|---|---|---|---|
| 5% | 0.453 | 0.490 | 0.500 | 0.659 |
| 10% | 0.588 | 0.615 | 0.607 | 0.771 |
| 15% | 0.677 | 0.688 | 0.685 | 0.833 |
| 20% | 0.741 | 0.745 | 0.745 | 0.874 |
| 30% | 0.831 | 0.823 | 0.823 | 0.917 |
| 50% | 0.914 | 0.910 | 0.904 | 0.958 |

next-use alpha sweep (0.05-0.4) moves it by at most ~1.5 points; no setting beats ClassLRU by 2.

## Parallel requests (prompts merged per step)

| b | cache | LRU | ClassLRU | next-use | Belady |
|---|---|---|---|---|---|
| 4 | 5% | 0.000 | 0.250 | 0.315 | 0.480 |
| 4 | 10% | 0.425 | 0.459 | 0.466 | 0.660 |
| 8 | 5% | 0.000 | 0.144 | 0.230 | 0.329 |
| 8 | 10% | 0.000 | 0.334 | 0.380 | 0.568 |
| 8 | 20% | 0.474 | 0.560 | 0.570 | 0.753 |

## Perfect short-horizon predictor (b=1)

next-use plus never evicting an expert truly used within H tokens:
10% cache: 1 tok 0.628, 4 tok 0.666, 16 tok 0.742 (vs next-use 0.607, Belady 0.771).

## Verdict

- b=1: no-go. Next-use ties ClassLRU.
- b>=4 with a 5-10% cache: next-use +4.6 to +8.5 points. Worth checking on real llama.cpp traces
  (`LLAMA_MOE_CACHE_TRACE`, needs a GPU backend) of a popular model before any PR.
- The Belady gap is long-range reuse; even perfect 1-token lookahead closes ~2 of 16 points.
- Caveats: one model, DEE's FP4 routing rather than a llama.cpp quant, short traces.
