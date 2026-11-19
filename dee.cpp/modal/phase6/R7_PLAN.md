# Run r7: tokens/sec across GPUs (plan, 2026-11-19, not launched)

Goal: measure DEE's decode speed on every Modal GPU that fits DSv4-Flash, at b=1 and b=8 and two
host-RAM sizes, so we can fit a TPS model ("given GPU X and Y GiB of RAM, expect Z tok/s") from
real numbers instead of guesses. Every row uses next-use eviction on both tiers (Addendum 6). Prefetch
hints stay off (killed in r6, Nivin's decision 2026-11-18).

Needs first: the harness-fix PR from the "Fix known DEE harness bugs" thread merged into
`research/phase6-modal`. This plan uses its `--max-usd` cap, its all-in cost estimate, and its
per-row decoded text for cohort rows.

## What each container runs

Same four rows on every GPU (one container per GPU, one build and one store copy each):

| tag | b | host GiB | tokens | why |
|---|---|---|---|---|
| `b1-h16-nu` | 1 | 16 | 48 | single-user speed, small RAM |
| `b1-h64-nu` | 1 | 64 | 48 | single-user speed, large RAM |
| `b8-h16-nu` | 8 | 16 | 65 (ignore EOS) | batch throughput, small RAM |
| `b8-h64-nu` | 8 | 64 | 65 (ignore EOS) | batch throughput, large RAM |

The L4 container also runs, first, a repeat of r6's `b8-h16-nu` (same position, cold container) to
measure run-to-run noise, and last, `b8-ask`: Nivin's 8 questions in
`modal/phase6/prompts_ask_dee.json`, 160 tokens each (the harness requires `ignore_eos` at b=8, so each answer is cut at its first EOS when shown). That row is the
"see DEE answer" demo and a TPS point on a different workload.

b=1 uses 48 tokens instead of 16: frozen decode tok/s is UNTESTABLE anyway (B_SSD gate), and 16
tokens is mostly warm-up. Host hit and cold rec/tok are still scored.

## Cost (all-in list price: GPU + 8 CPU + reserved RAM)

Timing is from r6 (3,002 s for setup + four b=8 rows on L4): setup ~6 min, b=8 65-token row ~11
min, b=1 48-token row ~5 min (estimate), ask row ~20 min. Faster GPUs are assumed no faster, so
these are upper estimates.

| order | GPU | $/h all-in | rows | est. min | est. $ | `--max-usd` cap |
|---|---|---|---|---|---|---|
| 1 | 1xL4 | 1.94 | 6 | 58 | 1.88 | 2.80 |
| 2 | 1xA10 | 2.25 | 4 | 38 | 1.43 | 2.20 |
| 2 | 2xL4 | 2.74 | 4 | 38 | 1.74 | 2.60 |
| 2 | 1xL40S | 3.35 | 4 | 38 | 2.12 | 3.20 |
| 3 | RTX PRO 6000 | 4.94 | 4 | 38 | 3.13 | 4.70 |
| | preflights (CPU) | | 5 | | 0.50 | |
| | **total** | | 22 | | **~10.80** | 15.50 |

Worst case if every container hits its cap and its last row runs to the 45 min row timeout: ~$20.
Expected leftover: ~$14 of the ~$25 remaining, kept for MiMo, a chat session, and re-runs.

Not in r7: host-256 rows (288 GiB containers, ~$3 each, availability unverified), T4 (DSv4 does not
fit), Addendum 7 LRU cliff rows (PR #19; LRU is no longer the policy we ship).

## Kill rules (decided before the run)

- **K1** No GPU launch unless its preflight prints `preflight_ok: true`.
- **K2** Exactness: within each container, every b=8 row must match the others' tokens and
  journals where the prompts match (`score.py --placement-exact`), and every row's runner verdict
  must be `ACCEPT_CORRECTNESS`. Any failure stops the campaign after that container; diagnose free.
- **K3** Spend: `--max-usd` skips remaining rows once the container's all-in estimate would cross it.
  If cumulative r7 spend passes **$15**, stop and re-plan.
- **K4** Noise: if L4 `b8-h16-nu` differs from r6 (1.348 tok/s) by more than 20%, pause before
  order 2 and report: the TPS model would need repeats on every GPU.
- **K5** Order: L4 first. Order 2 launches only after L4 is scored clean. RTX PRO 6000 last, and
  only if order 2 passed K2.

## Commands (run from `dee.cpp/`; each `modal run --detach` BILLS)

```
NU='"eviction_policy":"next_use","host_eviction_policy":"next_use"'
B1='"n_tokens":48'
B8='"n_tokens":65,"ignore_eos":true,"cohort":"0-7","prompts_file":"dee.cpp/aws/phase6/prompts_phase4_8.json"'
CORE="{\"tag\":\"b1-h16-nu\",\"host_gib\":16,$B1,$NU},
      {\"tag\":\"b1-h64-nu\",\"host_gib\":64,$B1,$NU},
      {\"tag\":\"b8-h16-nu\",\"host_gib\":16,$B8,$NU},
      {\"tag\":\"b8-h64-nu\",\"host_gib\":64,$B8,$NU}"
ASK="{\"tag\":\"b8-ask\",\"host_gib\":16,\"n_tokens\":160,\"ignore_eos\":true,\"cohort\":\"0-7\",\"prompts_file\":\"dee.cpp/modal/phase6/prompts_ask_dee.json\",$NU}"
L4CFGS="[{\"tag\":\"b8-h16-nu-rep\",\"host_gib\":16,$B8,$NU},$CORE,$ASK]"
CFGS="[$CORE]"

# order 1
modal run modal/phase6/modal_p6.py::preflight --gpu L4 --configs "$L4CFGS"
modal run --detach modal/phase6/modal_p6.py::bench_L4 --run-id p6m-r7-l4 --max-usd 2.8 --configs "$L4CFGS"

# order 2 (after L4 is scored clean; these three may run at the same time)
modal run modal/phase6/modal_p6.py::preflight --gpu A10 --configs "$CFGS"
modal run modal/phase6/modal_p6.py::preflight --gpu 2xL4 --configs "$CFGS"
modal run modal/phase6/modal_p6.py::preflight --gpu L40S --configs "$CFGS"
modal run --detach modal/phase6/modal_p6.py::bench_A10  --run-id p6m-r7-a10  --max-usd 2.2 --configs "$CFGS"
modal run --detach modal/phase6/modal_p6.py::bench_2xL4 --run-id p6m-r7-2xl4 --max-usd 2.6 --configs "$CFGS"
modal run --detach modal/phase6/modal_p6.py::bench_L40S --run-id p6m-r7-l40s --max-usd 3.2 --configs "$CFGS"

# order 3
modal run modal/phase6/modal_p6.py::preflight --gpu RTXPRO6000 --configs "$CFGS"
modal run --detach modal/phase6/modal_p6.py::bench_RTXPRO6000 --run-id p6m-r7-rtx --max-usd 4.7 --configs "$CFGS"
```

## The TPS model the data feeds

Per row, the stage profile splits a decode step into compute, host-to-GPU copies and SSD reads.
The model is `step_time = t_compute(GPU, b) + cold_reads(host_GiB, b) x 12.75 MiB / B_SSD
+ host_hits(host_GiB, b) x 12.75 MiB / B_h2d(GPU)`, with cold reads and host hits taken from the
next-use simulator (`theory/NEXT_USE_SIM.md`, free) and the three hardware terms measured per GPU.
It is judged by leave-one-GPU-out: fit on four GPUs, predict the fifth, report the error.
