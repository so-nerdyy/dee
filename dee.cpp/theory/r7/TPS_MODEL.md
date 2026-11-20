# r7 tokens/sec model (fitted 2026-11-20, after the data)

This was fitted after r7 ran, so it is not a pre-registration. It describes what r7 measured. The next
paid run can test it as a prediction. Source: `theory/r7/tps_fit.py` (stdlib only), output in
`theory/r7/tps_fit.json`, inputs from `runs/p6m-r7-*/scorecard.json`.

## Model

At batch 8, one decode step (8 tokens) takes

    T_step = t_fixed(GPU) + cold_reads_per_step x c_read(container)

`t_fixed` covers compute, host-to-GPU copies and overhead. `c_read` is the effective cost of one
12.75 MiB expert record that missed both caches. Each GPU's h16/h64 pair gives two equations, which
are solved for the two terms:

| GPU | t_fixed (s/step) | c_read (ms/record) | effective read GiB/s | fio B_SSD GiB/s |
|---|---|---|---|---|
| L4 | 3.41 | 3.73 | 3.3 | 4.52 |
| A10 | 2.21 | 1.19 | 10.4 | 1.43 |
| L40S | 1.65 | 1.81 | 6.9 | 89 (page cache, flagged) |
| RTX PRO 6000 | 1.14 | 0.59 | 21.0 | 149 (page cache, flagged) |
| 2xL4 | 3.96 | 1.78 | 7.0 | 3.51 |

## What it says

1. **The fixed cost follows GPU memory bandwidth.** `t_fixed ≈ 114 x BW^-0.62` s, with BW in GB/s
   from the datasheets (L4 300, A10G 600, L40S 864, RTX PRO 6000 1792). The ordering matches all
   four single-GPU cards.
2. **The read cost is set by the container, not the disk.** A10 had the slowest fio disk (1.43 GiB/s)
   but read records about 3x faster than L4. The store is 146.6 GiB and the containers have 96-192
   GiB RAM. Under gVisor, part of the "SSD" tier is actually served from the host page cache, and fio
   with O_DIRECT doesn't see that. `c_read` has to be measured per environment. It can't be read off
   a spec sheet.
3. **The two-GPU split buys nothing.** 2xL4 has a higher fixed cost than 1xL4 and gives
   byte-identical tokens.

## Leave-one-GPU-out check

Each single-GPU card is predicted from the other three: `t_fixed` from the bandwidth power law and
`c_read` from the others' median.

| GPU | host | measured | predicted | error |
|---|---|---|---|---|
| L4 | 16 GiB | 1.20 | 1.88 | +56% |
| L4 | 64 GiB | 1.87 | 2.29 | +23% |
| A10 | 16 GiB | 2.47 | 2.14 | -13% |
| A10 | 64 GiB | 3.23 | 3.10 | -4% |
| L40S | 16 GiB | 2.48 | 2.83 | +14% |
| L40S | 64 GiB | 3.87 | 3.88 | 0% |
| RTX PRO 6000 | 16 GiB | 4.86 | 3.13 | -36% |
| RTX PRO 6000 | 64 GiB | 6.27 | 5.62 | -10% |

Median absolute error is 14% and the worst is 56%. The h64 rows, which read little from storage, come
in within 23%. The h16 rows carry the error, because they depend on `c_read`, which varies 6x
between containers. A useful estimate for a new GPU therefore needs one cheap measurement of
`c_read` in the target environment. GPU specs alone aren't enough.

## Limits

- Four single-GPU points fit two power-law parameters, so this is a description of r7, not a tested
  law.
- b=1 is not fitted. The h64 row ran after h16 in the same container, read warmer page cache, and
  came out slower on 4 of 5 GPUs, so its pair gives a negative read cost. b=1 speeds are reported as
  measured: 0.41-2.22 tok/s.
- Page-cache warmth depends on row order (gVisor ignores DONTNEED), so `c_read` mixes disk and RAM
  reads.

## Raw evidence

Only scorecards, small summaries, and `runs/r7_environment.json` are in git. The full per-row
`result-c0.json`, journals, and logs (about 3.5 GB) are on the Modal volume `dee-p6-evidence`
under `runs/p6m-r7-*`. Fetch a row with
`modal volume get dee-p6-evidence runs/p6m-r7-<gpu>-<tag> runs/` and re-score it with
`aws/phase6/score.py`.
