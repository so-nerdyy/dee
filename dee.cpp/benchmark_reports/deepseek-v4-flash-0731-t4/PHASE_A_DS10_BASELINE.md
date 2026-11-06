# Phase A — Cache Characterization Report

## Run 20260806T035518Z (stage=final)
- verdict: ACCEPT_DUAL_T4_DECODE
- expert accesses: 5676
- overall cache hit rate: 0.0
- popularity: top1=0.01268 top10=0.18851 top20=0.34302 gini=0.24338 (distinct=256)
- consecutive-token reuse: 0.36241 (p90 reuse distance 5.0)
- Belady (oracle) ceiling: 1GiB→10.42%, 2GiB→20.9%, 4GiB→36.87%, 8GiB→51.24%
- decode: 0 tokens, median None ms/tok, p95 None ms, max None ms, None tok/s
- HTTP: 1609 requests, 8.85 GB

## Bottleneck ranking (by decode wall time)

| run | decode_s | tok/s | hit_rate | http_GB | estimate |
|---|---|---|---|---|---|
