# Ornith full-token T4 critical path and roofline

Status: reproducible analysis complete. Values are labelled as measured, calculated, or inferred; no static byte floor is represented as a hardware counter.

## Bottom line

- Accepted cap-32 unprofiled throughput: **6.487701 TPS**.
- Peak process VRAM: cuda:0 6.554 GiB, cuda:1 6.554 GiB.
- Calculated mandatory active-weight floor: **5,892,863,232 B/token** (5.488 GiB/token).
- Effective bandwidth at the measured rate: **38.231 GB/s**, or **14.16%** of the 270 GB/s sustained assumption.
- The trace launches **6,166 kernels** and its combined GPU kernel/memcpy timeline is busy only **18.80%**. This is a launch/GEMV-fragmentation problem, not an expert-H2D problem.
- Verdict: **20-30 TPS is physically plausible but not demonstrated.** The 270 GB/s single-sequence static floor is 45.82 TPS; actual DRAM counters and substantially lower launch latency are still required.

## Real tensor-map byte floor

| Device | Active static B/token | Routed top-8 B/token | Total B/token |
|---|---:|---:|---:|
| cuda:0 | 1,431,239,296 | 1,006,632,960 | 2,437,872,256 |
| cuda:1 | 2,448,358,016 | 1,006,632,960 | 3,454,990,976 |

Only one token-embedding row is counted; the full LM head is counted. The routed term is eight complete experts per layer across all 40 layers. The floor excludes activation traffic, allocator effects, cache-line effects, and implementation rereads.

## Measured latest decode-step critical path

The selected layer-timer step covers 40 layers and 202.612 ms. The profiler perturbs wall time, so use this ranking, not its TPS, as the baseline.

| Rank | Component | ms | Layer-wall share | Evidence |
|---:|---|---:|---:|---|
| 1 | linear_attention_inferred | 47.074 | 23.23% | inferred_differential |
| 2 | remaining_residual_and_unattributed | 32.536 | 16.06% | derived_residual |
| 3 | router | 25.086 | 12.38% | measured_layer_timer |
| 4 | expert_compute | 23.562 | 11.63% | measured_layer_timer |
| 5 | normalization | 21.159 | 10.44% | measured_layer_timer |
| 6 | expert_output_combination | 15.728 | 7.76% | measured_layer_timer |
| 7 | full_attention | 14.288 | 7.05% | measured_layer_timer |
| 8 | shared_expert | 9.000 | 4.44% | measured_layer_timer |
| 9 | shared_expert_gate | 4.934 | 2.44% | measured_layer_timer |
| 10 | inter_device_transfer | 4.506 | 2.22% | measured_layer_timer |
| 11 | explicit_synchronization | 0.199 | 0.10% | measured_layer_timer |
| 12 | expert_lookup | 0.018 | 0.01% | measured_layer_timer |

The linear-attention row is a labelled differential estimate because the current timer records linear attention inside residual/unattributed time. It subtracts the mean full-attention residual from each linear-attention layer; it is not a direct CUDA counter.

## CUDA trace

- Timeline: 330.222 ms span, 62.088 ms busy, 268.134 ms idle.
- Runtime calls: cudaLaunchKernel=6,166, cudaStreamSynchronize=409, cudaDeviceSynchronize=1, cudaStreamWaitEvent=132, cudaEventRecord=2,640, cudaEventRecordWithFlags=558.
- Expert H2D recorded by the layer timer: 0 B (the cap-32 experts were resident).
- Positive GPU gaps: 7,486; 5,244 exceed 10 us; median 22.752 us.

| Kernel family | Calls | CUDA duration ms |
|---|---:|---:|
| cublas_gemvx | 901 | 27.674 |
| elementwise | 3,084 | 13.458 |
| cublas_gemv2t | 370 | 6.247 |
| aten_direct_copy | 649 | 3.883 |
| reduction | 441 | 3.783 |
| dee_swiglu_activation | 320 | 1.027 |
| index_gather_topk | 41 | 0.646 |
| cublas_dot | 100 | 0.564 |
| aten_cat_copy | 91 | 0.509 |
| aten_bitonic_sort | 40 | 0.466 |

## Roofline and target bandwidth

| Assumption | Bandwidth GB/s | Sequential-device TPS | Ideal pipelined TPS |
|---|---:|---:|---:|
| realistic_sustained | 270.0 | 45.82 | 78.15 |
| theoretical_peak | 320.0 | 54.30 | 92.62 |

| Target | Required effective GB/s | Share of 270 GB/s |
|---:|---:|---:|
| 20 TPS | 117.857 | 43.65% |
| 30 TPS | 176.786 | 65.48% |

FP16 and BF16 have the same two-byte storage floor. Ideal INT8/FP8 and INT4 storage floors are included in JSON only as qualified arithmetic, not throughput promises: T4 has no native FP8 tensor-core path, and scales/dequantization are not free. No MTP/speculative multiplier is claimed without measured acceptance and verification cost.

## Historical approximately-30 result

The historical 30.571 figure is a single-T4 rate in **synthetic recurrent MoE steps/s**, at commit `75d5218`, with one physical expert layer replayed as 40 logical calls, intermediate width 64, generated weights, and no attention, recurrent state, vocabulary head, or dual-GPU path. It has no valid conversion to full-token Ornith TPS.

## Next optimization gate

The trace and layer timing nominate the linear-attention/recurrent path and Python/PyTorch launch fragmentation as the first target. A faithful next Pareto point must preserve exact token IDs/text, all 40 routed layers, zero host fallback, no trace abort, and <=8 GiB process VRAM per GPU. Before claiming a bandwidth-bound result, collect Nsight Compute `dram__bytes_read.sum` and achieved-DRAM-throughput counters on a steady decode token.

## Source integrity

- `tensor_map`: `791f1669879067f0aacd85e4cef608ae3ac8c749aec2adb51afbee0c257a928c` (12,980,885 bytes), `tmp/m3_v6_output/ornith-milestone3-evidence-20260727T003223Z-4ec6d17d/runs/dual-cold-primary/tensor-map.json`
- `control_report`: `324798b6616bf994109a2467c77b0dd296fe1daa6dcc485b2a8be1dee3e2de6a` (229,192 bytes), `tmp/m4_ledger_seal_redo/ornith-milestone4-evidence-20260728T183337Z-phase1-redo/runs/capacity-32-control/run-report.json`
- `profiled_report`: `0e5e23074d69de7f1bb39ee57af3282f56c9dfb673c16ae1ac4b4d7e104e4ed1` (327,522 bytes), `tmp/m4_ledger_seal_redo/ornith-milestone4-evidence-20260728T183337Z-phase1-redo/runs/capacity-32-profiled/run-report.json`
- `layer_timing`: `824a5609f8db53a2795409d54c95b70b924332e1be487e04bf9004cc6af53a76` (223,831 bytes), `tmp/m4_ledger_seal_redo/ornith-milestone4-evidence-20260728T183337Z-phase1-redo/runs/capacity-32-profiled/layer-timing.json`
- `torch_trace`: `a1ee56c82b17691b65492cd2d5d78882ae74de18aec8107e1f802ed17e884f1a` (1,298,797 bytes), `tmp/m4_ledger_seal_redo/ornith-milestone4-evidence-20260728T183337Z-phase1-redo/runs/capacity-32-profiled/torch-profiler-trace.json.gz`
- `historical_audit`: `0a045f5b7b0683561478d085bc3e578de9e017ed088a7da43761f7e75f17ef19` (21,462 bytes), `dee.cpp/benchmark_reports/milestone-2.5/work/prior-30-tps-audit.md`
