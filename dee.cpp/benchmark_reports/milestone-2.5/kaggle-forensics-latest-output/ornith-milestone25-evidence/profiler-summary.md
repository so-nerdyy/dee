# Milestone 2.5 profiler summary

All values below are measurements unless explicitly labeled as an inference. No optimization was performed.

## Representative warm decode token

Run `dual-warm-profiled`, decode step 2: 2249.124 ms wall time across 40 layers.

| Additive layer category | Wall ms | Share of token |
|---|---:|---:|
| expert_native_wall_ms | 2009.461 | 89.34% |
| residual_and_unattributed_wall_ms | 96.660 | 4.30% |
| router_wall_ms | 30.220 | 1.34% |
| normalization_wall_ms | 25.843 | 1.15% |
| expert_output_combination_wall_ms | 20.310 | 0.90% |
| attention_wall_ms | 17.254 | 0.77% |
| shared_expert_wall_ms | 11.254 | 0.50% |
| expert_output_h2d_wall_ms | 8.081 | 0.36% |
| shared_expert_gate_wall_ms | 7.859 | 0.35% |
| inter_device_transfer_wall_ms | 4.790 | 0.21% |
| expert_input_d2h_wall_ms | 4.746 | 0.21% |

Native CUDA/synchronization subphases below are inclusive within the expert wall category and are not additive to the table above.

| Native subphase | Measured ms |
|---|---:|
| pinning_ms | 1225.823 |
| pageable_to_pinned_copy_ms | 267.178 |
| h2d_completion_cuda_ms | 126.493 |
| expert_compute_cuda_ms | 29.329 |
| h2d_submission_ms | 11.309 |
| expert_output_d2h_cuda_ms | 5.954 |
| activation_h2d_cuda_ms | 4.335 |
| activation_conversion_cuda_ms | 2.955 |
| synchronization_ms | 2.781 |
| host_tensor_preparation_ms | 0.760 |
| expert_lookup_ms | 0.043 |

## GPU concurrency and idle evidence

Comparable host-monotonic layer intervals overlapped for 0.000 ms (0.00% of model wall).
NVML sampled both GPUs active in 1 of 18 representative-step samples.

Inference: the current 20/20 placement is a sequential pipeline for a single sequence, not concurrent model-parallel execution. Each native expert output is synchronized and returned to the host before the next layer proceeds.

## Cache and transfer evidence

The combined trace contains 8,011 cache requests and 276 measured expert-weight H2D copies totaling 1,736,441,856 B (1.617 GiB).
Measured expert-weight bytes per known miss: 257,365 B (0.000 GiB).

## Profiler overhead

Custom tracing later-decode overhead versus the unprofiled control: 6.02%.
The focused PyTorch-profiler decode-step overhead versus the control first decode: 277.74%.

## Earlier throughput comparison

| Workload | Reported throughput | Meaning |
|---|---:|---|
| Earlier isolated synthetic harness | 1.046735 | recurrent synthetic MoE steps/s |
| Full Ornith warm decode | 0.240229 | generated tokens/s through all 40 layers |

The values are not contradictory: the isolated harness excluded the dense model and used a much smaller synthetic expert payload. See `prior-30-tps-audit.md`.
