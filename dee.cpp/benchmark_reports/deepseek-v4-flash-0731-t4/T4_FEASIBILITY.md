# T4_FEASIBILITY — DeepSeek-V4-Flash-0731 on Tesla T4

Hardware: Tesla T4, SM75, 16 GB VRAM/GPU, PCIe, no native FP8/FP4 tensor
cores, no NVLink assumption. Initial platform: Kaggle dual T4.

## Memory arithmetic (from exact ledger, not estimates)

### Persistently resident dense state (never experts)

| Item | Compressed bytes |
|---|---|
| Embedding | 1.059 GB |
| LM head | 1.059 GB |
| Attention dense + norms | 4.60 GB |
| Shared experts | 1.08 GB |
| Router/gates | 0.11 GB |
| Hash/compress (incl. indexer) | 0.94 GB |
| **Persistent dense total** | **≈ 8.84 GB compressed** |

Persistent state alone is ≈8.8 GB — inside 16 GB but already heavy before a
single expert, KV cache, activations, or staging is counted. DSpark (10.86 GB
compressed) is only required for speculative decode, not base decode.

### Expert working set (per token per layer)

| Metric | FP16 execution (incl. scales) | INT8 execution (incl. scales) |
|---|---|---|
| One expert (w1+w2+w3 + scales) | 49.5 MiB | 24.75 MiB |
| Top-6 active | 297.0 MiB | 148.5 MiB |
| Top-6 across all 43 layers (upper bound) | 12.8 GiB | 6.4 GiB |

The top-6-per-layer bound assumes a *different* set of 6 experts at every
layer simultaneously — impossible for real routes but a valid worst case.
Dynamic Expert Eviction is required to bound this to the union of hot experts
actually observed (DS8 will measure real routing unions from DS5 traces).

### One-T4 plan (must be bounded)

Required under 16 GB (plus KV/activations/staging):

- dense persistent ≈ 8.8 GB (potentially shrinkable with FP8 dense execution
  later; the FP8 dense is already 1 byte/element in checkpoint);
- bounded expert cache: e.g. 48 FP16 experts ≈ 2.4 GB, or 96 INT8 ≈ 2.4 GB;
- KV cache: compress ratios make this a fraction of 1M context at decode;
  window 128 + compressed positions;
- pinned staging + mapped checkpoint via host, not duplicated in VRAM.

### Dual-T4 plan

Contiguous layer split (e.g., 21/22 layers) keeps persistent dense to
≈4.4 GB/GPU, leaving large headroom for expert caches and KV.

## Throughput reality

- No NVLink: hidden-state handoffs across GPUs must be host-staged or rely on
  PCIe P2P if available (Kaggle T4 pairs do not guarantee P2P).
- FP4→FP16 unpack is a real per-token cost on SM75; it must be fused into the
  GEMV to avoid a full extra pass over 147 GB of experts.
- First goal is **genuine output**, not 20 TPS. No TPS forecast before DS7
  produces a byte + timing ledger.

## Known SM75 constraints

1. No FP8 arithmetic support — all FP8 dense must dequantize to FP16/FP32.
2. No FP4 support — experts must unpack to FP16 or convert to INT8.
3. No NVLink — cross-GPU transfers are PCIe.
4. 16 GB is the hard envelope; the ledger shows full-FP16 is impossible
   (626.88 GB), so eviction is not optional.

## Verdict

DS7 (one expert on T4) is the first decision gate: it measures unpack + GEMV
time, achievable bandwidth, and temporary VRAM. Nothing beyond the ledger is
assumed until DS7 runs.
