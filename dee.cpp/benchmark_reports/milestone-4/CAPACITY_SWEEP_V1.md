# Milestone 4 controlled capacity sweep v1

Status: sealed initial capacity proof. This run predates the generation-aware
transfer ledger added in `b343d9d`, so it is not the final transfer
classification artifact.

## Identity and integrity

- Kaggle kernel: `nivind/dee-cpp-ornith-milestone-4-ledger-sweep`
- URL: <https://www.kaggle.com/code/nivind/dee-cpp-ornith-milestone-4-ledger-sweep>
- Run ID: `20260727T024309Z-capacity-sweep`
- Tested commit: `726fbfce22d676e9c86e1af1e1cb197d21ea8612`
- Accelerator: two dedicated Nvidia Tesla T4 GPUs
- Archive: `ornith-milestone4-capacity-sweep.tar.gz`
- Archive SHA256: `4F9C7743FDD5536CF119FBAB5EA0C23E229E17BB5A998AEBBF50D5E70A5C1322`
- Artifact manifest SHA256: `B858EED79010AED4684F84424627EC6C4E0543D9FA97A6308D103FF55A37D8D6`
- Capacity summary SHA256: `8CD08E939F184F4914843CC489715E6F39422D0EEA91A8BE94243973311CB5E7`
- Manifest verification: 84 artifacts and 71 required paths; every recorded
  byte count and SHA256 matched.
- Preserved local evidence:
  `tmp/m4_capacity_sweep_v1_output/ornith-milestone4-evidence-20260727T024309Z-capacity-sweep`

## Controlled result

All rows used Ornith-1.0-35B, prompt `Hello`, token IDs `[9419]`, expected
generated IDs `[11, 271, 40, 1044]`, top-k 8, BF16 transfer, FP16 cache, a
20/20 layer split, and one warmup generation. All rows passed all 40 layers,
exact tokens, 100% device path, and zero host fallback.

| Experts/layer | Control TPS | vs v17 2.8139 | Control TTFT | Profiled model span | Measured hit rate | Expert H2D | Loads/token | Unique loads/token | Repeat loads/token | Evictions/token | Peak process VRAM/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3.490 | +24.03% | 540.70 ms | 352.07 ms | 21.25% | 6,341,787,648 B | 252.00 | 216.00 | 36.00 | 252.00 | 4,033,740,800 B |
| 16 | 4.967 | +76.52% | 395.86 ms | 261.07 ms | 65.23% | 2,799,697,920 B | 111.25 | 98.25 | 13.00 | 111.25 | 5,040,373,760 B |
| 32 | 6.644 | +136.12% | 343.64 ms | 209.16 ms | 100.00% | 0 B | 0.00 | 0.00 | 0.00 | 0.00 | 7,053,639,680 B |

`Profiled model span` is the representative non-torch-profiler decode span,
not the sum of all layer wall records. Transfer counts and bytes come from the
profiled row; TPS and TTFT come from its paired unprofiled control.

Cap 32 retained all 875 layer/expert/GPU keys encountered during warmup and
then served all 1,280 measured expert requests without an expert H2D copy or
eviction. The 7.05 GB peak process allocation per GPU leaves substantial room
on a 16 GB T4, so actual T4 VRAM, scratch buffers, duplicated dense state,
conservative reservation, and fragmentation did not prevent this point.

## Important limitations

- The cap-8 control process spent 1,665 seconds in its outer driver because it
  was the first cold model setup. Decode TPS is based only on the three
  measured decode spans, but reproducibility still requires another warm
  control.
- Control-mode native counters include warmup plus measurement, while profiled
  counters are reset at the measurement boundary. Traffic comparisons in this
  report therefore use profiled counters only.
- This version corrected layer-scoped transfer IDs but does not contain
  residency generations, consumption state, before/after cache metadata, or
  the nine-category ledger. A follow-up build is required before attributing
  avoidable traffic.
- Profiler overhead reduces profiled TPS substantially. Performance conclusions
  use paired unprofiled controls; profiled rows provide causal counters and
  timing only.
