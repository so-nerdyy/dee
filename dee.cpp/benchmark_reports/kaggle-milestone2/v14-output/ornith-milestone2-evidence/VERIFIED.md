# Ornith Milestone 2 verified proof

Kaggle kernel `nivind/dee-cpp-ornith-milestone-2-dual-t4-proof`, version 14,
finished with Kaggle status `complete` and report result `PASS`.

- Implementation under test: `7cdc1571def629112cad9158f718640747c2b265`
- Notebook pin commit: `c3d2cdcdcb22e8519c4c6f706a60bab06cc6abe8`
- Branch: `opt/real-model-t1`
- Machine: 2 x Tesla T4 (15,636,037,632 bytes each), PyTorch 2.10.0+cu128,
  Transformers 5.14.1, CUDA runtime 12.8
- Native/CUDA/support tests: 11/11 passed
- Checkpoint inventory: 31,666 tensors across 16 shards; 70,214,363,872
  validated tensor bytes exactly matched the declared total
- Layout: layers 0-19 and embedding on cuda:0; layers 20-39, final norm, and
  LM head on cuda:1; eight FP16 cached experts per layer
- Permanent genuine layer-0 regression: PASS; worst expert max absolute error
  8.34465027e-7 and worst hidden-state max absolute error 3.27825546e-7
- Focused genuine router parity: PASS; logits and routing weights were bit-exact
  for layers 0, 3, 20, and 39. One exact-value layer-39 tie used the guarded
  Transformers ordering compatibility path.

## Full-model parity

All three prompts executed all 40 layers. Candidate and Transformers token IDs,
decoded text, embeddings, router logits, routing weights, ordered expert IDs,
selected expert outputs, every layer hidden state, final hidden state, and LM
head logits were exact (maximum absolute error 0 across the full trace).

| Prompt | Prompt IDs | Generated IDs | Decoded output |
| --- | --- | --- | --- |
| `Hello` | `[9419]` | `[11, 271, 40, 1044]` | `,\n\nI am` |
| `2+2=` | `[17, 10, 17, 28]` | `[19, 271, 248068, 198]` | `4\n\n<think>\n` |
| `Paris` | `[57590]` | `[11, 279, 4170, 314]` | `, the City of` |

## Measured benchmark

The benchmark prompt was `Hello` (one prompt token, four generated tokens).
Prefill throughput and decode throughput are reported separately.

| Measurement | Cold | Warm |
| --- | ---: | ---: |
| Prefill / time to first token | 655.203258 s | 2.370449 s |
| Prefill tokens/s | 0.001526 | 0.421861 |
| Per-token decode seconds | 484.142663, 344.034877, 279.381796 | 2.139442, 2.148716, 2.154179 |
| Single-stream decode tokens/s | 0.002709 | 0.465670 |
| Total generation time | 1762.783760 s | 8.818389 s |
| Phase peak host RSS | 13,449,666,560 B | 13,454,475,264 B |
| Phase peak VRAM, each GPU | 4,006,477,824 B | 4,006,477,824 B |
| Cache hits / misses / hit rate | 226 / 1054 / 17.65625% | 272 / 1008 / 21.25% |
| Host-to-device expert bytes | 6,631,194,624 B | 6,341,787,648 B |

Whole-process peak RSS was 38,818,201,600 bytes. Whole-run sampled peak VRAM
was 5,015,207,936 bytes on each GPU.

## Evidence integrity

The following compact evidence files are committed beside this note:

- `ornith-milestone2-report.json`: `eaf29c0b13cc84b508d3d2727e2374dd91a568f31afad0e390b11045455a8609`
- `ornith-router-parity.json`: `0b791f1787278fc1ea99b33e64d1473fdc5c7550740ea7eb845b85d4c85dd10d`
- `ornith-layer0-regression.json`: `aa43df402b8f420bb9675ce507a6eb216264f8b71e92e1289f44cc30137d41f0`
- `full-run.log`: `8d2a7a7666bc242e5d2f15c2cdf55adef6d3b3347a8e68dd05d58232ba2df300`
- `ornith-milestone2-summary.txt`: `91523611a6e167e840e7b788b63d32b7aa3973052adf4058e5a75da62e601fce`
- `evidence-manifest.json`: `43d7da3241d5fba6c40e7d18f7fdb72d363389ce8505feba9f6a884dab48ed14`

Large generated artifacts were deliberately not committed. The downloaded
local tensor map is 12,980,884 bytes with SHA-256
`e5efae2ccf213ba79ea1155122a1e328c2ec06c1a423e2ce132b6abae571b528`.
The Kaggle evidence archive has SHA-256
`9bffad77a06fcfa7b2c43a4951f0b3f7bd3389105e7d1f19595df528273e64a4`.
The downloaded raw kernel log has SHA-256
`0258373bc298ecef70e6dad9a91311e8c994b798171f98e3e3a7d870495e70be`.
