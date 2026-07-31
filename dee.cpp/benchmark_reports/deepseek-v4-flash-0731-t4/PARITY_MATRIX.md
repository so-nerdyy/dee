# PARITY_MATRIX — DeepSeek-V4-Flash-0731 reference agreement

Reference = official `inference/model.py` + pinned revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
executed on supported hardware (Family A). Candidates = Family B (FP4/FP16),
Family C (INT8), Family D (FP16 subset).

## Predeclared gates

| Category | Gate | Tolerance basis |
|---|---|---|
| Embedding output | EXACT | integer gather; must be bitwise |
| Layer norm outputs | BITWISE-OR-TIGHT | predeclared, per-tensor, not invented after results |
| Attention dense outputs | TIGHT | FP8 dequant + FP32 accumulation |
| Index scores / top-k indices | EXACT ROUTE | selected compressed positions must match |
| Routing scores | EXACT ROUTE | expert IDs must match; weights within declared band |
| Expert ID selection | EXACT | any ID flip = FAIL |
| Shared expert output | TIGHT | BF16/FP8 dense |
| Routed expert output | TIGHT | FP4 unpack + FP16/FP32 exec; tolerance predeclared |
| Residual / HC mixing | TIGHT | Hyper-Connections combine in FP32 |
| Final hidden state | TIGHT | bounded, no growth across 43 layers |
| LM head logits | TIGHT | top-1 margin preserved |
| Generated token IDs | EXACT | any token flip = FAIL |
| DSpark proposals | EXACT STRUCTURE | block size, noise token, verification flow |
| Accepted tokens | EXACT | speculative acceptance must equal reference |

## Rules

- No broad global tolerance invented after seeing results.
- Any route/token difference = semantic failure, not drift.
- Error growth across layers/tokens is measured and bounded.
- Quantized families are **never** labeled bitwise exact.
- Each family is validated independently; results are not merged.

## Current status

- DS1: reference semantics audited from official `inference/model.py`.
- DS2: exact per-tensor ledger complete.
- Gates above are predeclared and locked pending DS5 reference traces.
