# Phase-4 workload spec (draft for B7)

All prompts are pre-wrapped in the model chat format the runner feeds raw to the
tokenizer: `<｜begin▁of▁sentence｜>{text}<｜Assistant｜>` (fullwidth chars, same as
the sealed GPU-2 prompts).

## NATIVE_PROMPTS_JSON order per arm (one process, N_TOKENS=128)

| idx | name | type | prompt text (inside wrapper) |
|-----|------|------|------------------------------|
| 0 | q0 | factual | Explain how mRNA vaccines work, from injection to immune memory. |
| 1 | q1 | code | Write a Python function that finds the longest common subsequence of two strings, and explain its time and space complexity. |
| 2 | q2 | math/reasoning | A fair coin is flipped until two consecutive heads appear. What is the expected number of flips? Show your reasoning step by step. |
| 3 | q3 | long-form/open | Write the opening paragraph of a hard science fiction novel about a generation ship whose crew discovers the laws of physics are slightly different two light-years from Earth. |
| 4 | r0 | regression | Explain in one sentence why the sky is blue. |
| 5 | r1 | regression | Write a Python function that returns the nth Fibonacci number. |
| 6 | r2 | regression | List three causes of the French Revolution. |
| 7 | rep0 | cross-prompt | (verbatim repeat of q0 — measures warm cross-prompt reuse) |

## Regression gate

r0-r2 carry the sealed Phase-3 truth. Greedy decoding is prefix-consistent:
at N_TOKENS=128 the first 16 generated tokens AND the first 688 journal
records (43 layers x 16 forwards; journal prefix lines 1..688) MUST
byte-match the sealed artifacts:
- r0: tokens sha 4537a0525e77… journal sha 591072115feb…
- r1: tokens sha 15897d13b715… journal sha 0d4f1a12e469…
- r2: tokens sha bb9324463d67… journal sha ee0808ddfc21…
(journal compare = first 688 jsonl lines hashed identically to sealed file,
or whole-file if the sealed 16-token journal is written to a separate file.)

## Per-arm wall budget (est.)

prefill ~70s x8 + decode 128 x ~4.5s x8 cold ≈ ~75-85 min per arm process
(cached arms faster; that's the point). 4 arms ≈ 5-6 h + build/seal ≈ within
one ~9-12h Kaggle session if arms are the only work; otherwise split.

## Cold/warm/steady design

- q0..q3 run in fixed order on a fresh process: q0 is the coldest possible,
  later prompts inherit whatever the caches retain (cross-prompt warmth is
  measured, not controlled away).
- rep0 = isolated cross-prompt reuse probe: identical routing to q0, so its
  hit profile is a pure function of cache state after 4 prompts.
- Steady state = late decode windows of each prompt (tokens 64-128) reported
  separately from cold (first 16) per the event stream.
