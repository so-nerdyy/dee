# Phase-6 scorecard: p6m-r7-2xl4-b1-h16-nu

**Verdict (frozen section-1 rows): KILLED**  
cell `2xL4` · b=1 · host 16.0 GiB (TOTAL basis) · measured B_SSD 3.512 GiB/s · commit `7fc878ac50a7` · est. cost $0.246

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.703 | 2.328 | [1.847, 4.312] | <1.293 \| >6.167 | UNTESTABLE | decode tokens / decode wall; UNTESTABLE: measured B_SSD is 0.75x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.6433 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup, prompt-scoped (8592/13356) |
| cold_records_per_tok | 99.250 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 1.119 | 0.1023 | [0.01766, 0.1105] | <0.0106 \| >0.1846 | UNTESTABLE | 1000 x (2.8324800000000003 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 0.75x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **0.75x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 8.68x (expensive reading, P15b/P15c basis) / 31.25x (cheap reading)  

Warnings:
- OWN-REFERENCE gate (Addendum 4): exact_16_token_ids/exact_decoded_text vs the sealed T4 answer were waived because this run is bit-identical to p6m-r7-2xl4-b1-h64-nu at another host budget. Weaker than the exact gate; it does not show one-GPU mode is bug-free.
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.703 | 1.624 | [1.265, 2.765] | <0.8852 \| >3.953 | KILLED | decode tokens / decode wall |
| host_hit_rate | 0.6433 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup, prompt-scoped (8592/13356) |
| cold_records_per_tok | 99.250 | 179.4 | [74.108, 358.8] | <55.581 \| >448.5 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 1.084 | 0.2599 | [0.1286, 0.3228] | <0.07718 \| >0.5391 | KILLED | 1000 x registered rate 0.00076192 $/s / decode tok/s |

Predicted limiter: `storage`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
