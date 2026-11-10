# Phase-6 scorecard: p6m-l4-r5-b1-h16

**Verdict (frozen section-1 rows): PARTIAL**  
cell `1xL4` · b=1 · host 16.0 GiB (TOTAL basis) · measured B_SSD 3.955 GiB/s · commit `311d9d0bf821` · est. cost $0.128

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.654 | 1.295 | [1.002, 2.114] | <0.7017 \| >3.023 | UNTESTABLE | decode tokens / decode wall; UNTESTABLE: measured B_SSD is 1.70x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.5578 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | PASS | pooled_dedup, prompt-scoped (2845/5100) |
| cold_records_per_tok | 140.9 | 156.1 | [15.749, 312.2] | <11.812 \| >390.3 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 0.6287 | 0.08536 | [0.03203, 0.09976] | <0.01922 \| >0.1666 | UNTESTABLE | 1000 x (1.4801760000000002 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 1.70x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **1.70x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 4.88x (expensive reading, P15b/P15c basis) / 17.56x (cheap reading)  

Warnings:
- OWN-REFERENCE gate (Addendum 4): exact_16_token_ids/exact_decoded_text vs the sealed T4 answer were waived because this run is bit-identical to p6m-l4-r5-b1-h64 at another host budget. Weaker than the exact gate; it does not show one-GPU mode is bug-free.
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.654 | 1.581 | [1.228, 2.700] | <0.8597 \| >3.861 | KILLED | decode tokens / decode wall |
| host_hit_rate | 0.5578 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | PASS | pooled_dedup, prompt-scoped (2845/5100) |
| cold_records_per_tok | 140.9 | 185.0 | [81.895, 370.0] | <61.421 \| >462.6 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 0.8256 | 0.2077 | [0.1065, 0.2605] | <0.06389 \| >0.435 | KILLED | 1000 x registered rate 0.00053992 $/s / decode tok/s |

Predicted limiter: `storage`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
