# Phase-6 scorecard: p6m-r7-a10-b1-h16-nu

**Verdict (frozen section-1 rows): KILLED**  
cell `1xA10` · b=1 · host 16.0 GiB (TOTAL basis) · measured B_SSD 1.426 GiB/s · commit `7fc878ac50a7` · est. cost $0.221

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.411 | 1.486 | [1.155, 2.481] | <0.8085 \| >3.549 | UNTESTABLE | decode tokens / decode wall; UNTESTABLE: measured B_SSD is 0.51x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.6462 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup, prompt-scoped (8633/13359) |
| cold_records_per_tok | 98.458 | 156.1 | [15.749, 312.2] | <11.812 \| >390.3 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 1.205 | 0.09767 | [0.0349, 0.1134] | <0.02094 \| >0.1893 | UNTESTABLE | 1000 x (1.7825760000000002 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 0.51x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **0.51x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 9.35x (expensive reading, P15b/P15c basis) / 33.64x (cheap reading)  

Warnings:
- OWN-REFERENCE gate (Addendum 4): exact_16_token_ids/exact_decoded_text vs the sealed T4 answer were waived because this run is bit-identical to p6m-r7-a10-b1-h64-nu at another host budget. Weaker than the exact gate; it does not show one-GPU mode is bug-free.
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.411 | 0.7333 | [0.5588, 1.128] | <0.3911 \| >1.613 | WEAKENED | decode tokens / decode wall |
| host_hit_rate | 0.6462 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup, prompt-scoped (8633/13359) |
| cold_records_per_tok | 98.458 | 185.0 | [81.895, 370.0] | <61.421 \| >462.6 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 1.518 | 0.4418 | [0.2655, 0.5703] | <0.1593 \| >0.9523 | KILLED | 1000 x registered rate 0.00062392 $/s / decode tok/s |

Predicted limiter: `storage`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
