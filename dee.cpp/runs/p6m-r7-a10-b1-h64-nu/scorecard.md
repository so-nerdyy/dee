# Phase-6 scorecard: p6m-r7-a10-b1-h64-nu

**Verdict (frozen section-1 rows): PARTIAL**  
cell `1xA10` · b=1 · host 64.0 GiB (TOTAL basis) · measured B_SSD 1.426 GiB/s · commit `7fc878ac50a7` · est. cost $0.256

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.756 | 1.546 | [1.204, 2.602] | <0.8428 \| >3.720 | UNTESTABLE | decode tokens / decode wall; UNTESTABLE: measured B_SSD is 0.51x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.7337 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup, prompt-scoped (9802/13359) |
| cold_records_per_tok | 74.104 | 147.8 | [2.619e-09, 295.5] | <1.964e-09 \| >369.4 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 0.7959 | 0.08877 | [0.02129, 0.09786] | <0.01277 \| >0.1634 | UNTESTABLE | 1000 x (2.166192 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 0.51x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **0.51x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 6.17x (expensive reading, P15b/P15c basis) / 22.23x (cheap reading)  

Warnings:
- OWN-REFERENCE gate (Addendum 4): exact_16_token_ids/exact_decoded_text vs the sealed T4 answer were waived because this run is bit-identical to p6m-r7-a10-b1-h16-nu at another host budget. Weaker than the exact gate; it does not show one-GPU mode is bug-free.
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU
- row 2 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.756 | 0.8825 | [0.6747, 1.382] | <0.4723 \| >1.976 | PASS | decode tokens / decode wall |
| host_hit_rate | 0.7337 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup, prompt-scoped (9802/13359) |
| cold_records_per_tok | 74.104 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 0.8253 | 0.1258 | [0.04235, 0.1488] | <0.02541 \| >0.2485 | KILLED | 1000 x registered rate 0.00062392 $/s / decode tok/s |

Predicted limiter: `h2d`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
