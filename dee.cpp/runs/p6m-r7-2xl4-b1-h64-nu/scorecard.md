# Phase-6 scorecard: p6m-r7-2xl4-b1-h64-nu

**Verdict (frozen section-1 rows): PARTIAL**  
cell `2xL4` · b=1 · host 64.0 GiB (TOTAL basis) · measured B_SSD 3.512 GiB/s · commit `7fc878ac50a7` · est. cost $0.321

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.672 | 2.328 | [1.847, 4.312] | <1.293 \| >6.167 | UNTESTABLE | decode tokens / decode wall; UNTESTABLE: measured B_SSD is 0.75x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.7362 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup, prompt-scoped (9833/13356) |
| cold_records_per_tok | 73.396 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 1.329 | 0.1162 | [0.02005, 0.1255] | <0.01203 \| >0.2096 | UNTESTABLE | 1000 x (3.2160960000000003 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 0.75x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **0.75x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 10.31x (expensive reading, P15b/P15c basis) / 37.12x (cheap reading)  

Warnings:
- OWN-REFERENCE gate (Addendum 4): exact_16_token_ids/exact_decoded_text vs the sealed T4 answer were waived because this run is bit-identical to p6m-r7-2xl4-b1-h16-nu at another host budget. Weaker than the exact gate; it does not show one-GPU mode is bug-free.
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU
- row 2 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.672 | 1.865 | [1.460, 3.272] | <1.022 \| >4.680 | KILLED | decode tokens / decode wall |
| host_hit_rate | 0.7362 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup, prompt-scoped (9833/13356) |
| cold_records_per_tok | 73.396 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 1.134 | 0.1181 | [0.02852, 0.1337] | <0.01711 \| >0.2232 | KILLED | 1000 x registered rate 0.00076192 $/s / decode tok/s |

Predicted limiter: `compute`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
