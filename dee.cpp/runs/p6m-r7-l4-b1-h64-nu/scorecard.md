# Phase-6 scorecard: p6m-r7-l4-b1-h64-nu

**Verdict (frozen section-1 rows): PARTIAL**  
cell `1xL4` · b=1 · host 64.0 GiB (TOTAL basis) · measured B_SSD 4.522 GiB/s · commit `7fc878ac50a7` · est. cost $0.345

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.732 | 1.351 | [1.047, 2.219] | <0.9 \| >2.100 | UNTESTABLE | decode tokens / decode wall; UNTESTABLE: measured B_SSD is 1.94x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.7362 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup, prompt-scoped (9833/13356) |
| cold_records_per_tok | 73.396 | 147.8 | [2.619e-09, 295.5] | <1.964e-09 \| >369.4 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 0.7073 | 0.0766 | [0.01853, 0.08441] | <0.01112 \| >0.141 | UNTESTABLE | 1000 x (1.8637920000000001 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 1.94x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **1.94x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 5.49x (expensive reading, P15b/P15c basis) / 19.75x (cheap reading)  

Warnings:
- OWN-REFERENCE gate (Addendum 4): exact_16_token_ids/exact_decoded_text vs the sealed T4 answer were waived because this run is bit-identical to p6m-r7-l4-b1-h16-nu at another host budget. Weaker than the exact gate; it does not show one-GPU mode is bug-free.
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU
- row 3 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.732 | 1.968 | [1.543, 3.540] | <1.080 \| >5.063 | KILLED | decode tokens / decode wall |
| host_hit_rate | 0.7362 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup, prompt-scoped (9833/13356) |
| cold_records_per_tok | 73.396 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 0.7376 | 0.1091 | [0.03687, 0.129] | <0.02212 \| >0.2154 | KILLED | 1000 x registered rate 0.00053992 $/s / decode tok/s |

Predicted limiter: `h2d`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
