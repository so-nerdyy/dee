# Phase-6 scorecard: p6m-r7-a10-b8-h16-nu

**Verdict (frozen section-1 rows): KILLED**  
cell `1xA10` · b=8 · host 16.0 GiB (TOTAL basis) · measured B_SSD 1.426 GiB/s · commit `7fc878ac50a7` · est. cost $0.367

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 2.470 | 1.854 | [1.417, 2.954] | <0.9921 \| >4.224 | UNTESTABLE | aggregate, last 32 of 64 decode steps; UNTESTABLE: measured B_SSD is 0.51x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.433 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup over the window (21052/48617) |
| cold_records_per_tok | 107.7 | 156.1 | [124.0, 312.2] | <93.009 \| >390.3 | WEAKENED | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.2005 | 0.267 | [0.1676, 0.3494] | <0.1006 \| >0.5835 | UNTESTABLE | 1000 x (1.7825760000000002 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 0.51x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **0.51x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 1.55x (expensive reading, P15b/P15c basis) / 5.60x (cheap reading)  

Warnings:
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU
- row 3 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast
- frozen b=8 rows carry the units defect (Addendum 2 item 3): scored as registered against AGGREGATE last-32-step TPS; expect them to read low


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 2.470 | 2.068 | [1.590, 3.299] | <1.113 \| >4.717 | PASS | aggregate, last 32 of 64 decode steps |
| host_hit_rate | 0.433 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup over the window (21052/48617) |
| cold_records_per_tok | 107.7 | 60.306 | [60.306, 370.0] | <45.230 \| >462.6 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.2526 | 0.3017 | [0.1891, 0.3923] | <0.1135 \| >0.6551 | PASS | 1000 x registered rate 0.00062392 $/s / decode tok/s |

b=8 steady-state gate: **OPEN**  
Predicted limiter: `storage`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
