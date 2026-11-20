# Phase-6 scorecard: p6m-r7-l4-b8-ask

**Verdict (frozen section-1 rows): KILLED**  
cell `1xL4` · b=8 · host 16.0 GiB (TOTAL basis) · measured B_SSD 4.522 GiB/s · commit `7fc878ac50a7` · est. cost $1.019

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 1.064 | 1.619 | [1.234, 2.537] | <0.8638 \| >3.628 | UNTESTABLE | aggregate, last 32 of 159 decode steps; UNTESTABLE: measured B_SSD is 1.94x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.3978 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup over the window (22182/55761) |
| cold_records_per_tok | 131.2 | 156.1 | [124.0, 312.2] | <93.009 \| >390.3 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.3864 | 0.254 | [0.1621, 0.3332] | <0.09723 \| >0.5564 | UNTESTABLE | 1000 x (1.4801760000000002 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 1.94x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **1.94x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 3.00x (expensive reading, P15b/P15c basis) / 10.79x (cheap reading)  

Warnings:
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU
- row 6 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast
- frozen b=8 rows carry the units defect (Addendum 2 item 3): scored as registered against AGGREGATE last-32-step TPS; expect them to read low


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 1.064 | 4.375 | [3.465, 8.117] | <2.425 \| >11.608 | KILLED | aggregate, last 32 of 159 decode steps |
| host_hit_rate | 0.3978 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup over the window (22182/55761) |
| cold_records_per_tok | 131.2 | 60.306 | [60.306, 370.0] | <45.230 \| >462.6 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.5074 | 0.1234 | [0.06651, 0.1558] | <0.03991 \| >0.2602 | KILLED | 1000 x registered rate 0.00053992 $/s / decode tok/s |

b=8 steady-state gate: **OPEN**  
Predicted limiter: `storage`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
