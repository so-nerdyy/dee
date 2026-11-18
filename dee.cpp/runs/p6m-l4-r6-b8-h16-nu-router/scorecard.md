# Phase-6 scorecard: p6m-l4-r6-b8-h16-nu-router

**Verdict (frozen section-1 rows): PARTIAL**  
cell `1xL4` · b=8 · host 16.0 GiB (TOTAL basis) · measured B_SSD 6.697 GiB/s · commit `8e77d2e44c68` · est. cost $0.509

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 1.269 | 1.619 | [1.234, 2.537] | <0.8638 \| >3.628 | UNTESTABLE | aggregate, last 32 of 64 decode steps; UNTESTABLE: measured B_SSD is 2.88x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.5433 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | PASS | pooled_dedup over the window (26875/49463) |
| cold_records_per_tok | 144.8 | 156.1 | [124.0, 312.2] | <93.009 \| >390.3 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.3241 | 0.254 | [0.1621, 0.3332] | <0.09723 \| >0.5564 | UNTESTABLE | 1000 x (1.4801760000000002 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 2.88x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **2.88x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 2.51x (expensive reading, P15b/P15c basis) / 9.05x (cheap reading)  

Warnings:
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU
- row 3 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast
- frozen b=8 rows carry the units defect (Addendum 2 item 3): scored as registered against AGGREGATE last-32-step TPS; expect them to read low


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 1.269 | 5.251 | [4.207, 10.391] | <2.945 \| >14.859 | KILLED | aggregate, last 32 of 64 decode steps |
| host_hit_rate | 0.5433 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | PASS | pooled_dedup over the window (26875/49463) |
| cold_records_per_tok | 144.8 | 60.306 | [60.306, 370.0] | <45.230 \| >462.6 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.4256 | 0.1028 | [0.05196, 0.1283] | <0.03118 \| >0.2143 | KILLED | 1000 x registered rate 0.00053992 $/s / decode tok/s |

b=8 steady-state gate: **OPEN**  
Predicted limiter: `storage`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
