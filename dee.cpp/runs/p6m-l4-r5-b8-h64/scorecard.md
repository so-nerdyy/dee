# Phase-6 scorecard: p6m-l4-r5-b8-h64

**Verdict (frozen section-1 rows): PARTIAL**  
cell `1xL4` · b=8 · host 64.0 GiB (TOTAL basis) · measured B_SSD 3.955 GiB/s · commit `311d9d0bf821` · est. cost $0.307

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.8382 | 6.751 | [5.443, 16.020] | <3.810 \| >22.909 | UNTESTABLE | aggregate, last 32 of 64 decode steps; UNTESTABLE: measured B_SSD is 1.70x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.8351 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup over the window (41307/49463) |
| cold_records_per_tok | 31.859 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.6177 | 0.07668 | [0.03232, 0.09512] | <0.01939 \| >0.1588 | UNTESTABLE | 1000 x (1.8637920000000001 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 1.70x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **1.70x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 4.79x (expensive reading, P15b/P15c basis) / 17.25x (cheap reading)  

Warnings:
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU
- frozen b=8 rows carry the units defect (Addendum 2 item 3): scored as registered against AGGREGATE last-32-step TPS; expect them to read low


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 0.8382 | 9.006 | [7.586, 24.890] | <5.310 \| >35.593 | KILLED | aggregate, last 32 of 64 decode steps |
| host_hit_rate | 0.8351 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup over the window (41307/49463) |
| cold_records_per_tok | 31.859 | 0 | [0, 295.5] | <0 \| >369.4 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.6442 | 0.05995 | [0.02169, 0.07117] | <0.01302 \| >0.1189 | KILLED | 1000 x registered rate 0.00053992 $/s / decode tok/s |

b=8 steady-state gate: **OPEN**  
Predicted limiter: `h2d`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
