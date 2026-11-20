# Phase-6 scorecard: p6m-r7-a10-b8-h64-nu

**Verdict (frozen section-1 rows): PARTIAL**  
cell `1xA10` · b=8 · host 64.0 GiB (TOTAL basis) · measured B_SSD 1.426 GiB/s · commit `7fc878ac50a7` · est. cost $0.466

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 3.230 | 6.765 | [5.452, 16.095] | <3.816 \| >23.016 | UNTESTABLE | aggregate, last 32 of 64 decode steps; UNTESTABLE: measured B_SSD is 0.51x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.853 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup over the window (41471/48617) |
| cold_records_per_tok | 27.914 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.1863 | 0.08895 | [0.03739, 0.1104] | <0.02243 \| >0.1843 | UNTESTABLE | 1000 x (2.166192 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 0.51x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **0.51x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 1.45x (expensive reading, P15b/P15c basis) / 5.20x (cheap reading)  

Warnings:
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU
- row 4 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast
- frozen b=8 rows carry the units defect (Addendum 2 item 3): scored as registered against AGGREGATE last-32-step TPS; expect them to read low


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 3.230 | 9.029 | [7.603, 25.071] | <5.322 \| >35.852 | KILLED | aggregate, last 32 of 64 decode steps |
| host_hit_rate | 0.853 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup over the window (41471/48617) |
| cold_records_per_tok | 27.914 | 0 | [0, 295.5] | <0 \| >369.4 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.1932 | 0.0691 | [0.02489, 0.08206] | <0.01493 \| >0.137 | KILLED | 1000 x registered rate 0.00062392 $/s / decode tok/s |

b=8 steady-state gate: **OPEN**  
Predicted limiter: `h2d`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
