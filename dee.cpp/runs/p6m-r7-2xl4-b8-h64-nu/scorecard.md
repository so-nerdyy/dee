# Phase-6 scorecard: p6m-r7-2xl4-b8-h64-nu

**Verdict (frozen section-1 rows): PARTIAL**  
cell `2xL4` · b=8 · host 64.0 GiB (TOTAL basis) · measured B_SSD 3.512 GiB/s · commit `7fc878ac50a7` · est. cost $0.784

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 1.826 | 14.628 | [13.549, 78.299] | <9.484 \| >112.0 | UNTESTABLE | aggregate, last 32 of 64 decode steps; UNTESTABLE: measured B_SSD is 0.75x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.8459 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup over the window (41840/49463) |
| cold_records_per_tok | 29.777 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.4892 | 0.06107 | [0.01141, 0.06594] | <0.006846 \| >0.1101 | UNTESTABLE | 1000 x (3.2160960000000003 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 0.75x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **0.75x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 3.79x (expensive reading, P15b/P15c basis) / 13.66x (cheap reading)  

Warnings:
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU
- row 4 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast
- frozen b=8 rows carry the units defect (Addendum 2 item 3): scored as registered against AGGREGATE last-32-step TPS; expect them to read low


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 1.826 | 11.831 | [10.370, 44.511] | <7.259 \| >63.651 | KILLED | aggregate, last 32 of 64 decode steps |
| host_hit_rate | 0.8459 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup over the window (41840/49463) |
| cold_records_per_tok | 29.777 | 0 | [0, 295.5] | <0 \| >369.4 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.4172 | 0.0644 | [0.01712, 0.07347] | <0.01027 \| >0.1227 | KILLED | 1000 x registered rate 0.00076192 $/s / decode tok/s |

b=8 steady-state gate: **OPEN**  
Predicted limiter: `h2d`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
