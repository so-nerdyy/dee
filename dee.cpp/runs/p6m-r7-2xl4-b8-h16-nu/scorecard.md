# Phase-6 scorecard: p6m-r7-2xl4-b8-h16-nu

**Verdict (frozen section-1 rows): KILLED**  
cell `2xL4` · b=8 · host 16.0 GiB (TOTAL basis) · measured B_SSD 3.512 GiB/s · commit `7fc878ac50a7` · est. cost $0.552

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 1.452 | 14.628 | [13.549, 78.299] | <9.484 \| >112.0 | UNTESTABLE | aggregate, last 32 of 64 decode steps; UNTESTABLE: measured B_SSD is 0.75x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.4342 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup over the window (21478/49463) |
| cold_records_per_tok | 109.3 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.542 | 0.05379 | [0.01005, 0.05807] | <0.006029 \| >0.09698 | UNTESTABLE | 1000 x (2.8324800000000003 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 0.75x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **0.75x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 4.20x (expensive reading, P15b/P15c basis) / 15.14x (cheap reading)  

Warnings:
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 3584 MiB vs the frozen row's 22528 MiB per GPU
- row 3 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast
- frozen b=8 rows carry the units defect (Addendum 2 item 3): scored as registered against AGGREGATE last-32-step TPS; expect them to read low


## Addendum 3 (Modal, measured B_SSD): **KILLED**

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 1.452 | 4.620 | [3.671, 8.649] | <2.570 \| >12.368 | KILLED | aggregate, last 32 of 64 decode steps |
| host_hit_rate | 0.4342 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup over the window (21478/49463) |
| cold_records_per_tok | 109.3 | 52.551 | [52.551, 358.8] | <39.413 \| >448.5 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.5248 | 0.1649 | [0.08809, 0.2075] | <0.05286 \| >0.3466 | KILLED | 1000 x registered rate 0.00076192 $/s / decode tok/s |

b=8 steady-state gate: **OPEN**  
Predicted limiter: `storage`  

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
