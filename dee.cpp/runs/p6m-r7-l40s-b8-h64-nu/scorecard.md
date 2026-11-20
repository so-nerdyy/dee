# Phase-6 scorecard: p6m-r7-l40s-b8-h64-nu

**Verdict (frozen section-1 rows): PARTIAL**  
cell `1xL40S` · b=8 · host 64.0 GiB (TOTAL basis) · measured B_SSD 89.331 GiB/s · commit `7fc878ac50a7` · est. cost $0.669

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 3.867 | 14.959 | [13.838, 88.756] | <9.687 \| >126.9 | UNTESTABLE | aggregate, last 32 of 64 decode steps; UNTESTABLE: measured B_SSD is 15.99x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.848 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup over the window (41879/49387) |
| cold_records_per_tok | 29.328 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.2587 | 0.06687 | [0.01127, 0.07228] | <0.006762 \| >0.1207 | UNTESTABLE | 1000 x (3.6008639999999996 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 15.99x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **15.99x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 2.01x (expensive reading, P15b/P15c basis) / 7.22x (cheap reading)  

Warnings:
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 26624 MiB vs the frozen row's 49152 MiB per GPU
- row 4 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast
- measured B_SSD 89.3308 GiB/s is above any plausible NVMe; treat it as page cache, not disk
- frozen b=8 rows carry the units defect (Addendum 2 item 3): scored as registered against AGGREGATE last-32-step TPS; expect them to read low


## Addendum 3 (Modal, measured B_SSD): **UNTESTABLE**

- VRAM budget 26624 MiB != registered 28672 MiB
- measured B_SSD 89.3308 GiB/s flagged implausible for NVMe (page cache / virtualized FS); Addendum 3 is evaluated AT B_SSD

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
