# Phase-6 scorecard: p6m-r7-rtx-b8-h16-nu

**Verdict (frozen section-1 rows): KILLED**  
cell `1xRTXPRO6000` · b=8 · host 16.0 GiB (TOTAL basis) · measured B_SSD 149.0 GiB/s · commit `7fc878ac50a7` · est. cost $0.424

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 4.856 | 15.296 | [14.132, 102.0] | <9.892 \| >145.9 | UNTESTABLE | aggregate, last 32 of 64 decode steps; UNTESTABLE: measured B_SSD is 20.00x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.4472 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup over the window (22009/49219) |
| cold_records_per_tok | 106.3 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | window source reads / (8 x 32) |
| usd_per_1k_tok | 0.2677 | 0.085 | [0.01274, 0.09201] | <0.007647 \| >0.1537 | UNTESTABLE | 1000 x (4.680864 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 20.00x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **20.00x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 2.08x (expensive reading, P15b/P15c basis) / 7.48x (cheap reading)  

Warnings:
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 61440 MiB vs the frozen row's 98304 MiB per GPU
- row 3 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast
- measured B_SSD 149.024 GiB/s is above any plausible NVMe; treat it as page cache, not disk
- frozen b=8 rows carry the units defect (Addendum 2 item 3): scored as registered against AGGREGATE last-32-step TPS; expect them to read low


## Addendum 3 (Modal, measured B_SSD): **UNTESTABLE**

- measured B_SSD 149.024 GiB/s flagged implausible for NVMe (page cache / virtualized FS); Addendum 3 is evaluated AT B_SSD

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
