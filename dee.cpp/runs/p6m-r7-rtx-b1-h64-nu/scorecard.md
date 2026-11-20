# Phase-6 scorecard: p6m-r7-rtx-b1-h64-nu

**Verdict (frozen section-1 rows): PARTIAL**  
cell `1xRTXPRO6000` · b=1 · host 64.0 GiB (TOTAL basis) · measured B_SSD 149.0 GiB/s · commit `7fc878ac50a7` · est. cost $0.27

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 1.948 | 3.187 | [2.578, 6.685] | <1.805 \| >9.560 | UNTESTABLE | decode tokens / decode wall; UNTESTABLE: measured B_SSD is 20.00x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.7358 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup, prompt-scoped (9828/13356) |
| cold_records_per_tok | 73.500 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 0.7222 | 0.1769 | [0.02558, 0.1915] | <0.01535 \| >0.3198 | UNTESTABLE | 1000 x (5.06448 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 20.00x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **20.00x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 5.60x (expensive reading, P15b/P15c basis) / 20.17x (cheap reading)  

Warnings:
- OWN-REFERENCE gate (Addendum 4): exact_16_token_ids/exact_decoded_text vs the sealed T4 answer were waived because this run is bit-identical to p6m-r7-rtx-b1-h16-nu at another host budget. Weaker than the exact gate; it does not show one-GPU mode is bug-free.
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 61440 MiB vs the frozen row's 98304 MiB per GPU
- row 2 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast
- measured B_SSD 149.024 GiB/s is above any plausible NVMe; treat it as page cache, not disk


## Addendum 3 (Modal, measured B_SSD): **UNTESTABLE**

- measured B_SSD 149.024 GiB/s flagged implausible for NVMe (page cache / virtualized FS); Addendum 3 is evaluated AT B_SSD

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
