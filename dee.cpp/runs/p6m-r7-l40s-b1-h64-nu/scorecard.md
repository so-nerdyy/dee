# Phase-6 scorecard: p6m-r7-l40s-b1-h64-nu

**Verdict (frozen section-1 rows): PARTIAL**  
cell `1xL40S` · b=1 · host 64.0 GiB (TOTAL basis) · measured B_SSD 89.331 GiB/s · commit `7fc878ac50a7` · est. cost $0.263

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 1.351 | 2.626 | [2.096, 5.081] | <1.468 \| >7.266 | UNTESTABLE | decode tokens / decode wall; UNTESTABLE: measured B_SSD is 15.99x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.7359 | 0.54 | [0.47, 1.000] | <0.4 \| >1.000 | PASS | pooled_dedup, prompt-scoped (9828/13355) |
| cold_records_per_tok | 73.479 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 0.7404 | 0.1279 | [0.02029, 0.1383] | <0.01218 \| >0.231 | UNTESTABLE | 1000 x (3.6008639999999996 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 15.99x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **15.99x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 5.74x (expensive reading, P15b/P15c basis) / 20.67x (cheap reading)  

Warnings:
- OWN-REFERENCE gate (Addendum 4): exact_16_token_ids/exact_decoded_text vs the sealed T4 answer were waived because this run is bit-identical to p6m-r7-l40s-b1-h16-nu at another host budget. Weaker than the exact gate; it does not show one-GPU mode is bug-free.
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 26624 MiB vs the frozen row's 49152 MiB per GPU
- row 2 of a multi-config container: earlier rows warmed the page cache (not droppable under gVisor); timings read fast
- measured B_SSD 89.3308 GiB/s is above any plausible NVMe; treat it as page cache, not disk


## Addendum 3 (Modal, measured B_SSD): **UNTESTABLE**

- VRAM budget 26624 MiB != registered 28672 MiB
- measured B_SSD 89.3308 GiB/s flagged implausible for NVMe (page cache / virtualized FS); Addendum 3 is evaluated AT B_SSD

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
