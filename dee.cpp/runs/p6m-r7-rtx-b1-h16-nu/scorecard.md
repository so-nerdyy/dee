# Phase-6 scorecard: p6m-r7-rtx-b1-h16-nu

**Verdict (frozen section-1 rows): KILLED**  
cell `1xRTXPRO6000` · b=1 · host 16.0 GiB (TOTAL basis) · measured B_SSD 149.0 GiB/s · commit `7fc878ac50a7` · est. cost $0.22

| metric | measured | nominal | range | kill band | grade | basis |
|---|---|---|---|---|---|---|
| decode_tps | 2.219 | 3.187 | [2.578, 6.685] | <1.805 \| >9.560 | UNTESTABLE | decode tokens / decode wall; UNTESTABLE: measured B_SSD is 20.00x the cell spec (registered band 0.9-1.1x) |
| host_hit_rate | 0.6426 | 0.51 | [0.44, 0.7531] | <0.45 \| >0.57 | KILLED | pooled_dedup, prompt-scoped (8583/13356) |
| cold_records_per_tok | 99.438 | 147.8 | [0, 295.5] | <0 \| >369.4 | PASS | storage_requests_per_generated_token |
| usd_per_1k_tok | 0.586 | 0.1635 | [0.02365, 0.177] | <0.01419 \| >0.2956 | UNTESTABLE | 1000 x (4.680864 USD/h / 3600) / decode tok/s; UNTESTABLE: measured B_SSD is 20.00x the cell spec (registered band 0.9-1.1x) |

Measured B_SSD / cell spec: **20.00x** (decode tok/s and $/1k are scorable only inside 0.9-1.1x)  
$/1k vs dense residency: 4.55x (expensive reading, P15b/P15c basis) / 16.36x (cheap reading)  

Warnings:
- OWN-REFERENCE gate (Addendum 4): exact_16_token_ids/exact_decoded_text vs the sealed T4 answer were waived because this run is bit-identical to p6m-r7-rtx-b1-h64-nu at another host budget. Weaker than the exact gate; it does not show one-GPU mode is bug-free.
- no billing receipt (add summary.json 'billed_usd' from the Modal dashboard); $/1k scored on the 1000 x rate_s / TPS cross-check
- VRAM expert budget 61440 MiB vs the frozen row's 98304 MiB per GPU
- measured B_SSD 149.024 GiB/s is above any plausible NVMe; treat it as page cache, not disk


## Addendum 3 (Modal, measured B_SSD): **UNTESTABLE**

- measured B_SSD 149.024 GiB/s flagged implausible for NVMe (page cache / virtualized FS); Addendum 3 is evaluated AT B_SSD

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
