# Phase-6 scorecard: p6m-r7-2xt4-b1-h64-nu

**Verdict (frozen section-1 rows): UNTESTABLE**  
cell `None` · b=1 · host 64.0 GiB (TOTAL basis) · measured B_SSD 26.758 GiB/s · commit `7fc878ac50a7` · est. cost $0.21

## Prerequisites not met (run is UNTESTABLE)

- Modal GPU '2xT4' is not a frozen cell


Warnings:
- OWN-REFERENCE gate (Addendum 4): exact_16_token_ids/exact_decoded_text vs the sealed T4 answer were waived because this run is bit-identical to p6m-r7-2xt4-b1-h16-nu at another host budget. Weaker than the exact gate; it does not show one-GPU mode is bug-free.

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
