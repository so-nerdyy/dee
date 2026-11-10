# Phase-6 scorecard: p6m-l4-r5-b1-h16

**Verdict (frozen section-1 rows): UNTESTABLE**  
cell `1xL4` · b=1 · host 16.0 GiB (TOTAL basis) · measured B_SSD 3.955 GiB/s · commit `311d9d0bf821` · est. cost $0.128

## Prerequisites not met (run is UNTESTABLE)

- runner correctness gate: REJECT_NUMERICAL (failed: ?)

ID tightening applied: row prediction_ids per section 4 (P1, P1b, P2, P14, P15b/P15c vs the EXPENSIVE dense reading 0.12892 $/1k)  
Not applied here: P5/P8b/P16/P17 (need a stage-profile bucket analysis; read the profile by hand), P2b (needs a MIN replay of the route journal)
