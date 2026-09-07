# FINAL_SUMMARY.md — pack-cap forensics (misses without wall, order asymmetry)

Branch: `experiment/pack-cap-ab` · Base: `45b2a16` · Date: 2026-09-07
Track: OFFLINE evidence-forensics on the COMPLETED pack-cap A/B. The A/B was
not rerun; 20 GiB was not promoted; no production change; nothing merged.

Evidence: the four sealed arms (all ACCEPT_CORRECTNESS, byte-identical route
journal `f6ec7024…`), their `per_token_accounting`, `expert_store`,
`host_pack`, `memory.json` telemetry, and one new storage-only rider kernel.
Machine-readable: `results/{eliminated-misses,miss-criticality,order-effect,
pread-rider,revised-service-model}.json`. Documents:
`ELIMINATED_MISS_LEDGER.md`, `MISS_CRITICALITY.md`,
`REVISED_SERVICE_MODEL.md`, `ORDER_EFFECT_FORENSICS.md`,
`PREAD_RIDER_RESULT.md`, `REPRODUCIBILITY_POLICY.md`.

## Answers to the required questions

**1. Which exact 52 misses disappeared?**
The replay-produced ledger (`results/eliminated-misses.json`) identifies 51
eliminated misses at replay granularity (live delta 52; the ±1 is the
replay's documented v65-era tolerance). All are (device, forward_step,
layer, expert) identities on cuda:0 (40) / cuda:1 (11), tokens 3–15, layers
0–38, 681.8 MB (≈ the measured 695 MB). **All 51 are capacity re-reads —
zero compulsory** — with previous-use distance 873–1900 distinct records
(median 1298). The +2 GiB rescued only long-recency re-reads.

**2. How many were actually on the critical path?**
None can be demonstrated demand-blocking. 21/51 sit on tokens where the
B−A wall delta flips sign with arm order (position-dominated); 17 sit on
tokens where B is faster in both sessions but by ≤ 0.26 s (sub-noise;
same-arm cross-session variation reaches ~0.5 s); 13 on tokens where B is
slower in both sessions. Per-request classification is UNKNOWN (stage
profiling was disabled; no read submit/complete/demand timestamps exist).

**3. How much measured critical storage wait did they represent?**
Not directly measurable per request. Bounds: read-worker overlap was 62.2%
(cuda0) / 62.3% (cuda1); batches average ~3.4 concurrent reads so batch
wall ≈ max, not sum; the removed reads barely reduce batch count (~764
batches). Even the most favorable same-sign tokens bound the total saving
at ≲ 0.26 s vs the serial model's 2.53 s prediction.

**4. Why did ~695 MB fewer reads yield ~zero wall improvement?**
Three measured mechanisms: (a) batch wall = max of concurrent reads —
removing 1–2 reads from a multi-read batch shortens nothing; (b) 62% of
read-worker time was already overlapped, and the removed reads were
disproportionately inside overlapped windows; (c) the rider proves the
device never saturates during decode (~470 MB/s demanded vs ~2.9 GB/s
cold capability), so bytes saved have no bandwidth value — only
demand-blocking lateness would, and these reads weren't late (prefetch
ring: 2613 prefetches issued vs 2285 cold loads).

**5. Is the global 48.679 ms/miss model permanently rejected?**
As a causal model, yes — prospectively falsified (predicted −2.53 s,
observed +0.03 s, sign set by arm order). It remains a retrospective
correlation on v65 where miss count co-varied with concurrency structure.
Replacement: observational per-token model
`wall ≈ floor_token + 0.38 × read_batch_wall_token` (read-batch wall
predicts per-token wall with r = 0.971 vs 0.933 for miss count, slope
0.384 s/s). Sharpening it requires one profiler-enabled arm.

**6. Why is B order-sensitive while A is stable?**
Not a property of the cap. Two arm-agnostic effects compose differently
across the four (arm, position) cells: a **session effect** on tokens 1–11
(both session-2 arms +2.3–2.5 s slower; coincides with cuda1 read
bandwidth 124→108 MiB/s) and a **position effect** on tokens 12–15 (the
second-position arm's read-batch wall halves: −1.1 to −3.6 s/token).
A-first happens to pair session-fast-mid with cold-tail (70.65), B-second
session-fast-mid with warm-tail (68.48), etc. B's "swings" are the session
and position effects landing on opposite cells.

**7. Is page-cache/warmness supported or rejected?**
SUPPORTED as the tail mechanism, but NOT candidate-specific: the second
arm's tail reads are faster whichever arm it is (p50 read 102.5 vs 108.5 ms
for B; tail rd-wall −1.1…−3.6 s for both arms). The rider measures warm
page-cache reads at 3.2–5.7× cold — ample to explain the tail. B-specific
costs exist but are small (fill-reservation +1.3 s; MemAvailable 4.9 vs
7.8 GiB). The stated hypothesis ("larger pack costs cold-start, benefits
warm") is partially supported and incomplete — it cannot explain A's
stability without the session effect.

**8. What did the corrected pread rider measure?**
Same T4 storage class, post-A/B, bank identity validated
(`c83462ba…` PASS): cold aggregate saturates at **~2.9 GB/s by depth 2–4**,
flat through depth 16 (per-read p50 ≈ 4.7 ms × depth — a shared back-end,
not per-lane bandwidth); warm 8.8–9.4 GB/s (3.0–5.7×); depth 1 cold is
425 MB/s (dispersed); pattern barely matters at saturation; zero
short/empty reads. Production qdepth 6 is beyond the saturation knee →
more depth/lanes ≈ 0 (consistent with sealed lane-count neutrality).

**9. Is another full-model order experiment justified?**
Not yet — and not an A/B of caps. The remaining ambiguity is the session
effect (mid-run, cuda1-bandwidth-linked). The minimal discriminator, only
if the profiler path is unavailable: a same-session **A/B/A triple**
(identical arm, randomized middle) — it separates within-session position
from between-session drift with ONE kernel. Pre-register it; do not launch
without accepting that it cannot measure any cap effect.

**10. What should Muse's live profiler focus on most?**
Per-request storage timestamps + readiness attribution, per the staging
telemetry design: request layer, route-known, read-submit, read-complete,
H2D-submit/complete, compute-needed, ready-before-demand. Concretely:
`host_waits.cache_readiness` and `cache_readiness_attribution`
(`copy_not_submitted` vs `copy_in_flight` vs `consumer_reached_entry_too_
early`) on ONE profiler-enabled cap-17 arm. That converts every per-request
UNKNOWN in this report into measured critical wait.

**11. What experiment should happen after the profiler?**
A single profiler-enabled 17-GiB arm (n_tokens 16, lanes 3, qdepth 6) to
measure critical wait per request. Its expected finding, per this
forensics: ≥ 60% of cold loads ready before demand; critical wait
concentrated in early tokens. That calibrates `floor_token + κ·exposure`
into a predictive model — which is the prerequisite for any future
optimization ranking. A/B/A above is the fallback if profiling perturbs.

**12. What permanent source-preservation policy should seals use?**
`REPRODUCIBILITY_POLICY.md`: (1) standalone (prerequisite-free) bundle of
the engine commit embedded in every seal, verified fail-closed
(`rev-parse HEAD == pinned`); (2) the same bundle pushed as a Kaggle
dataset + sha256 in `provenance.json` (survives GitHub rewrites — this
incident proved incremental-against-remote-tip bundles break on
force-push); (3) mandatory reconstruction contract (commit+parent, bundle
sha, model REV, 48 shard-header hashes, bank data sha, run-config hash,
torch/CUDA, GPU UUIDs); (4) `REPAIR_LOG.md` for any future rewrite; never
rewrite old seals, repair forward only. The `217a3335` repair bundle
(bit-identical commit keyed to current remote tip `7b137846`) is committed
here and is now load-bearing for every live kernel.

## One-line verdict

The cap increase did exactly what the replay said (−51/−52 capacity
re-reads, −682/−695 MB) and nothing to the wall because those reads were
not on the critical path; the wall deltas that do exist are two
arm-agnostic confounds (session drift, tail warmness) misread as an arm
effect; the ms-per-miss era is over — decode wall follows measured
read-batch exposure at κ ≈ 0.38, and the disk was never the constraint.
