# REPRODUCIBILITY_POLICY.md — source-preservation after the force-push incident

Branch: `experiment/pack-cap-ab` · Date: 2026-09-07

## 1. Incident summary (audit)

During dispatch of the pack-cap sessions it was discovered that GitHub
history for `so-nerdyy/dee` had been **rewritten**: branch
`freebuff/deepseek-v4-flash-0731-t4` moved to `7b137846` (the old baseline
era) and engine commit `217a33359b06a0453444a698ec52e4078b77e388` — the
canonical commit every host-reuse-era seal ran on — **no longer existed on
the remote**.

Failure chain in the seal-era harness (reproduced locally before repair):

1. `git clone --branch freebuff/deepseek-v4-flash-0731-t4 --single-branch …`
   now yields a history without `217a3335`;
2. the harness's embedded bundle (`ac2bac46…`) was **incremental** — it
   carried `39acc76` but relied on the then-current remote history to supply
   `217a3335`;
3. fetch of the bundle against the force-pushed clone → missing prerequisite
   → `checkout 217a3335` fails → kernel dies at setup (~15 s).

Both session kernels died this way on v1/v2. **No performance data was
produced by the failed runs**; they were superseded by repaired kernels
(new slugs, `…r1`), which is why the A/B evidence is intact.

## 2. The repair (audited, in-repo)

`bundle/repair-217a3335-prereq-7b137846.bundle` (69,963 bytes,
sha256 `76e1b437c6bf521d…`):

- incremental bundle whose **prerequisite is the remote's current tip**
  (`7b137846`) — present in every fresh clone — and whose head is
  `refs/heads/codex/dee4-bounded-fill-storage` carrying the **bit-identical**
  commit `217a3335` (same hash ⇒ same tree, by git content addressing);
- head name matches the harness's fetch refspec, so only two embedded
  constants changed (bundle sha256 + b64 payload); arm A = base + repair,
  arm B = base + repair + the single cap constant;
- validated end-to-end locally: fresh clone of the force-pushed branch +
  fetch bundle + `checkout 217a3335` + `rev-parse HEAD` verification
  (the harness then double-verifies HEAD == pinned commit and fails closed).

Both repaired sessions and the pread rider use this mechanism. Bit-identical
engine source is thereby reconstructable **even under another force-push of
the same branch**, as long as `7b137846` remains the remote tip.

## 3. Residual risks in the current repair

- The bundle is again **incremental**: if the remote tip moves again
  (another rewrite of `freebuff/…`), the prerequisite disappears and a new
  repair bundle would be needed. Each repair keys to a moving target.
- The only durable copies of `217a3335` today are: this repository (git
  objects), local clones/worktrees, and the embedded bundles inside pushed
  Kaggle kernel versions. All are under our control; none are independent.
- Build inputs (torch/CUDA versions, CMake flags) are recorded in run
  evidence but not pinned by hash.

## 4. Durable evidence policy (for future seals)

Every future sealed experiment MUST make its source reconstructable without
trusting the mutable remote:

1. **Standalone bundle at seal time.** Embed a bundle created with
   `git bundle create seal.bundle <engine-commit>` (no prerequisites) OR an
   incremental bundle whose prerequisite is a commit archived in a
   **standalone** bundle also embedded. Embed as base64 with sha256, as
   today; verify by `rev-parse HEAD == pinned` after checkout (fail closed).
2. **Second home for the bundle.** Push the seal bundle as a private Kaggle
   **dataset** (e.g. `nivind/dee-seal-bundles`, one file per commit) and
   record its sha256 in both the kernel and `provenance.json`. Kaggle
   datasets are outside GitHub's rewrite reach; the embedded copy inside the
   kernel version is itself a third copy.
3. **Reconstruction contract in evidence** (all already collected by the
   harness — keep mandatory): source commit sha + parent hash, bundle
   sha256(s), model revision, per-shard header sha256s (48 shards),
   expert-bank `data_sha256` + trace-journal sha256/terminal chain,
   run-config hash + effective `engine_config`, torch/CUDA versions,
   GPU UUIDs, timestamps. `provenance.json` must carry all of them.
4. **Incident log.** Any future history rewrite gets a `REPAIR_LOG.md`
   entry (old tip, new tip, repair bundle sha, validation transcript).
   Never rewrite existing seals; repair forward only.
5. **Pre-flight rule (unchanged, now explicit):** a kernel that cannot
   verify `HEAD == pinned commit` and `bundle sha256` must fail closed at
   setup — never fall back to "closest available source".

## 5. Applying the policy retroactively (allowed actions only)

- The repair bundle + validation transcript are committed here (done).
- `provenance.json` of the pack-cap seal already records source commit,
  bundle sha, model revision, shard hashes, kernel identity (done in the
  A/B evidence).
- Older seals are NOT modified; their evidence continues to reference
  commits reconstructable from this repository's objects.
