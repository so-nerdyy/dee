# Phase 3 — Build & publish plan for the full-universe expert store

Status: **execution-ready**; the driver (`dee.cpp/tools/phase3/p3_kaggle_job.py`)
is implemented and proven end-to-end on the synthetic universe
(`test_p3_kaggle_job.py`: 15 tests, all local-pass).
Branch: `research/phase3-build-execution` · Model:
`deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`

> Companion to `PHASE3_FULL_EXPERT_STORE.md` (the design/prototype doc). This
> doc is the *operational* plan: what it costs, what gets pushed where, and
> the explicit go/no-go for spending Kaggle budget.

---

## 1. What the driver does (and what was proven locally)

`p3_kaggle_job.py` wraps the proven `p3_builder.build_segmented` and adds:

- **Source**: `RemoteRangeSource` on the pinned revision by default (no local
  checkpoint copy needed — the 166.9 GB source is *read* via HTTP ranges,
  never staged); `--source local` reads a mounted/local shard dir;
  `--source synthetic` is a zero-IO pipeline smoke.
- **Bounded-transient publish**: each committed segment is *hardlinked*
  (never copied) into a one-file staging dir, uploaded, journal-committed
  to `publish.journal.jsonl`, then evicted — the job never holds more than
  ~2 segments locally (~6.4 GiB ≪ 19.5 GiB working budget) and never stages
  source+dest simultaneously.
- **Backpressure**: `PushGate` blocks the first fetch of bucket K's shard
  until bucket K−1 is pushed; a push failure releases the gate (degrades to
  "build all, push at end"), never deadlocks.
- **Resume**: same-session via the builder's own journal/tail-reverify;
  cross-session via `--seed-dir` (restore journals + in-flight `.partial`
  from a previous run's kernel output; pushed buckets are recreated as
  same-size sparse tombstones so `build_segmented`'s committed-file checks
  pass without the bytes).
- **Prefetch**: `--prefetch N` wraps the source with a bounded look-ahead
  (window ≈ 3×N requests, ~≤150 MiB RAM) — required for the remote path to
  be bandwidth-bound rather than request-latency-bound.
- **`concat` subcommand**: v4-segmented → v2 single file (streamed concat,
  per-segment rehash vs the published table, atomic `os.replace`, dee4-v2
  metadata, optional full `verify_store` pass).

Local proof (synthetic 3×4 universe): end-to-end build+publish+evict,
publish-failure resume (no re-push of shipped buckets), cross-session
`seed_from` resume (zero rebuild, zero re-push), keep-segments mode,
concat→v2 byte-exact + verified, prefetch concurrency + byte-exactness,
gate ordering. `pytest dee.cpp/tools/phase3` → 24 passed.

No `p3_*.py` file was modified — the driver is purely additive.

---

## 2. Cost model (honest numbers)

Universe: 46 buckets (43 main + 3 mtp) × 256 experts = **11,776 records**
× 13,369,344 B = **157,437,394,944 B = 146.625 GiB**, shipped as 46
segments of **3,422,552,064 B = 3.1875 GiB**.
Remote fetch volume = 157.4 GB over **70,702 HTTP requests**
(6 per record + 46 header probes, ~2.2 MiB average response).

| Stage | Bound | Basis | Time (46 buckets) |
|---|---|---|---|
| Pipeline sans I/O (assemble+sha256+fsync) | CPU/disk | measured 168.4 MiB/s on this box | ~15 min |
| Write side | Kaggle SSD ~126 MB/s | doc §5.2 | ~21 min floor |
| **Build, mount-sourced** (`--source local`, `/kaggle/input/…-shards`) | mount read ~13 MB/s | doc §5.2 | **~3.4 h read floor → ~3.5–4 h total** |
| **Build, remote-sourced serial** (`--source remote`, prefetch 0) | request latency ~70.7k × 100–300 ms | measured 2.91 MiB/s residential | **~2.9–5.9 h request-bound; 14.3 h at residential BW — must prefetch** |
| **Build, remote + prefetch 8–16** | datacenter HF BW (unknown until measured) | Kaggle egress typically ≥30–100 MB/s | **~0.5–1.5 h expected; measured in-run** |
| Prior sealed-bank repack (reference) | 70.42 MiB/s achieved on Kaggle T4 | doc §5.1 | ~35.5 min equivalent |
| Publish 46 datasets | Kaggle inbound ~30–60 MB/s (unknown) | — | ~45–130 min, interleaved with build |
| v4→v2 concat + full verify (in-run, needs +157.4 GB scratch) | ~126 MB/s r/w | — | ~21 min copy + ~21 min hash |
| Consumer-side concat later (reads mounted segment datasets) | 13 MB/s mount | doc §5.2 | ~3.4 h + verify |

**Session reality**: Kaggle CPU session ≈ 9–12 h wall. Both source paths fit;
the mount path fits comfortably, the remote path fits *only with prefetch
or a fast link* — which is why the run measures remote throughput on the
first buckets before committing to it.

**The request-rate wall is the remote path's real risk**, not bandwidth:
70,702 serial requests at ~150 ms each ≈ 2.9 h even at infinite bandwidth.
`--prefetch 8` collapses that floor to ~22 min. Default for the run:
`--prefetch 8`.

### Which source

- **Default: `--source remote --prefetch 8`**, pinned revision, HF public
  ranges (no token needed). Fallback inside the same session if the pilot
  buckets measure a poor rate: restart with `--source local --shards
  /kaggle/input/deepseek-v4-flash-0731-shards` — the journal makes the
  switch free (already-built buckets are skipped; mixing sources across
  buckets is safe because integrity lines record per-record source shards).
- Mount-sourced is the *safe* path (known 3.4 h floor); remote is the *fast*
  path if Kaggle's HF link is good. The run decides with data, not hope.

---

## 3. Publish plan

Kaggle dataset versions are **complete snapshots** — files are not carried
across versions (Kaggle/kaggle-api#274), and a single 146.6 GiB version can
never be staged inside 19.5 GiB anyway. So the publish unit is the segment:

- **46 segment datasets**: `dee4-p3-full-u<universe_sha256[:12]>-b00` … `-b45`
  (on this branch's universe, `u2081ada5e37e`). One version, one 3.1875 GiB
  file each; upload atomicity = per-segment retry.
- **1 index dataset**: `…-index` carrying `metadata.json`
  (dee4-v4-segmented, per-segment sha256 table), `integrity.jsonl`
  (~11,776 lines), `build.journal.jsonl`, `publish.journal.jsonl`,
  `build_report.json`, `job_report.json`, `p3_manifest.json`,
  `p3_records.jsonl` — everything needed to locate, reassemble, and verify
  the whole store.
- Naming binds every artifact to `universe_sha256`; records also carry
  `source_revision` = `9e165c30…`.
- Datasets should be **public** (bytes derive from a public checkpoint;
  the precedent `nivind/deepseek-v4-flash-0731-shards` at ~155 GiB shows
  the account can host this scale; private quota would not).
- Per-push read-back gate: `--verify-before-push` (default on) rehashes the
  finished segment against its journal sha256 before upload — corrupt
  local bytes can never be published.
- Preflight: the index dataset is created FIRST in the real run script so
  auth/quota/slug problems fail before any build time is spent.

### Why not publish the concatenated v2 file

One 157,437,394,944 B file exceeds practical per-file dataset limits and
can't be staged in the working budget regardless. The segmented store *is*
the durable artifact; consumers either concat it locally or read segments
directly (see §5).

---

## 4. Resume semantics (what a dead session costs)

- Same-session crash/kill: rerun the identical command. Journal replays,
  torn tail truncated+reverified, committed+pushed buckets become
  tombstones, build continues. Zero rework beyond the in-flight record.
- Kaggle session loss (12 h cap, spot preemption): the failed run's
  `/kaggle/working` output is still saved. Attach it as an input to the
  continuation run; `P3KaggleJob.seed_from()` restores journals +
  in-flight `.partial`; pushed buckets resume as tombstones. Worst-case
  rework ≈ one partially built bucket.
- Push-side idempotence: a bucket already in `publish.journal.jsonl` is
  never re-uploaded; a journal line lost after a successful upload is
  healed by the publisher's remote file check (same name+size ⇒ skip).

---

## 5. dee4-v4 → dee4-v2 concat + verify

`python p3_kaggle_job.py concat --store <dir> --out <v2dir>`:

1. Streams `metadata.json`'s segment table in `first_record` order into
   `experts.dee4.partial`, hashing the whole file and re-hashing each
   segment against its published `sha256` as it goes (free source-side
   integrity check).
2. `os.replace` → `experts.dee4`; copies `integrity.jsonl`; emits dee4-v2
   `metadata.json` (same component tables the shipped
   `dee::Dee4ExpertStore` reads; `num_layers=46` ⇒ runtime layers 43–45 =
   mtp draft buckets) + `data_sha256` + `total_bytes` +
   `assembled_from` provenance.
3. `--verify` (default) runs the full `p3_builder.verify_store` pass —
   every record rehashed against `integrity.jsonl`.

Where it runs: needs 2× the store on fast scratch (segments + output).
On Kaggle `/tmp` (~1 TiB overlay observed) it fits: ~21 min copy + ~21 min
verify. Or the consumer does it — reading the mounted segment datasets at
~13 MB/s is a ~3.4 h read floor per session. **A leaner long-term answer
is the optional segmented reader** (design-doc §3.2/§7): mmap each mounted
segment file directly, zero assembly, ~50 lines — worth adding before any
run that would otherwise concat repeatedly.

---

## 6. Eager build vs dee4-v5-lazy — decision for the first arbitrary-prompt run

| | Eager (this plan) | Lazy (`p3_lazy_store`, dee4-v5) |
|---|---|---|
| Artifact | durable published datasets | ephemeral `/tmp` sparse file; dies with the session |
| First-arbitrary-prompt latency | must wait for build+publish+concat/mount | none — fills on demand |
| Cold-miss cost | zero (bank complete) | ~4.7 s/record residential HF; ~0.1–0.3 s/record datacenter-class link (12.75 MiB + RTT) |
| Fill bound for a 16-token decode | — | ≤ 16×46×6 = 4,416 pairs ≈ **55.0 GiB worst-case**; the sealed 16-token journal actually touched 2,364 pairs ≈ **29.4 GiB realistic** → ~10–19 min at 50–100 MB/s datacenter HF, ~2.9–5.4 h at residential 2.91 MiB/s |
| Disk | ~6.4 GiB peak (interleaved) | sparse 157.4 GB logical; real ≈ touched bytes (~30–55 GiB) ≪ 1 TiB overlay |
| Repeatability/perf evidence | yes — identical bytes, `data_sha256` sealed | yes after `finalize()`, but each session refills |

**Decision**: the first arbitrary-prompt *validation* does not need the
bank — run it on a dee4-v5 lazy store on `/tmp` (fill bounded at ~30–55 GiB,
~10–20 min at datacenter HF speeds, hidden behind decode if desired) —
**while** the eager build runs as the CPU job below. The bank is the
durable deliverable every later step depends on (repeatable runs, perf
measurement, the segmented reader, dee-serve). Lazy de-risks the timeline;
eager owns the artifact. Both produce byte-identical dee4-v2 content.

---

## 7. Kaggle run spec (pre-flight gate per AGENTS.md)

```
RUN TYPE:            CPU batch (no GPU needed; pure I/O + hashing job)
RUN NUMBER:          CPU 1/10 (worst case CPU 2/10 — a continuation run if
                     the first dies at the session cap; seed-dir resume makes
                     the second run incremental, not a restart)
QUESTION ANSWERED:   Can the full 11,776-record universe be built on Kaggle
                     from pinned-revision range fetches, published as
                     datasets, and verified — inside one session? Side-
                     measurement: real Kaggle↔HF range-fetch throughput and
                     Kaggle dataset-upload throughput (decides remote vs
                     mount for all later builds).
WHY LOCAL INSUFFICIENT: this box has ~20-24 GiB free vs a 146.6 GiB store
                     and measures 2.91 MiB/s residential HF (~14.3 h serial —
                     beyond feasibility); only Kaggle provides the datacenter
                     link, the scratch, and the dataset-publish endpoint.
ARMS:                Single job, two internal phases — (a) pilot: build
                     buckets 0-1 remote+prefetch, push under a -pilot dataset
                     prefix, measure MiB/s + req/s + upload rate;
                     (b) if remote ETA ≤ ~2 h, continue remote; else restart
                     same session mount-sourced (journal skips nothing since
                     pilot used a separate dataset prefix; the 2 pilot
                     buckets rebuild in ~4-5 min at mount speed — cheap
                     insurance, and keeps every published artifact under
                     one uniform -full naming).
SUCCESS CRITERIA:    job_report success=true; 46 segment datasets + index
                     dataset created; every publish journal sha256 equals
                     the builder journal sha256 (read-back verified pre-push);
                     metadata.json carries universe_sha256 +
                     source_revision=9e165c30…; spot-check: re-download 2
                     segment datasets in a later session and rehash == table.
FAILURE CRITERIA:    measured remote rate implies ETA > session even at
                     prefetch 16 AND mount read path errors; dataset create
                     refused (quota/auth) at preflight; any verify-before-push
                     sha mismatch (fail-closed: corrupt bytes never ship).
ARTIFACTS:           46 datasets dee4-p3-full-u2081ada5e37e-b{00..45} +
                     …-index (metadata/integrity/journals/manifest/records/
                     reports); kernel output copy of the same small files
                     (the resume seed).
```

Kernel wiring notes: run script invokes the driver exactly as
`python dee.cpp/tools/phase3/p3_kaggle_job.py build --build-dir
/kaggle/working/p3-store --headers dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers --source remote --prefetch 8 --publisher kaggle --dataset-owner nivind --dataset-prefix dee4-p3-full --public`.
The kernel needs (i) this branch's code — push `research/phase3-build-execution`
to a reachable remote and clone it (repo convention) or paste the 4 module
files + this driver into the kernel, and (ii) `KAGGLE_USERNAME`/`KAGGLE_KEY`
attached as kernel Secrets for the dataset push. HF fetches are anonymous
(public repo). `enable_internet` must be on; GPU off.

---

## 8. Go / no-go

**GO** — spend CPU 1/10 (with CPU 2/10 reserved as the resume contingency,
not a parallel spend).

Rationale: the build is the only remaining gate between the proven
prototype and a durable arbitrary-prompt store; it is resumable at every
level (record, segment, session), fail-closed on integrity at every level
(record/segment/store/universe hashes), bounded to ~6.4 GiB transient disk,
and self-measuring (the pilot phase retires the one unknown — Kaggle's real
HF/upload rates — in the first ~10 minutes). Worst-case total: ~5 h inside
a ~12 h session via the mount-sourced fallback; expected ~1.5–3 h remote.
No GPU budget is touched; the eager/lazy split in §6 means the first
arbitrary-prompt validation is not blocked on this run at all.

**NO-GO conditions to check before submit**: dataset create must succeed at
preflight (otherwise nothing ships — abort early, don't burn the session);
if the pilot measures remote throughput that implies > ~6 h total AND the
checkpoint-shards dataset mount is unavailable, abort and re-plan rather
than gamble on the session cap.
