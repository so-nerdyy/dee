# json_min `null` literal support (T17)

Branch `fix/json-min-null`.

## Problem

`dee.cpp/src/json_min.cpp` had no `null` literal: `parse_value()` fell
through to `error = true` on any `'n'`-led token. Conforming JSON such as
`"data_file": null` — legitimately emitted by
`tools/phase3/p3_builder.py::build_segmented` (`metadata["data_file"] = None`)
— failed to parse outright. The segmented-store reader
(`feat/dee4-segmented-reader` @ 48923198) worked around this inside
`Dee4ExpertStore::open()` with `scrub_json_nulls()`, a string-aware
value-position `null` -> `""` rewrite used as a retry after a failed parse.

## Decision: `null` parses to `Value::Type == Null`

The `Value` enum already carried `Null` — used until now only as the
default/error sentinel. Rather than inventing a new convention (e.g.
null -> empty string, which is what the scrub emulated), the parser now
produces a real `Null`-typed `Value` for the literal, and `Value` gains an
`is_null()` predicate symmetric with `is_object()/is_array()/...`.

Semantics for existing call sites (unchanged, opt-in tolerated):

- `find(key)` returns non-null `Value*` for a present-but-null key, still
  `nullptr` for a missing key — present-null and absent are distinguishable.
- `is_string()/is_int()/is_array()/is_object()` all return false for Null,
  so the existing typed accessors in `expert_store.cpp`/`weight_mmap.cpp`
  (`json_string`, `json_nonnegative_int`, `json_size_array`, direct
  `is_*` checks) treat a null field exactly like a missing or wrong-typed
  one — same verdict the scrub produced via `""`, minus one subtlety: the
  scrub made `is_string()` true with `s == ""`; native null makes it false.
  That difference is observable in exactly one place: the segmented gate in
  T10's `Dee4ExpertStore::open()` (see below).
- Parse failure still returns a Null-typed root with `*ok == false`; use
  `*ok`, not `is_null()`, to detect errors (documented in json_min.h).

## Strictness

`parse_null()` requires the literal to end on a token boundary (next char
not `[A-Za-z0-9_]`), so `nul`, `nulll`, `nullx`, `nullify`, `NULL` are
rejected even at top level where `parse()` does not check trailing garbage.
Inside arrays/objects the container `,`/`]`/`}` check already rejected such
suffixes. Only `'n'`-led inputs change behavior, and every one of them
previously errored — no previously-accepted input is now rejected, and no
previously-rejected input is now accepted. `truex`-style leniency for other
literals is untouched (out of scope; noted for the record).

## Interplay with the reader's scrub — REQUIRED follow-up on T10's branch

Two changes are needed on `feat/dee4-segmented-reader` (@ 48923198,
`dee.cpp/src/expert_store.cpp`) once it shares a line with this fix —
verified against that commit's source:

1. **The segmented `data_file` gate (~line 716) must accept `is_null()` —
   this is a correctness fix, not cleanup.** The gate is
   `if (data_file_value && !(data_file_value->is_string() &&
   data_file_value->s.empty()))` -> "must not name a data_file". Under the
   scrub, `"data_file": null` was rewritten to `""` and passed. With native
   null, the first-pass parse now succeeds (so the scrub retry never runs)
   and `find("data_file")` returns a Null-typed value: `is_string()` is
   false, the gate fails, and `open()` rejects every conforming
   dee4-v4-segmented store. Required repair, e.g.:

       !(data_file_value->is_null() ||
         (data_file_value->is_string() && data_file_value->s.empty()))

   (Keeping the `""` arm is optional back-compat for hand-edited metadata.)

2. **`scrub_json_nulls()` (~line 239) and the `!parsed` retry in `open()`
   (~lines 622-628) become unreachable dead code** — they only fire when
   the first-pass parse fails, and the sole defect they repaired (no `null`
   literal) no longer exists. The retry also cannot rescue anything else
   (it rewrites only nulls), and its value-position rules are strictly
   subsumed by `parse_null()`'s boundary rule. Safe to delete in the same
   follow-up; also update the gate comment that references the scrub.

NOT changed here — `expert_store.cpp` is owned by T10's line. Until that
follow-up lands, do not merge this fix together with 48923198 as-is.

## Files

- `dee.cpp/include/dee/json_min.h` — `is_null()` predicate + docs.
- `dee.cpp/src/json_min.cpp` — `'n'` dispatch + `parse_null()`.
- `dee.cpp/tests/test_json_min.cpp` — new standalone test (52 checks).

## Build / run (standalone)

    cd dee.cpp/tests
    g++ -std=c++17 -I../include test_json_min.cpp ../src/json_min.cpp -o test_json_min
    ./test_json_min

## CMake registration

Registered in `dee.cpp/CMakeLists.txt` via
`list(APPEND DEE_TEST_SOURCES tests/test_json_min.cpp)` after the existing
`test_real_router.cpp` append; the foreach loop builds/links/registers it
like the other host tests (`add_test(NAME test_json_min ...)`).
