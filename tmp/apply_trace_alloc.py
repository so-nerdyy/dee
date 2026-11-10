#!/usr/bin/env python3
"""Apply DEE_TA_* instrumentation wrappers to the C++ source files.

Mechanical rewrite: cudaXxx call sites get wrapped in DEE_TA_* macros; map
insertions (emplace / operator[]) on the three target maps get a preceding
DEE_TA_INSERT(...) marker line. Lifetime / ownership / destruction order
are UNCHANGED.

The walker is balance-aware (handles nested parens + string templates) and
skips string literals + comments via a masked-copy comparison:
  - For each candidate identifier at position [i..j], if every byte in the
    masked copy at the same range is space, the identifier is inside a
    "" string or /* */ comment - skip the rename.
  - For the opening '(' after the identifier, the masked byte at that
    position must also be non-space; otherwise we would rename a literal
    `cudaFree(` written as a comment-string arg.
"""
import os
import re
import sys

ROOT = r"C:/Users/carth/Downloads/dynamic_expert_eviction/dee.cpp"

RENAMES = [
    ("cudaMalloc",                 "DEE_TA_MALLOC"),
    ("cudaMallocHost",             "DEE_TA_MALLOC_HOST"),
    ("cudaFree",                   "DEE_TA_FREE"),
    ("cudaFreeHost",               "DEE_TA_FREE_HOST"),
    ("cudaEventCreate",            "DEE_TA_EVENT_CREATE"),
    ("cudaEventCreateWithFlags",   "DEE_TA_EVENT_CREATE_FLAGS"),
    ("cudaEventDestroy",           "DEE_TA_EVENT_DESTROY"),
    ("cudaStreamCreate",           "DEE_TA_STREAM_CREATE"),
    ("cudaStreamCreateWithFlags",  "DEE_TA_STREAM_CREATE_FLAGS"),
    ("cudaStreamDestroy",          "DEE_TA_STREAM_DESTROY"),
    ("cublasCreate",               "DEE_TA_CUBLAS_CREATE"),
    ("cublasDestroy",              "DEE_TA_CUBLAS_DESTROY"),
]

DEFAULT_OWNER = "unlabeled"

TARGET_MAPS = ("pinned_staging_bf16_", "staging_int8_", "registered_mmap_views_bf16_")

#  MILESTONE 3 v5 / FIX 5: per-site origin tag. The walker used to stamp a
#  uniform `"unlabeled_origin"` on every insert, which made the Breach-A
#  cross-check structurally inert (origin_matches short-circuits to true
#  for that passive tag). Each target map has a defensible expected
#  allocator, so we stamp it explicitly:
#     - pinned_staging_bf16_  -> host-flavour pool (cudaMallocHost / cudaHostAlloc)
#     - staging_int8_         -> per-expert pinned weights (cudaMallocHost)
#     - registered_mmap_views_bf16_ -> mmap-resident views (no allocation,
#       they're a resolved mmap window; tag is "mmap_resolve")
MAP_ORIGIN_HINTS = {
    "pinned_staging_bf16_":          "cudaMallocHost_or_cudaHostAlloc",
    "staging_int8_":                 "cudaMallocHost",
    "registered_mmap_views_bf16_":   "mmap_resolve",
}

#  MILESTONE 3 v5 / FIX 6: dict-key + value invariant. A typo in
#  MAP_ORIGIN_HINTS (e.g. dropping the trailing underscore) would silently
#  fall back to "unknown_origin", which origin_matches treats as a passive
#  observer tag and would silently re-disable the Breach-A diagnostic for
#  that map. Assert at module-import time that:
#    (a) every TARGET_MAPS entry has an explicit hint,
#    (b) no hint value is a passive observer tag.
_PAS_TAGS = ("unlabeled_origin", "unknown_origin", "unbalanced_origin")
assert set(MAP_ORIGIN_HINTS.keys()) == set(TARGET_MAPS), (
    "MAP_ORIGIN_HINTS must cover every TARGET_MAPS entry: "
    f"missing={sorted(set(TARGET_MAPS) - set(MAP_ORIGIN_HINTS.keys()))}, "
    f"extra={sorted(set(MAP_ORIGIN_HINTS.keys()) - set(TARGET_MAPS))}"
)
assert all(v not in _PAS_TAGS for v in MAP_ORIGIN_HINTS.values()), (
    "MAP_ORIGIN_HINTS values must not be passive observer tags; "
    f"got passive values: {[v for v in MAP_ORIGIN_HINTS.values() if v in _PAS_TAGS]}"
)

INCLUDE_ANCHORS = (
    "engine.h",
    "async_prefetcher.h",
    "profiling.h",
    "vram_cache.h",
)

FILES = [
    os.path.join(ROOT, "src", "engine.cpp"),
    os.path.join(ROOT, "src", "async_prefetcher.cpp"),
    os.path.join(ROOT, "src", "profiling.cpp"),
    os.path.join(ROOT, "src", "vram_cache.cpp"),
]


def strip_quoted_and_commented(src: str) -> str:
    """Replace every byte inside a C++ string literal or comment with a
    space of equal length. The result has the same indices as the input.
    """
    out = list(src)
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            while i < n and src[i] != '\n':
                out[i] = ' '
                i += 1
            continue
        if c == '/' and i + 1 < n and src[i + 1] == '*':
            out[i] = out[i + 1] = ' '
            i += 2
            while i < n and not (src[i] == '*' and i + 1 < n and src[i + 1] == '/'):
                if src[i] != '\n':
                    out[i] = ' '
                i += 1
            if i + 1 < n:
                out[i] = out[i + 1] = ' '
                i += 2
            continue
        if c == '"':
            out[i] = ' '
            i += 1
            while i < n and src[i] != '"':
                if src[i] == '\\' and i + 1 < n:
                    out[i] = out[i + 1] = ' '
                    i += 2
                else:
                    if src[i] != '\n':
                        out[i] = ' '
                    i += 1
            if i < n:
                out[i] = ' '
                i += 1
            continue
        if c == '\'':
            out[i] = ' '
            i += 1
            while i < n and src[i] != '\'':
                if src[i] == '\\' and i + 1 < n:
                    out[i] = out[i + 1] = ' '
                    i += 2
                else:
                    if src[i] != '\n':
                        out[i] = ' '
                    i += 1
            if i < n:
                out[i] = ' '
                i += 1
            continue
        i += 1
    return "".join(out)


def find_matching_paren(src: str, open_idx: int) -> int:
    """Given src[open_idx] == '(', return the index of its matching ')'."""
    depth = 0
    i = open_idx
    n = len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            while i < n and src[i] != '\n':
                i += 1
            continue
        if c == '/' and i + 1 < n and src[i + 1] == '*':
            i += 2
            while i < n and not (src[i] == '*' and i + 1 < n and src[i + 1] == '/'):
                i += 1
            i += 2 if i < n else 0
            continue
        if c == '"':
            i += 1
            while i < n and src[i] != '"':
                if src[i] == '\\' and i + 1 < n:
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        if c == '\'':
            i += 1
            while i < n and src[i] != '\'':
                if src[i] == '\\' and i + 1 < n:
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def extract_owner(args_text: str) -> str:
    """Auto-extract the FIRST pointer-like identifier from the call's args."""
    s = args_text.strip()
    if not s:
        return DEFAULT_OWNER
    for _ in range(4):
        m = re.match(
            r'\s*(const\s+)?(static|reinterpret|dynamic|explicit)\s*<[^>]+>\s*',
            s)
        if not m:
            break
        s = s[m.end():]
    s = s.lstrip()
    for _ in range(4):
        if s and s[0] in '&*':
            s = s[1:]
        else:
            break
    for _ in range(4):
        m = re.match(
            r'\s*(const\s+)?(static|reinterpret|dynamic|explicit)\s*<[^>]+>\s*',
            s)
        if not m:
            break
        s = s[m.end():]
    s = s.lstrip()
    for _ in range(4):
        if s and s[0] in '&*':
            s = s[1:]
        else:
            break
    m = re.match(r'(reinterpret|static|dynamic|const)_cast\s*<[^>]+>\s*\(', s)
    if m:
        open_p = s.find('(', m.end() - 1)
        close_p = find_matching_paren(s, open_p)
        if close_p > 0:
            return extract_owner(s[open_p + 1: close_p])
    if s.startswith('('):
        close_p = find_matching_paren(s, 0)
        if close_p > 0:
            return extract_owner(s[1: close_p])
    m = re.match(r'([A-Za-z_][A-Za-z_0-9]*)', s)
    if m:
        return m.group(1)
    return DEFAULT_OWNER


def _split_top_level_commas(text: str) -> list:
    """Split a comma-separated argument list into individual args, honoring
    ( ) [ ] < > nesting (so e.g. `staging_key(0, 1), x` becomes two parts).
    """
    depth = 0
    parts = []
    start = 0
    for k_, c in enumerate(text):
        if c in '([{<':
            depth += 1
        elif c in ')]}>':
            depth -= 1
        elif c == ',' and depth == 0:
            parts.append(text[start: k_].strip())
            start = k_ + 1
    parts.append(text[start:].strip())
    return parts


def apply_rename(src: str) -> tuple[str, int]:
    masked = strip_quoted_and_commented(src)
    out = []
    i = 0
    n = len(src)
    renames = 0
    while i < n:
        if (src[i].isalpha() or src[i] == '_') and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == '_')):
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == '_'):
                j += 1
            ident = src[i:j]
            # MILESTONE 3 v5 / FIX 1: skip identifiers entirely inside
            # a string or comment.
            if all(masked[k_] == ' ' for k_ in range(i, j)):
                out.append(src[i:j])
                i = j
                continue
            k = j
            while k < n and src[k] in ' \t':
                k += 1
            # MILESTONE 3 v5 / FIX 2: skip if opening '(' is inside a
            # string/comment (its masked byte is a space).
            opening_paren_is_real = (k < n and src[k] == '(' and
                                     k < len(masked) and masked[k] != ' ')
            if opening_paren_is_real:
                target = None
                for cuda_name, ta_name in RENAMES:
                    if ident == cuda_name:
                        target = ta_name
                        break
                if target:
                    close = find_matching_paren(masked, k)
                    if close < 0:
                        out.append(src[i:k + 1])
                        i = k + 1
                        continue
                    args_text = src[k + 1: close]
                    owner = extract_owner(args_text)
                    new_call = f'{target}({args_text}, "{owner}")'
                    # MILESTONE 3 v5 / FIX 4: emit ONLY the new call. The old
                    # v2 walker emitted both src[i:k] (`cudaFree`) AND
                    # `DEE_TA_FREE(...)`, producing `cudaFreeDEE_TA_FREE(...)`.
                    # The original `cudaFree` is consumed by advancing i.
                    out.append(new_call)
                    renames += 1
                    i = close + 1
                    continue
            out.append(src[i:j])
            i = j
            continue
        if src[i] == '#' and (i == 0 or src[i - 1] == '\n'):
            nl = src.find('\n', i)
            if nl < 0:
                nl = n
            line = src[i:nl]
            for anchor in INCLUDE_ANCHORS:
                if re.match(r'#\s*include\s*"dee/' + re.escape(anchor) + '"', line) \
                        and 'dee/trace_alloc.h' not in src:
                    out.append(line + "\n")
                    out.append('#include "dee/trace_alloc.h"  '
                               '// Milestone 3 v5 teardown-forensics sentinel\n')
                    i = nl + 1
                    break
            else:
                pass
            # The `for...else` doesn't break the outer while; we still need
            # to emit the original line. Detect whether the include line
            # matched by checking i > nl above; fall through otherwise.
            if i != nl + 1:
                out.append(line + "\n")
                i = nl + 1
                continue
            continue
        out.append(src[i])
        i += 1
    return ("".join(out), renames)


def apply_map_inserts(src: str) -> tuple[str, int]:
    """MILESTONE 3 v5 / FIX 3: emplace-aware. Detects the map reference on a
    line, then either:
      Form 1: `.emplace(<key>, <value>)` or `.insert(<key>, <value>)` --
              uses balanced-paren splitting to extract key/value, then
              prepends a DEE_TA_INSERT(...) marker line.
      Form 2: `MAP[k]` (rare for these maps) -- uses bracket-matching.
    """
    n_inserts = 0
    lines = src.splitlines(keepends=True)
    out_lines = []
    for line in lines:
        stripped = line.lstrip()
        is_marker = False
        for prefix in (
            "DEE_TA_INSERT(",
            "DEE_TA_MALLOC(",
            "DEE_TA_FREE(",
            "DEE_TA_FREE_HOST(",
            "DEE_TA_EVENT_",
            "DEE_TA_STREAM_",
            "DEE_TA_CUBLAS_",
            "// ",
            "/*",
            "*",
        ):
            if stripped.startswith(prefix):
                is_marker = True
                break
        if is_marker:
            out_lines.append(line)
            continue
        replaced = False
        for map_name in TARGET_MAPS:
            idx = line.find(map_name)
            if idx < 0:
                continue
            j = idx + len(map_name)
            while j < len(line) and line[j] in ' \t':
                j += 1
            if j >= len(line):
                continue
            # Form 1: .emplace(...) or .insert(...)
            kind_token = None
            if line.startswith('.emplace(', j):
                kind_token = '.emplace('
            elif line.startswith('.insert(', j):
                kind_token = '.insert('
            if kind_token:
                open_p = j + len(kind_token) - 1
                close_p = find_matching_paren(line, open_p)
                if close_p < 0:
                    continue
                args = line[open_p + 1: close_p]
                parts = _split_top_level_commas(args)
                if len(parts) != 2:
                    continue
                key, value = parts[0], parts[1]
                indent = line[: len(line) - len(line.lstrip())]
                #  MILESTONE 3 v5 / FIX 5: per-site origin tag from
                #  MAP_ORIGIN_HINTS (was: monolithic `"unlabeled_origin"`).
                origin_hint = MAP_ORIGIN_HINTS.get(map_name, "unknown_origin")
                marker = (f'{indent}DEE_TA_INSERT("{map_name}", '
                          f'{key}, {value}, "{origin_hint}");  '
                          f'// Milestone 3 v5: assert origin tag in post-mortem\n')
                out_lines.append(marker)
                n_inserts += 1
                replaced = True
                break
            # Form 2: operator[]= (rare; handled if encountered)
            if line[j] == '[':
                close_b = line.find(']', j)
                if close_b < 0:
                    continue
                eq = line.find('=', close_b)
                if eq < 0:
                    continue
                semi = line.find(';', eq)
                if semi < 0:
                    semi = len(line)
                key = line[j: close_b + 1]
                value = line[eq + 1: semi].strip()
                indent = line[: len(line) - len(line.lstrip())]
                #  MILESTONE 3 v5 / FIX 5: per-site origin tag from
                #  MAP_ORIGIN_HINTS (was: monolithic `"unlabeled_origin"`).
                origin_hint = MAP_ORIGIN_HINTS.get(map_name, "unknown_origin")
                marker = (f'{indent}DEE_TA_INSERT("{map_name}", '
                          f'{key}, {value}, "{origin_hint}");  '
                          f'// Milestone 3 v5: assert origin tag in post-mortem\n')
                out_lines.append(marker)
                n_inserts += 1
                replaced = True
                break
        out_lines.append(line)
    return ("".join(out_lines), n_inserts)


def main(argv):
    check_only = (len(argv) > 1 and argv[1] == 'check-only')
    summary = []
    for path in FILES:
        if not os.path.exists(path):
            summary.append(f"  MISSING: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        new_src, n_renames = apply_rename(src)
        new_src, n_inserts = apply_map_inserts(new_src)
        if check_only:
            summary.append(
                f"  {os.path.basename(path)}: {n_renames} renames + "
                f"{n_inserts} inserts (check only)")
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            summary.append(
                f"  {os.path.basename(path)}: {n_renames} renames + "
                f"{n_inserts} inserts APPLIED")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
