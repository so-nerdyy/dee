#!/usr/bin/env python3
"""Lossless codec battery for dee4 bank records (research only).

Every codec entry: compress(bytes) -> bytes, decompress -> bytes, with
byte-exact roundtrip verification done by the caller (scan.py) via sha256.
Entropy estimators bound what any entropy coder (incl. ANS) can attain.
All codecs here preserve per-record independence and random expert access;
anything needing global state (dictionaries) is measured separately.
"""

from __future__ import annotations

import time

import lz4.frame
import numpy as np
import zstandard as zstd

REGIONS: dict[str, tuple[int, int]] = {
    "w1w": (0, 4194304),
    "w3w": (4194304, 4194304),
    "w2w": (8388608, 4194304),
    "w1s": (12582912, 262144),
    "w3s": (12845056, 262144),
    "w2s": (13107200, 262144),
    "whole": (0, 13369344),
}
PACKED = ("w1w", "w3w", "w2w")
SCALES = ("w1s", "w3s", "w2s")


def _timed(fn, blob: bytes, reps: int = 2):
    best = None
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(blob)
        dt = (time.perf_counter() - t0) * 1000.0
        if best is None or dt < best:
            best = dt
    return out, best


def c_lz4_fast(blob: bytes):
    return _timed(lambda b: lz4.frame.compress(
        b, compression_level=0, block_size=lz4.frame.BLOCKSIZE_MAX1MB), blob)


def c_lz4_hc(blob: bytes):
    return _timed(lambda b: lz4.frame.compress(
        b, compression_level=16, block_size=lz4.frame.BLOCKSIZE_MAX4MB), blob)


def d_lz4(blob: bytes):
    return _timed(lz4.frame.decompress, blob)


def _zstd_cctx(level: int):
    return zstd.ZstdCompressor(level=level,
                               write_content_size=True).compress


def c_zstd(level: int):
    def go(blob: bytes):
        return _timed(_zstd_cctx(level), blob)
    return go


def d_zstd(blob: bytes):
    return _timed(zstd.ZstdDecompressor().decompress, blob)


def nibble_split(blob: bytes) -> tuple[bytes, bytes]:
    """FP4-packed bytes -> (low-nibble stream, high-nibble stream).

    Low nibble holds the even-index element, high nibble the odd-index
    element (cuda_convert.cu). Splitting lets entropy coders exploit the
    different statistics of the two positions.
    """
    a = np.frombuffer(blob, dtype=np.uint8)
    return ((a & 0x0F).tobytes(), ((a >> 4) & 0x0F).tobytes())


def order0_bits_per_byte(blob: bytes) -> float:
    a = np.frombuffer(blob, dtype=np.uint8)
    hist = np.bincount(a, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def order1_bits_per_byte(blob: bytes) -> float:
    """Conditional entropy H(x_i | x_{i-1}): attainable bound for an
    order-1 context entropy coder (excludes model cost)."""
    a = np.frombuffer(blob, dtype=np.uint8).astype(np.uint32)
    if len(a) < 2:
        return 0.0
    joint = np.zeros(65536, dtype=np.float64)
    np.add.at(joint, (a[:-1] << 8) | a[1:], 1)
    joint /= joint.sum()
    ctx = joint.reshape(256, 256).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = joint.reshape(256, 256) / ctx[:, None]
        terms = np.where(joint.reshape(256, 256) > 0,
                         joint.reshape(256, 256) * np.log2(cond), 0.0)
    return float(-np.nansum(terms))


def rans_encode_size(blob: bytes) -> tuple[int, float]:
    """Real rANS encode (constriction, order-0 categorical from own
    histogram). Model cost (~1 KiB) excluded; labeled as bound."""
    import constriction
    a = np.frombuffer(blob, dtype=np.uint8)
    hist = np.bincount(a, minlength=256)
    probs = (hist / hist.sum()).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(
        probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    t0 = time.perf_counter()
    encoder.encode_reverse(np.ascontiguousarray(a, dtype=np.int32),
                           entropy_model)
    # get_compressed() returns u32 words; exact size via valid bits.
    dt = (time.perf_counter() - t0) * 1000.0
    return (encoder.num_valid_bits() + 7) // 8, dt


def nibble_stream_stats(blob: bytes) -> dict:
    """Nibble-level structure of FP4-packed bytes. Returns order-0 bits
    per nibble for lo/hi streams, conditional H(hi|lo) and H(lo|prev_hi),
    and per-bit-position binary entropy of nibbles (bit3=sign for E2M1).
    All are attainable bounds for nibble-aware entropy coders."""
    a = np.frombuffer(blob, dtype=np.uint8)
    lo = (a & 0x0F).astype(np.uint32)
    hi = ((a >> 4) & 0x0F).astype(np.uint32)

    def h0(x: np.ndarray, depth: int) -> float:
        p = np.bincount(x, minlength=depth).astype(np.float64)
        p = p / p.sum()
        nz = p[p > 0]
        return float(-(nz * np.log2(nz)).sum())

    def hcond(x: np.ndarray, ctx: np.ndarray, depth: int) -> float:
        joint = np.zeros((depth, depth))
        np.add.at(joint, (ctx, x), 1)
        joint /= joint.sum()
        c = joint.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = np.where(c > 0, joint / c, 0.0)
            t = np.where(joint > 0, joint * np.log2(cond), 0.0)
        return float(-np.nansum(t))

    nib = np.concatenate([lo, hi]).astype(np.uint32)
    bit_h = {}
    for b in range(4):
        p1 = float(((nib >> b) & 1).mean())
        if p1 in (0.0, 1.0):
            bit_h[f"bit{b}"] = 0.0
        else:
            bit_h[f"bit{b}"] = float(
                -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1)))
    return {"lo_bits_per_nibble": h0(lo, 16),
            "hi_bits_per_nibble": h0(hi, 16),
            "h_hi_given_lo": hcond(hi, lo, 16),
            "h_lo_given_prev_hi": hcond(
                lo[1:], hi[:-1], 16) if len(lo) > 1 else 0.0,
            "bit_plane_binary_entropy": bit_h}


def zero_stats(blob: bytes) -> dict:
    a = np.frombuffer(blob, dtype=np.uint8)
    z = int((a == 0).sum())
    # longest zero run via diff on nonzero positions
    nz = np.flatnonzero(a)
    if len(nz) == 0:
        longest = len(a)
    else:
        gaps = np.diff(nz, prepend=-1, append=len(a)) - 1
        longest = int(gaps.max())
    return {"zero_frac": z / len(a), "longest_zero_run": longest,
            "distinct_bytes": int((np.bincount(a, minlength=256) > 0).sum())}
