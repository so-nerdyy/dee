"""HF byte-range source with HTTP status accounting (stdlib only).

Same convention as tools/phase3/p3_builder.RemoteRangeSource: the absolute
file offset is 8 + header_length + data_offset, and the header length comes
from one 8-byte probe per shard. Unlike RemoteRangeSource, every HTTP error
is counted by status code, so a 429 is visible in the progress evidence
instead of being retried silently.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request

USER_AGENT = "dee-p6-aws/1.0"
RETRY_BASE_S = 2.0
RETRY_CAP_S = 300.0
MAX_ATTEMPTS = 20


def retry_delay_s(status: int | None, attempt: int, retry_after: str | None = None) -> float:
    """Seconds to wait before retry `attempt` (0-based).

    429 and 5xx back off up to RETRY_CAP_S, honoring a numeric Retry-After.
    Other failures back off briefly.
    """
    if status == 429 or (status is not None and status >= 500):
        if retry_after and retry_after.strip().isdigit():
            return min(float(retry_after), RETRY_CAP_S)
        return min(RETRY_CAP_S, 30.0 * (2 ** attempt))
    return min(60.0, RETRY_BASE_S * (2 ** attempt))


class CountingRangeSource:
    def __init__(self, *, repository: str, revision: str, max_attempts: int = MAX_ATTEMPTS):
        self.repository = repository
        self.revision = revision
        self.max_attempts = max_attempts
        self._header_len: dict[str, int] = {}
        self._lock = threading.Lock()
        self.stats = {"requests": 0, "bytes": 0, "retries": 0}
        self.status_counts: dict[str, int] = {}

    def _url(self, shard: str) -> str:
        return (f"https://huggingface.co/{self.repository}/resolve/"
                f"{self.revision}/{shard}")

    def _count(self, key: str, field: str = "") -> None:
        with self._lock:
            if field:
                self.stats[field] += 1
            else:
                self.status_counts[key] = self.status_counts.get(key, 0) + 1

    def _range(self, url: str, start: int, end: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=300) as resp:
                    if resp.status != 206:
                        raise RuntimeError(
                            f"server did not honor Range (status {resp.status})")
                    data = resp.read()
                self._count("", "requests")
                with self._lock:
                    self.stats["bytes"] += len(data)
                return data
            except urllib.error.HTTPError as exc:
                self._count(str(exc.code))
                last = exc
                delay = retry_delay_s(exc.code, attempt, exc.headers.get("Retry-After"))
            except Exception as exc:  # noqa: BLE001
                self._count("exception")
                last = exc
                delay = retry_delay_s(None, attempt)
            self._count("", "retries")
            time.sleep(delay)
        raise ConnectionError(f"range fetch failed: {last!r}")

    def _hlen(self, shard: str) -> int:
        if shard not in self._header_len:
            raw = self._range(self._url(shard), 0, 7)
            if len(raw) != 8:
                raise RuntimeError(f"{shard}: bad prefix")
            hlen = int.from_bytes(raw, "little")
            if hlen <= 0 or hlen > (1 << 31):
                raise RuntimeError(f"{shard}: implausible header length {hlen}")
            self._header_len[shard] = hlen
        return self._header_len[shard]

    def fetch(self, shard: str, data_offset: int, nbytes: int) -> bytes:
        absolute = 8 + self._hlen(shard) + data_offset
        return self._range(self._url(shard), absolute, absolute + nbytes - 1)
