from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    """Small in-process limiter for a public demo endpoint.

    It is deliberately conservative and has no external dependency. For a
    multi-instance deployment, replace it with a shared Redis/KV limiter.
    """

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0, max_clients: int = 10_000):
        self.max_requests = max(1, max_requests)
        self.window_seconds = max(1.0, window_seconds)
        self.max_clients = max(100, max_clients)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client_id: str) -> bool:
        now = time.monotonic()
        key = client_id or "anonymous"
        with self._lock:
            events = self._events[key]
            cutoff = now - self.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.max_requests:
                return False
            events.append(now)
            if len(self._events) > self.max_clients:
                oldest_key = min(self._events, key=lambda candidate: self._events[candidate][0] if self._events[candidate] else now)
                self._events.pop(oldest_key, None)
            return True


_SECRET_PATTERNS = [
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(api-subscription-key\s*[:=]\s*)\S+"), r"\1<redacted>"),
    (re.compile(r"\b(?:sk|gsk|hf)_[A-Za-z0-9_-]+\b"), "<redacted>"),
    (re.compile(r"(?i)(qdrant[_ -]?api[_ -]?key\s*[:=]\s*)\S+"), r"\1<redacted>"),
]


def sanitize_error(error: object, limit: int = 500) -> str:
    """Redact known credentials before writing an error to a trusted log."""
    text = str(error or "request failed")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text[:limit]


def public_error(code: str) -> str:
    """Return a stable error code; never expose provider exception details."""
    return code
