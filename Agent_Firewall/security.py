"""API key hashing and per-key rate limiting for the firewall proxy."""

import hashlib
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

KEY_PREFIX = "agf-live-"
HASH_PREFIX = "sha256:"


def hash_api_key(key: str) -> str:
    """Value stored in the DB. Keys are 128-bit random tokens, so plain SHA-256 is enough
    (no salt/KDF needed, unlike passwords). The last 4 chars are kept for display."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}:{key[-4:]}"


def display_key(stored: str) -> str:
    if stored.startswith(HASH_PREFIX):
        return f"{KEY_PREFIX}…{stored.rsplit(':', 1)[-1]}"
    return f"{KEY_PREFIX}…"  # legacy plaintext row, migrated at startup


def migrate_plaintext_keys(db, model) -> int:
    """Replace any plaintext keys left over from older versions with their hash."""
    migrated = 0
    for row in db.query(model).filter(model.api_key.like(f"{KEY_PREFIX}%")).all():
        row.api_key = hash_api_key(row.api_key)
        migrated += 1
    if migrated:
        db.commit()
    return migrated


class RateLimiter:
    """Sliding 60-second window per key. In-memory, so limits are per server process."""

    def __init__(self, window_s: float = 60.0):
        self.window_s = window_s
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key_id: str, limit_per_window: int) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds). limit <= 0 means unlimited."""
        if not limit_per_window or limit_per_window <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key_id]
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= limit_per_window:
                return False, max(1, int(self.window_s - (now - hits[0])) + 1)
            hits.append(now)
            return True, 0
