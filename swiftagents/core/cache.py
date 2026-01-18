from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self, *, max_size: int = 256, ttl_s: float = 600.0) -> None:
        self._max_size = max_size
        self._ttl_s = ttl_s
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        entry = self._entries.get(key)
        if entry is None:
            self._stats.misses += 1
            return None
        if entry.expires_at < now:
            self._entries.pop(key, None)
            self._stats.misses += 1
            return None
        self._entries.move_to_end(key)
        self._stats.hits += 1
        return entry.value

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = CacheEntry(value=value, expires_at=now + self._ttl_s)
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)
            self._stats.evictions += 1

    def stats(self) -> CacheStats:
        return self._stats

    def clear(self) -> None:
        self._entries.clear()


def make_cache_key(parts: Dict[str, Any]) -> str:
    items = sorted((k, str(v)) for k, v in parts.items())
    return "|".join(f"{k}={v}" for k, v in items)
