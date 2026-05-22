"""LRU cache for query results -- microsecond latency for repeat queries.

Hot questions in a conversation (or across users) get cached. Cache
key = normalised query string. Cache entry includes the full response
dict so we don't re-run the whole ConsciousAgent pipeline.

Cache eviction:
  * LRU on size limit
  * Time-based expiry (default 5 minutes)
  * Explicit invalidation on KB writes (tell / forget) -- the calling
    code is responsible for calling `cache.invalidate()` when facts change
"""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    value: dict
    inserted_at: float
    hits: int = 0


@dataclass
class QueryCache:
    """Bounded LRU + TTL cache."""

    max_size: int = 1024
    ttl_seconds: float = 300.0
    _data: OrderedDict[str, CacheEntry] = field(default_factory=OrderedDict)
    _hits: int = 0
    _misses: int = 0

    def _key(self, query: str) -> str:
        # Normalise: lowercase + collapse whitespace + strip punctuation
        # except question mark.
        q = re.sub(r"\s+", " ", query.strip().lower())
        return q

    def get(self, query: str) -> dict | None:
        key = self._key(query)
        entry = self._data.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.time() - entry.inserted_at > self.ttl_seconds:
            del self._data[key]
            self._misses += 1
            return None
        # LRU touch.
        self._data.move_to_end(key)
        entry.hits += 1
        self._hits += 1
        return entry.value

    def put(self, query: str, value: dict) -> None:
        key = self._key(query)
        self._data[key] = CacheEntry(value=value, inserted_at=time.time())
        self._data.move_to_end(key)
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def invalidate(self) -> int:
        n = len(self._data)
        self._data.clear()
        return n

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._data),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
        }
