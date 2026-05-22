"""Memoize discovered chains.

Chain discovery is BFS over the KB -- relatively cheap but not free
(measured ~15-55ms per query in earlier studies). When the same
`(start, target)` query repeats, we can serve it from cache in O(1).

The cache is keyed by `(start, target)` and stores the relation
sequence + directions + cached confidence. On cache hit we re-walk
the chain to refresh confidence against the current KB state (the
KB may have grown since the discovery), but we skip the BFS.

Invalidation is conservative: any new fact stored in the KB might
have created a better chain, so we expose a `clear()` method that
agents call after batch ingest.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CachedChain:
    start: str
    target: str
    relations: list[str]
    directions: list[str]
    discovery_confidence: float  # confidence at the time of discovery
    hits: int = 0
    last_walked_confidence: Optional[float] = None
    kb_version_at_insert: int = 0


@dataclass
class ChainCache:
    """Bounded LRU-ish cache for discovered chains.

    Versioned: callers bump `kb_version` after bulk KB writes that
    may have created shorter chains. Cached entries inserted at an
    older version are treated as cache misses (returned as None).
    """

    max_size: int = 256
    kb_version: int = 0
    _store: OrderedDict[tuple[str, str], CachedChain] = field(
        default_factory=OrderedDict
    )

    # ---- basic ops -------------------------------------------------------

    def put(self, start: str, target: str,
            relations: list[str], directions: list[str],
            discovery_confidence: float) -> CachedChain:
        key = (start.lower(), target.lower())
        if key in self._store:
            # Refresh and move to MRU position.
            entry = self._store.pop(key)
            entry.relations = list(relations)
            entry.directions = list(directions)
            entry.discovery_confidence = float(discovery_confidence)
            entry.kb_version_at_insert = self.kb_version
            self._store[key] = entry
            return entry
        entry = CachedChain(
            start=key[0], target=key[1],
            relations=list(relations),
            directions=list(directions),
            discovery_confidence=float(discovery_confidence),
            kb_version_at_insert=self.kb_version,
        )
        self._store[key] = entry
        # Evict oldest if over capacity.
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)
        return entry

    def get(self, start: str, target: str) -> Optional[CachedChain]:
        key = (start.lower(), target.lower())
        entry = self._store.get(key)
        if entry is None:
            return None
        # Version check: entries inserted before the current kb_version
        # are treated as stale.
        if entry.kb_version_at_insert != self.kb_version:
            # Lazy-delete the stale entry so we don't keep checking it.
            self._store.pop(key, None)
            return None
        # Promote to MRU.
        self._store.move_to_end(key)
        entry.hits += 1
        return entry

    def bump_version(self) -> int:
        """Mark all currently-cached entries as stale.

        Returns the new version. Existing entries stay in the OrderedDict
        until a get() lazy-deletes them, OR until clear() / eviction.
        """
        self.kb_version += 1
        return self.kb_version

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "kb_version": self.kb_version,
            "total_hits": sum(c.hits for c in self._store.values()),
        }
