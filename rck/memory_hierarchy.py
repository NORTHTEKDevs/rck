"""Memory hierarchies -- the four-tier biological model.

LLMs have ONE memory mechanism (the context window). Humans have four:

  * Working memory:    ~7 items, held for seconds (the current focus)
  * Episodic memory:   time-stamped events, "what happened"
  * Semantic memory:   general world knowledge, "what is true"
  * Procedural memory: how-to skills, sequences of actions

RCK gives each its own substrate and the consolidation pathways
between them. Working -> episodic happens automatically (recent
operations log). Episodic -> semantic happens on consolidation
(recurring events promote to general facts).

Working memory is the small set of HVs the agent is currently
"thinking about." Episodic is a time-stamped log. Semantic is the
existing ShardedKnowledgeBase. Procedural is a registry of named
programs (already partially in `rck/actions.py`).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Hashable


@dataclass
class WorkingMemoryItem:
    """A single item currently in focus."""
    content: str
    salience: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkingMemory:
    """Bounded ring buffer of items currently being reasoned over.

    Capacity = 16 by default (Miller's 7 +/- 2, generous).
    """

    capacity: int = 16
    _items: deque = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._items = deque(maxlen=self.capacity)

    def push(self, content: str, salience: float = 1.0) -> None:
        self._items.append(WorkingMemoryItem(content=content, salience=salience))

    def all(self) -> list[WorkingMemoryItem]:
        return list(self._items)

    def recent(self, n: int = 5) -> list[WorkingMemoryItem]:
        return list(self._items)[-n:]

    def clear(self) -> None:
        self._items.clear()

    def size(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
#  Episodic memory
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One thing that happened at a specific time."""
    timestamp: float
    actor: str   # "user" / "system" / external
    kind: str    # e.g. "told", "asked", "answered", "ingested"
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EpisodicMemory:
    """Time-stamped log of events. Searchable by time / actor / kind."""

    _episodes: list[Episode] = field(default_factory=list)
    max_size: int = 10_000

    def record(self, actor: str, kind: str, content: str,
               metadata: dict | None = None) -> Episode:
        ep = Episode(
            timestamp=time.time(), actor=actor, kind=kind,
            content=content, metadata=dict(metadata) if metadata else {},
        )
        self._episodes.append(ep)
        # Bounded growth: drop oldest once we exceed max_size.
        if len(self._episodes) > self.max_size:
            self._episodes = self._episodes[-self.max_size:]
        return ep

    def by_actor(self, actor: str) -> list[Episode]:
        return [e for e in self._episodes if e.actor == actor]

    def by_kind(self, kind: str) -> list[Episode]:
        return [e for e in self._episodes if e.kind == kind]

    def in_window(self, start_t: float, end_t: float) -> list[Episode]:
        return [e for e in self._episodes if start_t <= e.timestamp <= end_t]

    def recent(self, n: int = 10) -> list[Episode]:
        return self._episodes[-n:]

    def size(self) -> int:
        return len(self._episodes)


# ---------------------------------------------------------------------------
#  Procedural memory (named programs the agent has learned)
# ---------------------------------------------------------------------------

@dataclass
class Procedure:
    """A named sequence of operations the agent can invoke."""

    name: str
    description: str
    steps: list[str]  # symbolic steps: e.g. ["lookup", "filter", "render"]
    usage_count: int = 0
    success_count: int = 0

    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def record_use(self, succeeded: bool) -> None:
        self.usage_count += 1
        if succeeded:
            self.success_count += 1


@dataclass
class ProceduralMemory:
    """Named procedures the agent has learned."""

    _procs: dict[str, Procedure] = field(default_factory=dict)

    def store(self, name: str, description: str, steps: list[str]) -> Procedure:
        proc = Procedure(name=name, description=description, steps=list(steps))
        self._procs[name] = proc
        return proc

    def get(self, name: str) -> Procedure | None:
        return self._procs.get(name)

    def all(self) -> list[Procedure]:
        return list(self._procs.values())

    def by_success_rate(self, min_uses: int = 3) -> list[Procedure]:
        return sorted(
            (p for p in self._procs.values() if p.usage_count >= min_uses),
            key=lambda p: p.success_rate(),
            reverse=True,
        )


# ---------------------------------------------------------------------------
#  Consolidation: episodic -> semantic
# ---------------------------------------------------------------------------

def consolidate_episodic_to_semantic(
    episodic: EpisodicMemory,
    threshold: int = 3,
    actor: str = "user",
    kind: str = "told",
) -> list[tuple[str, int]]:
    """Find recurring patterns in episodic memory worth promoting to
    semantic memory.

    Returns a list of (content_pattern, occurrence_count) for episodes
    that have happened >= `threshold` times. The caller can decide which
    to actually promote to the semantic KB.
    """
    from collections import Counter
    counter: Counter[str] = Counter()
    for ep in episodic._episodes:
        if ep.actor != actor or ep.kind != kind:
            continue
        counter[ep.content] += 1
    return [(c, n) for c, n in counter.items() if n >= threshold]
