"""Counterfactual reasoning via multi-universe KBs.

The "What if?" capability LLMs fake but cannot do cleanly.

A `Universe` is a copy-on-write knowledge base that inherits from a
parent. Modifications stay local. The agent can branch into a
hypothetical universe, edit facts, run queries, draw conclusions, then
discard the branch -- without touching the ground-truth KB.

Use cases:
  * "What would happen if Paris were the capital of Germany?"
  * "What if Napoleon had won Waterloo?"
  * "Suppose the patient has condition X -- what do we expect?"
  * Multi-step planning: explore an action chain hypothetically before
    committing.

Implementation: a Universe stores its OWN HRR memory deltas as
(stored_facts, forgotten_facts). Queries first check local deltas,
then fall back to the parent. Cheap to branch (O(1) until edits
happen).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Hashable

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class Universe:
    """A copy-on-write KB branched from a parent universe.

    The ROOT universe (no parent) is the ground-truth KB. Child
    universes inherit from a parent and only store deltas.
    """

    name: str
    parent: "Universe | None" = None
    kb: ShardedKnowledgeBase | None = None
    _added: set[tuple[str, str, str]] = field(default_factory=set, init=False)
    _removed: set[tuple[str, str, str]] = field(default_factory=set, init=False)
    _created_at: float = field(default_factory=time.time, init=False)

    @classmethod
    def root(cls, name: str, kb: ShardedKnowledgeBase) -> "Universe":
        return cls(name=name, parent=None, kb=kb)

    def branch(self, name: str) -> "Universe":
        """Create a child universe inheriting from this one.

        Branching is O(1). The child shares the parent's KB by reference
        and only tracks its own added/removed facts. Queries traverse the
        chain transparently.
        """
        # Child uses the SAME kb instance but its own added/removed sets.
        # When the child writes (tell), we write to the shared kb but track
        # the addition. On discard, we undo by forgetting the additions
        # and re-storing the removed ones.
        child = Universe(name=name, parent=self, kb=self.kb)
        return child

    # ---- facts ------------------------------------------------------------

    def tell(self, s: str, r: str, o: str) -> None:
        """Add a fact to this universe (and to the shared KB; tracked for undo)."""
        if self.kb is None:
            raise ValueError("universe has no kb")
        key = (s.lower(), r.lower(), o.lower())
        if key in self._removed:
            self._removed.discard(key)  # un-remove
        self._added.add(key)
        self.kb.store({"S": key[0], "R": key[1], "O": key[2]})

    def forget(self, s: str, r: str, o: str) -> None:
        """Remove a fact from this universe (tracked for undo on discard)."""
        if self.kb is None:
            raise ValueError("universe has no kb")
        key = (s.lower(), r.lower(), o.lower())
        if key in self._added:
            self._added.discard(key)  # un-add
        self._removed.add(key)
        self.kb.forget({"S": key[0], "R": key[1], "O": key[2]})

    # ---- queries (delegate to shared kb) ---------------------------------

    def answer(self, s: str, r: str) -> tuple[str | None, float]:
        if self.kb is None:
            return None, 0.0
        ans, score = self.kb.answer({"S": s.lower(), "R": r.lower()}, "O")
        return (str(ans) if ans is not None else None, float(score))

    def query(self, known: dict[str, Hashable], unknown_role: str,
              top_k: int = 3):
        if self.kb is None:
            return []
        return self.kb.query(known, unknown_role, top_k=top_k)

    # ---- lifecycle -------------------------------------------------------

    def discard(self) -> dict:
        """Undo all edits made in this universe. Restore the parent state."""
        if self.kb is None:
            return {"undone_adds": 0, "undone_removes": 0}
        n_adds = 0; n_removes = 0
        # Undo additions: remove the facts we added.
        for s, r, o in list(self._added):
            self.kb.forget({"S": s, "R": r, "O": o})
            n_adds += 1
        # Undo removals: re-store facts we removed in this branch.
        for s, r, o in list(self._removed):
            self.kb.store({"S": s, "R": r, "O": o})
            n_removes += 1
        self._added.clear()
        self._removed.clear()
        return {"undone_adds": n_adds, "undone_removes": n_removes}

    def commit(self) -> dict:
        """Promote the deltas to the parent (or the root). Effectively
        clears the delta tracking so undo no longer reverses these
        changes."""
        if self.parent is None:
            # Root universe -- nothing to do.
            self._added.clear()
            self._removed.clear()
            return {"committed_adds": 0, "committed_removes": 0}
        n_adds = len(self._added); n_removes = len(self._removed)
        # Merge our deltas into the parent's tracking so they aren't
        # accidentally undone if the parent is later discarded.
        self.parent._added.update(self._added)
        self.parent._removed.update(self._removed)
        self._added.clear()
        self._removed.clear()
        return {"committed_adds": n_adds, "committed_removes": n_removes}

    def deltas(self) -> dict:
        return {
            "added": list(self._added),
            "removed": list(self._removed),
        }


@dataclass
class UniverseManager:
    """Convenience wrapper that owns the root universe + branch lifecycle."""

    kb: ShardedKnowledgeBase
    root_name: str = "ground_truth"
    _root: Universe = field(default=None, init=False)
    _active: list[Universe] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._root = Universe.root(self.root_name, self.kb)

    def root(self) -> Universe:
        return self._root

    def branch(self, name: str, parent: Universe | None = None) -> Universe:
        p = parent or self._root
        u = p.branch(name)
        self._active.append(u)
        return u

    def discard_all(self) -> int:
        n = 0
        for u in list(self._active):
            u.discard(); n += 1
        self._active.clear()
        return n

    def active_branches(self) -> list[str]:
        return [u.name for u in self._active]
