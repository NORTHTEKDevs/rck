"""Counterfactual reasoning.

A `Counterfactual` context manager lets the caller add temporary
facts to the KB, run queries, and have those facts automatically
rolled back when the context exits. Useful for "what if"
exploration without polluting permanent state.

Implementation note: the rollback uses `kb.forget()` for each fact
added during the context. Provenance entries are also removed.
The chain_cache version is bumped on enter and exit so cached
chains computed during the counterfactual don't leak into the real
KB's cache view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rck.conscious_agent import ConsciousAgent


@dataclass
class Counterfactual:
    """Context manager for transient KB additions.

    Usage:
        with agent.counterfactual([("dog", "isa", "fish")]) as cf:
            # within this block the agent thinks dogs are fish
            ans, score = agent.knowledge.answer(
                {"S": "dog", "R": "isa"}, "O"
            )
        # after exit, the temporary fact is gone
    """
    agent: "ConsciousAgent"
    facts: list[tuple[str, str, str]]
    _added: list[tuple[str, str, str]] = field(default_factory=list)
    _entered: bool = False

    def __enter__(self) -> "Counterfactual":
        for s, r, o in self.facts:
            s_l, r_l, o_l = s.lower(), r.lower(), o.lower()
            self.agent.knowledge.store({"S": s_l, "R": r_l, "O": o_l})
            if self.agent.provenance is not None:
                self.agent.provenance.store(
                    s_l, r_l, o_l, source="counterfactual",
                    tags={"counterfactual"},
                )
            self._added.append((s_l, r_l, o_l))
        if self.agent.chain_cache is not None:
            self.agent.chain_cache.bump_version()
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for s, r, o in reversed(self._added):
            self.agent.knowledge.forget({"S": s, "R": r, "O": o})
            if self.agent.provenance is not None:
                self.agent.provenance.forget(s, r, o)
        if self.agent.chain_cache is not None:
            self.agent.chain_cache.bump_version()
        self._added.clear()
        self._entered = False
