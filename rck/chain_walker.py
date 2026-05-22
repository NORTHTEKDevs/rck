"""N-hop chain reasoning over the sharded KB.

`multistep.two_step` handles fixed 2-hop patterns parsed from English.
This module is the general substrate: given a start entity and a chain
of (relation, direction) edges, walk the KB hop by hop, propagating
confidence through `rck.confidence_propagation`.

A chain edge is a `Hop`:
  Hop(relation="capital", direction="forward")    # (entity, capital, ?)
  Hop(relation="locatedin", direction="reverse")  # (?, locatedin, entity)

This lets us answer:
  "What is the continent of the capital of france?"
    chain: [Hop("capital"), Hop("locatedin")]

  "What did the author of the great gatsby write before that?"
    chain: [Hop("author", direction="reverse"), Hop("wrote_before")]

  "Where does the CEO of the company that owns X live?"
    chain: [Hop("owns", direction="reverse"), Hop("ceo"), Hop("lives_in")]

The walker can ALSO record a successful chain pattern into a
`SkillLibrary` so the same chain shape is recognised in the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rck.confidence_propagation import (
    PropagationConfig, propagate, verbalize_chain_confidence,
)
from rck.knowledge_base import ShardedKnowledgeBase
from rck.skills import SkillLibrary


@dataclass
class Hop:
    """One edge in a reasoning chain."""
    relation: str
    direction: str = "forward"   # "forward" or "reverse"
    min_score: float = 0.10      # abort if a hop falls below this


@dataclass
class ChainResult:
    answer: Optional[str]
    confidence: float
    hedge: str
    trace: list[tuple[str, str, str, float]]  # (s, r, o, score) per hop
    aborted_at: int = -1                       # hop index that broke the chain

    def verbalize(self) -> str:
        if self.answer is None:
            if self.trace:
                last_s = self.trace[-1][0]
                return (f"I traced as far as {last_s!r} but the next hop "
                        f"({self.aborted_at}) had no answer.")
            return "I couldn't start the chain (no first-hop answer)."
        proposition = self._proposition_phrase()
        return verbalize_chain_confidence(
            {"final_confidence": self.confidence, "hedge": self.hedge},
            base_answer=proposition,
        )

    def _proposition_phrase(self) -> str:
        if not self.trace:
            return str(self.answer)
        parts: list[str] = []
        for s, r, o, _ in self.trace:
            parts.append(f"the {r} of {s} is {o}")
        return ", and ".join(parts) + "."


def walk_chain(kb: ShardedKnowledgeBase, start: str, hops: list[Hop],
               *, config: PropagationConfig | None = None,
               skills: SkillLibrary | None = None) -> ChainResult:
    """Execute an N-hop chain.

    On success, the result's `answer` is the last hop's object (or
    subject if the last hop is reverse). Confidence is propagated via
    `propagate()`.

    If `skills` is given, a successful chain registers the
    pattern (role, relation) into the library.
    """
    cfg = config or PropagationConfig()
    trace: list[tuple[str, str, str, float]] = []
    link_scores: list[float] = []
    current = start.lower()

    for idx, hop in enumerate(hops):
        if hop.direction == "forward":
            ans, score = kb.answer({"S": current, "R": hop.relation}, "O")
            s, r, o = current, hop.relation, str(ans) if ans is not None else ""
        elif hop.direction == "reverse":
            ans, score = kb.answer({"R": hop.relation, "O": current}, "S")
            s, r, o = str(ans) if ans is not None else "", hop.relation, current
        else:
            raise ValueError(f"unknown direction {hop.direction!r}")

        if ans is None or score < hop.min_score:
            propagated = propagate(link_scores, config=cfg) if link_scores else {
                "final_confidence": 0.0,
                "weakest_link_index": idx,
                "hedge": "unknown",
            }
            return ChainResult(
                answer=None,
                confidence=propagated["final_confidence"],
                hedge=propagated["hedge"],
                trace=trace,
                aborted_at=idx,
            )

        trace.append((s, r, o, float(score)))
        link_scores.append(float(score))
        current = str(ans).lower()

    propagated = propagate(link_scores, config=cfg)

    if skills is not None and trace:
        pattern = [(("O" if hop.direction == "forward" else "S"), hop.relation)
                   for hop in hops]
        skills.record_success(pattern)

    return ChainResult(
        answer=current,
        confidence=propagated["final_confidence"],
        hedge=propagated["hedge"],
        trace=trace,
    )


def try_chains(kb: ShardedKnowledgeBase, start: str,
               candidates: list[list[Hop]],
               *, config: PropagationConfig | None = None,
               skills: SkillLibrary | None = None) -> ChainResult | None:
    """Try several candidate chains, return the highest-confidence success.

    Useful when we have a `SkillLibrary` of known successful patterns
    and want to try each one in confidence order.
    """
    best: ChainResult | None = None
    for chain in candidates:
        result = walk_chain(kb, start, chain, config=config, skills=skills)
        if result.answer is None:
            continue
        if best is None or result.confidence > best.confidence:
            best = result
    return best
