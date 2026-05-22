"""Multi-agent consensus.

Given several agents (e.g. one per domain, or per training run), this
module aggregates their answers to a single query by:

  * MAJORITY voting on top_symbol
  * CONFIDENCE-WEIGHTED scoring (sum of top_score per candidate)
  * BOTH (default): tie-break majority by confidence

Each agent's `ask_with_idk` result is folded in. Agents that answer
IDK don't vote.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rck.conscious_agent import ConsciousAgent


@dataclass
class ConsensusVote:
    symbol: str
    votes: int = 0
    total_score: float = 0.0
    contributors: list[str] = field(default_factory=list)


@dataclass
class ConsensusResult:
    chosen: str | None
    chosen_votes: int
    chosen_score: float
    candidates: list[ConsensusVote] = field(default_factory=list)
    n_voters: int = 0
    n_abstain: int = 0


def majority(agents: list["ConsciousAgent"],
             known: dict, unknown_role: str,
             *, mode: str = "both") -> ConsensusResult:
    """Run `ask_with_idk` on every agent and aggregate results.

    Mode:
      * "majority"    -- highest vote count wins
      * "confidence"  -- highest sum of top_score wins
      * "both"        -- majority first, ties broken by confidence

    Returns ConsensusResult with the chosen symbol and all candidates.
    """
    by_symbol: dict[str, ConsensusVote] = defaultdict(
        lambda: ConsensusVote(symbol="")
    )
    abstain = 0
    for i, ag in enumerate(agents):
        res = ag.ask_with_idk(known, unknown_role)
        if res.top_symbol is None or res.state.value == "idk":
            abstain += 1
            continue
        sym = str(res.top_symbol).lower()
        vote = by_symbol[sym]
        if not vote.symbol:
            vote.symbol = sym
        vote.votes += 1
        vote.total_score += float(res.top_score)
        vote.contributors.append(f"agent_{i}")

    candidates = sorted(
        by_symbol.values(),
        key=lambda v: (-v.votes, -v.total_score),
    )
    if not candidates:
        return ConsensusResult(
            chosen=None, chosen_votes=0, chosen_score=0.0,
            candidates=[], n_voters=len(agents), n_abstain=abstain,
        )
    if mode == "confidence":
        candidates.sort(key=lambda v: -v.total_score)
    elif mode == "majority":
        candidates.sort(key=lambda v: -v.votes)
    # "both" already sorted correctly above.
    top = candidates[0]
    return ConsensusResult(
        chosen=top.symbol,
        chosen_votes=top.votes,
        chosen_score=top.total_score,
        candidates=candidates,
        n_voters=len(agents),
        n_abstain=abstain,
    )
