"""Branching set reasoning: union and intersection of KB queries.

Single-fact queries return one answer. Some questions need set
operations: "what European countries use the euro?" =
  {X | (X, locatedin, europe)} INTERSECT {X | (X, currency, euro)}

This module exposes:
  * `query_set(kb, partial_fact, unknown_role, threshold, ...)`
      Returns the SET of symbols above threshold for one constraint.
  * `intersect_queries(kb, partials, unknown_role)`
      Intersection across multiple constraints (all unknown_role
      must agree).
  * `union_queries(kb, partials, unknown_role)`
      Union -- "satisfies at least one of these".

Each constraint is a partial fact (a `dict` with all roles except
`unknown_role`).

The output is a list of `SetCandidate(symbol, scores_per_constraint,
aggregate_score)`. Scores are kept per-constraint so callers can
audit WHICH constraints each candidate satisfied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class SetCandidate:
    symbol: Hashable
    scores_per_constraint: dict[int, float] = field(default_factory=dict)
    aggregate_score: float = 0.0

    def n_satisfied(self) -> int:
        return len(self.scores_per_constraint)


def query_set(kb: ShardedKnowledgeBase,
              partial: dict[str, Hashable], unknown_role: str,
              *, top_k: int = 20, min_score: float = 0.10
              ) -> dict[Hashable, float]:
    """Return all candidates for one constraint above `min_score`."""
    out: dict[Hashable, float] = {}
    for sym, score in kb.query(partial, unknown_role, top_k=top_k):
        if float(score) >= min_score:
            out[sym] = float(score)
    return out


def intersect_queries(kb: ShardedKnowledgeBase,
                      partials: list[dict[str, Hashable]],
                      unknown_role: str,
                      *, top_k: int = 20, min_score: float = 0.10
                      ) -> list[SetCandidate]:
    """Candidates that satisfy ALL constraints.

    Aggregate score = geometric mean of per-constraint scores
    (penalises weak hits across many constraints).
    """
    if not partials:
        return []
    sets = [
        query_set(kb, p, unknown_role, top_k=top_k, min_score=min_score)
        for p in partials
    ]
    # Intersection of keys.
    common = set(sets[0])
    for s in sets[1:]:
        common &= set(s)
    out: list[SetCandidate] = []
    for sym in common:
        scores = {i: sets[i][sym] for i in range(len(sets))}
        # Geometric mean across constraints.
        import math
        log_sum = sum(math.log(max(v, 1e-9)) for v in scores.values())
        agg = math.exp(log_sum / len(scores))
        out.append(SetCandidate(
            symbol=sym, scores_per_constraint=scores,
            aggregate_score=agg,
        ))
    out.sort(key=lambda c: -c.aggregate_score)
    return out


def union_queries(kb: ShardedKnowledgeBase,
                  partials: list[dict[str, Hashable]],
                  unknown_role: str,
                  *, top_k: int = 20, min_score: float = 0.10
                  ) -> list[SetCandidate]:
    """Candidates that satisfy AT LEAST ONE constraint.

    Aggregate score = max(scores) for each candidate. The number of
    constraints satisfied is exposed via SetCandidate.n_satisfied().
    """
    if not partials:
        return []
    sets = [
        query_set(kb, p, unknown_role, top_k=top_k, min_score=min_score)
        for p in partials
    ]
    accumulated: dict[Hashable, dict[int, float]] = {}
    for i, s in enumerate(sets):
        for sym, score in s.items():
            accumulated.setdefault(sym, {})[i] = score
    out: list[SetCandidate] = []
    for sym, scores in accumulated.items():
        agg = max(scores.values()) if scores else 0.0
        out.append(SetCandidate(
            symbol=sym, scores_per_constraint=scores,
            aggregate_score=agg,
        ))
    out.sort(key=lambda c: (-c.n_satisfied(), -c.aggregate_score))
    return out


def difference_queries(kb: ShardedKnowledgeBase,
                       positive: dict[str, Hashable],
                       negative: dict[str, Hashable],
                       unknown_role: str,
                       *, top_k: int = 20, min_score: float = 0.10
                       ) -> list[SetCandidate]:
    """Candidates satisfying `positive` but NOT `negative`."""
    pos = query_set(kb, positive, unknown_role,
                    top_k=top_k, min_score=min_score)
    neg = query_set(kb, negative, unknown_role,
                    top_k=top_k, min_score=min_score)
    out: list[SetCandidate] = []
    for sym, score in pos.items():
        if sym in neg:
            continue
        out.append(SetCandidate(
            symbol=sym, scores_per_constraint={0: score},
            aggregate_score=score,
        ))
    out.sort(key=lambda c: -c.aggregate_score)
    return out
