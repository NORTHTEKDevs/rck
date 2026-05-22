"""Negative facts -- explicit (X, NOT_R, Y).

A negative fact says "we KNOW X is NOT Y under relation R." This is
structurally different from "we don't know" (handled by IDK):
negation is positive certainty about non-membership.

Encoding: a negative fact uses the same KB substrate but with a
`NOT_` prefix on the relation. So `(fish, NOT_isa, vegetable)` is
stored as `kb.store({"S": "fish", "R": "not_isa", "O": "vegetable"})`.

This module provides:
  * `deny(kb, s, r, o)` -- store a negative fact.
  * `negate(relation)` / `denegate(relation)` -- convert between
    positive and negative forms.
  * `is_negative(relation)` -- predicate.
  * `denied_pairs_for(kb, s, r)` -- list explicit denials for (s, r, ?).
  * `filter_against_negatives(kb, s, r, candidates)` -- given a
    candidate list for (s, r, ?), remove answers that are explicitly
    denied by stored negative facts.

The contradiction detector is extended elsewhere to flag the case
where (X, R, Y) and (X, NOT_R, Y) are both stored.
"""
from __future__ import annotations

from typing import Hashable, Iterable

from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


NEGATION_PREFIX: str = "not_"


def is_negative(relation: str) -> bool:
    return relation.lower().startswith(NEGATION_PREFIX)


def negate(relation: str) -> str:
    r = relation.lower()
    if r.startswith(NEGATION_PREFIX):
        return r
    return NEGATION_PREFIX + r


def denegate(relation: str) -> str:
    r = relation.lower()
    if r.startswith(NEGATION_PREFIX):
        return r[len(NEGATION_PREFIX):]
    return r


def deny(kb: ShardedKnowledgeBase, subject: str, relation: str, obj: str,
         *, provenance: ProvenanceStore | None = None) -> None:
    """Store the negative fact `(subject, NOT_relation, obj)`."""
    s = subject.lower()
    r = negate(relation)
    o = obj.lower()
    kb.store({"S": s, "R": r, "O": o})
    if provenance is not None:
        provenance.store(s, r, o, source="user", tags={"negative"})


def denied_pairs_for(kb: ShardedKnowledgeBase, subject: str, relation: str,
                      *, top_k: int = 10, min_score: float = 0.10
                      ) -> list[tuple[Hashable, float]]:
    """Return all (object, score) pairs for which `(subject, NOT_R, object)`
    is stored above `min_score`."""
    s = subject.lower()
    neg_r = negate(relation)
    raw = kb.query({"S": s, "R": neg_r}, "O", top_k=top_k)
    return [(sym, float(score)) for sym, score in raw
            if float(score) >= min_score]


def filter_against_negatives(kb: ShardedKnowledgeBase,
                              subject: str, relation: str,
                              candidates: Iterable[tuple[Hashable, float]],
                              *, min_denial_score: float = 0.10
                              ) -> list[tuple[Hashable, float]]:
    """Given a list of candidate (object, score) pairs for
    `(subject, relation, ?)`, drop any whose object is explicitly
    denied by a stored negative fact.

    The denied-set is collected once via `denied_pairs_for` so this
    is O(k) per candidate vs O(k * |candidates|).
    """
    denied_pairs = denied_pairs_for(
        kb, subject, relation, min_score=min_denial_score,
    )
    denied_objs = {str(sym).lower() for sym, _ in denied_pairs}
    if not denied_objs:
        return list(candidates)
    return [(sym, score) for sym, score in candidates
            if str(sym).lower() not in denied_objs]
