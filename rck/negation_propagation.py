"""Transitive propagation of negative facts.

If `(mammal, NOT_has, feathers)` is asserted AND `(cat, isa, mammal)`
holds, the implied fact `(cat, NOT_has, feathers)` follows by the
SAME logic that lets `(cat, has, fur)` follow from
`(mammal, has, fur)` for positive lifting relations.

This module scans the KB for stored negative facts whose subject
is a CLASS (something other entities are `isa` of), and propagates
the denial to every member. The propagation is gated by the same
lifting-relation logic as positive induction: we only propagate
through `isa`, `partof`, `locatedin`, `memberof`, etc.

The output is a list of newly stored negative facts, each tagged
with provenance `source="rule"` and `tags={"negative_propagated"}`.
Re-verification is light (just check the new fact wasn't already
positively asserted, which would be a contradiction).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rck.chain_induction import InductionPolicy
from rck.knowledge_base import ShardedKnowledgeBase
from rck.negative_facts import (
    NEGATION_PREFIX, denegate, is_negative, negate,
)
from rck.provenance import ProvenanceStore


@dataclass
class PropagatedNegation:
    subject: str          # the member entity
    relation: str         # the NEGATIVE relation (e.g. "not_has")
    obj: str              # the denied object
    parent: str           # the class the negation came from
    via_relation: str     # the lifting relation (e.g. "isa")
    stored: bool = False
    rejected_reason: str | None = None


def propagate_negations(kb: ShardedKnowledgeBase,
                        *, policy: InductionPolicy | None = None,
                        provenance: ProvenanceStore | None = None,
                        min_link_score: float = 0.10,
                        max_facts: int = 500,
                        ) -> list[PropagatedNegation]:
    """Find every stored (CLASS, NOT_R, OBJ) and propagate to every
    (MEMBER, NOT_R, OBJ) where (MEMBER lifting-relation CLASS) holds.

    Only LIFTING relations (isa, partof, locatedin, ...) are followed.
    """
    cfg = policy or InductionPolicy()
    out: list[PropagatedNegation] = []
    seen: set[tuple[str, str, str]] = set()

    # Find every stored negative fact.
    negative_facts: list[tuple[str, str, str]] = []
    for shard in kb._shards:
        for fact in shard.facts():
            r = str(fact.get("R", "")).lower()
            if is_negative(r):
                negative_facts.append((
                    str(fact.get("S", "")).lower(),
                    r,
                    str(fact.get("O", "")).lower(),
                ))

    # For each negative, find members of that subject via lifting relations.
    for parent, neg_r, obj in negative_facts:
        for lifting_r in cfg.lifting_relations:
            # Query (?, lifting_r, parent) -> member candidates.
            members = kb.query(
                {"R": lifting_r, "O": parent}, "S", top_k=10,
            )
            for sym, score in members:
                if float(score) < min_link_score:
                    continue
                member = str(sym).lower()
                if member == parent:
                    continue
                key = (member, neg_r, obj)
                if key in seen:
                    continue
                seen.add(key)
                # Skip if the negative already exists.
                existing = kb.query({"S": member, "R": neg_r}, "O", top_k=5)
                if any(str(s_).lower() == obj
                        and float(sc) >= min_link_score
                        for s_, sc in existing):
                    continue
                # Skip if the POSITIVE is asserted (contradiction guard).
                positive_r = denegate(neg_r)
                pos_candidates = kb.query(
                    {"S": member, "R": positive_r}, "O", top_k=5,
                )
                pos_match = any(
                    str(s_).lower() == obj
                    and float(sc) >= min_link_score
                    for s_, sc in pos_candidates
                )
                pn = PropagatedNegation(
                    subject=member, relation=neg_r, obj=obj,
                    parent=parent, via_relation=lifting_r,
                )
                if pos_match:
                    pn.rejected_reason = (
                        f"positive {member} {positive_r} {obj} already stored"
                    )
                    out.append(pn)
                    continue
                # Commit.
                kb.store({"S": member, "R": neg_r, "O": obj})
                pn.stored = True
                out.append(pn)
                if provenance is not None:
                    provenance.store(
                        member, neg_r, obj,
                        source="rule",
                        tags={"negative_propagated", f"from_{parent}",
                              f"via_{lifting_r}"},
                        derivation=[
                            (member, lifting_r, parent),
                            (parent, neg_r, obj),
                        ],
                    )
                if len(out) >= max_facts:
                    return out
    return out
