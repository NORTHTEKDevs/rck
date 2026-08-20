"""Relation co-occurrence map.

For each pair of relations (R1, R2), count how many SUBJECTS have
facts for both R1 and R2. High-co-occurrence pairs reveal that the
KB tends to know both attributes of those subjects together --
useful for:
  * detecting clusters of "always-known-together" relations
  * priming chain discovery (R1 -> R2 is a likely path)
  * spotting opportunities for hierarchical abstraction
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class RelationPair:
    r1: str
    r2: str
    co_occurrence: int            # subjects that have both R1 and R2
    r1_only: int = 0              # subjects with R1 but not R2
    r2_only: int = 0              # subjects with R2 but not R1

    @property
    def jaccard(self) -> float:
        total = self.co_occurrence + self.r1_only + self.r2_only
        return self.co_occurrence / max(1, total)


@dataclass
class CooccurrenceMap:
    pairs: list[RelationPair] = field(default_factory=list)
    n_subjects: int = 0

    def top_pairs(self, n: int = 10) -> list[RelationPair]:
        return sorted(
            self.pairs,
            key=lambda p: (-p.co_occurrence, -p.jaccard),
        )[:n]


def cooccurrence(kb: ShardedKnowledgeBase,
                  *, min_co_occurrence: int = 2,
                  ignore_negative: bool = True) -> CooccurrenceMap:
    """Walk the KB and produce a relation-pair co-occurrence map."""
    # Subject -> set of relations.
    subj_to_rels: dict[str, set[str]] = defaultdict(set)
    for fact in kb.all_facts():
        r = str(fact.get("R", "")).lower()
        if ignore_negative and r.startswith("not_"):
            continue
        if r == "isa":
            # isa is everywhere; not informative for co-occurrence.
            continue
        s = str(fact.get("S", "")).lower()
        if s:
            subj_to_rels[s].add(r)

    rel_set: defaultdict[str, set[str]] = defaultdict(set)
    for s, rels in subj_to_rels.items():
        for r in rels:
            rel_set[r].add(s)

    relations = sorted(rel_set)
    pairs: list[RelationPair] = []
    for r1, r2 in combinations(relations, 2):
        s1 = rel_set[r1]
        s2 = rel_set[r2]
        co = len(s1 & s2)
        if co < min_co_occurrence:
            continue
        pairs.append(RelationPair(
            r1=r1, r2=r2,
            co_occurrence=co,
            r1_only=len(s1 - s2),
            r2_only=len(s2 - s1),
        ))
    return CooccurrenceMap(
        pairs=pairs,
        n_subjects=len(subj_to_rels),
    )
