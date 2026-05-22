"""Active gap detection.

For a given subject, find which relations are COMMONLY known about
its peers (entities that share its `isa` parent) but are MISSING
for the subject. Surfaces what the user could helpfully tell the
agent next.

Algorithm:
  1. Find the subject's isa parent(s).
  2. Find sibling subjects with the same parent.
  3. Collect every relation those siblings have facts for.
  4. For each common relation R, check whether the subject already
     has a fact `(subject, R, ?)` above threshold.
  5. Emit a `Gap` for every R where the subject has no record.

Each Gap carries the relation, how many siblings have it (support),
and the most common object value across siblings (a hint for the
user).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class Gap:
    subject: str
    relation: str
    sibling_support: int           # how many siblings have this relation
    common_objects: list[tuple[str, int]] = field(default_factory=list)
    suggested_question: str = ""

    def verbalize(self) -> str:
        return self.suggested_question or (
            f"What is the {self.relation} of {self.subject}?"
        )


def _siblings_via_isa(kb: ShardedKnowledgeBase, subject: str,
                      max_siblings: int = 25,
                      min_score: float = 0.10
                      ) -> list[str]:
    """Return entities sharing the subject's isa parent."""
    subj = subject.lower()
    parents = kb.query({"S": subj, "R": "isa"}, "O",
                        top_k=3)
    parents = [str(p).lower() for p, s in parents if float(s) >= min_score]
    if not parents:
        return []
    siblings: set[str] = set()
    for parent in parents:
        peers = kb.query({"R": "isa", "O": parent}, "S",
                          top_k=max_siblings)
        for sym, score in peers:
            if float(score) < min_score:
                continue
            cand = str(sym).lower()
            if cand != subj:
                siblings.add(cand)
    return list(siblings)


def _relations_with_facts(kb: ShardedKnowledgeBase,
                           subjects: Iterable[str]
                           ) -> Counter:
    """Count which relations each subject has at least one fact for."""
    subject_set = {s.lower() for s in subjects}
    counts: Counter = Counter()
    for shard in kb._shards:
        for fact in shard.facts():
            s = str(fact.get("S", "")).lower()
            if s not in subject_set:
                continue
            r = str(fact.get("R", "")).lower()
            if r.startswith("not_"):
                continue
            if r == "isa":
                continue
            counts[r] += 1
    return counts


def _has_fact_for(kb: ShardedKnowledgeBase, subject: str, relation: str,
                  *, min_score: float = 0.10) -> bool:
    rows = kb.query({"S": subject, "R": relation}, "O", top_k=3)
    return any(float(score) >= min_score for _, score in rows)


def _common_objects_for(kb: ShardedKnowledgeBase, subjects: Iterable[str],
                         relation: str) -> Counter:
    """Across the sibling subjects, which OBJECT values are most common
    for `relation`?"""
    subj_set = {s.lower() for s in subjects}
    counter: Counter = Counter()
    for shard in kb._shards:
        for fact in shard.facts():
            if str(fact.get("S", "")).lower() not in subj_set:
                continue
            if str(fact.get("R", "")).lower() != relation:
                continue
            counter[str(fact.get("O", "")).lower()] += 1
    return counter


def find_gaps(kb: ShardedKnowledgeBase, subject: str,
              *, min_sibling_support: int = 2,
              max_gaps: int = 10) -> list[Gap]:
    """Return the relations that the subject's peers have but the
    subject is missing.

    Ranked by sibling_support descending (the higher the support,
    the more peer-shared the relation is).
    """
    subj = subject.lower()
    siblings = _siblings_via_isa(kb, subj)
    if not siblings:
        return []
    rel_counts = _relations_with_facts(kb, siblings)
    gaps: list[Gap] = []
    for relation, support in rel_counts.most_common():
        if support < min_sibling_support:
            continue
        if _has_fact_for(kb, subj, relation):
            continue
        common_objs_counter = _common_objects_for(kb, siblings, relation)
        common_objs = common_objs_counter.most_common(3)
        suggestion = _question_for(subj, relation, common_objs)
        gaps.append(Gap(
            subject=subj, relation=relation,
            sibling_support=support,
            common_objects=common_objs,
            suggested_question=suggestion,
        ))
        if len(gaps) >= max_gaps:
            break
    return gaps


def _question_for(subject: str, relation: str,
                   common_objects: list[tuple[str, int]]) -> str:
    if not common_objects:
        return f"What is the {relation} of {subject}?"
    if relation in {"has", "haspart", "partof", "madeof", "contains",
                     "color", "size", "kind", "category", "locatedin"}:
        examples = ", ".join(o for o, _ in common_objects[:2])
        return (f"What is the {relation} of {subject}? "
                f"(its peers include {examples})")
    return f"What is the {relation} of {subject}?"
