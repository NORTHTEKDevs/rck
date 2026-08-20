"""Entity similarity by relation-overlap.

Given a subject, find the most similar OTHER subjects in the KB.
Similarity is measured as the Jaccard overlap of their (R, O)
attribute sets:

    sim(a, b) = |attrs(a) intersect attrs(b)| / |attrs(a) union attrs(b)|

where `attrs(x) = {(r, o) | (x, r, o) in KB and r != "isa" and not r.startswith("not_")}`.

This is the structural analogue of word-vector similarity: two
entities are similar if they share many (R, O) pairs. Useful for:
  * "find entities like X"
  * detecting near-duplicates / alias candidates
  * suggesting siblings even without an isa parent
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class SimilarEntity:
    subject: str
    overlap: int
    similarity: float
    shared: list[tuple[str, str]]


def _subject_attrs(kb: ShardedKnowledgeBase) -> dict[str, set[tuple[str, str]]]:
    attrs: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    for fact in kb.all_facts():
        s = str(fact.get("S", "")).lower()
        r = str(fact.get("R", "")).lower()
        o = str(fact.get("O", "")).lower()
        if r == "isa" or r.startswith("not_"):
            continue
        attrs[s].add((r, o))
    return attrs


def similar_entities(kb: ShardedKnowledgeBase, subject: str,
                      *, top_k: int = 10,
                      min_overlap: int = 1) -> list[SimilarEntity]:
    """Return up to `top_k` entities similar to `subject`."""
    subj = subject.lower()
    attrs = _subject_attrs(kb)
    target = attrs.get(subj, set())
    if not target:
        return []
    out: list[SimilarEntity] = []
    for other, other_attrs in attrs.items():
        if other == subj:
            continue
        shared = target & other_attrs
        if len(shared) < min_overlap:
            continue
        union = target | other_attrs
        sim = len(shared) / max(1, len(union))
        out.append(SimilarEntity(
            subject=other,
            overlap=len(shared),
            similarity=sim,
            shared=sorted(shared),
        ))
    out.sort(key=lambda e: (-e.overlap, -e.similarity))
    return out[:top_k]
