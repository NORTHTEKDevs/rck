"""Active curiosity / gap-detection.

LLMs don't know what they don't know. RCK can detect knowledge gaps
explicitly: when sibling entities share a property that one entity
doesn't have recorded, that's a candidate gap.

Algorithm:
  1. For each entity E in the KB:
       a. Find all "siblings" S(E) -- entities that share at least one
          isa parent with E.
       b. For each relation R appearing on >50% of siblings, check if E
          has (E, R, ?) recorded.
       c. If not -- that's a GAP. Generate a question to fill it.

Returned gaps are SORTED by sibling-agreement (more siblings sharing
the property -> stronger candidate for the gap to be real).

Examples:
  - 10 mammals have fur. The platypus doesn't have fur recorded.
    -> "Does the platypus have fur?"
  - 8 countries have capitals. France doesn't have a capital recorded.
    -> "What is the capital of France?"

Gaps that are answered get stored; those that conflict with sibling
expectations get flagged for review.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class Gap:
    """A candidate knowledge gap to fill."""
    entity: str
    relation: str
    expected_kind: str
    sibling_count: int
    agreement: float  # fraction of siblings that have this relation
    question: str


def _siblings_of(kb: ShardedKnowledgeBase, entity: str, max_siblings: int = 50) -> list[str]:
    """Return entities that share at least one `isa` parent with `entity`."""
    parents: set[str] = set()
    for rel in ("isa", "kind", "category"):
        results = kb.query({"S": entity, "R": rel}, "O", top_k=3)
        for sym, score in results:
            if score >= 0.10:
                parents.add(str(sym))
    if not parents:
        return []
    siblings: set[str] = set()
    for parent in parents:
        for rel in ("isa", "kind", "category"):
            results = kb.query({"R": rel, "O": parent}, "S", top_k=max_siblings)
            for sym, score in results:
                if score >= 0.10:
                    siblings.add(str(sym))
    siblings.discard(entity)
    return sorted(siblings)


def _relations_for_subjects(
    kb: ShardedKnowledgeBase, subjects: list[str],
) -> dict[str, set[str]]:
    """For each subject, list the relations it has facts about.

    Returns {relation: set_of_subjects_with_this_relation}.
    """
    relation_to_subjects: dict[str, set[str]] = defaultdict(set)
    for s in subjects:
        for shard in kb._shards:
            for fact in shard._facts:
                if str(fact.get("S", "")) == s:
                    relation_to_subjects[str(fact.get("R", ""))].add(s)
    return dict(relation_to_subjects)


def detect_gaps(
    kb: ShardedKnowledgeBase, entity: str,
    *,
    min_agreement: float = 0.4,
    min_siblings: int = 3,
) -> list[Gap]:
    """Find knowledge gaps for `entity` based on sibling expectations."""
    entity = entity.lower()
    siblings = _siblings_of(kb, entity)
    if len(siblings) < min_siblings:
        return []

    # What relations do the siblings cover?
    rel_subjects = _relations_for_subjects(kb, siblings + [entity])
    entity_relations = {
        r for r, subs in rel_subjects.items() if entity in subs
    }
    gaps: list[Gap] = []
    for relation, subs_with_rel in rel_subjects.items():
        if relation in {"isa", "kind", "category"}:
            continue  # we're using these to find siblings already
        if entity in subs_with_rel:
            continue  # we already know this for our entity
        sibling_hits = len(subs_with_rel & set(siblings))
        agreement = sibling_hits / max(len(siblings), 1)
        if agreement < min_agreement:
            continue
        gaps.append(Gap(
            entity=entity,
            relation=relation,
            expected_kind="value",
            sibling_count=sibling_hits,
            agreement=agreement,
            question=_render_gap_question(entity, relation),
        ))
    gaps.sort(key=lambda g: -g.agreement)
    return gaps


def _render_gap_question(entity: str, relation: str) -> str:
    """Best-effort natural-language question for a gap."""
    # Tiny dispatch -- could be expanded.
    if relation == "color":
        return f"What color is the {entity}?"
    if relation == "size":
        return f"What is the size of the {entity}?"
    if relation == "has":
        return f"What does the {entity} have?"
    if relation in ("locatedin", "lives_in"):
        return f"Where is the {entity}?"
    if relation == "capital":
        return f"What is the capital of {entity}?"
    return f"What is the {relation} of the {entity}?"


def detect_global_gaps(
    kb: ShardedKnowledgeBase,
    sample_size: int = 50,
    *,
    min_agreement: float = 0.4,
    min_siblings: int = 3,
) -> list[Gap]:
    """Sample entities from the KB and detect gaps across all of them."""
    # Collect entities (subjects of any fact).
    entities: set[str] = set()
    for shard in kb._shards:
        for fact in shard._facts:
            entities.add(str(fact.get("S", "")))
        if len(entities) >= sample_size * 5:
            break
    entities_list = sorted(entities)[:sample_size]
    all_gaps: list[Gap] = []
    for e in entities_list:
        all_gaps.extend(
            detect_gaps(kb, e,
                        min_agreement=min_agreement,
                        min_siblings=min_siblings)
        )
    # Deduplicate and sort by agreement.
    seen = set()
    deduped: list[Gap] = []
    for g in all_gaps:
        key = (g.entity, g.relation)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(g)
    deduped.sort(key=lambda g: -g.agreement)
    return deduped
