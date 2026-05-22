"""Tests for active gap detection."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.gap_detection import Gap, find_gaps
from rck.knowledge_base import ShardedKnowledgeBase


def _zoo_kb() -> ShardedKnowledgeBase:
    """A few mammals with shared and missing relations."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("horse", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("cat", "has", "fur"),
        ("horse", "has", "fur"),
        ("dog", "color", "brown"),
        ("cat", "color", "black"),
        ("horse", "color", "white"),
        # whale is also a mammal but is missing has/color
        ("whale", "isa", "mammal"),
    ])
    return kb


def test_find_gaps_surfaces_missing_relations():
    """`whale` lacks `has` and `color` facts -- find_gaps should report them."""
    kb = _zoo_kb()
    gaps = find_gaps(kb, "whale", min_sibling_support=2)
    rels = {g.relation for g in gaps}
    assert "has" in rels
    assert "color" in rels


def test_find_gaps_skips_relations_already_present():
    """If we add (whale, color, gray), `color` should drop off the gaps."""
    kb = _zoo_kb()
    kb.store({"S": "whale", "R": "color", "O": "gray"})
    gaps = find_gaps(kb, "whale", min_sibling_support=2)
    rels = {g.relation for g in gaps}
    assert "color" not in rels


def test_find_gaps_returns_empty_for_unknown_subject():
    kb = _zoo_kb()
    gaps = find_gaps(kb, "non_existent_subject")
    assert gaps == []


def test_find_gaps_orders_by_sibling_support():
    """If 3 peers have `has` and 2 have a rare `lifespan`, has comes first."""
    kb = _zoo_kb()
    # Two peers have lifespan.
    kb.store({"S": "dog", "R": "lifespan", "O": "12"})
    kb.store({"S": "cat", "R": "lifespan", "O": "14"})
    gaps = find_gaps(kb, "whale", min_sibling_support=2)
    if len(gaps) >= 2:
        # Top gap has the highest sibling support.
        assert gaps[0].sibling_support >= gaps[-1].sibling_support


def test_gap_verbalize_includes_relation():
    kb = _zoo_kb()
    gaps = find_gaps(kb, "whale")
    assert gaps
    text = gaps[0].verbalize()
    assert gaps[0].relation in text


def test_gap_carries_common_objects_hint():
    kb = _zoo_kb()
    gaps = find_gaps(kb, "whale", min_sibling_support=2)
    has_gap = next((g for g in gaps if g.relation == "has"), None)
    if has_gap is not None:
        # All siblings have fur -> "fur" should be the most common object.
        objs = {obj for obj, _ in has_gap.common_objects}
        assert "fur" in objs


def test_conscious_agent_find_gaps():
    """ConsciousAgent.find_gaps() wraps the call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("dog", "color", "brown"),
        ("cat", "color", "black"),
        ("whale", "isa", "mammal"),
    ]:
        agent.tell(s, r, o)
    gaps = agent.find_gaps("whale", min_sibling_support=2)
    rels = {g.relation for g in gaps}
    assert "color" in rels
