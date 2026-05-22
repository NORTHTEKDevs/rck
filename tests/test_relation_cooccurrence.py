"""Tests for relation co-occurrence map."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.conscious_agent import ConsciousAgent
from rck.knowledge_base import ShardedKnowledgeBase
from rck.relation_cooccurrence import (
    CooccurrenceMap, RelationPair, cooccurrence,
)


def test_cooccurrence_pairs_relations():
    """has and color appear together on dog and cat."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "has", "fur"),
        ("dog", "color", "brown"),
        ("cat", "has", "fur"),
        ("cat", "color", "black"),
        ("eagle", "has", "feathers"),
    ])
    m = cooccurrence(kb, min_co_occurrence=2)
    assert isinstance(m, CooccurrenceMap)
    assert m.pairs
    top = m.top_pairs(1)[0]
    rels = {top.r1, top.r2}
    assert rels == {"has", "color"}
    assert top.co_occurrence == 2


def test_cooccurrence_respects_min():
    """Pairs below min_co_occurrence are filtered out."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "has", "fur"),
        ("dog", "color", "brown"),
    ])
    m = cooccurrence(kb, min_co_occurrence=3)
    assert m.pairs == []


def test_cooccurrence_ignores_isa():
    """isa is universal and not informative -> excluded."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("cat", "isa", "mammal"),
        ("cat", "has", "fur"),
    ])
    m = cooccurrence(kb, min_co_occurrence=2)
    for p in m.pairs:
        assert p.r1 != "isa"
        assert p.r2 != "isa"


def test_jaccard_computed_correctly():
    """3 subjects, R1+R2 on 2, R1 only on 1 -> jaccard = 2/3."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "has", "fur"),
        ("dog", "color", "brown"),
        ("cat", "has", "fur"),
        ("cat", "color", "black"),
        ("eagle", "has", "feathers"),  # no color
    ])
    m = cooccurrence(kb, min_co_occurrence=1)
    p = next((p for p in m.pairs
              if {p.r1, p.r2} == {"has", "color"}), None)
    assert p is not None
    assert p.co_occurrence == 2
    assert abs(p.jaccard - (2.0 / 3.0)) < 1e-9


def test_conscious_agent_relation_cooccurrence():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    for s, r, o in [
        ("dog", "has", "fur"),
        ("dog", "color", "brown"),
        ("cat", "has", "fur"),
        ("cat", "color", "black"),
    ]:
        agent.tell(s, r, o)
    m = agent.relation_cooccurrence(min_co_occurrence=2)
    assert m.pairs
    assert any({p.r1, p.r2} == {"has", "color"} for p in m.pairs)
