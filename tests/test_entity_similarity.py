"""Tests for entity similarity by relation overlap."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.conscious_agent import ConsciousAgent
from rck.entity_similarity import SimilarEntity, similar_entities
from rck.knowledge_base import ShardedKnowledgeBase


def test_similar_finds_overlapping_entities():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "has", "fur"),
        ("dog", "has", "tail"),
        ("cat", "has", "fur"),
        ("cat", "has", "tail"),
        ("cat", "color", "black"),
        ("eagle", "has", "feathers"),
    ])
    results = similar_entities(kb, "dog", top_k=5, min_overlap=1)
    assert results
    top = results[0]
    assert isinstance(top, SimilarEntity)
    assert top.subject == "cat"
    assert top.overlap == 2


def test_similar_returns_empty_for_unknown():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [("dog", "has", "fur")])
    results = similar_entities(kb, "no_such_subject")
    assert results == []


def test_similar_jaccard_computed():
    """Two subjects each with 2 attrs sharing 2 attrs -> jaccard = 1.0."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "has", "fur"),
        ("dog", "has", "tail"),
        ("cat", "has", "fur"),
        ("cat", "has", "tail"),
    ])
    results = similar_entities(kb, "dog")
    cat = next((r for r in results if r.subject == "cat"), None)
    assert cat is not None
    assert abs(cat.similarity - 1.0) < 1e-9


def test_similar_ignores_isa_and_negative():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("cat", "has", "fur"),
    ])
    results = similar_entities(kb, "dog")
    # isa is excluded -> overlap is just `has=fur`.
    cat = next(r for r in results if r.subject == "cat")
    assert cat.overlap == 1
    assert ("isa", "mammal") not in cat.shared


def test_conscious_agent_similar_entities():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    for s, r, o in [
        ("dog", "has", "fur"),
        ("cat", "has", "fur"),
    ]:
        agent.tell(s, r, o)
    results = agent.similar_entities("dog")
    assert any(r.subject == "cat" for r in results)
