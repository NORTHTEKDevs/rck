"""Tests for concept density map."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.concept_density import DensityMap, density_map
from rck.conscious_agent import ConsciousAgent
from rck.knowledge_base import ShardedKnowledgeBase


def test_density_map_returns_structured_report():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("dog", "color", "brown"),
        ("cat", "isa", "mammal"),
        ("whale", "isa", "mammal"),
    ])
    d = density_map(kb)
    assert isinstance(d, DensityMap)
    assert d.total_subjects >= 3
    assert d.total_facts >= 5


def test_density_map_finds_top_subject():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("dog", "color", "brown"),
        ("dog", "lifespan", "12"),
        ("cat", "isa", "mammal"),
    ])
    d = density_map(kb)
    top_subj = d.top_subjects[0][0]
    assert top_subj == "dog"


def test_density_map_identifies_stubs():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("whale", "isa", "mammal"),  # only 1 fact, a stub
    ])
    d = density_map(kb, stub_max=1)
    stub_set = {s for s, _ in d.stub_subjects}
    assert "whale" in stub_set


def test_density_map_histogram_shape():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("cat", "isa", "mammal"),
        ("cat", "has", "fur"),
        ("eagle", "isa", "bird"),
    ])
    d = density_map(kb)
    # Two subjects with 2 facts, one with 1 fact -> histogram has
    # both bins present.
    assert 1 in d.histogram
    assert 2 in d.histogram


def test_density_map_verbalize():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("dog", "has", "fur"),
    ])
    d = density_map(kb)
    text = d.verbalize()
    assert "subjects" in text
    assert "facts" in text


def test_conscious_agent_concept_density():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.tell("dog", "has", "fur")
    d = agent.concept_density()
    assert isinstance(d, DensityMap)
    assert d.total_facts >= 2
