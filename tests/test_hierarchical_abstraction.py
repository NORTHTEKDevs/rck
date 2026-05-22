"""Tests for hierarchical fact abstraction."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.hierarchical_abstraction import (
    Abstraction, commit_abstractions, find_abstractions,
)
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


def _zoo() -> ShardedKnowledgeBase:
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("horse", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("cat", "has", "fur"),
        ("horse", "has", "fur"),
        # one that doesn't share:
        ("eagle", "isa", "bird"),
        ("eagle", "has", "feathers"),
    ])
    return kb


def test_find_abstractions_returns_parent_facts():
    kb = _zoo()
    abs_list = find_abstractions(kb, min_support=3)
    assert abs_list
    top = abs_list[0]
    assert isinstance(top, Abstraction)
    assert top.parent == "mammal"
    assert top.relation == "has"
    assert top.obj == "fur"
    assert top.sibling_support == 3
    sib_set = set(top.siblings)
    assert {"dog", "cat", "horse"} <= sib_set


def test_find_respects_min_support():
    kb = _zoo()
    # min_support=4 means we need 4 siblings; we only have 3.
    abs_list = find_abstractions(kb, min_support=4)
    assert all(a.sibling_support >= 4 for a in abs_list)


def test_commit_abstractions_stores_parent_fact():
    kb = _zoo()
    prov = ProvenanceStore()
    abs_list = find_abstractions(kb, min_support=3)
    n = commit_abstractions(kb, abs_list, provenance=prov)
    assert n >= 1
    # (mammal, has, fur) should now be in the KB and provenance.
    ans, score = kb.answer({"S": "mammal", "R": "has"}, "O")
    assert ans == "fur" or score > 0.10
    rec = prov.get("mammal", "has", "fur")
    assert rec is not None
    assert rec.source == "abstracted"
    assert any("abstracted" in t for t in rec.tags)


def test_commit_skips_already_existing():
    kb = _zoo()
    kb.store({"S": "mammal", "R": "has", "O": "fur"})  # pre-existing
    abs_list = find_abstractions(kb, min_support=3)
    n = commit_abstractions(kb, abs_list)
    # The already_existed flag should suppress storage.
    for a in abs_list:
        if a.parent == "mammal" and a.relation == "has" and a.obj == "fur":
            assert a.already_existed is True


def test_find_ignores_negative_relations():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("horse", "isa", "mammal"),
    ])
    from rck.negative_facts import deny
    deny(kb, "dog", "has", "feathers")
    deny(kb, "cat", "has", "feathers")
    deny(kb, "horse", "has", "feathers")
    abs_list = find_abstractions(kb, min_support=3)
    # Negative relations (not_has) should NOT yield abstractions here.
    assert not any(a.relation.startswith("not_") for a in abs_list)


def test_conscious_agent_abstract_facts():
    """ConsciousAgent.abstract_facts() wraps the call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    for s, r, o in [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("horse", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("cat", "has", "fur"),
        ("horse", "has", "fur"),
    ]:
        agent.tell(s, r, o)
    found, committed = agent.abstract_facts(min_support=3, commit=True)
    assert found
    assert committed >= 1
    rec = agent.provenance.get("mammal", "has", "fur")
    assert rec is not None
    assert rec.source == "abstracted"
