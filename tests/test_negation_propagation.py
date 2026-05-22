"""Tests for transitive negation propagation."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase
from rck.negation_propagation import (
    PropagatedNegation, propagate_negations,
)
from rck.negative_facts import deny
from rck.provenance import ProvenanceStore


def test_propagate_through_isa():
    """If (mammal, NOT_has, feathers) and (cat, isa, mammal),
    derive (cat, NOT_has, feathers)."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("cat", "isa", "mammal"),
        ("dog", "isa", "mammal"),
    ])
    deny(kb, "mammal", "has", "feathers")
    results = propagate_negations(kb)
    propagated = {(r.subject, r.relation, r.obj) for r in results if r.stored}
    # Both cat and dog should inherit the negation.
    assert ("cat", "not_has", "feathers") in propagated
    assert ("dog", "not_has", "feathers") in propagated


def test_propagate_skips_already_stored_negative():
    """Don't re-emit a negative that's already in the KB."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [("cat", "isa", "mammal")])
    deny(kb, "mammal", "has", "feathers")
    deny(kb, "cat", "has", "feathers")
    results = propagate_negations(kb)
    cat_results = [r for r in results
                    if r.subject == "cat" and r.obj == "feathers"]
    assert all(not r.stored for r in cat_results)


def test_propagate_does_not_overwrite_positive():
    """If (cat, has, feathers) is asserted -- weird but possible --
    don't propagate the negation. Mark it as rejected."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("cat", "isa", "mammal"),
        ("cat", "has", "feathers"),
    ])
    deny(kb, "mammal", "has", "feathers")
    results = propagate_negations(kb)
    cat_results = [r for r in results
                    if r.subject == "cat" and r.obj == "feathers"]
    if cat_results:
        assert not cat_results[0].stored
        assert "positive" in (cat_results[0].rejected_reason or "")


def test_propagate_records_provenance():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [("cat", "isa", "mammal")])
    deny(kb, "mammal", "has", "feathers")
    prov = ProvenanceStore()
    propagate_negations(kb, provenance=prov)
    rec = prov.get("cat", "not_has", "feathers")
    assert rec is not None
    assert rec.source == "rule"
    assert "negative_propagated" in rec.tags
    assert rec.derivation


def test_propagate_only_via_lifting_relations():
    """A non-lifting relation like `causes` shouldn't propagate negations."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    kb.store({"S": "rain", "R": "causes", "O": "wet"})
    deny(kb, "wet", "has", "feathers")
    results = propagate_negations(kb)
    # rain isa wet?  No, rain causes wet -- causes isn't lifting.
    propagated = {(r.subject, r.relation, r.obj) for r in results if r.stored}
    assert ("rain", "not_has", "feathers") not in propagated


def test_conscious_agent_propagate_negations():
    """ConsciousAgent.propagate_negations() wraps the call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    agent.tell("cat", "isa", "mammal")
    agent.deny("mammal", "has", "feathers")
    results = agent.propagate_negations()
    assert any(r.stored and r.subject == "cat"
               and r.relation == "not_has"
               and r.obj == "feathers" for r in results)
