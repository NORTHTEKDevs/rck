"""Tests for negative facts."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase
from rck.negative_facts import (
    NEGATION_PREFIX, denegate, denied_pairs_for, deny,
    filter_against_negatives, is_negative, negate,
)
from rck.provenance import ProvenanceStore


# ---- pure helpers ---------------------------------------------------------

def test_negate_adds_prefix():
    assert negate("isa") == "not_isa"
    assert negate("not_isa") == "not_isa"  # idempotent
    assert negate("ISA") == "not_isa"      # lowercased


def test_denegate_strips_prefix():
    assert denegate("not_isa") == "isa"
    assert denegate("isa") == "isa"


def test_is_negative_predicate():
    assert is_negative("not_isa") is True
    assert is_negative("isa") is False


def test_negation_prefix_constant():
    assert NEGATION_PREFIX == "not_"


# ---- kb-level ops --------------------------------------------------------

def test_deny_stores_negative_fact():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    deny(kb, "fish", "isa", "vegetable")
    ans, score = kb.answer({"S": "fish", "R": "not_isa"}, "O")
    assert ans == "vegetable"
    assert score > 0.10


def test_deny_records_provenance_with_negative_tag():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    prov = ProvenanceStore()
    deny(kb, "fish", "isa", "vegetable", provenance=prov)
    rec = prov.get("fish", "not_isa", "vegetable")
    assert rec is not None
    assert "negative" in rec.tags


def test_denied_pairs_for_finds_explicit_denials():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    deny(kb, "fish", "isa", "vegetable")
    deny(kb, "fish", "isa", "mineral")
    pairs = denied_pairs_for(kb, "fish", "isa")
    objs = {str(sym) for sym, _ in pairs}
    assert "vegetable" in objs
    assert "mineral" in objs


def test_filter_drops_denied_candidates():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    # Populate positive facts (could return vegetable as a noisy answer).
    bulk_load_triples(kb, [("fish", "isa", "animal")])
    # Deny an alternative answer.
    deny(kb, "fish", "isa", "vegetable")
    candidates = [("animal", 0.5), ("vegetable", 0.4), ("mineral", 0.3)]
    filtered = filter_against_negatives(kb, "fish", "isa", candidates)
    objs = {str(s) for s, _ in filtered}
    assert "animal" in objs
    assert "vegetable" not in objs


def test_filter_passthrough_when_no_denials():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    candidates = [("animal", 0.5), ("vegetable", 0.4)]
    filtered = filter_against_negatives(kb, "fish", "isa", candidates)
    assert filtered == candidates


# ---- conscious agent integration -----------------------------------------

def test_conscious_agent_deny_and_filter():
    """Agent gets a deny() method and ask_with_idk filters denied candidates."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    agent.tell("fish", "isa", "animal")
    agent.deny("fish", "isa", "vegetable")
    rec = agent.provenance.get("fish", "not_isa", "vegetable")
    assert rec is not None
    # Now check filter_against_negatives helper at agent level.
    candidates = agent.knowledge.query(
        {"S": "fish", "R": "isa"}, "O", top_k=5,
    )
    filtered = agent.filter_negatives("fish", "isa", candidates)
    objs = {str(s) for s, _ in filtered}
    assert "vegetable" not in objs
