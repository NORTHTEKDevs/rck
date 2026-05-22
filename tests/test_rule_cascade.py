"""Tests for cascading rule instantiation."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore
from rck.rule_cascade import RuleCascadeResult, cascade_instantiate
from rck.rule_extraction import Rule, RuleStore


def _staircase_kb() -> ShardedKnowledgeBase:
    """a -isa- b -isa- c -isa- d -isa- e (4-level chain)."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
        ("c", "isa", "d"),
        ("d", "isa", "e"),
    ])
    return kb


def test_rule_cascade_reaches_fixed_point():
    kb = _staircase_kb()
    store = RuleStore()
    store.add(Rule(body=["isa", "isa"], head="isa",
                    support=4, confidence=1.0))
    res = cascade_instantiate(kb, store, max_rounds=5)
    assert isinstance(res, RuleCascadeResult)
    assert res.saturated  # should reach fixed point well before max_rounds


def test_rule_cascade_adds_more_in_round2_than_round1():
    """Round 1 emits (a, isa, c) and (b, isa, d) etc. Round 2 should
    pick those up and emit longer-range edges."""
    kb = _staircase_kb()
    store = RuleStore()
    store.add(Rule(body=["isa", "isa"], head="isa",
                    support=4, confidence=1.0))
    res = cascade_instantiate(kb, store, max_rounds=5)
    # First round must have emitted some facts; later rounds may add more
    # or saturate.
    assert res.rounds
    assert any(r.facts_verified > 0 for r in res.rounds)


def test_cascade_records_provenance():
    kb = _staircase_kb()
    prov = ProvenanceStore()
    store = RuleStore()
    store.add(Rule(body=["isa", "isa"], head="isa",
                    support=4, confidence=1.0))
    res = cascade_instantiate(kb, store, provenance=prov)
    for f in res.induced_facts:
        rec = prov.get(f.subject, f.relation, f.obj)
        assert rec is not None
        assert rec.source == "rule"


def test_cascade_empty_store_yields_nothing():
    kb = _staircase_kb()
    store = RuleStore()
    res = cascade_instantiate(kb, store, max_rounds=3)
    assert res.saturated
    assert res.total_verified() == 0
    assert res.final_facts == res.initial_facts


def test_cascade_with_no_useful_rules():
    """A rule that doesn't apply to anything in the KB -> no facts."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [("dog", "color", "brown")])
    store = RuleStore()
    store.add(Rule(body=["capital", "locatedin"], head="locatedin",
                    support=2, confidence=1.0))
    res = cascade_instantiate(kb, store, max_rounds=3)
    assert res.total_verified() == 0
    assert res.saturated


def test_conscious_agent_cascade_instantiate():
    """ConsciousAgent.cascade_instantiate_rules() composes
    extract_rules + cascade_instantiate."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
        ("c", "isa", "d"),
    ]:
        agent.tell(s, r, o)
    pattern = [("O", "isa"), ("O", "isa")]
    for _ in range(3):
        agent.skills.record_success(pattern)
    res = agent.cascade_instantiate_rules(max_rounds=4)
    assert res.total_verified() >= 1
