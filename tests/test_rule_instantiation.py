"""Tests for forward-chaining rule instantiation."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore
from rck.rule_extraction import Rule, RuleStore
from rck.rule_instantiation import (
    InstantiatedFact, instantiate_all, instantiate_rule,
)


def _staircase_kb() -> ShardedKnowledgeBase:
    """isa transitivity: a -isa- b -isa- c -isa- d."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
        ("c", "isa", "d"),
    ])
    return kb


def test_instantiate_isa_transitive():
    """isa->isa => isa instantiates (a, isa, c) and (b, isa, d)."""
    kb = _staircase_kb()
    rule = Rule(body=["isa", "isa"], head="isa",
                support=3, confidence=1.0)
    facts = instantiate_rule(kb, rule)
    # Either (a, isa, c) or (b, isa, d) (or both) should emit.
    pairs = {(f.subject, f.obj) for f in facts}
    assert ("a", "c") in pairs or ("b", "d") in pairs


def test_instantiate_records_provenance():
    kb = _staircase_kb()
    prov = ProvenanceStore()
    rule = Rule(body=["isa", "isa"], head="isa",
                support=3, confidence=1.0)
    facts = instantiate_rule(kb, rule, provenance=prov)
    for f in facts:
        if f.verified:
            rec = prov.get(f.subject, f.relation, f.obj)
            assert rec is not None
            assert rec.source == "rule"
            assert any("rule_" in t for t in rec.tags)


def test_instantiate_skips_already_stored_direct_facts():
    """If (a, isa, c) is ALREADY stored directly, don't re-emit it."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
        ("a", "isa", "c"),  # already direct
    ])
    rule = Rule(body=["isa", "isa"], head="isa",
                support=3, confidence=1.0)
    facts = instantiate_rule(kb, rule)
    # No NEW facts for (a, isa, c) because it's already a direct edge.
    pairs = {(f.subject, f.obj) for f in facts}
    assert ("a", "c") not in pairs


def test_instantiate_filters_inverse_pair():
    """A rule body with an inverse pair should not emit anything."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dickens", "wrote", "olivertwist"),
    ])
    # author and wrote are an inverse pair.
    rule = Rule(body=["author", "wrote"], head="implies",
                support=2, confidence=1.0)
    facts = instantiate_rule(kb, rule)
    assert facts == []


def test_instantiate_filters_non_transitive_same_relation():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dickens", "wrote", "olivertwist"),
        ("dickens", "wrote", "greatexpectations"),
    ])
    # `wrote -> wrote` is non-transitive same-relation -> rejected.
    rule = Rule(body=["wrote", "wrote"], head="implies",
                support=2, confidence=1.0)
    facts = instantiate_rule(kb, rule)
    assert facts == []


def test_instantiate_too_short_body_returns_empty():
    kb = _staircase_kb()
    rule = Rule(body=["isa"], head="isa", support=2, confidence=1.0)
    assert instantiate_rule(kb, rule) == []


def test_instantiate_all_processes_store():
    """instantiate_all walks every rule in a RuleStore."""
    kb = _staircase_kb()
    store = RuleStore()
    store.add(Rule(body=["isa", "isa"], head="isa",
                    support=3, confidence=1.0))
    facts = instantiate_all(kb, store)
    assert len(facts) >= 1


def test_instantiate_three_clause_rule():
    """A 3-clause rule (R1, R2, R3) emits (X0, head, X3) facts."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dog", "partof", "house"),
        ("house", "locatedin", "city"),
        ("city", "isa", "place"),
    ])
    rule = Rule(body=["partof", "locatedin", "isa"], head="isa",
                support=3, confidence=1.0)
    facts = instantiate_rule(kb, rule)
    pairs = {(f.subject, f.obj) for f in facts}
    assert ("dog", "place") in pairs


def test_three_clause_filters_inverse_pair_anywhere():
    """An inverse-pair seam anywhere in the body rejects the rule."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("a", "partof", "b"),
        ("b", "author", "c"),   # author/wrote inverse pair would block this
        ("c", "wrote", "d"),
    ])
    # body = [partof, author, wrote]; pair (author, wrote) is inverse.
    rule = Rule(body=["partof", "author", "wrote"], head="implies",
                support=3, confidence=1.0)
    facts = instantiate_rule(kb, rule)
    assert facts == []


def test_three_clause_filters_non_transitive_same_relation():
    """Non-transitive same-relation seam in a 3-clause body is rejected."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("a", "partof", "b"),
        ("b", "wrote", "c"),
        ("c", "wrote", "d"),
    ])
    # `wrote -> wrote` mid-chain is non-transitive same-relation.
    rule = Rule(body=["partof", "wrote", "wrote"], head="implies",
                support=3, confidence=1.0)
    facts = instantiate_rule(kb, rule)
    assert facts == []


def test_four_clause_chain_works():
    """4-clause body chains through 4 intermediate nodes."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
        ("c", "isa", "d"),
        ("d", "isa", "e"),
    ])
    rule = Rule(body=["isa", "isa", "isa", "isa"], head="isa",
                support=4, confidence=1.0)
    facts = instantiate_rule(kb, rule)
    # We expect at least (a, isa, e).
    pairs = {(f.subject, f.obj) for f in facts if f.verified}
    assert ("a", "e") in pairs


def test_instantiate_respects_negative_facts():
    """If (X, NOT_R_head, Z) is stored, the rule must NOT emit (X, R_head, Z)."""
    from rck.negative_facts import deny
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
    ])
    # Deny the would-be derived fact.
    deny(kb, "a", "isa", "c")
    rule = Rule(body=["isa", "isa"], head="isa",
                support=3, confidence=1.0)
    facts = instantiate_rule(kb, rule)
    pairs = {(f.subject, f.obj) for f in facts}
    assert ("a", "c") not in pairs


def test_conscious_agent_instantiate_rules():
    """ConsciousAgent.instantiate_rules() walks the agent's RuleStore."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
    ]:
        agent.tell(s, r, o)
    # Manually seed the skill library so extract_rules has support.
    pattern = [("O", "isa"), ("O", "isa")]
    for _ in range(3):
        agent.skills.record_success(pattern)
    facts = agent.instantiate_rules()
    # At least one verified emission of (a, isa, c).
    verified = [f for f in facts if f.verified]
    assert any((f.subject, f.relation, f.obj) == ("a", "isa", "c")
               for f in verified)
