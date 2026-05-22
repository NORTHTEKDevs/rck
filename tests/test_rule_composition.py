"""Tests for symbolic rule composition."""
from __future__ import annotations

from rck.rule_composition import (
    can_compose, compose, compose_all, compose_chain,
)
from rck.rule_extraction import Rule, RuleStore


def test_compose_two_simple_rules():
    r1 = Rule(body=["partof", "locatedin"], head="locatedin",
              support=5, confidence=0.9)
    r2 = Rule(body=["locatedin", "continent"], head="continent",
              support=10, confidence=0.95)
    composed = compose(r1, r2)
    assert composed is not None
    assert composed.body == ["partof", "locatedin", "continent"]
    assert composed.head == "continent"
    assert abs(composed.confidence - 0.9 * 0.95) < 1e-9
    assert composed.support == min(5, 10)


def test_compose_rejects_when_head_doesnt_match():
    r1 = Rule(body=["partof", "color"], head="color",
              support=3, confidence=0.9)
    r2 = Rule(body=["locatedin", "continent"], head="continent",
              support=5, confidence=0.9)
    assert compose(r1, r2) is None
    assert not can_compose(r1, r2)


def test_compose_rejects_inverse_pair_at_seam():
    """If composition would create an inverse-pair seam, reject."""
    # author/wrote are inverse pairs.
    r1 = Rule(body=["partof", "author"], head="author",
              support=3, confidence=0.9)
    r2 = Rule(body=["author", "wrote"], head="implies",
              support=3, confidence=0.9)
    # The seam in the new body would be author -> wrote.
    # NOTE: r2's body[0]=author matches r1.head=author so head match ok.
    # But the composed body inserts author followed by wrote -> reject.
    composed = compose(r1, r2)
    assert composed is None


def test_compose_all_adds_to_store():
    store = RuleStore()
    store.add(Rule(body=["partof", "locatedin"], head="locatedin",
                    support=5, confidence=0.9))
    store.add(Rule(body=["locatedin", "continent"], head="continent",
                    support=10, confidence=0.95))
    pre = store.size()
    new_rules = compose_all(store)
    assert len(new_rules) >= 1
    assert store.size() > pre


def test_compose_all_respects_max_body_length():
    store = RuleStore()
    # All composable but produce 4-clause rules.
    store.add(Rule(body=["partof", "locatedin"], head="locatedin",
                    support=5, confidence=0.9))
    store.add(Rule(body=["locatedin", "continent"], head="continent",
                    support=5, confidence=0.9))
    # max_body_length=2 should prevent the 3-clause composed rule.
    new_rules = compose_all(store, max_body_length=2)
    assert new_rules == []


def test_compose_chain_for_n_rules():
    r1 = Rule(body=["partof", "x"], head="x",
              support=5, confidence=0.9)
    r2 = Rule(body=["x", "y"], head="y",
              support=5, confidence=0.9)
    r3 = Rule(body=["y", "z"], head="z",
              support=5, confidence=0.9)
    composed = compose_chain([r1, r2, r3])
    assert composed is not None
    assert composed.body == ["partof", "x", "y", "z"]
    assert composed.head == "z"


def test_compose_chain_returns_none_on_break():
    """If any link in the chain doesn't compose, return None."""
    r1 = Rule(body=["partof", "locatedin"], head="locatedin",
              support=5, confidence=0.9)
    r2 = Rule(body=["mismatch", "thing"], head="thing",
              support=5, confidence=0.9)
    assert compose_chain([r1, r2]) is None


def test_conscious_agent_compose_rules():
    """ConsciousAgent.compose_rules() composes extracted rules."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    # Pre-warm skills so extract_rules finds patterns.
    p1 = [("O", "partof"), ("O", "locatedin")]
    p2 = [("O", "locatedin"), ("O", "continent")]
    for _ in range(3):
        agent.skills.record_success(p1)
        agent.skills.record_success(p2)
    new = agent.compose_rules()
    # We expect a composed (partof, locatedin, continent) rule.
    bodies = {tuple(r.body) for r in new}
    assert ("partof", "locatedin", "continent") in bodies
