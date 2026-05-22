"""Tests for rule effectiveness report."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent
from rck.rule_extraction import Rule, RuleStore


def test_effectiveness_report_returns_rows():
    store = RuleStore()
    r1 = Rule(body=["partof", "locatedin"], head="locatedin",
              support=5, confidence=0.9)
    r1.record_instantiation(True)
    r1.record_instantiation(True)
    r1.record_instantiation(False)
    store.add(r1)
    rows = store.effectiveness_report()
    assert rows
    assert "body" in rows[0]
    assert "uses" in rows[0]
    assert rows[0]["uses"] == 3


def test_effectiveness_report_sorted_by_uses():
    store = RuleStore()
    r1 = Rule(body=["a", "b"], head="b", support=1, confidence=1.0)
    r2 = Rule(body=["c", "d"], head="d", support=1, confidence=1.0)
    for _ in range(5):
        r1.record_instantiation(True)
    for _ in range(2):
        r2.record_instantiation(True)
    store.add(r1); store.add(r2)
    rows = store.effectiveness_report()
    assert rows[0]["uses"] == 5
    assert rows[1]["uses"] == 2


def test_effectiveness_report_top_k_limits():
    store = RuleStore()
    for i in range(5):
        rule = Rule(body=[f"r{i}"], head=f"r{i}",
                    support=1, confidence=1.0)
        rule.record_instantiation(True)
        store.add(rule)
    rows = store.effectiveness_report(top_k=3)
    assert len(rows) == 3


def test_conscious_agent_rule_effectiveness_report():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    p = [("O", "partof"), ("O", "locatedin")]
    for _ in range(3):
        agent.skills.record_success(p)
    rows = agent.rule_effectiveness_report()
    assert isinstance(rows, list)
