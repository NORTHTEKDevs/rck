"""Tests for agent.status_report text renderer."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_status_report_basic_fields():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    report = agent.status_report()
    assert isinstance(report, str)
    assert "KB facts" in report
    assert "Provenance" in report
    assert "shards" in report


def test_status_report_after_queries():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    for _ in range(3):
        agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    report = agent.status_report()
    assert "Query memory" in report
    assert "states" in report


def test_status_report_with_skills_and_rules():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    p = [("O", "partof"), ("O", "locatedin")]
    for _ in range(3):
        agent.skills.record_success(p)
    report = agent.status_report()
    assert "Skills" in report
    assert "Rules" in report
