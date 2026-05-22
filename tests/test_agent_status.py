"""Tests for ConsciousAgent.status() dashboard."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_status_returns_dict_with_expected_keys():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    s = agent.status()
    for key in (
        "kb_size", "n_shards", "provenance_records",
        "provenance_sources", "induced_via_hops",
        "skills", "rules", "query_memory", "chain_cache",
        "calibration_relations",
    ):
        assert key in s


def test_status_counts_provenance_sources():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.tell("cat", "isa", "mammal")
    agent.deny("fish", "isa", "vegetable")
    s = agent.status()
    sources = s["provenance_sources"]
    assert sources.get("user", 0) >= 3  # dog, cat, deny  (deny is also user)


def test_status_query_memory_state_breakdown():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    agent.ask_with_idk({"S": "xxx_unknown", "R": "isa"}, "O")
    s = agent.status()
    bd = s["query_memory"]["state_breakdown"]
    assert bd.get("known", 0) >= 1
    assert bd.get("idk", 0) >= 1


def test_status_rules_summary_when_skills_present():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    p = [("O", "partof"), ("O", "locatedin")]
    for _ in range(3):
        agent.skills.record_success(p)
    s = agent.status()
    assert s["rules"]["n_rules"] >= 1
