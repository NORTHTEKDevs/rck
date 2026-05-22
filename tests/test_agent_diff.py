"""Tests for agent diff."""
from __future__ import annotations

from rck.agent_diff import AgentDiffReport, diff_agents
from rck.conscious_agent import ConsciousAgent


def test_diff_identical_agents_has_no_unique_facts():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    b.tell("dog", "isa", "mammal")
    report = diff_agents(a, b)
    assert isinstance(report, AgentDiffReport)
    assert report.only_in_a_facts == []
    assert report.only_in_b_facts == []
    assert report.shared_facts >= 1


def test_diff_finds_unique_facts():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    b.tell("cat", "isa", "mammal")
    report = diff_agents(a, b)
    only_a_subjects = {key[0] for key in report.only_in_a_facts}
    only_b_subjects = {key[0] for key in report.only_in_b_facts}
    assert "dog" in only_a_subjects
    assert "cat" in only_b_subjects


def test_diff_compares_skills():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.skills.record_success([("O", "partof")])
    b.skills.record_success([("O", "locatedin")])
    report = diff_agents(a, b)
    assert report.only_in_a_skills
    assert report.only_in_b_skills


def test_diff_summary_dict():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    report = diff_agents(a, b)
    s = report.summary()
    for key in (
        "facts_only_in_a", "facts_only_in_b", "facts_shared",
        "skills_only_in_a", "skills_only_in_b", "skills_shared",
        "rules_only_in_a", "rules_only_in_b", "rules_shared",
    ):
        assert key in s


def test_conscious_agent_diff_with():
    """ConsciousAgent.diff_with(other) wraps the call."""
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    report = a.diff_with(b)
    assert isinstance(report, AgentDiffReport)
    assert report.only_in_a_facts
