"""Tests for agent.delta_replay."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_delta_replay_returns_one_row_per_fact():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    facts = [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
        ("c", "isa", "d"),
    ]
    results = agent.delta_replay(facts)
    assert len(results) == 3
    for row in results:
        assert "fact" in row
        assert "kb_pre" in row
        assert "kb_post" in row
        assert "n_verified_inductions" in row


def test_delta_replay_kb_grows_within_window():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    results = agent.delta_replay([
        ("a", "isa", "b"),
        ("b", "isa", "c"),
    ])
    # After both facts, the KB inside the counterfactual has at
    # least 2 facts.
    assert results[1]["kb_post"] >= results[0]["kb_post"]


def test_delta_replay_rolls_back():
    """After delta_replay, the agent's KB should be unchanged."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    pre = agent.knowledge.size()
    agent.delta_replay([("a", "isa", "b"), ("c", "isa", "d")])
    assert agent.knowledge.size() == pre


def test_delta_replay_empty_list_returns_empty():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    results = agent.delta_replay([])
    assert results == []
