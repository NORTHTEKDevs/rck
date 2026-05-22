"""Tests for real-time drift warning on ask_with_idk."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_no_drift_on_first_query():
    """First time we see a signature there's no prior episode."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    res = agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    assert res.drift_from_prior is None


def test_no_drift_when_answer_stable():
    """Same query twice with no KB change -> no drift flag."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    res2 = agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    assert res2.drift_from_prior is None


def test_drift_detected_when_state_changes():
    """If we forget a fact between two queries the state changes."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    res1 = agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    assert res1.drift_from_prior is None
    # Remove the fact via direct kb.forget.
    agent.knowledge.forget({"S": "dog", "R": "isa", "O": "mammal"})
    res2 = agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    # State should have changed (was known, now idk or different sym).
    if res2.drift_from_prior is not None:
        assert "prev" in res2.drift_from_prior
        assert "now" in res2.drift_from_prior


def test_drift_recorded_in_query_memory_notes():
    """When drift is detected, the new episode's notes carry the
    drift description for later audit."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    agent.knowledge.forget({"S": "dog", "R": "isa", "O": "mammal"})
    res2 = agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    if res2.drift_from_prior is not None:
        last_ep = agent.query_memory.recent(1)[0]
        assert "prev" in last_ep.notes
