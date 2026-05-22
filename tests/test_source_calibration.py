"""Tests for agent.source_calibration."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_source_calibration_returns_per_source_stats():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.tell("cat", "isa", "mammal")
    stats = agent.source_calibration()
    assert "user" in stats
    user = stats["user"]
    assert user["count"] >= 2
    assert user["avg_confidence"] > 0
    assert user["avg_reinforce"] >= 1


def test_source_calibration_handles_multiple_sources():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.knowledge.store({"S": "x", "R": "isa", "O": "y"})
    agent.provenance.store("x", "isa", "y", source="induced",
                            confidence=0.4)
    stats = agent.source_calibration()
    assert "user" in stats
    assert "induced" in stats
    assert stats["induced"]["count"] == 1
    assert stats["induced"]["avg_confidence"] == 0.4


def test_source_calibration_empty_agent():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    stats = agent.source_calibration()
    assert stats == {}
