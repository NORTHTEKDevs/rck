"""Tests for fact pruning."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent
from rck.fact_pruning import PruningPolicy, prune


def test_prune_drops_low_confidence_induced_facts():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    # Manually inject an induced fact with low confidence.
    agent.knowledge.store({"S": "x", "R": "isa", "O": "y"})
    rec = agent.provenance.store(
        "x", "isa", "y", source="induced", confidence=0.05,
        tags={"induced", "via_3_hops"},
    )
    rec.confidence = 0.05  # ensure low
    policy = PruningPolicy(min_confidence=0.1)
    report = prune(agent.knowledge, agent.provenance, policy=policy)
    assert report.dropped >= 1
    assert agent.provenance.get("x", "isa", "y") is None


def test_prune_keeps_user_facts_by_default():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    rec = agent.provenance.get("dog", "isa", "mammal")
    rec.confidence = 0.05  # artificially low
    report = prune(agent.knowledge, agent.provenance,
                    policy=PruningPolicy(min_confidence=0.1))
    # User fact stays.
    assert agent.provenance.get("dog", "isa", "mammal") is not None
    assert report.dropped == 0


def test_prune_can_drop_user_facts_when_flag_set():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    rec = agent.provenance.get("dog", "isa", "mammal")
    rec.confidence = 0.05
    report = prune(
        agent.knowledge, agent.provenance,
        policy=PruningPolicy(min_confidence=0.1, prune_user_facts=True),
    )
    assert report.dropped >= 1


def test_prune_keeps_high_count_facts():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    rec = agent.provenance.store(
        "x", "isa", "y", source="induced", confidence=0.05,
    )
    rec.count = 10  # heavily reinforced
    agent.knowledge.store({"S": "x", "R": "isa", "O": "y"})
    report = prune(
        agent.knowledge, agent.provenance,
        policy=PruningPolicy(min_confidence=0.1, min_count=1),
    )
    # The count >= min_count+1 escape clause keeps it.
    assert agent.provenance.get("x", "isa", "y") is not None


def test_prune_drops_negative_facts_when_flag_set():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.deny("fish", "isa", "vegetable")
    rec = agent.provenance.get("fish", "not_isa", "vegetable")
    rec.confidence = 0.05
    report = prune(
        agent.knowledge, agent.provenance,
        policy=PruningPolicy(
            min_confidence=0.1, prune_negative_facts=True,
            prune_user_facts=True,
        ),
    )
    assert report.dropped >= 1


def test_conscious_agent_prune_facts():
    """ConsciousAgent.prune_facts() wraps the call."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.knowledge.store({"S": "x", "R": "isa", "O": "y"})
    rec = agent.provenance.store(
        "x", "isa", "y", source="induced", confidence=0.05,
    )
    report = agent.prune_facts(min_confidence=0.1)
    assert report.dropped >= 1
