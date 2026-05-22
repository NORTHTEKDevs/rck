"""Tests for counterfactual reasoning context."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_counterfactual_adds_fact_during_context():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    pre = agent.knowledge.size()
    with agent.counterfactual([("dog", "isa", "fish")]):
        ans, score = agent.knowledge.answer(
            {"S": "dog", "R": "isa"}, "O",
        )
        assert ans == "fish"
        assert score > 0.10
        assert agent.knowledge.size() == pre + 1


def test_counterfactual_rolls_back_on_exit():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    pre = agent.knowledge.size()
    with agent.counterfactual([("dog", "isa", "fish")]):
        pass
    assert agent.knowledge.size() == pre
    candidates = agent.knowledge.query(
        {"S": "dog", "R": "isa"}, "O", top_k=5,
    )
    objs = {str(s).lower() for s, sc in candidates if sc >= 0.10}
    assert "fish" not in objs


def test_counterfactual_handles_multiple_facts():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    pre = agent.knowledge.size()
    with agent.counterfactual([
        ("dog", "isa", "fish"),
        ("cat", "isa", "robot"),
    ]):
        ans1, _ = agent.knowledge.answer({"S": "dog", "R": "isa"}, "O")
        ans2, _ = agent.knowledge.answer({"S": "cat", "R": "isa"}, "O")
        assert ans1 == "fish"
        assert ans2 == "robot"
    assert agent.knowledge.size() == pre


def test_counterfactual_rollback_after_exception():
    """Rollback must run even if an exception is raised inside the context."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    pre = agent.knowledge.size()
    try:
        with agent.counterfactual([("dog", "isa", "fish")]):
            raise ValueError("simulated failure")
    except ValueError:
        pass
    assert agent.knowledge.size() == pre


def test_counterfactual_bumps_cache_version():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    pre_v = agent.chain_cache.kb_version
    with agent.counterfactual([("dog", "isa", "fish")]):
        mid_v = agent.chain_cache.kb_version
        assert mid_v > pre_v
    post_v = agent.chain_cache.kb_version
    assert post_v > mid_v
