"""Tests for causal chain reasoning."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.causal import CausalNode, downstream_effects, root_causes
from rck.knowledge_base import ShardedKnowledgeBase


def _weather_kb() -> ShardedKnowledgeBase:
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    kb.store({"S": "rain", "R": "causes", "O": "wet"})
    kb.store({"S": "wet", "R": "causes", "O": "slippery"})
    kb.store({"S": "slippery", "R": "causes", "O": "fall"})
    kb.store({"S": "fall", "R": "causes", "O": "injury"})
    return kb


def test_downstream_effects_walks_forward():
    kb = _weather_kb()
    effects = downstream_effects(kb, "rain", max_depth=3)
    assert effects
    entities = [n.entity for n in effects]
    assert "wet" in entities
    assert "slippery" in entities


def test_root_causes_walks_backward():
    kb = _weather_kb()
    causes = root_causes(kb, "injury", max_depth=3)
    assert causes
    entities = [n.entity for n in causes]
    # Walking backward from injury -> fall -> slippery -> wet (within depth 3).
    assert any(e in {"fall", "slippery", "wet"} for e in entities)


def test_causal_walk_respects_max_depth():
    kb = _weather_kb()
    shallow = downstream_effects(kb, "rain", max_depth=1)
    deep = downstream_effects(kb, "rain", max_depth=4)
    assert len(deep) >= len(shallow)


def test_causal_walk_no_chain_returns_empty():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    kb.store({"S": "dog", "R": "isa", "O": "mammal"})  # not causes
    effects = downstream_effects(kb, "dog", max_depth=3)
    assert effects == []


def test_causal_nodes_carry_chain_via():
    kb = _weather_kb()
    effects = downstream_effects(kb, "rain", max_depth=3)
    assert effects
    for n in effects:
        assert isinstance(n, CausalNode)
        assert n.via  # non-empty chain


def test_conscious_agent_causal_methods():
    """ConsciousAgent.downstream_effects and root_causes wrap the call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    for s, r, o in [
        ("rain", "causes", "wet"),
        ("wet", "causes", "slippery"),
    ]:
        agent.tell(s, r, o)
    effects = agent.downstream_effects("rain", max_depth=2)
    assert any(n.entity == "wet" for n in effects)
    causes = agent.root_causes("slippery", max_depth=2)
    assert any(n.entity == "wet" for n in causes)
