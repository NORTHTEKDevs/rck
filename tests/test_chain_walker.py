"""Tests for the n-hop chain walker."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.chain_walker import ChainResult, Hop, try_chains, walk_chain
from rck.confidence_propagation import PropagationConfig
from rck.knowledge_base import ShardedKnowledgeBase
from rck.skills import SkillLibrary


def _kb_with_geo_chain() -> ShardedKnowledgeBase:
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("france", "capital", "paris"),
        ("paris", "locatedin", "europe"),
        ("europe", "isa", "continent"),
        ("germany", "capital", "berlin"),
        ("berlin", "locatedin", "europe"),
        ("japan", "capital", "tokyo"),
        ("tokyo", "locatedin", "asia"),
        ("asia", "isa", "continent"),
    ])
    return kb


# ---- 2-hop -----------------------------------------------------------------

def test_two_hop_forward_chain():
    kb = _kb_with_geo_chain()
    chain = [Hop("capital"), Hop("locatedin")]
    res = walk_chain(kb, "france", chain)
    assert isinstance(res, ChainResult)
    assert res.answer == "europe"
    assert res.aborted_at == -1
    assert len(res.trace) == 2
    assert res.confidence > 0.05


def test_two_hop_aborts_on_missing_relation():
    kb = _kb_with_geo_chain()
    # `capital` exists; `industry` does not.
    chain = [Hop("capital"), Hop("industry")]
    res = walk_chain(kb, "france", chain)
    assert res.answer is None
    assert res.aborted_at == 1


# ---- 3-hop -----------------------------------------------------------------

def test_three_hop_chain_propagates_confidence():
    kb = _kb_with_geo_chain()
    chain = [Hop("capital"), Hop("locatedin"), Hop("isa")]
    res = walk_chain(kb, "france", chain)
    assert res.answer == "continent"
    # Confidence should be the product of link scores * decay; checks
    # that it is non-zero and floored sensibly.
    assert res.confidence > 0.0
    assert res.hedge in {"strong", "moderate", "weak"}


def test_chain_records_skill_on_success():
    kb = _kb_with_geo_chain()
    skills = SkillLibrary()
    chain = [Hop("capital"), Hop("locatedin")]
    res = walk_chain(kb, "france", chain, skills=skills)
    assert res.answer == "europe"
    assert len(skills.skills) == 1
    sk = next(iter(skills.skills.values()))
    assert sk.success_count == 1


# ---- reverse hops ---------------------------------------------------------

def test_reverse_hop_resolves_subject():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("paris", "locatedin", "france"),
        ("berlin", "locatedin", "germany"),
        ("france", "isa", "country"),
    ])
    # Chain: from "france", go REVERSE on locatedin to find a city,
    # then no further hop. Use a 1-hop chain.
    chain = [Hop("locatedin", direction="reverse")]
    res = walk_chain(kb, "france", chain)
    assert res.answer == "paris"


# ---- multi-candidate ------------------------------------------------------

def test_try_chains_picks_best():
    kb = _kb_with_geo_chain()
    chains = [
        [Hop("capital"), Hop("industry")],          # aborts
        [Hop("capital"), Hop("locatedin")],         # succeeds
        [Hop("capital"), Hop("locatedin"), Hop("isa")],  # also succeeds
    ]
    res = try_chains(kb, "france", chains)
    assert res is not None
    assert res.answer in {"europe", "continent"}


# ---- verbalisation --------------------------------------------------------

def test_verbalize_includes_chain():
    kb = _kb_with_geo_chain()
    chain = [Hop("capital"), Hop("locatedin")]
    res = walk_chain(kb, "france", chain)
    text = res.verbalize()
    assert "france" in text
    assert "paris" in text or "europe" in text


def test_conscious_agent_reason_walks_chain_and_records_skill():
    """The agent's reason() method walks an n-hop chain and updates skills."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("france", "capital", "paris"),
        ("paris", "locatedin", "europe"),
        ("europe", "isa", "continent"),
    ]:
        agent.tell(s, r, o)
    res = agent.reason("france", ["capital", "locatedin", "isa"])
    assert res["answer"] == "continent"
    assert res["confidence"] > 0.0
    assert agent.skills.stats()["n"] == 1


def test_propagation_config_affects_confidence():
    kb = _kb_with_geo_chain()
    chain = [Hop("capital"), Hop("locatedin"), Hop("isa")]
    aggressive = PropagationConfig(rule="product", chain_decay=0.5)
    gentle = PropagationConfig(rule="product", chain_decay=0.99)
    r1 = walk_chain(kb, "france", chain, config=aggressive)
    r2 = walk_chain(kb, "france", chain, config=gentle)
    # Both find an answer; gentle is at least as confident.
    assert r1.answer == r2.answer == "continent"
    assert r2.confidence >= r1.confidence
