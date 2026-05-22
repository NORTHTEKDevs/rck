"""Tests for explicit IDK detection."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.idk_detection import (
    EpistemicAnswer, EpistemicState, IDKPolicy, ask_with_idk, classify,
)
from rck.knowledge_base import ShardedKnowledgeBase


# ---- pure classification --------------------------------------------------

def test_classify_known_when_clear_winner():
    res = classify([("paris", 0.7), ("london", 0.05)])
    assert res.state is EpistemicState.KNOWN
    assert res.top_symbol == "paris"


def test_classify_idk_when_no_candidates():
    res = classify([])
    assert res.state is EpistemicState.IDK
    assert "No candidates" in res.explanation


def test_classify_idk_when_top_below_floor():
    res = classify([("paris", 0.05)])
    assert res.state is EpistemicState.IDK
    assert res.top_symbol == "paris"


def test_classify_ambiguous_when_top2_close():
    res = classify([("paris", 0.30), ("london", 0.28)])
    assert res.state is EpistemicState.AMBIGUOUS
    assert len(res.alternatives) == 2


def test_classify_known_when_top_2_far_apart():
    res = classify([("paris", 0.50), ("london", 0.10)])
    assert res.state is EpistemicState.KNOWN


def test_classify_respects_custom_policy():
    """Tighter known_threshold makes a moderate-score answer IDK."""
    policy = IDKPolicy(known_threshold=0.80, idk_threshold=0.05)
    res = classify([("paris", 0.50)], policy=policy)
    assert res.state is EpistemicState.IDK


# ---- verbalisation --------------------------------------------------------

def test_verbalize_known():
    res = classify([("paris", 0.7), ("london", 0.05)])
    text = res.verbalize()
    assert "confident" in text and "paris" in text


def test_verbalize_idk():
    res = classify([("paris", 0.05)])
    text = res.verbalize()
    assert "don't know" in text


# ---- end-to-end with KB ---------------------------------------------------

def test_ask_with_idk_returns_known_for_stored_fact():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [("dog", "isa", "mammal")])
    res = ask_with_idk(kb, {"S": "dog", "R": "isa"}, "O")
    assert res.state is EpistemicState.KNOWN
    assert res.top_symbol == "mammal"


def test_conscious_agent_ask_with_idk():
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    res = agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    assert res.state is EpistemicState.KNOWN
    res2 = agent.ask_with_idk({"S": "qqqqqq_nope", "R": "isa"}, "O")
    assert res2.state is EpistemicState.IDK


def test_ask_with_idk_returns_idk_for_unknown_subject():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [("dog", "isa", "mammal")])
    # Subject not in codebook -> tightly low cosine on everything.
    res = ask_with_idk(kb, {"S": "xyzzy_unknown", "R": "isa"}, "O")
    assert res.state is EpistemicState.IDK
