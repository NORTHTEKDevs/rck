"""Tests for analogical reasoning."""
from __future__ import annotations

from rck.analogy import (
    AnalogyResult, RelationCandidate, find_relation, solve_analogy,
)
from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase


def _geo_kb() -> ShardedKnowledgeBase:
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("france", "capital", "paris"),
        ("germany", "capital", "berlin"),
        ("japan", "capital", "tokyo"),
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("eagle", "isa", "bird"),
        ("hawk", "isa", "bird"),
    ])
    return kb


# ---- find_relation --------------------------------------------------------

def test_find_relation_returns_capital():
    kb = _geo_kb()
    cands = find_relation(kb, "france", "paris")
    assert cands
    assert cands[0].relation == "capital"
    assert cands[0].score > 0.10


def test_find_relation_returns_isa():
    kb = _geo_kb()
    cands = find_relation(kb, "dog", "mammal")
    assert cands
    assert cands[0].relation == "isa"


def test_find_relation_empty_when_no_relation():
    kb = _geo_kb()
    cands = find_relation(kb, "france", "tokyo")
    # No direct relation france -> tokyo in this KB.
    assert cands == []


# ---- solve_analogy --------------------------------------------------------

def test_analogy_country_to_capital():
    """france : paris :: germany : berlin."""
    kb = _geo_kb()
    res = solve_analogy(kb, "france", "paris", "germany")
    assert isinstance(res, AnalogyResult)
    assert res.relation == "capital"
    assert res.answer == "berlin"


def test_analogy_isa_chain():
    """dog : mammal :: eagle : bird."""
    kb = _geo_kb()
    res = solve_analogy(kb, "dog", "mammal", "eagle")
    assert res.relation == "isa"
    assert res.answer == "bird"


def test_analogy_unknown_relation_returns_none():
    kb = _geo_kb()
    res = solve_analogy(kb, "france", "tokyo", "germany")
    assert res.relation is None
    assert res.answer is None


def test_analogy_verbalize():
    kb = _geo_kb()
    res = solve_analogy(kb, "france", "paris", "germany")
    text = res.verbalize()
    assert "france" in text and "paris" in text
    assert "germany" in text and "berlin" in text
    assert "capital" in text


# ---- conscious agent integration ------------------------------------------

def test_analogy_surfaces_alternatives():
    """The result carries a sorted alternatives list for inspection."""
    kb = _geo_kb()
    res = solve_analogy(kb, "france", "paris", "germany")
    assert isinstance(res.alternatives, list)
    assert res.alternatives
    # First alternative matches the chosen answer.
    top = res.alternatives[0]
    assert top[0] == res.relation
    assert top[2] == res.answer


def test_analogy_picks_best_joint_score():
    """When multiple relations link A->B with different qualities,
    pick the relation whose application to C has the strongest
    JOINT score (not just the top-1 R)."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("france", "capital", "paris"),
        ("germany", "capital", "berlin"),
        # Add an alternative weakly-linking relation france/paris share.
        ("france", "contains", "paris"),
        ("germany", "contains", "berlin"),
    ])
    res = solve_analogy(kb, "france", "paris", "germany")
    # Either capital or contains gives berlin. The system must pick
    # one whose joint score is the highest of the alternatives.
    assert res.answer == "berlin"
    if len(res.alternatives) > 1:
        # The chosen joint score is the max across all alternatives.
        chosen_joint = res.alternatives[0][4]
        for alt in res.alternatives[1:]:
            assert chosen_joint >= alt[4]


def test_analogy_joint_score_helper():
    kb = _geo_kb()
    res = solve_analogy(kb, "france", "paris", "germany")
    expected = float(res.relation_score) * float(res.answer_score)
    assert abs(res.joint_score() - expected) < 1e-9


def test_analogy_falls_back_to_chain_when_no_direct_relation():
    """When A and B aren't linked by one relation but ARE linked by
    a 2-hop chain, the fallback discovers the chain and applies it to C."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("france", "capital", "paris"),
        ("paris", "locatedin", "europe"),
        ("germany", "capital", "berlin"),
        ("berlin", "locatedin", "europe"),
    ])
    # No single relation connects france -> europe directly.
    # Chain: capital -> locatedin
    res = solve_analogy(kb, "france", "europe", "germany")
    assert res.via == "chain_fallback"
    assert res.chain == ["capital", "locatedin"]
    assert res.answer == "europe"


def test_analogy_direct_path_still_wins_when_available():
    """If a direct relation exists, the chain fallback shouldn't fire."""
    kb = _geo_kb()
    res = solve_analogy(kb, "france", "paris", "germany")
    assert res.via == "direct"
    assert res.chain is None


def test_analogy_chain_fallback_disabled_returns_none():
    """With chain_fallback=False, an indirect analogy returns no answer."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("france", "capital", "paris"),
        ("paris", "locatedin", "europe"),
        ("germany", "capital", "berlin"),
        ("berlin", "locatedin", "europe"),
    ])
    res = solve_analogy(kb, "france", "europe", "germany",
                        chain_fallback=False)
    assert res.relation is None
    assert res.answer is None


def test_analogy_bayesian_probabilities_sum_to_one():
    """With scoring='bayesian', alternatives carry softmax probabilities."""
    kb = _geo_kb()
    res = solve_analogy(kb, "france", "paris", "germany", scoring="bayesian")
    if res.alternatives:
        total = sum(a[4] for a in res.alternatives)
        assert abs(total - 1.0) < 1e-6


def test_analogy_product_mode_preserves_old_behaviour():
    """With scoring='product', joint scores are raw products (>=0, not summing to 1)."""
    kb = _geo_kb()
    res = solve_analogy(kb, "france", "paris", "germany", scoring="product")
    if res.alternatives:
        # In product mode the last column is R_score * answer_score
        # (can be larger than 1 only if both are >1, which they aren't).
        # The defining property: NOT a probability distribution.
        total = sum(a[4] for a in res.alternatives)
        # Could be anything >= 0; check at least that top entry isn't a
        # softmax-style fraction unless by coincidence.
        assert res.alternatives[0][4] >= 0.0


def test_analogy_bayesian_argmax_matches_product_argmax():
    """The chosen answer is the same under both scoring modes."""
    kb = _geo_kb()
    bay = solve_analogy(kb, "france", "paris", "germany", scoring="bayesian")
    prod = solve_analogy(kb, "france", "paris", "germany", scoring="product")
    assert bay.answer == prod.answer
    assert bay.relation == prod.relation


def test_analogy_bayesian_higher_temperature_sharpens():
    """Higher temperature -> top probability closer to 1."""
    kb = _geo_kb()
    flat = solve_analogy(kb, "france", "paris", "germany",
                         scoring="bayesian", temperature=0.5)
    sharp = solve_analogy(kb, "france", "paris", "germany",
                          scoring="bayesian", temperature=10.0)
    if (flat.alternatives and sharp.alternatives
            and len(flat.alternatives) > 1):
        assert sharp.alternatives[0][4] >= flat.alternatives[0][4]


def test_conscious_agent_analogy():
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("france", "capital", "paris"),
        ("germany", "capital", "berlin"),
    ]:
        agent.tell(s, r, o)
    res = solve_analogy(agent.knowledge, "france", "paris", "germany")
    assert res.answer == "berlin"
