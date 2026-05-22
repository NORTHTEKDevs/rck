"""Tests for chain discovery."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.chain_discover import DiscoveredChain, Goal, discover_chains
from rck.chain_walker import Hop, walk_chain
from rck.knowledge_base import ShardedKnowledgeBase


def _build_geo_kb() -> ShardedKnowledgeBase:
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


# ---- exact-symbol goal ----------------------------------------------------

def test_discover_two_hop_to_symbol():
    kb = _build_geo_kb()
    chains = discover_chains(kb, "france", Goal.symbol("europe"), max_depth=3)
    assert chains
    best = chains[0]
    assert isinstance(best, DiscoveredChain)
    assert best.relations == ["capital", "locatedin"]


def test_discovered_chain_is_walkable():
    """A discovered chain can be passed to walk_chain and produce the same answer."""
    kb = _build_geo_kb()
    chains = discover_chains(kb, "france", Goal.symbol("europe"), max_depth=3)
    assert chains
    best = chains[0]
    hops = [Hop(r, d) for r, d in zip(best.relations, best.directions)]
    res = walk_chain(kb, "france", hops)
    assert res.answer == "europe"


# ---- relation-value goal --------------------------------------------------

def test_discover_chain_to_continent():
    """Find a chain from france to any node X with (X, isa, continent)."""
    kb = _build_geo_kb()
    chains = discover_chains(
        kb, "france", Goal.relation_value("isa", "continent"),
        max_depth=3,
    )
    assert chains
    # The reached node should be europe.
    last_obj = chains[0].trace[-1][2]
    assert last_obj == "europe"


def test_no_chain_when_disconnected():
    kb = _build_geo_kb()
    chains = discover_chains(
        kb, "france", Goal.symbol("nonexistent_node"), max_depth=3,
    )
    assert chains == []


# ---- relation restriction -------------------------------------------------

def test_discover_respects_relation_whitelist():
    """If we restrict to ['capital'] we should NOT reach europe in one hop."""
    kb = _build_geo_kb()
    chains = discover_chains(
        kb, "france", Goal.symbol("europe"), max_depth=3,
        relations=["capital"],
    )
    assert chains == []


# ---- beam width -----------------------------------------------------------

def test_discover_uses_reverse_edges_when_allowed():
    """With allow_reverse=True, search can traverse (?, R, O) to find paths
    that don't exist in pure forward direction. We avoid bulk_load_triples
    here so its auto-inverse symmetrisation doesn't make the test trivial."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    for s, r, o in [
        ("paris", "locatedin", "france"),
        ("france", "isa", "country"),
    ]:
        kb.store({"S": s, "R": r, "O": o})
    no_reverse = discover_chains(
        kb, "country", Goal.symbol("paris"), max_depth=3,
        allow_reverse=False,
    )
    assert no_reverse == []
    with_reverse = discover_chains(
        kb, "country", Goal.symbol("paris"), max_depth=3,
        allow_reverse=True, min_link_score=0.20,
    )
    assert with_reverse
    chain = with_reverse[0]
    # The reverse direction MUST appear at least once: there's no forward
    # path from country to paris in the unsymmetrised KB.
    assert "reverse" in chain.directions
    assert chain.trace[-1][2] == "paris"


def test_conscious_agent_discover_then_reason():
    """ConsciousAgent.discover() returns a chain spec ready for reason()."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("france", "capital", "paris"),
        ("paris", "locatedin", "europe"),
        ("europe", "isa", "continent"),
    ]:
        agent.tell(s, r, o)
    spec = agent.discover("france", "continent", max_depth=4)
    assert spec is not None
    assert spec["relations"] == ["capital", "locatedin", "isa"]
    res = agent.reason("france", spec["relations"],
                       directions=spec["directions"])
    assert res["answer"] == "continent"


def test_discover_with_skill_prior_finds_same_chains():
    """Skill-prior is a heuristic reordering; it must not change which
    chains are findable, only the order they come out."""
    from rck.skills import SkillLibrary
    kb = _build_geo_kb()
    skills = SkillLibrary()
    # Pre-warm skills with the canonical capital+locatedin pattern.
    for _ in range(3):
        skills.record_success([("O", "capital"), ("O", "locatedin")])
    no_prior = discover_chains(kb, "france", Goal.symbol("europe"), max_depth=3)
    with_prior = discover_chains(kb, "france", Goal.symbol("europe"),
                                  max_depth=3, skills_prior=skills)
    assert no_prior and with_prior
    # Both must find the canonical chain.
    sigs_no = {tuple(c.relations) for c in no_prior}
    sigs_with = {tuple(c.relations) for c in with_prior}
    assert ("capital", "locatedin") in sigs_no
    assert ("capital", "locatedin") in sigs_with


def test_relation_priority_orders_by_success_weight():
    """The internal priority map is the success_count * confidence sum."""
    from rck.chain_discover import _relation_priority_from_skills
    from rck.skills import SkillLibrary
    skills = SkillLibrary()
    # `partof` should rank higher than `capital`.
    for _ in range(10):
        skills.record_success([("O", "partof"), ("O", "locatedin")])
    skills.record_success([("O", "capital"), ("O", "locatedin")])
    pri = _relation_priority_from_skills(skills)
    assert pri["partof"] > pri["capital"]
    assert pri["locatedin"] >= pri["partof"]  # appears in both


def test_top_n_returns_distinct_chains_when_possible():
    """A KB with multiple paths from france to a continent should yield
    one chain (the canonical capital->locatedin->isa)."""
    kb = _build_geo_kb()
    chains = discover_chains(
        kb, "france", Goal.relation_present("isa"),
        max_depth=3, top_n=5,
    )
    assert len(chains) >= 1
    # Distinct chain signatures.
    sigs = {tuple(c.relations) for c in chains}
    assert len(sigs) == len(chains)
