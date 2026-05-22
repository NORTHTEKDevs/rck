"""Tests for provenance-aware confidence calibration."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.confidence_calibration import (
    CalibratedAnswer, CalibrationPolicy, calibrated_lookup, calibrated_score,
)
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


# ---- pure scoring ---------------------------------------------------------

def test_user_source_keeps_full_score():
    s = calibrated_score(0.5, "user", via_hops=1)
    assert abs(s - 0.5) < 1e-9


def test_induced_discount_grows_with_hops():
    one_hop = calibrated_score(0.5, "induced", via_hops=1)
    two_hop = calibrated_score(0.5, "induced", via_hops=2)
    three_hop = calibrated_score(0.5, "induced", via_hops=3)
    assert one_hop > two_hop > three_hop


def test_induced_floor_caps_decay():
    """Long-chain induced facts never decay below the floor."""
    extreme = calibrated_score(0.5, "induced", via_hops=50)
    policy = CalibrationPolicy()
    assert extreme >= 0.5 * policy.induced_floor


def test_multi_source_gets_small_bonus():
    s = calibrated_score(0.5, "multi", via_hops=1)
    assert s > 0.5


def test_unknown_source_falls_through():
    s = calibrated_score(0.5, "from_some_external_thing", via_hops=1)
    assert abs(s - 0.5) < 1e-9  # unknown_factor = 1.0


# ---- end-to-end lookup ----------------------------------------------------

def test_calibrated_lookup_prefers_user_over_induced():
    """If a (S, R) has two candidate Os, one user, one induced, the user
    answer wins after calibration even if the induced was slightly
    higher in raw HRR score."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    prov = ProvenanceStore()
    # Two facts with the same S+R.
    bulk_load_triples(kb, [("city", "in", "europe")])
    prov.store("city", "in", "europe", source="user")
    kb.store({"S": "city", "R": "in", "O": "atlantis"})
    prov.store("city", "in", "atlantis", source="induced",
               tags={"induced", "via_3_hops"})
    results = calibrated_lookup(kb, prov,
                                 {"S": "city", "R": "in"}, "O", top_k=3)
    assert results
    top = results[0]
    assert top.symbol == "europe"
    assert top.source == "user"


def test_calibrated_lookup_records_source_and_hops():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    prov = ProvenanceStore()
    kb.store({"S": "leaf", "R": "locatedin", "O": "forest"})
    prov.store("leaf", "locatedin", "forest", source="induced",
               tags={"induced", "via_2_hops"})
    results = calibrated_lookup(kb, prov,
                                 {"S": "leaf", "R": "locatedin"}, "O")
    forest = next(r for r in results if r.symbol == "forest")
    assert forest.source == "induced"
    assert forest.via_hops == 2
    assert forest.calibrated_score < forest.raw_score


def test_conscious_agent_records_provenance_on_tell():
    """ConsciousAgent.tell() now also writes a provenance record
    with source='user'."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    rec = agent.provenance.get("dog", "isa", "mammal")
    assert rec is not None
    assert rec.source == "user"


def test_conscious_agent_calibrated_ask():
    """calibrated_ask returns CalibratedAnswer rows pulled from the
    agent's own provenance store."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    rows = agent.calibrated_ask({"S": "dog", "R": "isa"}, "O")
    assert rows
    top = rows[0]
    assert top.symbol == "mammal"
    assert top.source == "user"
    assert abs(top.calibrated_score - top.raw_score) < 1e-9


def test_calibrated_lookup_unknown_provenance_keeps_raw():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    prov = ProvenanceStore()  # empty -- no records at all
    kb.store({"S": "dog", "R": "isa", "O": "mammal"})
    results = calibrated_lookup(kb, prov,
                                 {"S": "dog", "R": "isa"}, "O")
    top = next(r for r in results if r.symbol == "mammal")
    assert top.source == "unknown"
    # With unknown_factor = 1.0, calibrated == raw.
    assert abs(top.calibrated_score - top.raw_score) < 1e-9
