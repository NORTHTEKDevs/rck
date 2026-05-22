"""Tests for v5 modules: provenance, memory hierarchies, universes,
curiosity, abduction."""
import time

from rck.abduction import candidates_for_effect, candidates_for_property, explain
from rck.bulk_ingest import bulk_load_triples
from rck.curiosity import _siblings_of, detect_gaps, detect_global_gaps
from rck.knowledge_base import ShardedKnowledgeBase
from rck.memory_hierarchy import (
    EpisodicMemory, ProceduralMemory, WorkingMemory,
    consolidate_episodic_to_semantic,
)
from rck.provenance import ProvenanceRecord, ProvenanceStore
from rck.universes import Universe, UniverseManager


# ---- provenance ------------------------------------------------------------

def test_provenance_store_and_retrieve():
    store = ProvenanceStore()
    store.store("sky", "color", "blue", source="user_test")
    rec = store.get("sky", "color", "blue")
    assert rec is not None
    assert rec.source == "user_test"
    assert rec.confidence == 1.0
    assert rec.count == 1


def test_provenance_reinforce_increments_count():
    store = ProvenanceStore()
    store.store("sky", "color", "blue", source="A")
    store.store("sky", "color", "blue", source="A")
    rec = store.get("sky", "color", "blue")
    assert rec.count == 2


def test_provenance_decay_lowers_confidence():
    rec = ProvenanceRecord(confidence=1.0)
    rec.decay(factor=0.5)
    assert rec.confidence == 0.5


def test_provenance_low_confidence_facts():
    store = ProvenanceStore()
    store.store("a", "is", "b", confidence=0.5)
    store.store("c", "is", "d", confidence=0.05)
    low = store.low_confidence_facts(threshold=0.1)
    assert ("c", "is", "d") in low
    assert ("a", "is", "b") not in low


# ---- working memory ------------------------------------------------------

def test_working_memory_capacity_bounded():
    wm = WorkingMemory(capacity=3)
    for x in ["a", "b", "c", "d"]:
        wm.push(x)
    items = [it.content for it in wm.all()]
    assert items == ["b", "c", "d"]


# ---- episodic memory -----------------------------------------------------

def test_episodic_record_and_filter():
    ep = EpisodicMemory()
    ep.record("user", "asked", "What is X?")
    ep.record("system", "answered", "X is Y.")
    ep.record("user", "asked", "What is Z?")
    by_user = ep.by_actor("user")
    assert len(by_user) == 2
    answered = ep.by_kind("answered")
    assert len(answered) == 1


def test_episodic_in_time_window():
    ep = EpisodicMemory()
    t0 = time.time()
    ep.record("user", "asked", "Q1")
    time.sleep(0.01)
    t1 = time.time()
    ep.record("user", "asked", "Q2")
    in_window = ep.in_window(t0, t1 + 1.0)
    assert len(in_window) == 2


def test_consolidate_finds_recurring_patterns():
    ep = EpisodicMemory()
    for _ in range(4):
        ep.record("user", "told", "the sky is blue")
    ep.record("user", "told", "the grass is green")
    consolidated = consolidate_episodic_to_semantic(ep, threshold=3)
    patterns = [c for c, n in consolidated]
    assert "the sky is blue" in patterns
    assert "the grass is green" not in patterns


# ---- procedural memory ----------------------------------------------------

def test_procedural_records_usage_and_success():
    pm = ProceduralMemory()
    proc = pm.store("lookup", "fact lookup", ["query", "render"])
    proc.record_use(True)
    proc.record_use(True)
    proc.record_use(False)
    assert proc.success_rate() == 2 / 3


# ---- universes ------------------------------------------------------------

def test_universe_root_and_branch():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [("paris", "capital_of", "france")], symmetrize=False)
    mgr = UniverseManager(kb=kb)
    root = mgr.root()
    branch = mgr.branch("hypothetical")
    # Both can read the existing fact.
    ans, _ = root.answer("paris", "capital_of")
    assert ans == "france"
    ans, _ = branch.answer("paris", "capital_of")
    assert ans == "france"


def test_universe_modification_then_discard():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [("sky", "color", "blue")], symmetrize=False)
    mgr = UniverseManager(kb=kb)
    branch = mgr.branch("what_if")
    branch.tell("sky", "color", "purple")
    # In the branch, querying "color of sky" might now return purple
    # OR blue (both stored). The point is discard() must restore.
    branch.discard()
    # Should be only "blue" again.
    ans, _ = mgr.root().answer("sky", "color")
    assert ans == "blue"


def test_universe_forget_then_discard_restores():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [("sky", "color", "blue")], symmetrize=False)
    mgr = UniverseManager(kb=kb)
    branch = mgr.branch("what_if")
    branch.forget("sky", "color", "blue")
    # Should be gone in the branch (and in shared kb until undo).
    ans, score = branch.answer("sky", "color")
    assert ans != "blue" or score < 0.1
    # Discard restores.
    branch.discard()
    ans, _ = mgr.root().answer("sky", "color")
    assert ans == "blue"


# ---- curiosity ------------------------------------------------------------

def test_siblings_via_shared_isa_parent():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("elephant", "isa", "mammal"),
    ], symmetrize=False)
    siblings = _siblings_of(kb, "dog")
    assert "cat" in siblings
    assert "elephant" in siblings
    assert "dog" not in siblings


def test_detect_gaps_finds_missing_property():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    facts = [
        # 4 mammals all have fur, but platypus doesn't have fur recorded.
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("rabbit", "isa", "mammal"),
        ("bear", "isa", "mammal"),
        ("platypus", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("cat", "has", "fur"),
        ("rabbit", "has", "fur"),
        ("bear", "has", "fur"),
    ]
    bulk_load_triples(kb, facts, symmetrize=False)
    gaps = detect_gaps(kb, "platypus", min_agreement=0.5, min_siblings=2)
    # Expected: gap for relation 'has' (most siblings have 'has' fur).
    relations = {g.relation for g in gaps}
    assert "has" in relations


# ---- abduction ------------------------------------------------------------

def test_abduction_for_effect():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("rain", "causes", "wetness"),
        ("sweat", "causes", "wetness"),
        ("spill", "causes", "wetness"),
    ], symmetrize=False)
    candidates = candidates_for_effect(kb, "causes", "wetness")
    causes = [c.cause for c in candidates]
    assert any(c in causes for c in ("rain", "sweat", "spill"))


def test_abduction_for_property():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("bird", "has", "feathers"),
        ("eagle", "isa", "bird"),
        ("sparrow", "isa", "bird"),
    ], symmetrize=False)
    candidates = candidates_for_property(kb, "has", "feathers")
    causes = [c.cause for c in candidates]
    assert "bird" in causes


def test_explain_returns_verbal_answer():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("rain", "causes", "wetness"),
        ("sweat", "causes", "wetness"),
    ], symmetrize=False)
    res = explain(kb, "wetness")
    assert res["best"] is not None
    assert "wetness" in res["verbal"] or res["best"].cause in res["verbal"]
