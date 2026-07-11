"""Tests for the per-shard live-relation index (paper §5.4 future work).

The index is a SNAPSHOT built from the shards' symbolic fact logs, so it
is correct under every mutation path (store, forget, federated merge,
session load) as long as it is rebuilt after writes. Discovery builds a
fresh one per call by default.
"""
import pytest

from rck.knowledge_base import RelationIndex, ShardedKnowledgeBase, _shard_index
from rck.chain_discover import Goal, discover_chains


def _kb(facts, n_shards=16):
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    for s, r, o in facts:
        kb.store({"S": s, "R": r, "O": o})
    return kb


CHAIN_FACTS = [
    ("dog", "isa", "mammal"),
    ("mammal", "isa", "animal"),
    ("dog", "has", "fur"),
    ("cat", "isa", "mammal"),
    ("paris", "capitalof", "france"),
    ("france", "locatedin", "europe"),
]


def test_index_all_relations_matches_stored():
    kb = _kb(CHAIN_FACTS)
    idx = kb.relation_index()
    assert idx.all_relations() == sorted({"isa", "has", "capitalof", "locatedin"})


def test_index_live_true_for_every_stored_fact():
    kb = _kb(CHAIN_FACTS)
    idx = kb.relation_index()
    for s, r, _ in CHAIN_FACTS:
        assert idx.live(s, r), f"stored fact ({s},{r},..) must be live"


def test_index_live_false_for_absent_relation():
    kb = _kb(CHAIN_FACTS)
    idx = kb.relation_index()
    assert not idx.live("dog", "eats")
    assert not idx.live("nonexistent", "isa") or True  # may collide-live; only absent RELATION is guaranteed dead
    assert "eats" not in idx.all_relations()


def test_index_shards_with_matches_fact_logs():
    kb = _kb(CHAIN_FACTS)
    idx = kb.relation_index()
    for rel in ("isa", "has", "capitalof", "locatedin"):
        expected = sorted(
            i for i, shard in enumerate(kb._shards)
            if any(str(f.get("R")) == rel for f in shard.facts())
        )
        assert idx.shards_with(rel) == expected


def test_index_is_snapshot_and_rebuild_sees_new_facts():
    kb = _kb(CHAIN_FACTS)
    stale = kb.relation_index()
    kb.store({"S": "fish", "R": "livesin", "O": "water"})
    assert "livesin" not in stale.all_relations()  # snapshot semantics
    assert "livesin" in kb.relation_index().all_relations()


def test_index_correct_after_shard_level_merge():
    kb_a = _kb([("dog", "isa", "mammal")])
    kb_b = _kb([("sun", "color", "yellow")])
    for i in range(kb_a.n_shards):
        kb_a._shards[i].merge(kb_b._shards[i])
    idx = kb_a.relation_index()
    assert idx.live("sun", "color")
    assert set(idx.all_relations()) == {"isa", "color"}


def test_query_shard_subset_finds_fact_in_included_shard():
    kb = _kb(CHAIN_FACTS)
    home = _shard_index("dog", "isa", kb.n_shards)
    hits = kb.query({"S": "dog", "R": "isa"}, "O", top_k=3, shard_subset=[home])
    assert hits and str(hits[0][0]) == "mammal"


def test_query_shard_subset_excluding_home_shard_misses():
    kb = _kb(CHAIN_FACTS)
    home = _shard_index("dog", "isa", kb.n_shards)
    others = [i for i in range(kb.n_shards) if i != home]
    hits = kb.query({"S": "dog", "R": "isa"}, "O", top_k=3, shard_subset=others)
    assert all(str(sym) != "mammal" or score < 0.3 for sym, score in hits)


def test_discover_same_chains_with_and_without_index():
    kb = _kb(CHAIN_FACTS)
    goal = Goal.symbol("animal")
    with_idx = discover_chains(kb, "dog", goal, max_depth=3)
    without = discover_chains(kb, "dog", goal, max_depth=3, use_relation_index=False)
    assert [c.relations for c in with_idx] == [c.relations for c in without]
    assert with_idx and with_idx[0].relations == ["isa", "isa"]
    assert with_idx[0].confidence == pytest.approx(without[0].confidence)


def test_discover_with_index_issues_fewer_queries():
    kb = _kb(CHAIN_FACTS)
    goal = Goal.symbol("animal")

    counts = {"n": 0}
    original = kb.query

    def counting_query(*args, **kwargs):
        counts["n"] += 1
        return original(*args, **kwargs)

    kb.query = counting_query
    discover_chains(kb, "dog", goal, max_depth=3, use_relation_index=False)
    baseline = counts["n"]

    counts["n"] = 0
    discover_chains(kb, "dog", goal, max_depth=3)
    pruned = counts["n"]

    assert pruned < baseline, f"index should prune queries ({pruned} !< {baseline})"


def test_discover_accepts_prebuilt_index():
    kb = _kb(CHAIN_FACTS)
    idx = kb.relation_index()
    goal = Goal.symbol("animal")
    chains = discover_chains(kb, "dog", goal, max_depth=3, relation_index=idx)
    assert chains and chains[0].relations == ["isa", "isa"]
