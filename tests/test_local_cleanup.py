"""Tests for shard-local cleanup (v15.3).

A shard's bundle is the sum of its own facts' vectors, so the only
signal that can be unbound from it is a symbol stored in that shard.
Cleaning up against the shard's own unknown-role symbols instead of
the global codebook makes per-query cost independent of vocabulary
size (the enabler for six-figure KBs) and strictly reduces
cross-shard false positives. Candidates are collected fresh from the
fact log on every query, so no cache can go stale under store /
forget / merge / session-load.
"""
import numpy as np
import pytest

from rck.codebook import Codebook
from rck.knowledge_base import ShardedKnowledgeBase
from rck.relational import RelationalMemory


def _mem(facts, dim=4096):
    cb = Codebook(dim=dim, seed=0)
    mem = RelationalMemory(dim=dim, seed=0, role_names=("S", "R", "O"))
    for s, r, o in facts:
        mem.store(cb, {"S": s, "R": r, "O": o})
    return cb, mem


FACTS = [
    ("dog", "isa", "mammal"),
    ("cat", "isa", "mammal"),
    ("paris", "capitalof", "france"),
]


def test_local_cleanup_finds_stored_fact():
    cb, mem = _mem(FACTS)
    hits = mem.query(cb, {"S": "dog", "R": "isa"}, "O", top_k=2)
    assert hits and str(hits[0][0]) == "mammal"


def test_local_matches_global_score_for_winner():
    # The winning symbol's cosine is probe-vs-atom either way; only the
    # candidate set differs, so the top hit and score must agree with
    # the legacy global cleanup on a clean present fact.
    cb, mem = _mem(FACTS)
    local = mem.query(cb, {"S": "dog", "R": "isa"}, "O", top_k=1)
    global_ = mem.query(cb, {"S": "dog", "R": "isa"}, "O", top_k=1,
                        cleanup="global")
    assert str(local[0][0]) == str(global_[0][0]) == "mammal"
    assert local[0][1] == pytest.approx(global_[0][1])


def test_local_candidates_are_only_shard_unknown_role_symbols():
    cb, mem = _mem(FACTS)
    # Mint plenty of distractor atoms in the global codebook.
    for i in range(200):
        cb.encode(f"distractor_{i}")
    hits = mem.query(cb, {"S": "dog", "R": "isa"}, "O", top_k=10)
    returned = {str(sym) for sym, _ in hits}
    assert returned <= {"mammal", "france"}  # O-role symbols only


def test_local_cleanup_excludes_other_role_symbols():
    cb, mem = _mem(FACTS)
    hits = mem.query(cb, {"S": "dog", "R": "isa"}, "O", top_k=10)
    returned = {str(sym) for sym, _ in hits}
    assert "dog" not in returned and "isa" not in returned


def test_empty_memory_returns_empty():
    cb = Codebook(dim=2048, seed=0)
    mem = RelationalMemory(dim=2048, seed=0, role_names=("S", "R", "O"))
    assert mem.query(cb, {"S": "x", "R": "y"}, "O") == []


def test_local_reflects_forget_immediately():
    cb, mem = _mem([("dog", "isa", "mammal")])
    mem.forget(cb, {"S": "dog", "R": "isa", "O": "mammal"})
    hits = mem.query(cb, {"S": "dog", "R": "isa"}, "O", top_k=3)
    assert all(str(sym) != "mammal" or score < 0.15 for sym, score in hits)
    # With no facts left there are no local candidates at all.
    assert hits == []


def test_local_reflects_shard_merge_immediately():
    cb, a = _mem([("dog", "isa", "mammal")])
    _, b = _mem([("sun", "color", "yellow")])
    a.merge(b)
    hits = a.query(cb, {"S": "sun", "R": "color"}, "O", top_k=2)
    assert hits and str(hits[0][0]) == "yellow"


def test_sharded_kb_uses_local_cleanup_by_default():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    for s, r, o in FACTS:
        kb.store({"S": s, "R": r, "O": o})
    for i in range(300):
        kb.codebook.encode(f"noise_{i}")
    ans, score = kb.answer({"S": "paris", "R": "capitalof"}, "O")
    assert str(ans) == "france" and score > 0.3


def test_sharded_kb_can_opt_into_global_cleanup():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    for s, r, o in FACTS:
        kb.store({"S": s, "R": r, "O": o})
    hits = kb.query({"S": "dog", "R": "isa"}, "O", top_k=1,
                    cleanup="global")
    assert hits and str(hits[0][0]) == "mammal"


def test_unknown_cleanup_mode_raises():
    cb, mem = _mem(FACTS)
    with pytest.raises(ValueError, match="cleanup"):
        mem.query(cb, {"S": "dog", "R": "isa"}, "O", cleanup="banana")
