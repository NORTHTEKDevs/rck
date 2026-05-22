"""Tests for SparseRelationalMemory + SparseShardedKnowledgeBase.

These tests pin down the regime where sparse-binary HRR bundling
actually works (small N per shard) and document the capacity cliff
discovered in docs/design/v13-sparse-substrate-finding.md.
"""
from __future__ import annotations

from rck.sparse_hrr import SparseCodebook
from rck.sparse_relational import (
    SparseRelationalMemory, SparseShardedKnowledgeBase,
)


# ---- single-memory recall in the working regime ---------------------------

def test_sparse_memory_recalls_in_working_regime():
    """At N=3 facts in a SINGLE memory (D=8192, k=160) recall is 100%.

    This is well below the empirical capacity cliff (~10 facts/shard);
    a regression here means cleanup is fundamentally broken.
    """
    cb = SparseCodebook(dim=8192, k=160, seed=0)
    mem = SparseRelationalMemory(dim=8192, k=160, seed=0,
                                  role_names=("S", "R", "O"))
    facts = [
        ("dog", "isa", "mammal"),
        ("elephant", "has", "tusks"),
        ("eagle", "isa", "bird"),
    ]
    for s, r, o in facts:
        for tok in (s, r, o):
            cb.encode(tok)
        mem.store(cb, {"S": s, "R": r, "O": o})
    correct = sum(
        1 for s, r, o in facts
        if mem.answer(cb, {"S": s, "R": r}, "O")[0] == o
    )
    assert correct == len(facts)


def test_sparse_memory_forget_restores():
    """forget() removes the fact and frees position counts."""
    cb = SparseCodebook(dim=8192, k=160, seed=0)
    mem = SparseRelationalMemory(dim=8192, k=160, seed=0,
                                  role_names=("S", "R", "O"))
    fact = {"S": "dog", "R": "isa", "O": "mammal"}
    cb.encode("dog"); cb.encode("isa"); cb.encode("mammal")
    mem.store(cb, fact)
    assert mem.size() == 1
    ans, _ = mem.answer(cb, {"S": "dog", "R": "isa"}, "O")
    assert ans == "mammal"
    mem.forget(cb, fact)
    assert mem.size() == 0
    # Empty memory -> nothing to score against -> returns (None, 0.0)
    ans, score = mem.answer(cb, {"S": "dog", "R": "isa"}, "O")
    assert ans is None and score == 0.0


def test_sparse_memory_query_excludes_subject_atoms():
    """Subject atoms get filtered from candidates (no XOR cancellation ties)."""
    cb = SparseCodebook(dim=8192, k=160, seed=0)
    mem = SparseRelationalMemory(dim=8192, k=160, seed=0,
                                  role_names=("S", "R", "O"))
    cb.encode("dog"); cb.encode("isa"); cb.encode("mammal")
    mem.store(cb, {"S": "dog", "R": "isa", "O": "mammal"})
    results = mem.query(cb, {"S": "dog", "R": "isa"}, "O", top_k=5)
    # "dog" should NOT appear among the candidates.
    syms = [str(s) for s, _ in results]
    assert "dog" not in syms
    # "isa" likewise.
    assert "isa" not in syms


# ---- capacity regression (documents the cliff) ----------------------------

def test_sparse_capacity_cliff_documented():
    """Regression check on the empirical capacity cliff. At D=8192/k=320
    the study reports recall=95% at N=20. We require >= 85% to absorb
    seed-to-seed variation."""
    cb = SparseCodebook(dim=8192, k=320, seed=0)
    mem = SparseRelationalMemory(dim=8192, k=320, seed=0,
                                  role_names=("S", "R", "O"))
    triples = [(f"s_{i}", f"r_{i % 4}", f"o_{i}") for i in range(20)]
    for s, r, o in triples:
        cb.encode(s); cb.encode(r); cb.encode(o)
        mem.store(cb, {"S": s, "R": r, "O": o})
    correct = sum(
        1 for s, r, o in triples
        if mem.answer(cb, {"S": s, "R": r}, "O")[0] == o
    )
    assert correct >= 17, f"got {correct}/20 -- regression vs documented cliff"


# ---- sharded API ----------------------------------------------------------

def test_sparse_sharded_kb_round_trip():
    kb = SparseShardedKnowledgeBase(dim=8192, k=160, n_shards=16, seed=0)
    facts = [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("eagle", "isa", "bird"),
    ]
    for s, r, o in facts:
        kb.store({"S": s, "R": r, "O": o})
    assert kb.size() == 4
    ans, _ = kb.answer({"S": "dog", "R": "isa"}, "O")
    assert ans == "mammal"
    ans, _ = kb.answer({"S": "eagle", "R": "isa"}, "O")
    assert ans == "bird"


def test_sparse_sharded_kb_memory_bytes_is_small():
    """Sparse KB memory footprint is dominated by the count tensors, not by
    the codebook -- and it stays within expectations."""
    kb = SparseShardedKnowledgeBase(dim=8192, k=160, n_shards=8, seed=0)
    for i in range(10):
        kb.store({"S": f"sub_{i}", "R": "isa", "O": f"obj_{i}"})
    # 8 shards * 8192 positions * 4 bytes = 262144 just for counts.
    bytes_ = kb.memory_bytes()
    assert bytes_ < 1_000_000  # ~1MB upper bound at this config
    assert bytes_ >= 8 * 8192 * 4
