"""Tests for the shard-balance report."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase
from rck.shard_balance import ShardBalanceReport, report


def test_report_returns_basic_structure():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
    ])
    r = report(kb)
    assert isinstance(r, ShardBalanceReport)
    assert r.n_shards == 8
    assert r.dim == 4096
    assert len(r.fills) == 8
    assert r.target_fill > 0


def test_report_overloaded_flagged():
    """Cram many facts into a small shard count and check overloaded list."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=2, seed=0)
    # auto_reshard would grow this KB out of the overloaded state before we
    # could observe it. Reporting still has to work for KBs that opt out or
    # are restored from a checkpoint, which is what this test covers.
    kb.auto_reshard = False
    # Use distinct subjects so the bundle accumulates per shard.
    triples = [(f"s_{i}", "isa", f"o_{i}") for i in range(200)]
    bulk_load_triples(kb, triples)
    r = report(kb)
    # At least one shard should be overloaded for D=4096, target=80.
    assert r.overloaded


def test_report_suggests_reshard_when_overloaded():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=2, seed=0)
    kb.auto_reshard = False   # see test_report_overloaded_flagged
    bulk_load_triples(kb, [(f"s_{i}", "isa", f"o_{i}") for i in range(200)])
    r = report(kb)
    # Unconditional: with auto_reshard off this KB is definitely overloaded, so
    # an `if r.overloaded` guard here would let the assert pass vacuously.
    assert r.overloaded
    assert "reshard" in r.suggested_action


def test_report_sparse_shards_with_balanced_kb():
    """A KB with very few facts has many sparse shards."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=64, seed=0)
    bulk_load_triples(kb, [("dog", "isa", "mammal")])
    r = report(kb)
    assert r.sparse


def test_report_verbalize_text():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [("dog", "isa", "mammal")])
    r = report(kb)
    text = r.verbalize()
    assert "shards" in text
    assert "target_fill" in text


def test_conscious_agent_shard_balance():
    """ConsciousAgent.shard_balance() wraps the call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    r = agent.shard_balance()
    assert isinstance(r, ShardBalanceReport)
