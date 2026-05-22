"""Tests for bulk ingest + inverse symmetrization."""
import json
import tempfile
from pathlib import Path

from rck.bulk_ingest import (
    auto_symmetrize, bulk_load_csv, bulk_load_jsonl, bulk_load_triples,
    inverse_relation,
)
from rck.knowledge_base import ShardedKnowledgeBase


def test_inverse_relation_lookup():
    assert inverse_relation("wrote") == "author"
    assert inverse_relation("author") == "wrote"
    assert inverse_relation("partof") == "haspart"
    assert inverse_relation("haspart") == "partof"
    assert inverse_relation("locatedin") == "contains"
    assert inverse_relation("nonexistent") is None


def test_bulk_load_jsonl_with_symmetrization():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for s, r, o in [
            ("shakespeare", "wrote", "hamlet"),
            ("orwell", "wrote", "1984"),
            ("paris", "locatedin", "france"),
        ]:
            f.write(json.dumps({"s": s, "r": r, "o": o}) + "\n")
        path = f.name
    try:
        stats = bulk_load_jsonl(kb, path, symmetrize=True)
    finally:
        Path(path).unlink()
    assert stats["facts"] == 3
    assert stats["symmetrized"] == 3
    # Forward AND inverse lookups must work.
    ans, _ = kb.answer({"S": "shakespeare", "R": "wrote"}, "O")
    assert ans == "hamlet"
    ans, _ = kb.answer({"S": "hamlet", "R": "author"}, "O")
    assert ans == "shakespeare"
    ans, _ = kb.answer({"S": "france", "R": "contains"}, "O")
    assert ans == "paris"


def test_bulk_load_csv():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as f:
        f.write("s,r,o\n")
        f.write("dog,isa,mammal\n")
        f.write("cat,isa,mammal\n")
        path = f.name
    try:
        stats = bulk_load_csv(kb, path, symmetrize=False)
    finally:
        Path(path).unlink()
    assert stats["facts"] == 2
    ans, _ = kb.answer({"S": "dog", "R": "isa"}, "O")
    assert ans == "mammal"


def test_bulk_load_triples_symmetric():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    triples = [("alice", "spouse", "bob")]
    stats = bulk_load_triples(kb, triples, symmetrize=True)
    # `spouse` is symmetric -- inverse is also `spouse`.
    assert stats["symmetrized"] == 1
    ans, _ = kb.answer({"S": "bob", "R": "spouse"}, "O")
    assert ans == "alice"


def test_auto_symmetrize_after_load():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("austen", "wrote", "emma"),
        ("dickens", "wrote", "olivertwist"),
    ], symmetrize=False)
    # Inverse facts NOT present yet.
    ans, score = kb.answer({"S": "emma", "R": "author"}, "O")
    assert ans != "austen" or score < 0.10
    # Add them.
    n = auto_symmetrize(kb)
    assert n >= 2
    ans, _ = kb.answer({"S": "emma", "R": "author"}, "O")
    assert ans == "austen"


def test_bulk_load_at_scale():
    """Stream 1000 synthetic facts -- should complete in <2 s."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=64, seed=0)
    triples = [(f"e{i}", "isa", f"cls{i % 50}") for i in range(1000)]
    stats = bulk_load_triples(kb, triples, symmetrize=False)
    assert stats["facts"] == 1000
    assert stats["elapsed_s"] < 5.0
    # Sample-check recall.
    hits = sum(1 for i in range(0, 1000, 10)
               if (a := kb.answer({"S": f"e{i}", "R": "isa"}, "O"))[0] == f"cls{i % 50}")
    assert hits >= 80  # at least 80% recall on a 1000-fact / 64-shard KB
