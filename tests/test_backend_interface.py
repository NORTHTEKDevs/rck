"""Backend-interface tests.

Phase 1 of the pluggable-backend plan (docs/plans/2026-08-19-pluggable-backend.md)
makes the reasoning layer's independence from the HRR substrate a property
enforced by a test rather than asserted in a paper. Two things live here:

1. `all_facts()` -- the substrate-agnostic way to enumerate a knowledge
   base, added to `ShardedKnowledgeBase` so reasoning modules never need
   to reach into `_shards` directly.
2. A guard test that fails if any reasoning-layer module reaches into
   `_shards` again.
"""
from __future__ import annotations

from pathlib import Path

from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase

REPO_ROOT = Path(__file__).parent.parent


# ---- all_facts() -----------------------------------------------------------

def test_all_facts_empty_kb_returns_empty_list():
    kb = ShardedKnowledgeBase(dim=1024, n_shards=4, seed=0)
    assert kb.all_facts() == []


def test_all_facts_returns_every_stored_fact():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("france", "capital", "paris"),
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
    ], symmetrize=False)
    facts = kb.all_facts()
    assert len(facts) == 3
    subjects = {f["S"] for f in facts}
    assert subjects == {"france", "dog", "cat"}


def test_all_facts_count_matches_size_after_symmetrization():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    result = bulk_load_triples(kb, [
        ("paris", "locatedin", "france"),
        ("dog", "isa", "mammal"),
    ], symmetrize=True)
    assert result["symmetrized"] > 0
    assert len(kb.all_facts()) == kb.size()


def test_all_facts_stable_across_reshard():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=4, seed=0)
    bulk_load_triples(kb, [
        (f"subj{i}", "rel", f"obj{i}") for i in range(20)
    ], symmetrize=False)
    before = {(f["S"], f["R"], f["O"]) for f in kb.all_facts()}
    kb.reshard(16)
    after = {(f["S"], f["R"], f["O"]) for f in kb.all_facts()}
    assert before == after
    assert len(kb.all_facts()) == kb.size()


# ---- guard: reasoning layer must not reach into _shards --------------------

SUBSTRATE_OWNED = {
    "knowledge_base.py", "shard_balance.py", "shard_sizing.py",
    "sparse_relational.py", "capacity_profiler.py", "snapshot_hash.py",
    "session.py",
    # generative subsystem, HRR-only by design (agent.py's char-LM /
    # RCKAgent path, not ConsciousAgent's reasoning path)
    "agent.py", "compose.py", "generative.py", "bigram.py", "fep.py",
    "server.py", "gen_server.py", "mcp_server.py",
}

# Modules that legitimately need shard-level access; each entry needs a
# reason in the comment above it, and Phase 2 gives them a backend hook.
ALLOWED_EXCEPTIONS = {
    # filled in once Task 2's migration is complete
}


def test_reasoning_layer_does_not_reach_into_shards():
    """The layer above the substrate must not depend on HRR internals.

    Paper 5.0/5.10 measure that the substrate does not earn its place;
    this test is what keeps the layer portable off it.
    """
    offenders = []
    for path in (REPO_ROOT / "rck").rglob("*.py"):
        if path.name in SUBSTRATE_OWNED or path.name in ALLOWED_EXCEPTIONS:
            continue
        if "_shards" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, f"reasoning modules reaching into _shards: {offenders}"
