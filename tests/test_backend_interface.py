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
    # RCKAgent path, not ConsciousAgent's reasoning path). Includes the
    # v7 polisher's training-data generators, which enumerate kb._shards
    # directly to build synthetic training corpora -- training-pipeline
    # code, not reasoning, so this plan does not touch it.
    "agent.py", "compose.py", "generative.py", "bigram.py", "fep.py",
    "server.py", "gen_server.py", "mcp_server.py",
    "multi_task_corpus.py", "polisher_training.py",
}

# Modules that legitimately need shard-level access; each entry needs a
# reason, and Phase 2 gives them a backend hook.
ALLOWED_EXCEPTIONS = {
    # Shard-to-shard HRR bundle merge: RelationalMemory.merge() sums two
    # shards' _memory tensors directly and requires matching shard count
    # + dim between the two KBs. Not expressible as fact-level
    # enumeration -- this *is* the substrate-specific operation.
    "federated_merge.py",
    # compress_duplicates() writes `shard._facts = keep` directly, per
    # shard -- a write to shard internals, not a read, and it bypasses
    # store()/forget() entirely. detect_contradictions() and
    # generate_abstractions() in the same file were migrated to
    # all_facts(); this one function could not be.
    "dreaming.py",
    # detect_global_gaps() samples entities with an early-exit break that
    # fires once per shard (after each shard's fact loop, not after each
    # fact), so which entities get collected before the sample_size*5 cap
    # depends on shard boundaries. _relations_for_subjects() in the same
    # file was migrated to all_facts(); this one function could not be
    # without changing which entities get sampled.
    "curiosity.py",
    # _related_entities()'s incoming-facts loop breaks once
    # max_each*10 is reached, but the break only exits the *current*
    # shard's inner loop (there is no matching break on the outer shard
    # loop), so the exact set collected depends on shard boundaries.
    # Migrating to a flat kb.all_facts() loop with a single break would
    # stop scanning at a different point for the same KB under a
    # different shard count -- a real, if obscure, behaviour change.
    "research.py",
    # summarize_subject() has the identical per-shard early-exit quirk
    # as research.py above (break only exits the current shard's loop
    # once max_facts is reached), for the same reason.
    "subject_summary.py",
}


def test_reasoning_layer_does_not_reach_into_shards():
    """The layer above the substrate must not depend on HRR internals.

    Paper 5.0/5.10 measure that the substrate does not earn its place;
    this test is what keeps the layer portable off it.

    Checks for `._shards` (attribute access on the private shard-list),
    not a bare "_shards" substring. A naive substring check is unreliable
    here: this codebase alone has "n_shards" (the public shard-count
    field that legitimately stays on the KnowledgeBackend interface),
    "recommend_shards" (a public function in shard_sizing.py),
    "sparse_shards" and "max_shards" (local names in shard_balance.py /
    shard_sizing.py) -- all literal substring matches for "_shards" that
    have nothing to do with reaching into shard internals. The genuine
    coupling this test guards against always takes the form
    `<something>._shards`, e.g. `kb._shards` or `source.knowledge._shards`.
    """
    offenders = []
    for path in (REPO_ROOT / "rck").rglob("*.py"):
        if path.name in SUBSTRATE_OWNED or path.name in ALLOWED_EXCEPTIONS:
            continue
        if "._shards" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, f"reasoning modules reaching into _shards: {offenders}"
