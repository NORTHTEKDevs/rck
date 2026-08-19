import pytest
from rck.knowledge_base import ShardedKnowledgeBase, _shard_index


def _kb_with(n_facts, n_shards=8):
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    for i in range(n_facts):
        kb.store({"S": f"s{i}", "R": "isa", "O": f"o{i}"})
    return kb


def test_reshard_preserves_every_fact_and_routes_correctly():
    kb = _kb_with(500, n_shards=8)
    before = {tuple(sorted(f.items())) for sh in kb._shards for f in sh.facts()}

    kb.reshard(64)

    assert kb.n_shards == 64
    assert len(kb._shards) == 64
    after = {tuple(sorted(f.items())) for sh in kb._shards for f in sh.facts()}
    assert after == before, "reshard lost or duplicated facts"

    # Every fact must now live in the shard its (S, R) routes to at n=64.
    for idx, sh in enumerate(kb._shards):
        for f in sh.facts():
            assert _shard_index(str(f["S"]), str(f["R"]), 64) == idx


def test_reshard_does_not_inflate_fact_count():
    kb = _kb_with(500, n_shards=8)
    assert kb.size() == 500
    kb.reshard(64)
    assert kb.size() == 500, "reshard double-counted _fact_count"


def test_reshard_is_a_noop_when_target_equals_current():
    kb = _kb_with(100, n_shards=16)
    codebook_before = kb.codebook
    kb.reshard(16)
    assert kb.n_shards == 16
    assert kb.codebook is codebook_before, "reshard must not rebuild the codebook"
