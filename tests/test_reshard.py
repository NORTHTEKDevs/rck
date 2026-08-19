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


import json, random
from collections import defaultdict
from pathlib import Path
from rck.conscious_agent import ConsciousAgent

DATA = Path(__file__).parent.parent / "data" / "conceptnet_scale_100k.jsonl"


def _load(n):
    facts = []
    with open(DATA, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            d = json.loads(line)
            facts.append((d["s"], d["r"], d["o"]))
    return facts


@pytest.mark.skipif(not DATA.exists(), reason="ConceptNet subset not present")
def test_underprovisioned_agent_recovers_recall_after_reshard():
    """Regression for the measured v15.3.1 blocker: a 200-provisioned agent
    holding 5,000 facts scored 24.0% recall@1 with no warning."""
    facts = _load(5000)
    valid = defaultdict(set)
    for s, r, o in facts:
        valid[(s, r)].add(o)
    sample = random.Random(1).sample(facts, 400)

    agent = ConsciousAgent(expected_facts=200)
    for s, r, o in facts:
        agent.tell(s, r, o)

    agent.knowledge.reshard()   # explicit for now; Task 3 makes this automatic

    hits = sum(1 for s, r, o in sample
               if agent.ask_with_idk({"S": s, "R": r}, "O").top_symbol in valid[(s, r)])
    recall = hits / len(sample)
    assert recall >= 0.99, f"recall {recall:.1%} after reshard, expected >=99%"


def test_growth_past_provisioning_auto_reshards():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    for i in range(2000):
        kb.store({"S": f"s{i}", "R": "isa", "O": f"o{i}"})

    assert kb.n_shards > 8, "KB never resharded while growing 250x past capacity"
    assert max(kb.shard_sizes()) <= kb.target_fill, \
        f"a shard is still over the cliff: {max(kb.shard_sizes())} > {kb.target_fill}"
    assert kb.size() == 2000


def test_auto_reshard_can_be_disabled():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    kb.auto_reshard = False
    for i in range(1000):
        kb.store({"S": f"s{i}", "R": "isa", "O": f"o{i}"})
    assert kb.n_shards == 8
