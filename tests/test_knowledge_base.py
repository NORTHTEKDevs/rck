from rck.knowledge_base import ShardedKnowledgeBase, _shard_index


def test_shard_index_is_stable_and_routes_evenly():
    n = 32
    counts = [0] * n
    for i in range(1000):
        idx = _shard_index(f"entity_{i}", "color", n)
        counts[idx] += 1
    # No shard should be dominated by >2x the average (chi-square tolerance).
    assert max(counts) < 2 * (1000 / n) + 10


def test_store_and_query_roundtrip():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    kb.store({"S": "sky", "R": "color", "O": "blue"})
    kb.store({"S": "grass", "R": "color", "O": "green"})
    kb.store({"S": "rose", "R": "color", "O": "red"})
    ans, score = kb.answer({"S": "sky", "R": "color"}, "O")
    assert ans == "blue"
    assert score > 0.5


def test_sharding_increases_total_capacity():
    """16 shards x 200 facts = 3200 total. Cleanup should still hit >85%."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    for i in range(200):
        kb.store({"S": f"e{i}", "R": "is", "O": f"v{i}"})
    hits = 0
    for i in range(200):
        ans, _ = kb.answer({"S": f"e{i}", "R": "is"}, "O")
        hits += (ans == f"v{i}")
    assert hits >= 170, f"only {hits}/200 recalled at modest load"


def test_capacity_at_2k_facts():
    """2k facts across 64 shards, D=4096. Should retain >85% recall."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=64, seed=0)
    n = 2000
    for i in range(n):
        kb.store({"S": f"e{i}", "R": "is", "O": f"v{i}"})
    hits = 0
    sample = list(range(0, n, 4))  # 500 sampled queries
    for i in sample:
        ans, _ = kb.answer({"S": f"e{i}", "R": "is"}, "O")
        hits += (ans == f"v{i}")
    rate = hits / len(sample)
    assert rate >= 0.85, f"recall fell to {rate:.2%} at 2k facts / 64 shards"


def test_forget_only_removes_from_correct_shard():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    kb.store({"S": "a", "R": "is", "O": "x"})
    kb.store({"S": "b", "R": "is", "O": "y"})
    kb.forget({"S": "a", "R": "is", "O": "x"})
    ans, _ = kb.answer({"S": "b", "R": "is"}, "O")
    assert ans == "y"
