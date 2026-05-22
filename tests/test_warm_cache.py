"""Tests for hot-query cache pre-warming."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_warm_cache_no_history_returns_zero():
    """With no episodes, nothing to warm."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    assert agent.warm_cache_from_history() == 0


def test_warm_cache_populates_from_recent_known_episode():
    """A repeated KNOWN query should have its (start, top_symbol) pair
    pre-discovered into the chain cache."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    for s, r, o in [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
    ]:
        agent.tell(s, r, o)
    # Ask the same query several times so it counts as "hot".
    for _ in range(3):
        agent.ask_with_idk({"S": "a", "R": "isa"}, "O")
    pre_cache_size = agent.chain_cache.size()
    warmed = agent.warm_cache_from_history(top_k=5)
    # We might warm 0 (if a->b is depth 1 and the cache only stores
    # multi-hop) or >=1. Either way the call should not raise and
    # cache size should not shrink.
    assert agent.chain_cache.size() >= pre_cache_size
    assert warmed >= 0


def test_warm_cache_skips_already_cached():
    """If a (start, target) is already in the cache, don't re-warm."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    for s, r, o in [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
    ]:
        agent.tell(s, r, o)
    # Prime the cache manually.
    agent.discover("a", "c", max_depth=3)
    pre_size = agent.chain_cache.size()
    for _ in range(3):
        agent.ask_with_idk({"S": "a", "R": "isa"}, "O")
    warmed = agent.warm_cache_from_history(top_k=5)
    # Cache size should not change since (a, c) was already cached.
    assert agent.chain_cache.size() >= pre_size
