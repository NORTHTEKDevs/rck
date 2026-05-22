"""Tests for the chain memoization cache."""
from __future__ import annotations

from rck.chain_cache import CachedChain, ChainCache


def test_put_then_get_round_trip():
    c = ChainCache()
    c.put("france", "europe", ["capital", "locatedin"],
           ["forward", "forward"], 0.5)
    e = c.get("france", "europe")
    assert isinstance(e, CachedChain)
    assert e.relations == ["capital", "locatedin"]
    assert e.hits == 1


def test_get_returns_none_on_miss():
    c = ChainCache()
    assert c.get("xyzzy", "qux") is None


def test_put_is_case_insensitive():
    c = ChainCache()
    c.put("France", "Europe", ["capital"], ["forward"], 0.5)
    e = c.get("france", "europe")
    assert e is not None


def test_clear_removes_all():
    c = ChainCache()
    c.put("a", "b", ["r"], ["forward"], 0.5)
    c.put("c", "d", ["r"], ["forward"], 0.5)
    c.clear()
    assert c.size() == 0


def test_eviction_at_capacity():
    c = ChainCache(max_size=2)
    c.put("a", "b", ["r"], ["forward"], 0.5)
    c.put("c", "d", ["r"], ["forward"], 0.5)
    c.put("e", "f", ["r"], ["forward"], 0.5)
    # Oldest ("a","b") should be evicted.
    assert c.get("a", "b") is None
    assert c.get("c", "d") is not None
    assert c.get("e", "f") is not None


def test_lru_promotion_on_get():
    c = ChainCache(max_size=2)
    c.put("a", "b", ["r"], ["forward"], 0.5)
    c.put("c", "d", ["r"], ["forward"], 0.5)
    # Access ("a","b") to promote it.
    c.get("a", "b")
    # Add a third entry; "c","d" (the least-recently-used) gets evicted.
    c.put("e", "f", ["r"], ["forward"], 0.5)
    assert c.get("a", "b") is not None
    assert c.get("c", "d") is None
    assert c.get("e", "f") is not None


def test_repeated_put_updates_in_place():
    c = ChainCache()
    c.put("a", "b", ["r1"], ["forward"], 0.5)
    c.put("a", "b", ["r2", "r3"], ["forward", "forward"], 0.8)
    e = c.get("a", "b")
    assert e.relations == ["r2", "r3"]
    assert e.discovery_confidence == 0.8


def test_conscious_agent_caches_discover():
    """Second call to agent.discover() for the same (start, target)
    uses the cache."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("france", "capital", "paris"),
        ("paris", "locatedin", "europe"),
    ]:
        agent.tell(s, r, o)
    first = agent.discover("france", "europe", max_depth=3)
    second = agent.discover("france", "europe", max_depth=3)
    assert first is not None and second is not None
    assert second["from_cache"] is True
    assert second["relations"] == first["relations"]
    assert agent.chain_cache.size() == 1


def test_bump_version_invalidates_existing_entries():
    """After bump_version(), previously-cached entries miss."""
    c = ChainCache()
    c.put("a", "b", ["r"], ["forward"], 0.5)
    assert c.get("a", "b") is not None
    c.bump_version()
    assert c.get("a", "b") is None  # treated as miss after version bump


def test_bump_version_does_not_affect_subsequently_inserted():
    """Entries inserted AFTER a bump are valid at the new version."""
    c = ChainCache()
    c.bump_version()
    c.put("a", "b", ["r"], ["forward"], 0.5)
    assert c.get("a", "b") is not None


def test_agent_load_jsonl_bumps_cache(tmp_path):
    """ConsciousAgent.load_jsonl() bumps the cache version after batch
    ingest so stale chains aren't returned from earlier topology."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    agent.tell("a", "isa", "b")
    agent.tell("b", "isa", "c")
    spec = agent.discover("a", "c", max_depth=3)
    assert spec is not None
    assert agent.chain_cache.size() == 1
    # Write a JSONL file and load it.
    f = tmp_path / "more.jsonl"
    f.write_text('{"s":"c","r":"isa","o":"d"}\n')
    pre = agent.chain_cache.kb_version
    agent.load_jsonl(str(f))
    assert agent.chain_cache.kb_version == pre + 1


def test_agent_induce_bumps_cache():
    """After a successful induce(), the chain cache version bumps so the
    next discover() re-runs BFS and can pick up the new direct shortcut."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
    ]:
        agent.tell(s, r, o)
    pre = agent.chain_cache.kb_version
    induced = agent.induce("a", "c")
    if induced is not None and induced.verified:
        assert agent.chain_cache.kb_version == pre + 1


def test_stats_summarises():
    c = ChainCache()
    c.put("a", "b", ["r"], ["forward"], 0.5)
    c.get("a", "b")
    c.get("a", "b")
    s = c.stats()
    assert s["size"] == 1
    assert s["total_hits"] == 2
