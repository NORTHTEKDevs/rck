"""Tests for parallel batch chain discovery."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase
from rck.parallel_discover import BatchDiscoveryResult, batch_discover


def _geo_kb() -> ShardedKnowledgeBase:
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("france", "capital", "paris"),
        ("paris", "locatedin", "europe"),
        ("germany", "capital", "berlin"),
        ("berlin", "locatedin", "europe"),
        ("japan", "capital", "tokyo"),
        ("tokyo", "locatedin", "asia"),
    ])
    return kb


def test_batch_discover_returns_one_row_per_probe():
    kb = _geo_kb()
    probes = [("france", "europe"), ("japan", "asia"), ("germany", "europe")]
    results = batch_discover(kb, probes, max_depth=3, max_workers=2)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, BatchDiscoveryResult)


def test_batch_discover_preserves_order():
    """Results come back in the same order as input."""
    kb = _geo_kb()
    probes = [("japan", "asia"), ("france", "europe")]
    results = batch_discover(kb, probes, max_workers=2)
    assert results[0].start == "japan"
    assert results[1].start == "france"


def test_batch_discover_finds_chains():
    kb = _geo_kb()
    probes = [("france", "europe"), ("japan", "asia")]
    results = batch_discover(kb, probes, max_workers=2)
    found = [r for r in results if r.chain is not None]
    assert len(found) >= 1


def test_batch_discover_handles_missing_targets():
    """A probe with no reachable target returns a row with chain=None."""
    kb = _geo_kb()
    results = batch_discover(
        kb, [("france", "nonexistent_node")], max_workers=2,
    )
    assert len(results) == 1
    assert results[0].chain is None


def test_auto_worker_count_caps_and_floors():
    """auto_worker_count obeys cap + n_probes + cpu floor."""
    from rck.parallel_discover import (
        DEFAULT_MAX_WORKERS_CAP, auto_worker_count,
    )
    # 0 probes still returns >=1 (defensive).
    assert auto_worker_count(0) >= 1
    # Lots of probes -> capped at DEFAULT_MAX_WORKERS_CAP (or cpu).
    n = auto_worker_count(1000)
    assert n <= DEFAULT_MAX_WORKERS_CAP
    # Fewer probes than cap -> match the probe count.
    assert auto_worker_count(2) <= 2


def test_batch_discover_auto_tunes_when_none():
    """max_workers=None auto-tunes; results should still be correct."""
    kb = _geo_kb()
    probes = [("france", "europe"), ("japan", "asia")]
    results = batch_discover(kb, probes, max_workers=None)
    assert len(results) == 2


def test_batch_discover_empty_input():
    kb = _geo_kb()
    results = batch_discover(kb, [], max_workers=2)
    assert results == []
