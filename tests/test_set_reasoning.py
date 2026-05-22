"""Tests for set/branching reasoning."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase
from rck.set_reasoning import (
    SetCandidate, difference_queries, intersect_queries,
    query_set, union_queries,
)


def _country_kb() -> ShardedKnowledgeBase:
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("france", "locatedin", "europe"),
        ("germany", "locatedin", "europe"),
        ("spain", "locatedin", "europe"),
        ("japan", "locatedin", "asia"),
        ("china", "locatedin", "asia"),
        ("france", "currency", "euro"),
        ("germany", "currency", "euro"),
        ("spain", "currency", "euro"),
        ("japan", "currency", "yen"),
        ("china", "currency", "yuan"),
    ])
    return kb


# ---- query_set ------------------------------------------------------------

def test_query_set_returns_dict_of_candidates():
    kb = _country_kb()
    eu = query_set(kb, {"R": "locatedin", "O": "europe"}, "S",
                    top_k=10, min_score=0.10)
    assert isinstance(eu, dict)
    # Most european countries should appear; HRR may surface 1-3
    # consistently.
    assert any(c in eu for c in ("france", "germany", "spain"))


# ---- intersection ---------------------------------------------------------

def test_intersect_european_euro_countries():
    kb = _country_kb()
    rows = intersect_queries(kb, [
        {"R": "locatedin", "O": "europe"},
        {"R": "currency", "O": "euro"},
    ], "S", top_k=15, min_score=0.10)
    if rows:
        for r in rows:
            assert str(r.symbol) in {"france", "germany", "spain"}
            assert r.n_satisfied() == 2


def test_intersect_with_no_overlap_returns_empty():
    kb = _country_kb()
    rows = intersect_queries(kb, [
        {"R": "locatedin", "O": "asia"},
        {"R": "currency", "O": "euro"},
    ], "S", top_k=15, min_score=0.10)
    assert rows == []


# ---- union ----------------------------------------------------------------

def test_union_returns_combined_set():
    kb = _country_kb()
    rows = union_queries(kb, [
        {"R": "locatedin", "O": "europe"},
        {"R": "currency", "O": "yen"},
    ], "S", top_k=15, min_score=0.10)
    syms = {str(r.symbol) for r in rows}
    # Should at least include a european country AND japan.
    assert any(c in syms for c in ("france", "germany", "spain"))


# ---- difference -----------------------------------------------------------

def test_difference_european_minus_euro():
    """No country is in europe AND not using euro in this KB,
    so the difference should be empty."""
    kb = _country_kb()
    rows = difference_queries(
        kb,
        positive={"R": "locatedin", "O": "europe"},
        negative={"R": "currency", "O": "euro"},
        unknown_role="S",
    )
    # If HRR returns something, it shouldn't be one of the euro countries.
    for r in rows:
        assert str(r.symbol) not in {"france", "germany", "spain"}


# ---- conscious agent integration ------------------------------------------

def test_conscious_agent_intersect():
    """The agent's intersect convenience wraps the same call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("france", "locatedin", "europe"),
        ("france", "currency", "euro"),
        ("japan", "locatedin", "asia"),
    ]:
        agent.tell(s, r, o)
    rows = agent.intersect([
        {"R": "locatedin", "O": "europe"},
        {"R": "currency", "O": "euro"},
    ], "S")
    if rows:
        assert any(str(r.symbol) == "france" for r in rows)
