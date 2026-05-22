"""Tests for subject importance ranking."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.conscious_agent import ConsciousAgent
from rck.knowledge_base import ShardedKnowledgeBase
from rck.subject_importance import ImportanceScore, rank_subjects


def test_rank_returns_top_k():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("dog", "color", "brown"),
        ("cat", "isa", "mammal"),
        ("eagle", "isa", "bird"),
    ])
    scores = rank_subjects(kb, top_k=5)
    assert scores
    syms = [s.subject for s in scores]
    # dog has the most facts.
    assert syms[0] == "dog"


def test_score_combines_fact_object_query_counts():
    s = ImportanceScore(subject="dog", fact_count=3, object_count=1,
                         query_count=2, derivation_count=1)
    expected = 3 * 1.0 + 1 * 0.5 + 2 * 2.0 + 1 * 1.0
    assert abs(s.score() - expected) < 1e-9


def test_rank_uses_query_memory():
    """Queries should boost importance."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.tell("cat", "isa", "mammal")
    # Query dog several times -> higher importance.
    for _ in range(5):
        agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    scores = rank_subjects(
        agent.knowledge, provenance=agent.provenance,
        query_memory=agent.query_memory, top_k=5,
    )
    syms = [s.subject for s in scores]
    assert syms[0] == "dog"


def test_rank_uses_provenance_derivation():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    for s, r, o in [
        ("a", "partof", "b"),
        ("b", "locatedin", "c"),
    ]:
        agent.tell(s, r, o)
    # Induce so a derived fact lands with a derivation chain.
    agent.induce("a", "c")
    scores = rank_subjects(
        agent.knowledge, provenance=agent.provenance,
        top_k=10,
    )
    syms = {s.subject for s in scores}
    assert {"a", "b", "c"} <= syms


def test_conscious_agent_rank_subjects():
    """ConsciousAgent.rank_subjects() wraps the call."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    scores = agent.rank_subjects(top_k=3)
    assert scores
    assert scores[0].subject == "dog"
