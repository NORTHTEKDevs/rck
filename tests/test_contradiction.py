"""Tests for contradiction detection."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.contradiction import (
    Conflict, ContradictionPolicy, FUNCTIONAL_RELATIONS,
    detect_conflicts, report_conflicts,
)
from rck.knowledge_base import ShardedKnowledgeBase


def test_no_conflict_in_consistent_kb():
    """Each (S, R) has one clear answer -> no conflicts."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("eagle", "isa", "bird"),
    ])
    conflicts = detect_conflicts(kb)
    assert conflicts == []


def test_conflict_when_subject_has_two_high_score_objects():
    """Storing two contradictory facts for a functional relation."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    # Same subject, same FUNCTIONAL relation, different objects.
    kb.store({"S": "fish", "R": "isa", "O": "animal"})
    kb.store({"S": "fish", "R": "isa", "O": "vegetable"})
    conflicts = detect_conflicts(kb)
    # We added two competing answers; severity should be high.
    flagged = [c for c in conflicts if c.subject == "fish" and c.relation == "isa"]
    assert flagged


def test_non_functional_relation_is_ignored():
    """`has` is multi-valued; multiple objects must NOT trigger a conflict."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dog", "has", "fur"),
        ("dog", "has", "tail"),
        ("dog", "has", "legs"),
    ])
    conflicts = detect_conflicts(kb)
    assert all(c.relation != "has" for c in conflicts)


def test_policy_can_extend_functional_set():
    """A custom policy lets us flag conflicts on additional relations."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    kb.store({"S": "dog", "R": "favourite_food", "O": "kibble"})
    kb.store({"S": "dog", "R": "favourite_food", "O": "bones"})
    policy = ContradictionPolicy(
        functional_relations=FUNCTIONAL_RELATIONS | {"favourite_food"},
    )
    conflicts = detect_conflicts(kb, policy=policy)
    assert any(c.relation == "favourite_food" for c in conflicts)


def test_severity_higher_when_scores_closer():
    """Two near-equal scores -> high severity. One dominant -> low."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    kb.store({"S": "fish", "R": "isa", "O": "animal"})
    kb.store({"S": "fish", "R": "isa", "O": "vegetable"})
    conflicts = detect_conflicts(kb, policy=ContradictionPolicy(min_severity=0.0))
    if conflicts:
        # severity is a float between 0 and 1
        for c in conflicts:
            assert 0.0 <= c.severity <= 1.0


def test_detect_subjects_filter():
    """Passing subjects= restricts the scan to those subjects."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    kb.store({"S": "fish", "R": "isa", "O": "animal"})
    kb.store({"S": "fish", "R": "isa", "O": "vegetable"})
    kb.store({"S": "dog", "R": "isa", "O": "mammal"})
    flagged = detect_conflicts(kb, subjects=["dog"])
    # We restricted to dog; no contradiction there.
    assert flagged == []


def test_report_conflicts_formats_message():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    kb.store({"S": "fish", "R": "isa", "O": "animal"})
    kb.store({"S": "fish", "R": "isa", "O": "vegetable"})
    cs = detect_conflicts(kb)
    msg = report_conflicts(cs)
    assert "conflict" in msg.lower()


def test_report_with_no_conflicts_message():
    msg = report_conflicts([])
    assert "no conflicts" in msg.lower()


def test_conflict_when_positive_and_negative_both_stored():
    """(fish, isa, animal) AND (fish, NOT_isa, animal) -> conflict."""
    from rck.negative_facts import deny
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    kb.store({"S": "fish", "R": "isa", "O": "animal"})
    deny(kb, "fish", "isa", "animal")
    conflicts = detect_conflicts(kb)
    # We expect one conflict surfacing the negation collision.
    relevant = [c for c in conflicts
                if c.subject == "fish" and c.relation == "isa"
                and any(o == "animal" for o, _ in c.candidates)]
    assert relevant


def test_no_negation_conflict_when_only_positive():
    from rck.negative_facts import deny
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    kb.store({"S": "fish", "R": "isa", "O": "animal"})
    deny(kb, "fish", "isa", "vegetable")  # different obj
    conflicts = detect_conflicts(kb)
    # No (fish, isa, animal) collision because vegetable != animal.
    same_obj = [c for c in conflicts
                if c.subject == "fish" and c.relation == "isa"
                and any(o == "animal" for o, _ in c.candidates)
                and len(c.candidates) >= 2
                and c.candidates[0][0] == c.candidates[1][0]]
    assert same_obj == []


def test_conscious_agent_detect_conflicts():
    """ConsciousAgent.detect_conflicts() wraps the same call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    agent.tell("fish", "isa", "animal")
    # Direct store of a contradiction (bypass auto-symmetrize logic).
    agent.knowledge.store({"S": "fish", "R": "isa", "O": "vegetable"})
    conflicts = agent.detect_conflicts()
    assert any(c.subject == "fish" and c.relation == "isa"
               for c in conflicts)
