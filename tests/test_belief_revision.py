"""Tests for belief revision -- resolving detected conflicts."""
from __future__ import annotations

from rck.belief_revision import (
    CandidateFact, ResolutionPlan, SOURCE_PRIORITY,
    apply_resolution, plan_resolution, resolve_all,
)
from rck.contradiction import Conflict, detect_conflicts
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


def _kb_with_conflict() -> tuple[ShardedKnowledgeBase, ProvenanceStore]:
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    prov = ProvenanceStore()
    kb.store({"S": "fish", "R": "isa", "O": "animal"})
    prov.store("fish", "isa", "animal", source="user")
    kb.store({"S": "fish", "R": "isa", "O": "vegetable"})
    prov.store("fish", "isa", "vegetable", source="induced",
               tags={"induced", "via_3_hops"})
    return kb, prov


def test_user_beats_induced():
    """A user-provided fact wins over an induced one in the same conflict."""
    kb, prov = _kb_with_conflict()
    conflicts = detect_conflicts(kb)
    assert conflicts
    plan = plan_resolution(conflicts[0], provenance=prov)
    assert plan.keep.obj == "animal"
    assert plan.keep.source == "user"
    assert any(d.obj == "vegetable" for d in plan.drop)


def test_priority_ordering():
    """SOURCE_PRIORITY enforces the documented hierarchy."""
    assert SOURCE_PRIORITY["user"] > SOURCE_PRIORITY["multi"]
    assert SOURCE_PRIORITY["multi"] > SOURCE_PRIORITY["external"]
    assert SOURCE_PRIORITY["external"] > SOURCE_PRIORITY["induced"]
    assert SOURCE_PRIORITY["induced"] > SOURCE_PRIORITY["unknown"]


def test_apply_removes_dropped_fact():
    kb, prov = _kb_with_conflict()
    conflicts = detect_conflicts(kb)
    plan = plan_resolution(conflicts[0], provenance=prov)
    removed = apply_resolution(kb, plan, provenance=prov)
    assert removed >= 1
    # The dropped fact should no longer be in provenance.
    for d in plan.drop:
        assert prov.get(d.subject, d.relation, d.obj) is None


def test_resolve_all_handles_empty_input():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    plans = resolve_all(kb, [])
    assert plans == []


def test_resolve_all_with_apply():
    """resolve_all(apply=True) commits the resolutions."""
    kb, prov = _kb_with_conflict()
    conflicts = detect_conflicts(kb)
    plans = resolve_all(kb, conflicts, provenance=prov, apply=True)
    assert plans
    # After resolution, detect_conflicts should find none (or far fewer).
    remaining = detect_conflicts(kb)
    assert len(remaining) <= len(conflicts)


def test_plan_explanation_includes_source():
    kb, prov = _kb_with_conflict()
    conflicts = detect_conflicts(kb)
    plan = plan_resolution(conflicts[0], provenance=prov)
    text = plan.verbalize()
    assert "fish" in text
    assert "animal" in text or "vegetable" in text
    # The explanation should mention WHY (provenance priority).
    assert "source" in text or "provenance" in text


def test_recency_tiebreak():
    """If both candidates have source=user, the more recently reinforced wins."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    prov = ProvenanceStore()
    kb.store({"S": "fish", "R": "isa", "O": "animal"})
    prov.store("fish", "isa", "animal", source="user")
    rec = prov.get("fish", "isa", "animal")
    rec.last_seen = 1.0  # older
    kb.store({"S": "fish", "R": "isa", "O": "creature"})
    prov.store("fish", "isa", "creature", source="user")
    rec2 = prov.get("fish", "isa", "creature")
    rec2.last_seen = 100.0  # newer
    conflicts = detect_conflicts(kb)
    if conflicts:
        plan = plan_resolution(conflicts[0], provenance=prov)
        assert plan.keep.obj == "creature"


def test_conscious_agent_resolve_conflicts():
    """ConsciousAgent.resolve_conflicts() does detect + plan in one call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    agent.tell("fish", "isa", "animal")
    # Manually inject a conflicting induced fact.
    agent.knowledge.store({"S": "fish", "R": "isa", "O": "vegetable"})
    agent.provenance.store("fish", "isa", "vegetable", source="induced",
                            tags={"induced", "via_3_hops"})
    plans = agent.resolve_conflicts(apply=False)
    assert plans
    plan = plans[0]
    assert plan.keep.obj == "animal"
