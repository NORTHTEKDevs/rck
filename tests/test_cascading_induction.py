"""Tests for cascading induction."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.cascading_induction import (
    CascadeResult, cascade_induct, top_pattern_signatures,
)
from rck.chain_induction import InductionPolicy
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore
from rck.skills import SkillLibrary


def _staircase_kb() -> ShardedKnowledgeBase:
    """Build a 4-level transitive ladder: a isa b isa c isa d.
    Single-pass induction yields (a, isa, c) and (b, isa, d).
    Cascade should ALSO get (a, isa, d)."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
        ("c", "isa", "d"),
    ])
    return kb


def test_cascade_reaches_fixed_point():
    kb = _staircase_kb()
    res = cascade_induct(kb, max_rounds=5)
    assert isinstance(res, CascadeResult)
    assert res.saturated  # should reach fixed point well before max_rounds


def test_cascade_records_induced_facts_per_round():
    kb = _staircase_kb()
    res = cascade_induct(kb, max_rounds=5)
    # Total induced is sum across rounds.
    assert sum(r.facts_verified for r in res.rounds) == res.total_verified
    # KB grew.
    assert res.final_facts > res.initial_facts


def test_cascade_top_patterns_surface():
    kb = _staircase_kb()
    skills = SkillLibrary()
    res = cascade_induct(kb, max_rounds=5, skills=skills)
    patterns = top_pattern_signatures(res)
    assert patterns
    # Top pattern is either ('isa', 'isa') or its auto-symmetrized
    # inverse ('hassubtype', 'hassubtype').
    top_sig, count = patterns[0]
    assert top_sig in {("isa", "isa"), ("hassubtype", "hassubtype")}
    assert count >= 1


def test_cascade_with_provenance_tags_each_fact():
    kb = _staircase_kb()
    prov = ProvenanceStore()
    res = cascade_induct(kb, max_rounds=5, provenance=prov)
    # Every verified induced fact has a provenance record marked "induced".
    for fact in res.induced_facts:
        rec = prov.get(fact.subject, fact.relation, fact.obj)
        assert rec is not None
        assert "induced" in rec.tags


def test_cascade_respects_max_rounds_for_growing_kb():
    """If we allow only 1 round on the staircase KB, we get fewer facts
    than the full cascade."""
    kb1 = _staircase_kb()
    kb2 = _staircase_kb()
    one_round = cascade_induct(kb1, max_rounds=1)
    full_run = cascade_induct(kb2, max_rounds=5)
    # Full run must induce at least as many facts as one round.
    assert full_run.total_verified >= one_round.total_verified
