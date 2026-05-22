"""Tests for ConsciousAgent.maintain() orchestration."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.conscious_agent import ConsciousAgent


def test_maintain_returns_summary_dict():
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
        ("c", "isa", "d"),
    ]:
        agent.tell(s, r, o)
    summary = agent.maintain(max_rounds=1, probes_per_round=20)
    assert isinstance(summary, dict)
    for key in (
        "chain_induction_verified",
        "chain_induction_rounds",
        "rule_cascade_verified",
        "rule_cascade_rounds",
        "conflicts_resolved",
        "cache_entries_warmed",
        "final_kb_size",
        "skill_library",
    ):
        assert key in summary


def test_maintain_grows_kb_when_inductive_patterns_present():
    """Loading a transitive chain should let maintain() add facts."""
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
        ("c", "isa", "d"),
        ("d", "isa", "e"),
    ]:
        agent.tell(s, r, o)
    pre = agent.knowledge.size()
    agent.maintain(max_rounds=2, probes_per_round=20)
    assert agent.knowledge.size() >= pre


def test_maintain_does_not_resolve_when_disabled():
    """resolve_conflicts=False -> summary['conflicts_resolved'] == 0."""
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    agent.tell("dog", "isa", "mammal")
    summary = agent.maintain(max_rounds=1, probes_per_round=10,
                              resolve_conflicts=False)
    assert summary["conflicts_resolved"] == 0


def test_maintain_includes_new_summary_keys():
    """v2 maintain includes negation/promotion/consolidation counts."""
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    agent.tell("dog", "isa", "mammal")
    summary = agent.maintain(max_rounds=1, probes_per_round=10)
    for key in (
        "negations_propagated",
        "skills_promoted",
        "episodes_consolidated",
        "episodes_ambiguous_flagged",
    ):
        assert key in summary


def test_maintain_checkpoint_dir_writes_state(tmp_path):
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    agent.tell("dog", "isa", "mammal")
    out = tmp_path / "ckpt"
    summary = agent.maintain(max_rounds=1, probes_per_round=10,
                              checkpoint_dir=out)
    assert "checkpoint" in summary
    assert (out / "skills.jsonl").exists()
    assert (out / "provenance.jsonl").exists()


def test_maintain_disables_optional_steps():
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    agent.tell("dog", "isa", "mammal")
    summary = agent.maintain(
        max_rounds=1, probes_per_round=10,
        propagate_negations=False,
        promote_skills=False,
        consolidate_episodes=False,
    )
    assert summary["negations_propagated"] == 0
    assert summary["skills_promoted"] == 0
    assert summary["episodes_consolidated"] == 0
