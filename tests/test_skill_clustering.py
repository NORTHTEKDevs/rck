"""Tests for skill clustering."""
from __future__ import annotations

from rck.skill_clustering import (
    SkillFamily, cluster_skills_by_prefix, family_summary,
)
from rck.skills import SkillLibrary


def test_cluster_skills_groups_by_shared_prefix():
    skills = SkillLibrary()
    # Three skills share prefix [(O, partof), (O, locatedin)].
    p1 = [("O", "partof"), ("O", "locatedin"), ("O", "isa")]
    p2 = [("O", "partof"), ("O", "locatedin"), ("O", "continent")]
    p3 = [("O", "partof"), ("O", "locatedin"), ("O", "kind")]
    for _ in range(3):
        skills.record_success(p1)
        skills.record_success(p2)
        skills.record_success(p3)
    families = cluster_skills_by_prefix(skills, prefix_length=2)
    assert families
    top = families[0]
    assert isinstance(top, SkillFamily)
    assert len(top.members) >= 2
    # The shared prefix matches.
    assert top.prefix[0][1] == "partof"
    assert top.prefix[1][1] == "locatedin"


def test_cluster_respects_min_members():
    skills = SkillLibrary()
    # Only one skill with this prefix.
    skills.record_success([("O", "rare1"), ("O", "rare2")])
    skills.record_success([("O", "rare1"), ("O", "rare2")])
    families = cluster_skills_by_prefix(skills, prefix_length=2,
                                         min_members=2)
    # We have only ONE distinct pattern -> family of 1 member -> dropped.
    assert families == []


def test_cluster_skips_short_patterns():
    skills = SkillLibrary()
    skills.record_success([("O", "single")])  # length < prefix_length
    skills.record_success([("O", "single")])
    families = cluster_skills_by_prefix(skills, prefix_length=2,
                                         min_members=1)
    assert families == []


def test_family_confidence_aggregates_correctly():
    skills = SkillLibrary()
    p1 = [("O", "a"), ("O", "b"), ("O", "c")]
    p2 = [("O", "a"), ("O", "b"), ("O", "d")]
    for _ in range(2):
        skills.record_success(p1)
        skills.record_success(p2)
    # Inject one failure on p1.
    skills.record_failure(p1)
    families = cluster_skills_by_prefix(skills, prefix_length=2,
                                         min_members=2)
    assert families
    top = families[0]
    assert top.total_success >= 4
    assert top.total_failure >= 1
    assert 0.0 < top.confidence < 1.0


def test_family_verbalize_includes_prefix():
    skills = SkillLibrary()
    p1 = [("O", "partof"), ("O", "locatedin"), ("O", "isa")]
    p2 = [("O", "partof"), ("O", "locatedin"), ("O", "continent")]
    for _ in range(2):
        skills.record_success(p1)
        skills.record_success(p2)
    families = cluster_skills_by_prefix(skills, prefix_length=2)
    text = families[0].verbalize()
    assert "partof" in text
    assert "locatedin" in text


def test_family_summary_json_friendly():
    skills = SkillLibrary()
    p1 = [("O", "a"), ("O", "b"), ("O", "c")]
    p2 = [("O", "a"), ("O", "b"), ("O", "d")]
    for _ in range(2):
        skills.record_success(p1)
        skills.record_success(p2)
    families = cluster_skills_by_prefix(skills, prefix_length=2)
    summary = family_summary(families, top_k=5)
    assert isinstance(summary, list)
    if summary:
        assert "prefix" in summary[0]
        assert "n_members" in summary[0]


def test_conscious_agent_cluster_skills():
    """ConsciousAgent.cluster_skills() wraps the call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    p1 = [("O", "partof"), ("O", "locatedin"), ("O", "isa")]
    p2 = [("O", "partof"), ("O", "locatedin"), ("O", "continent")]
    for _ in range(2):
        agent.skills.record_success(p1)
        agent.skills.record_success(p2)
    families = agent.cluster_skills(prefix_length=2)
    assert families
    assert families[0].prefix[0][1] == "partof"
