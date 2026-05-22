"""Tests for promoting SkillFamilies into Rules."""
from __future__ import annotations

from rck.rule_extraction import RuleStore
from rck.skill_promotion import PromotionPolicy, promote_families
from rck.skills import SkillLibrary


def test_promote_qualifying_family():
    """Two skills sharing prefix [partof, locatedin] with high success
    should produce a Rule."""
    skills = SkillLibrary()
    p1 = [("O", "partof"), ("O", "locatedin"), ("O", "isa")]
    p2 = [("O", "partof"), ("O", "locatedin"), ("O", "continent")]
    for _ in range(3):
        skills.record_success(p1)
        skills.record_success(p2)
    store = RuleStore()
    new = promote_families(skills, store)
    assert new
    rule = new[0]
    assert rule.body == ["partof", "locatedin"]
    # partof is a lifting relation, so head should be the second hop.
    assert rule.head == "locatedin"


def test_skip_low_confidence_family():
    """A family below min_family_confidence is dropped."""
    skills = SkillLibrary()
    p1 = [("O", "partof"), ("O", "locatedin"), ("O", "isa")]
    p2 = [("O", "partof"), ("O", "locatedin"), ("O", "continent")]
    for _ in range(2):
        skills.record_success(p1)
        skills.record_success(p2)
    # Add failures to drag confidence down.
    for _ in range(5):
        skills.record_failure(p1)
        skills.record_failure(p2)
    new = promote_families(
        skills, policy=PromotionPolicy(min_family_confidence=0.7),
    )
    assert new == []


def test_promote_does_not_duplicate_existing_rule():
    """Re-running promotion shouldn't add the same rule again."""
    skills = SkillLibrary()
    p1 = [("O", "partof"), ("O", "locatedin"), ("O", "isa")]
    p2 = [("O", "partof"), ("O", "locatedin"), ("O", "continent")]
    for _ in range(3):
        skills.record_success(p1)
        skills.record_success(p2)
    store = RuleStore()
    promote_families(skills, store)
    pre = store.size()
    again = promote_families(skills, store)
    # Second call should add 0 new rules.
    assert again == []
    assert store.size() == pre


def test_promote_filters_inverse_pair_prefix():
    """A family whose prefix is an inverse pair should be filtered out."""
    skills = SkillLibrary()
    p1 = [("O", "author"), ("O", "wrote"), ("O", "isa")]
    for _ in range(4):
        skills.record_success(p1)
    new = promote_families(skills)
    # author/wrote is an inverse pair -> head() returns None -> no rule.
    assert new == []


def test_conscious_agent_promote_skills():
    """ConsciousAgent.promote_skills() composes the call and stores
    rules into its own RuleStore (returned to the caller)."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    p1 = [("O", "partof"), ("O", "locatedin"), ("O", "isa")]
    p2 = [("O", "partof"), ("O", "locatedin"), ("O", "continent")]
    for _ in range(3):
        agent.skills.record_success(p1)
        agent.skills.record_success(p2)
    store, new_rules = agent.promote_skills()
    assert new_rules
    assert any(r.body == ["partof", "locatedin"] for r in new_rules)
