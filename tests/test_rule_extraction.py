"""Tests for rule extraction from skill patterns."""
from __future__ import annotations

from rck.chain_induction import InductionPolicy
from rck.rule_extraction import (
    Rule, RuleStore, extract_rules, extract_rules_from_chains,
)
from rck.skills import SkillLibrary


def test_rule_signature_and_verbalize():
    rule = Rule(body=["partof", "locatedin"], head="locatedin",
                support=5, confidence=0.9)
    assert rule.signature() == (("partof", "locatedin"), "locatedin")
    text = rule.verbalize()
    assert "forall" in text
    assert "partof" in text and "locatedin" in text


def test_extract_rules_from_skill_library():
    """A pattern seen >=2 times with high confidence yields a rule."""
    skills = SkillLibrary()
    pattern = [("O", "partof"), ("O", "locatedin")]
    skills.record_success(pattern)
    skills.record_success(pattern)
    skills.record_success(pattern)
    store = extract_rules(skills, min_support=2)
    assert store.size() == 1
    rule = store.all_rules()[0]
    assert rule.body == ["partof", "locatedin"]
    assert rule.head == "locatedin"   # partof is lifting -> last hop relation
    assert rule.support == 3


def test_extract_filters_below_support():
    """A pattern seen only once is below default support threshold."""
    skills = SkillLibrary()
    skills.record_success([("O", "partof"), ("O", "locatedin")])
    store = extract_rules(skills, min_support=2)
    assert store.size() == 0


def test_extract_falls_back_to_implies_for_non_lifting_first_hop():
    """Pattern starting with non-lifting relation (causes) -> head=implies."""
    skills = SkillLibrary()
    pattern = [("O", "causes"), ("O", "isa")]
    skills.record_success(pattern)
    skills.record_success(pattern)
    store = extract_rules(skills, min_support=2)
    assert store.size() == 1
    rule = store.all_rules()[0]
    assert rule.head == "implies"


def test_extract_same_relation_transitive():
    """Same-relation transitive chain keeps the relation as head."""
    skills = SkillLibrary()
    pattern = [("O", "isa"), ("O", "isa")]
    for _ in range(3):
        skills.record_success(pattern)
    store = extract_rules(skills, min_support=2)
    assert store.size() == 1
    rule = store.all_rules()[0]
    assert rule.head == "isa"


def test_lookup_by_body():
    store = RuleStore()
    store.add(Rule(body=["partof", "locatedin"], head="locatedin",
                    support=3, confidence=1.0))
    found = store.lookup(["partof", "locatedin"])
    assert found is not None
    assert found.head == "locatedin"
    miss = store.lookup(["nope", "locatedin"])
    assert miss is None


def test_top_rules_orders_by_support():
    store = RuleStore()
    store.add(Rule(body=["a", "b"], head="b", support=1, confidence=1.0))
    store.add(Rule(body=["partof", "locatedin"], head="locatedin",
                    support=10, confidence=1.0))
    store.add(Rule(body=["isa", "isa"], head="isa", support=5, confidence=1.0))
    top = store.top_rules(n=2)
    assert top[0].support == 10
    assert top[1].support == 5


def test_extract_from_chains_groups_by_signature():
    """Direct extraction from a list of relation-only chains."""
    chains = [
        ["partof", "locatedin"],
        ["partof", "locatedin"],
        ["partof", "locatedin"],
        ["isa", "isa"],
    ]
    store = extract_rules_from_chains(chains, min_support=2)
    assert store.size() == 1  # only partof-locatedin has support >= 2
    assert store.all_rules()[0].body == ["partof", "locatedin"]


def test_extract_rules_filter_inverse_pair():
    """Patterns like author -> wrote should NOT emit a rule."""
    skills = SkillLibrary()
    pattern = [("O", "author"), ("O", "wrote")]
    for _ in range(5):
        skills.record_success(pattern)
    store = extract_rules(skills, min_support=2)
    assert store.size() == 0


def test_extract_rules_filter_non_transitive_same_relation():
    """wrote -> wrote should NOT emit a rule (non-transitive)."""
    skills = SkillLibrary()
    pattern = [("O", "wrote"), ("O", "wrote")]
    for _ in range(5):
        skills.record_success(pattern)
    store = extract_rules(skills, min_support=2)
    assert store.size() == 0


def test_record_instantiation_updates_confidence():
    """Successes raise confidence; failures lower it."""
    rule = Rule(body=["partof", "locatedin"], head="locatedin",
                support=3, confidence=1.0)
    # All successes -> confidence stays near 1.0 (Laplace smoothing).
    for _ in range(10):
        rule.record_instantiation(True)
    assert rule.confidence > 0.85
    # Now feed failures.
    for _ in range(10):
        rule.record_instantiation(False)
    # Confidence drops toward 0.5.
    assert rule.confidence < 0.6


def test_prune_drops_low_confidence_rules():
    """RuleStore.prune removes rules below the threshold."""
    store = RuleStore()
    bad = Rule(body=["a", "b"], head="implies",
               support=2, confidence=1.0)
    good = Rule(body=["c", "d"], head="implies",
                support=2, confidence=1.0)
    for _ in range(5):
        bad.record_instantiation(False)
    for _ in range(5):
        good.record_instantiation(True)
    store.add(bad); store.add(good)
    n = store.prune(min_confidence=0.30, min_attempts=3)
    assert n == 1
    assert store.size() == 1
    # The good rule remains.
    remaining = store.all_rules()[0]
    assert remaining.body == ["c", "d"]


def test_prune_keeps_unattempted_rules():
    """A freshly-extracted rule with no feedback survives the prune."""
    store = RuleStore()
    fresh = Rule(body=["a", "b"], head="implies",
                  support=2, confidence=1.0)
    store.add(fresh)
    n = store.prune(min_confidence=0.99, min_attempts=3)
    assert n == 0  # had no attempts, can't judge


def test_extract_rules_skip_failed_skills():
    """Skills with low confidence don't graduate to rules."""
    skills = SkillLibrary()
    pattern = [("O", "partof"), ("O", "locatedin")]
    skills.record_success(pattern)
    skills.record_failure(pattern)
    skills.record_failure(pattern)
    store = extract_rules(skills, min_support=1, min_confidence=0.5)
    assert store.size() == 0
