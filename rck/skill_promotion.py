"""Promote successful skill families into universal rules.

`skill_clustering` surfaces SkillFamily groups. When a family has
enough members at high enough confidence, it represents a stable,
generalisable pattern -- exactly the shape of a Rule.

This module looks at the families and emits a Rule for each
qualifying one, applying the same lifting/inverse/non-transitive
filters as `rule_extraction._induced_head` for consistency.
"""
from __future__ import annotations

from dataclasses import dataclass

from rck.chain_induction import InductionPolicy
from rck.rule_extraction import Rule, RuleStore, _induced_head
from rck.skill_clustering import SkillFamily, cluster_skills_by_prefix
from rck.skills import SkillLibrary


@dataclass
class PromotionPolicy:
    min_family_members: int = 2
    min_family_confidence: float = 0.7
    prefix_length: int = 2


def promote_families(skills: SkillLibrary,
                      store: RuleStore | None = None,
                      *,
                      policy: PromotionPolicy | None = None,
                      induction_policy: InductionPolicy | None = None,
                      ) -> list[Rule]:
    """Promote qualifying SkillFamilies into Rules added to `store`.

    If `store` is None a new RuleStore is created. Returns the list of
    newly added rules (excluding ones already present at higher
    confidence).
    """
    cfg = policy or PromotionPolicy()
    icfg = induction_policy or InductionPolicy()
    if store is None:
        store = RuleStore()
    families = cluster_skills_by_prefix(
        skills, prefix_length=cfg.prefix_length,
        min_members=cfg.min_family_members,
    )
    new_rules: list[Rule] = []
    for family in families:
        if family.confidence < cfg.min_family_confidence:
            continue
        body = [rel for _role, rel in family.prefix]
        head = _induced_head(body, icfg)
        if head is None:
            continue
        rule = Rule(
            body=body, head=head,
            support=family.total_success,
            confidence=family.confidence,
        )
        sig = rule.signature()
        existing = store.rules.get(sig)
        if existing is None:
            store.add(rule)
            new_rules.append(rule)
        elif rule.confidence > existing.confidence:
            existing.confidence = rule.confidence
            existing.support = max(existing.support, rule.support)
    return new_rules
