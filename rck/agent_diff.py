"""Diff two agents' auxiliary state.

Returns a structured report of what `b` knows that `a` doesn't,
what `a` knows that `b` doesn't, and what they agree on.

Compared dimensions:
  * Provenance keys (S, R, O triples)
  * Skill patterns
  * Rule signatures (extracted via extract_rules on each)

Does NOT compare the HRR knowledge tensor directly (it's a single
opaque vector). The provenance-key comparison covers the meaningful
content layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rck.conscious_agent import ConsciousAgent


@dataclass
class AgentDiffReport:
    only_in_a_facts: list[tuple[str, str, str]] = field(default_factory=list)
    only_in_b_facts: list[tuple[str, str, str]] = field(default_factory=list)
    shared_facts: int = 0

    only_in_a_skills: list[tuple] = field(default_factory=list)
    only_in_b_skills: list[tuple] = field(default_factory=list)
    shared_skills: int = 0

    only_in_a_rules: list[tuple] = field(default_factory=list)
    only_in_b_rules: list[tuple] = field(default_factory=list)
    shared_rules: int = 0

    def summary(self) -> dict:
        return {
            "facts_only_in_a": len(self.only_in_a_facts),
            "facts_only_in_b": len(self.only_in_b_facts),
            "facts_shared": self.shared_facts,
            "skills_only_in_a": len(self.only_in_a_skills),
            "skills_only_in_b": len(self.only_in_b_skills),
            "skills_shared": self.shared_skills,
            "rules_only_in_a": len(self.only_in_a_rules),
            "rules_only_in_b": len(self.only_in_b_rules),
            "rules_shared": self.shared_rules,
        }


def diff_agents(a: "ConsciousAgent", b: "ConsciousAgent") -> AgentDiffReport:
    """Compute the AgentDiffReport for two agents."""
    report = AgentDiffReport()

    # Provenance fact keys.
    a_facts = set(a.provenance._records) if a.provenance else set()
    b_facts = set(b.provenance._records) if b.provenance else set()
    report.only_in_a_facts = sorted(a_facts - b_facts)
    report.only_in_b_facts = sorted(b_facts - a_facts)
    report.shared_facts = len(a_facts & b_facts)

    # Skill signatures.
    a_skills = set(a.skills.skills) if a.skills else set()
    b_skills = set(b.skills.skills) if b.skills else set()
    report.only_in_a_skills = sorted(a_skills - b_skills)
    report.only_in_b_skills = sorted(b_skills - a_skills)
    report.shared_skills = len(a_skills & b_skills)

    # Rule signatures.
    try:
        a_store = a.extract_rules() if a.skills else None
        b_store = b.extract_rules() if b.skills else None
        a_rules = set(a_store.rules) if a_store else set()
        b_rules = set(b_store.rules) if b_store else set()
        report.only_in_a_rules = sorted(a_rules - b_rules)
        report.only_in_b_rules = sorted(b_rules - a_rules)
        report.shared_rules = len(a_rules & b_rules)
    except Exception:
        pass

    return report
