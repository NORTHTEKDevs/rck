"""Higher-order rule extraction from skill patterns.

`SkillLibrary` records every successful chain pattern (sequence of
(role, relation) edges). When a pattern fires K times across DIFFERENT
subjects, it stops being a one-off coincidence and starts looking
like a UNIVERSAL inference rule. This module extracts those rules
and stores them in a `RuleStore`.

A rule has the form:
    forall X_0, X_1, ..., X_n.
      (X_0, R_1, X_1) and (X_1, R_2, X_2) and ... and (X_{n-1}, R_n, X_n)
      => (X_0, R_induced, X_n)

`R_induced` is computed using the same lifting-relation logic as
`chain_induction`: if R_1 is a lifting relation, propagate R_n;
otherwise emit a generic `implies` rule.

Rules carry confidence (= success_count / (success+failure)) and
support count (= success_count). They expire when their confidence
falls below `min_confidence` or when overridden by a more specific
rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from rck.bulk_ingest import inverse_relation
from rck.chain_induction import InductionPolicy
from rck.skills import Skill, SkillLibrary


@dataclass
class Rule:
    """A symbolic implication extracted from a repeated chain pattern."""
    body: list[str]            # ordered list of relation names
    head: str                  # the induced relation
    support: int               # how many distinct chains instantiated this
    confidence: float          # success rate over attempts
    body_directions: list[str] = field(default_factory=list)
    # Feedback counters updated by instantiation outcomes.
    instantiation_successes: int = 0
    instantiation_failures: int = 0

    def signature(self) -> tuple[tuple[str, ...], str]:
        return (tuple(self.body), self.head)

    def record_instantiation(self, succeeded: bool) -> None:
        """Update success / failure counters from an instantiation outcome.

        Confidence is updated as a smoothed success ratio (Laplace
        smoothing with prior 1.0): conf = (succ + 1) / (succ + fail + 2).
        """
        if succeeded:
            self.instantiation_successes += 1
        else:
            self.instantiation_failures += 1
        total = self.instantiation_successes + self.instantiation_failures
        self.confidence = (self.instantiation_successes + 1) / (total + 2)

    def verbalize(self) -> str:
        n = len(self.body)
        variables = [f"X{i}" for i in range(n + 1)]
        clauses = [
            f"({variables[i]} {self.body[i]} {variables[i + 1]})"
            for i in range(n)
        ]
        head_clause = f"({variables[0]} {self.head} {variables[n]})"
        return (f"forall {', '.join(variables)}. "
                f"{' and '.join(clauses)} => {head_clause}")


@dataclass
class RuleStore:
    rules: dict[tuple[tuple[str, ...], str], Rule] = field(default_factory=dict)

    def add(self, rule: Rule) -> None:
        sig = rule.signature()
        existing = self.rules.get(sig)
        if existing is None:
            self.rules[sig] = rule
            return
        existing.support = max(existing.support, rule.support)
        existing.confidence = max(existing.confidence, rule.confidence)

    def lookup(self, body: list[str]) -> Rule | None:
        body_tuple = tuple(body)
        for sig, rule in self.rules.items():
            if sig[0] == body_tuple:
                return rule
        return None

    def all_rules(self) -> list[Rule]:
        return list(self.rules.values())

    def top_rules(self, n: int = 10) -> list[Rule]:
        return sorted(
            self.rules.values(),
            key=lambda r: (-r.support, -r.confidence),
        )[:n]

    def size(self) -> int:
        return len(self.rules)

    def effectiveness_report(self, *, top_k: int = 20) -> list[dict]:
        """Per-rule effectiveness summary. Rows sorted by total uses
        descending. Useful for cron-job logging and human inspection."""
        rows: list[dict] = []
        for rule in self.rules.values():
            total = (rule.instantiation_successes
                     + rule.instantiation_failures)
            rows.append({
                "body": list(rule.body),
                "head": rule.head,
                "support": rule.support,
                "uses": total,
                "successes": rule.instantiation_successes,
                "failures": rule.instantiation_failures,
                "confidence": rule.confidence,
                "verbal": rule.verbalize(),
            })
        rows.sort(key=lambda r: (-r["uses"], -r["confidence"]))
        return rows[:top_k]

    def prune(self, *, min_confidence: float = 0.30,
              min_attempts: int = 3) -> int:
        """Drop rules with low post-feedback confidence.

        Only considers rules with at least `min_attempts` instantiations
        (success + failure) -- newly-extracted rules with no feedback
        history are preserved. Returns the number pruned.
        """
        to_drop: list = []
        for sig, rule in self.rules.items():
            total = (rule.instantiation_successes
                     + rule.instantiation_failures)
            if total < min_attempts:
                continue
            if rule.confidence < min_confidence:
                to_drop.append(sig)
        for sig in to_drop:
            del self.rules[sig]
        return len(to_drop)


def extract_rules(skills: SkillLibrary, *,
                  min_support: int = 2,
                  min_confidence: float = 0.5,
                  policy: InductionPolicy | None = None) -> RuleStore:
    """Walk a SkillLibrary and emit a `Rule` for every pattern that
    meets `min_support` AND `min_confidence`.

    The induced-relation rules apply the same lifting-relation logic
    as `chain_induction`, so the extracted rules are consistent with
    the system's fact-induction behaviour.
    """
    cfg = policy or InductionPolicy()
    store = RuleStore()
    for skill in skills.skills.values():
        if skill.success_count < min_support:
            continue
        if skill.confidence < min_confidence:
            continue
        body_relations = [rel for _, rel in skill.pattern]
        body_dirs = ["forward"] * len(skill.pattern)
        if not body_relations:
            continue
        head = _induced_head(body_relations, cfg)
        if head is None:
            continue
        rule = Rule(
            body=body_relations,
            head=head,
            support=skill.success_count,
            confidence=skill.confidence,
            body_directions=body_dirs,
        )
        store.add(rule)
    return store


def _induced_head(body: list[str], cfg: InductionPolicy) -> str | None:
    """Mirrors the chain_induction gates exactly:
    * reject chains with an inverse-pair seam
    * reject non-transitive same-relation chains
    * same-relation transitive chains keep their relation
    * lifting first hop propagates the last relation
    * everything else emits `implies`
    """
    if not body:
        return None
    # Inverse-pair seam: reject entirely (no rule emitted).
    for i in range(len(body) - 1):
        inv = inverse_relation(body[i])
        if inv is not None and inv == body[i + 1]:
            return None
    # Non-transitive same-relation seam: also reject.
    for i in range(len(body) - 1):
        if body[i] == body[i + 1] and body[i] not in cfg.transitive_relations:
            return None
    if all(r == body[0] for r in body) and body[0] in cfg.transitive_relations:
        return body[0]
    if body[0] in cfg.lifting_relations:
        return body[-1]
    return cfg.induce_relation


def extract_rules_from_chains(observed_chains: Iterable[list[str]],
                              *, min_support: int = 2,
                              policy: InductionPolicy | None = None
                              ) -> RuleStore:
    """Alternative ingestion: extract rules directly from a list of
    chain signatures (lists of relation names) without going through
    a SkillLibrary. Useful when chains are produced by a process that
    doesn't populate skills yet."""
    cfg = policy or InductionPolicy()
    counts: dict[tuple[str, ...], int] = {}
    for chain in observed_chains:
        sig = tuple(chain)
        counts[sig] = counts.get(sig, 0) + 1
    store = RuleStore()
    for sig, count in counts.items():
        if count < min_support:
            continue
        body = list(sig)
        head = _induced_head(body, cfg)
        if head is None:
            continue
        store.add(Rule(
            body=body, head=head,
            support=count, confidence=1.0,
            body_directions=["forward"] * len(body),
        ))
    return store
