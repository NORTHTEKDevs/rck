"""Compose existing rules into longer ones.

Given:
  R1: body=[A1, ..., An], head=H1
  R2: body=[H1, B2, ..., Bm], head=H2

The composition R1 ; R2 has:
  body=[A1, ..., An, B2, ..., Bm]
  head=H2

Composed rules let the system reason at deeper depths WITHOUT
re-running chain discovery. They inherit the SAME filters
(inverse-pair, non-transitive same-relation) and a confidence that
is the product of the two source rules' confidences.

Each composition is a deterministic operation -- we don't need
chain_walker or kb queries here, just symbolic substitution. The
resulting rule is added to the RuleStore where instantiate_rule
can apply it like any other.
"""
from __future__ import annotations

from typing import Iterable

from rck.bulk_ingest import inverse_relation
from rck.chain_induction import InductionPolicy
from rck.rule_extraction import Rule, RuleStore


def can_compose(r1: Rule, r2: Rule,
                *, policy: InductionPolicy | None = None) -> bool:
    """True iff r1's head matches r2's first body relation AND the
    resulting body passes the standard filters."""
    cfg = policy or InductionPolicy()
    if not r1.body or not r2.body:
        return False
    if r1.head != r2.body[0]:
        return False
    # Build the candidate body and check seams.
    new_body = list(r1.body) + list(r2.body[1:])
    if len(new_body) < 2:
        return False
    for i in range(len(new_body) - 1):
        inv = inverse_relation(new_body[i])
        if inv is not None and inv == new_body[i + 1]:
            return False
        if (new_body[i] == new_body[i + 1]
                and new_body[i] not in cfg.transitive_relations):
            return False
    return True


def compose(r1: Rule, r2: Rule,
            *, policy: InductionPolicy | None = None) -> Rule | None:
    """Return R1 ; R2 as a new Rule, or None if composition is invalid."""
    if not can_compose(r1, r2, policy=policy):
        return None
    body = list(r1.body) + list(r2.body[1:])
    return Rule(
        body=body,
        head=r2.head,
        support=min(r1.support, r2.support),
        confidence=float(r1.confidence) * float(r2.confidence),
    )


def compose_all(store: RuleStore,
                *, policy: InductionPolicy | None = None,
                max_body_length: int = 5,
                require_min_support: int = 1) -> list[Rule]:
    """Enumerate every valid pairwise composition over `store`.

    Composed rules longer than `max_body_length` are skipped (they
    become rare and expensive to instantiate). Returns the list of
    NEWLY added rules (each is also added to the store).
    """
    sources = [r for r in store.all_rules()
               if r.support >= require_min_support]
    new_rules: list[Rule] = []
    for r1 in sources:
        for r2 in sources:
            if r1 is r2:
                continue
            composed = compose(r1, r2, policy=policy)
            if composed is None:
                continue
            if len(composed.body) > max_body_length:
                continue
            sig = composed.signature()
            if sig in store.rules:
                # Already known; refresh metadata if stronger.
                existing = store.rules[sig]
                if composed.confidence > existing.confidence:
                    existing.confidence = composed.confidence
                continue
            store.add(composed)
            new_rules.append(composed)
    return new_rules


def compose_chain(rules: Iterable[Rule],
                  *, policy: InductionPolicy | None = None) -> Rule | None:
    """Sequentially compose a list of rules: r1 ; r2 ; r3 ; ...
    Returns None if any pair is incompatible."""
    rs = list(rules)
    if not rs:
        return None
    if len(rs) == 1:
        return rs[0]
    out = rs[0]
    for r in rs[1:]:
        nxt = compose(out, r, policy=policy)
        if nxt is None:
            return None
        out = nxt
    return out
