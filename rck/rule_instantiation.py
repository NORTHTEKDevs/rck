"""Forward-chaining rule instantiation.

`rule_extraction` produces symbolic rules of the form

    forall X0, X1, X2. (X0 R1 X1) and (X1 R2 X2) => (X0 R_head X2)

This module is the OTHER HALF: given a rule and a KB, scan the KB
and emit every direct fact (X0, R_head, X2) the rule derives. The
output integrates with the existing induction pipeline: each
emitted fact runs through `chain_induction`'s self-verification
and provenance tagging.

Unlike chain induction (which discovers chains by BFS), rule
instantiation walks the KB by relation index: for every stored
(X0, R1, X1), it queries (X1, R2, ?) and proposes (X0, R_head, ?).
This is asymptotically faster for very productive rules (the
`locatedin -> continent => continent` rule in the commonsense study
fires 28 times after one cascade pass).

The same induction filters apply -- a rule that survived
`extract_rules` should already have a sane head, but we still run
the inverse-pair / non-transitive / lifting checks at the
instantiation level for defence in depth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from rck.bulk_ingest import inverse_relation
from rck.chain_induction import InductionPolicy
from rck.knowledge_base import ShardedKnowledgeBase
from rck.negative_facts import denied_pairs_for
from rck.provenance import ProvenanceStore
from rck.rule_extraction import Rule, RuleStore
from rck.self_verify import verify_roundtrip


@dataclass
class InstantiatedFact:
    """A fact derived by forward-applying a rule."""
    subject: str
    relation: str
    obj: str
    rule_signature: tuple
    rule_support: int
    confidence: float
    bindings: tuple  # (X1, X2, ...) intermediate values from the chain
    verified: bool = False
    rejected_reason: str | None = None


def _iter_facts(kb: ShardedKnowledgeBase, relation: str | None = None
                ) -> Iterable[tuple[str, str, str]]:
    """Yield every stored (s, r, o) optionally filtered by relation."""
    for fact in kb.all_facts():
        s = str(fact.get("S", "")); r = str(fact.get("R", "")); o = str(fact.get("O", ""))
        if relation is not None and r != relation:
            continue
        yield s, r, o


def instantiate_rule(kb: ShardedKnowledgeBase, rule: Rule,
                     *, policy: InductionPolicy | None = None,
                     min_link_score: float = 0.10,
                     verify: bool = True,
                     max_facts: int = 1000,
                     provenance: ProvenanceStore | None = None,
                     ) -> list[InstantiatedFact]:
    """Forward-apply `rule` (N-clause body) to the KB.

    For a rule `(X0 R1 X1) and (X1 R2 X2) and ... and (X_{N-1} R_N X_N)
    => (X0 R_head X_N)`:
      * Walk every stored (X0, R1, X1) seed.
      * Recursively extend X1 via R2 -> X2, X2 via R3 -> X3, etc.
      * Emit (X0, R_head, X_N) if it isn't already a stored direct fact.

    Filters (inverse-pair, non-transitive same-relation) apply to
    every consecutive pair in the body. Cycles within the chain are
    blocked.
    """
    cfg = policy or InductionPolicy()
    body = rule.body
    if len(body) < 2:
        return []
    head = rule.head

    # Defence-in-depth: re-check the filters that extraction applied.
    for i in range(len(body) - 1):
        inv = inverse_relation(body[i])
        if inv is not None and inv == body[i + 1]:
            return []
        if (body[i] == body[i + 1]
                and body[i] not in cfg.transitive_relations):
            return []

    out: list[InstantiatedFact] = []
    seen_emitted: set[tuple[str, str, str]] = set()

    def emit(x0: str, x_n: str, chain: list[tuple[str, str, str]],
             link_score: float) -> None:
        key = (x0, head, x_n)
        if key in seen_emitted:
            return
        seen_emitted.add(key)
        # Skip if already a direct fact (top-K membership check).
        existing = kb.query({"S": x0, "R": head}, "O", top_k=5)
        if any(str(s_).lower() == x_n and float(sc) >= min_link_score
               for s_, sc in existing):
            return
        # Respect explicit negation: don't emit (x0, head, x_n) when
        # (x0, NOT_head, x_n) has been asserted.
        denied = denied_pairs_for(kb, x0, head, min_score=0.10)
        if any(str(sym).lower() == x_n for sym, _ in denied):
            rule.record_instantiation(succeeded=False)
            return
        kb.store({"S": x0, "R": head, "O": x_n})
        if provenance is not None:
            provenance.store(
                x0, head, x_n, source="rule",
                confidence=rule.confidence,
                tags={"rule_instantiated", f"rule_{rule.signature()}",
                      f"body_{len(body)}_clauses"},
                derivation=list(chain),
            )
        inst = InstantiatedFact(
            subject=x0, relation=head, obj=x_n,
            rule_signature=rule.signature(),
            rule_support=rule.support,
            confidence=float(rule.confidence) * float(link_score),
            bindings=tuple(c[2] for c in chain[:-1]),
        )
        if verify:
            v = verify_roundtrip(kb, x0, head, x_n)
            inst.verified = v.verified
            if not inst.verified:
                kb.forget({"S": x0, "R": head, "O": x_n})
                if provenance is not None:
                    provenance.forget(x0, head, x_n)
                inst.rejected_reason = "roundtrip verification failed"
                # Feedback: this rule produced a fact that didn't verify.
                rule.record_instantiation(succeeded=False)
                return
        else:
            inst.verified = True
        # Feedback: this rule produced a verified fact.
        rule.record_instantiation(succeeded=True)
        out.append(inst)

    def walk(current: str, depth: int, chain: list[tuple[str, str, str]],
             score_so_far: float, visited: set[str]) -> None:
        """Recursively walk the rule body from `current` (X_depth)."""
        if len(out) >= max_facts:
            return
        if depth == len(body):
            # Chain consumed; current is X_N.
            x0 = chain[0][0]
            x_n = current
            if x_n in {x0} or x_n in visited - {current}:
                return
            emit(x0, x_n, chain, score_so_far)
            return
        rel = body[depth]
        candidates = kb.query({"S": current, "R": rel}, "O", top_k=3)
        for sym, score in candidates:
            if float(score) < min_link_score:
                continue
            next_node = str(sym).lower()
            if next_node in visited:
                # Cycle in this binding path.
                continue
            walk(
                next_node, depth + 1,
                chain + [(current, rel, next_node)],
                score_so_far * float(score),
                visited | {next_node},
            )

    # Seed: every stored (x0, R1, x1) starts a path.
    r1 = body[0]
    for x0, _, x1 in _iter_facts(kb, relation=r1):
        if len(out) >= max_facts:
            return out
        x0 = x0.lower(); x1 = x1.lower()
        walk(
            x1, depth=1,
            chain=[(x0, r1, x1)],
            score_so_far=1.0,
            visited={x0, x1},
        )
    return out


def instantiate_all(kb: ShardedKnowledgeBase, store: RuleStore,
                    *, policy: InductionPolicy | None = None,
                    min_link_score: float = 0.10,
                    provenance: ProvenanceStore | None = None,
                    ) -> list[InstantiatedFact]:
    """Forward-apply every rule in the store to the KB.

    Rules are applied in support order (most-supported first) so the
    cumulative KB growth tracks the dominant patterns.
    """
    out: list[InstantiatedFact] = []
    for rule in store.top_rules(n=len(store.rules)):
        emitted = instantiate_rule(
            kb, rule, policy=policy,
            min_link_score=min_link_score, provenance=provenance,
        )
        out.extend(emitted)
    return out
