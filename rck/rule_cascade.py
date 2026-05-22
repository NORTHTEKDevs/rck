"""Cascading rule instantiation -- forward chain to fixed point.

`instantiate_all` applies every rule once. But an instantiated fact
can ENABLE further rule applications in subsequent rounds: if rule
R1 produces (a, isa, c) from (a, isa, b) and (b, isa, c), then rule
R2 with body [isa, isa] now sees (a, isa, c) as a new R1 binding
and can derive yet-deeper facts.

This module loops `instantiate_all` until no new facts are added
or `max_rounds` is reached.

Pairs naturally with `cascading_induction` which does the same for
chain-discovered shortcuts. The DIFFERENCE is that rule cascade is
KB-driven (walks every R1 binding); cascade_induction is
probe-driven (picks (start, target) pairs and BFSes). They produce
overlapping but distinct fact sets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rck.chain_induction import InductionPolicy
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore
from rck.rule_extraction import RuleStore
from rck.rule_instantiation import InstantiatedFact, instantiate_all


@dataclass
class RuleCascadeRound:
    round: int
    facts_emitted: int
    facts_verified: int
    facts_after: int


@dataclass
class RuleCascadeResult:
    rounds: list[RuleCascadeRound]
    induced_facts: list[InstantiatedFact]
    saturated: bool
    initial_facts: int
    final_facts: int

    def total_verified(self) -> int:
        return sum(r.facts_verified for r in self.rounds)


def cascade_instantiate(kb: ShardedKnowledgeBase,
                        store: RuleStore,
                        *, max_rounds: int = 4,
                        policy: InductionPolicy | None = None,
                        min_link_score: float = 0.10,
                        provenance: ProvenanceStore | None = None,
                        ) -> RuleCascadeResult:
    """Apply every rule in `store` repeatedly until saturation."""
    initial = kb.size()
    rounds: list[RuleCascadeRound] = []
    all_induced: list[InstantiatedFact] = []

    for rnd in range(1, max_rounds + 1):
        pre = kb.size()
        emitted = instantiate_all(
            kb, store, policy=policy,
            min_link_score=min_link_score,
            provenance=provenance,
        )
        verified = [f for f in emitted if f.verified]
        all_induced.extend(verified)
        rounds.append(RuleCascadeRound(
            round=rnd,
            facts_emitted=len(emitted),
            facts_verified=len(verified),
            facts_after=kb.size(),
        ))
        # Saturation: no new facts were added.
        if kb.size() == pre:
            return RuleCascadeResult(
                rounds=rounds, induced_facts=all_induced,
                saturated=True, initial_facts=initial,
                final_facts=kb.size(),
            )

    return RuleCascadeResult(
        rounds=rounds, induced_facts=all_induced,
        saturated=False, initial_facts=initial,
        final_facts=kb.size(),
    )
