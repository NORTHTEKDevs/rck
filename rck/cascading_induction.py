"""Cascading chain induction -- iterate until the KB stops growing.

A single induction pass produces shortcuts. Those shortcuts ARE new
KB edges, which means they might open NEW chains that weren't
reachable before. Cascading induction runs `induce_from_chain` in
a loop, each pass discovering chains over the KB enriched by the
previous pass, until no new verified facts can be induced.

This is the closest RCK comes to "self-supervised knowledge graph
completion": no external data, just iterated chain reasoning + the
filters in `chain_induction`.

Three operational concerns:

  * **Termination**: bounded by `max_rounds` (default 5) and by the
    fixed-point condition that no new verified facts were added.
  * **Drift**: each pass adds shortcuts; later passes might chain
    THROUGH those shortcuts to derive less-direct shortcuts. The
    filters in `chain_induction` are local graph properties and stay
    valid no matter how many passes ran.
  * **Auditability**: every induced fact is tagged with the round it
    came from (via provenance tags).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from rck.chain_discover import Goal, discover_chains
from rck.chain_induction import (
    InducedFact, InductionPolicy, induce_from_chain,
)
from rck.chain_walker import Hop, walk_chain
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore
from rck.skills import SkillLibrary


@dataclass
class CascadeRound:
    round: int
    probes_tried: int
    chains_discovered: int
    facts_induced: int
    facts_verified: int
    facts_after: int          # KB size after this round
    new_pair_signatures: list[tuple] = field(default_factory=list)


@dataclass
class CascadeResult:
    rounds: list[CascadeRound]
    total_induced: int
    total_verified: int
    initial_facts: int
    final_facts: int
    induced_facts: list[InducedFact]
    saturated: bool             # True if reached fixed point


def _candidate_pairs(kb: ShardedKnowledgeBase,
                     limit: int) -> list[tuple[str, str]]:
    """Build 2-hop transitive (start, target) probe pairs from the
    current KB state. Targets that already have a direct (start, ?, X)
    edge to them are skipped (no new induction possible)."""
    by_subject: dict[str, list[tuple[str, str]]] = {}
    for shard in kb._shards:
        for fact in shard.facts():
            s = str(fact["S"]); r = str(fact["R"]); o = str(fact["O"])
            by_subject.setdefault(s, []).append((r, o))

    probes: list[tuple[str, str]] = []
    for s, edges in by_subject.items():
        direct_objs = {o for _, o in edges}
        for _, mid in edges:
            if mid in by_subject:
                for _, target in by_subject[mid]:
                    if target == s or target in direct_objs:
                        continue
                    probes.append((s, target))
                    break
            if len(probes) >= limit:
                return probes
    return probes


def cascade_induct(kb: ShardedKnowledgeBase, *,
                   max_rounds: int = 5,
                   probes_per_round: int = 80,
                   policy: InductionPolicy | None = None,
                   skills: SkillLibrary | None = None,
                   provenance: ProvenanceStore | None = None,
                   ) -> CascadeResult:
    """Run cascading induction until saturation or `max_rounds`.

    Args:
        probes_per_round: how many (start, target) pairs to try per round.
        policy: InductionPolicy to apply per fact.
        skills: optional SkillLibrary that records every successful chain.
        provenance: optional ProvenanceStore for source tagging.
    """
    cfg = policy or InductionPolicy()
    initial = kb.size()
    rounds: list[CascadeRound] = []
    all_induced: list[InducedFact] = []
    seen_pairs: set[tuple[str, str]] = set()

    for rnd in range(1, max_rounds + 1):
        probes = _candidate_pairs(kb, probes_per_round * 3)
        # Only try pairs we haven't already probed THIS cascade.
        fresh = [p for p in probes if p not in seen_pairs][:probes_per_round]
        seen_pairs.update(fresh)

        chains_found = 0
        induced_this_round = 0
        verified_this_round = 0
        new_sigs: list[tuple] = []

        for start, target in fresh:
            ch_list = discover_chains(
                kb, start, Goal.symbol(target), max_depth=3,
                beam_width=3, top_n=1, min_link_score=0.10,
            )
            if not ch_list:
                continue
            ch = ch_list[0]
            chains_found += 1
            hops = [Hop(r, d) for r, d in zip(ch.relations, ch.directions)]
            result = walk_chain(kb, start, hops, skills=skills)
            if result.answer is None:
                continue
            induced = induce_from_chain(
                kb, result, start,
                policy=cfg, skills=skills, provenance=provenance,
            )
            if induced is None:
                continue
            induced_this_round += 1
            if induced.verified:
                verified_this_round += 1
                all_induced.append(induced)
                new_sigs.append(tuple(ch.relations))

        rounds.append(CascadeRound(
            round=rnd,
            probes_tried=len(fresh),
            chains_discovered=chains_found,
            facts_induced=induced_this_round,
            facts_verified=verified_this_round,
            facts_after=kb.size(),
            new_pair_signatures=new_sigs,
        ))

        # Fixed point: no new verified facts.
        if verified_this_round == 0:
            return CascadeResult(
                rounds=rounds,
                total_induced=sum(r.facts_induced for r in rounds),
                total_verified=sum(r.facts_verified for r in rounds),
                initial_facts=initial,
                final_facts=kb.size(),
                induced_facts=all_induced,
                saturated=True,
            )

    return CascadeResult(
        rounds=rounds,
        total_induced=sum(r.facts_induced for r in rounds),
        total_verified=sum(r.facts_verified for r in rounds),
        initial_facts=initial,
        final_facts=kb.size(),
        induced_facts=all_induced,
        saturated=False,
    )


def top_pattern_signatures(result: CascadeResult,
                           top_k: int = 5) -> list[tuple[tuple, int]]:
    """Rank the chain shapes (relation signatures) that produced
    induced facts. Useful for surfacing reusable inference templates."""
    counter: Counter = Counter()
    for r in result.rounds:
        for sig in r.new_pair_signatures:
            counter[sig] += 1
    return counter.most_common(top_k)
