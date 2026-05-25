"""Fact induction from confident chains.

When a chain like `leaf -> partof -> tree -> locatedin -> forest` walks
correctly with high confidence, the system has effectively DERIVED
the new fact `(leaf, locatedin, forest)`. We can synthesise that
direct edge and store it -- next time the same question comes up the
agent answers in O(1) instead of O(chain_depth).

This is fact INDUCTION from chains, structurally the opposite of LLM
hallucination: we only ever induce facts whose chain has been WALKED
and VERIFIED. Synthesised facts carry a provenance tag indicating
they were induced (so we can roll them back or treat them with care).

Three rules govern induction:

1. **Min chain confidence**: only induce from chains whose propagated
   confidence is above a threshold (default 0.20 = moderate).
2. **Min path length**: don't bother inducing from 1-hop chains
   (they're already direct facts).
3. **Skill confirmation**: optionally require that the chain's
   pattern has been seen K times before -- only stable patterns
   get to emit new facts.

Inferred facts that fail self-verification are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rck.bulk_ingest import inverse_relation
from rck.chain_walker import ChainResult
from rck.knowledge_base import ShardedKnowledgeBase
from rck.negative_facts import denied_pairs_for
from rck.provenance import ProvenanceStore
from rck.self_verify import verify as _verify_kb
from rck.skills import SkillLibrary


@dataclass
class InductionPolicy:
    min_confidence: float = 0.20
    min_chain_length: int = 2
    require_skill_uses: int = 0   # 0 = no skill confirmation needed
    verify_after: bool = True
    induce_relation: str = "implies"  # the relation under which the
                                       # inferred fact is stored
    # Filter "phantom-shortcut" chains where consecutive relations are
    # inverses of each other (e.g. author -> wrote). Such chains are
    # round-trips through a hub entity and the induced shortcut is
    # almost always semantically wrong.
    filter_inverse_pairs: bool = True
    # Same-relation chains (X -[R]-> Y -[R]-> Z) only induce a shortcut
    # if R is in this transitive set. Otherwise the apparent chain is
    # usually a HRR cleanup artefact.
    transitive_relations: frozenset[str] = frozenset({
        "isa", "partof", "locatedin", "ancestorof",
        "hassubtype", "haspart", "contains", "descendantof",
    })
    # "Lifting" relations make the SOURCE inherit properties through
    # them: X partof Y means SOME things true about Y are potentially
    # true about X -- but not ALL. Each lifting family permits a
    # specific tail-relation set (see *_transfers below). A chain
    # passes the type-signature gate only if (first_hop_family,
    # last_hop_relation) is in the allowed transfer set.
    lifting_relations: frozenset[str] = frozenset({
        "isa", "partof", "locatedin", "memberof",
        "instanceof", "kind", "class", "subclass",
        "subtype", "category",
    })
    # Class-style lifting (isa/class/category/...): a subclass inherits
    # superclass type membership and component-having relations.
    isa_family: frozenset[str] = frozenset({
        "isa", "class", "subclass", "subtype", "category",
        "kind", "instanceof", "memberof",
    })
    isa_transfers: frozenset[str] = frozenset({
        "isa", "has", "kind", "class", "category",
    })
    # Partof: a part inherits the whole's location, material, and use.
    # It does NOT inherit what the whole "has" (a wing doesn't have
    # a beak even though the bird does).
    partof_transfers: frozenset[str] = frozenset({
        "locatedin", "madeof", "usedfor",
    })
    # Locatedin: a thing in a location inherits aggregate location
    # facts (the continent/country/region of where it sits), not
    # the location's own size/color/use.
    locatedin_transfers: frozenset[str] = frozenset({
        "continent", "country", "region", "city",
    })
    # When a chain doesn't qualify for relation forwarding (first hop is
    # not lifting AND chain isn't a same-relation transitive chain), the
    # legacy behavior was to fall back to a generic `implies` relation
    # and store the fact anyway. That produced semantically vapid
    # inductions like (cat, implies, miniature) via [size -> synonym].
    # Default is now to REJECT such chains; flip this flag to True to
    # recover the legacy behavior (useful for chain-trace recording or
    # research where any walked chain should be remembered).
    allow_generic_implies: bool = False


@dataclass
class InducedFact:
    subject: str
    relation: str
    obj: str
    via: list[tuple[str, str, str]]   # the chain that produced it (s, r, o triples)
    confidence: float
    verified: bool = False
    rejected_reason: Optional[str] = None


def induce_from_chain(kb: ShardedKnowledgeBase,
                      chain_result: ChainResult,
                      start: str,
                      *, policy: InductionPolicy | None = None,
                      skills: SkillLibrary | None = None,
                      provenance: ProvenanceStore | None = None) -> InducedFact | None:
    """Try to induce a single fact from a successful chain.

    The induced fact is `(start, induce_relation, final_answer)` unless
    the chain's last hop already names a more specific relation -- then
    we synthesise the direct shortcut `(start, last_relation, final_answer)`.

    Returns the InducedFact (with `rejected_reason` set if a rule
    blocked it), or None if the chain itself wasn't a candidate.
    """
    cfg = policy or InductionPolicy()
    if chain_result.answer is None:
        return None
    if len(chain_result.trace) < cfg.min_chain_length:
        return None
    if chain_result.confidence < cfg.min_confidence:
        return InducedFact(
            subject=start.lower(),
            relation=cfg.induce_relation,
            obj=chain_result.answer,
            via=[(s, r, o) for s, r, o, _ in chain_result.trace],
            confidence=chain_result.confidence,
            rejected_reason=f"confidence {chain_result.confidence:.3f} < {cfg.min_confidence}",
        )

    # Phantom-shortcut filter: reject chains where consecutive relations
    # form an inverse pair (author -> wrote, partof -> haspart, ...).
    # Such chains are round-trips through a hub and the induced direct
    # fact connecting unrelated siblings is almost always wrong.
    if cfg.filter_inverse_pairs and len(chain_result.trace) >= 2:
        rels = [r for _, r, _, _ in chain_result.trace]
        for i in range(len(rels) - 1):
            inv = inverse_relation(rels[i])
            if inv is not None and inv == rels[i + 1]:
                return InducedFact(
                    subject=start.lower(),
                    relation=cfg.induce_relation,
                    obj=chain_result.answer,
                    via=[(s, r, o) for s, r, o, _ in chain_result.trace],
                    confidence=chain_result.confidence,
                    rejected_reason=(
                        f"phantom shortcut: {rels[i]} -> {rels[i + 1]} "
                        "is an inverse pair (round-trip through hub)"
                    ),
                )
        # Same-relation chains are only meaningful for transitive relations.
        for i in range(len(rels) - 1):
            if rels[i] == rels[i + 1] and rels[i] not in cfg.transitive_relations:
                return InducedFact(
                    subject=start.lower(),
                    relation=cfg.induce_relation,
                    obj=chain_result.answer,
                    via=[(s, r, o) for s, r, o, _ in chain_result.trace],
                    confidence=chain_result.confidence,
                    rejected_reason=(
                        f"non-transitive same-relation chain on {rels[i]!r} "
                        "(likely HRR cleanup artefact)"
                    ),
                )
        # If the final answer equals an intermediate node we already
        # walked through, the chain is a degenerate cycle (HRR cleanup
        # returned the input back to us). Reject.
        intermediates = {start.lower()}
        for s, _, o, _ in chain_result.trace[:-1]:
            intermediates.add(s.lower()); intermediates.add(o.lower())
        if chain_result.answer.lower() in intermediates:
            return InducedFact(
                subject=start.lower(),
                relation=cfg.induce_relation,
                obj=chain_result.answer,
                via=[(s, r, o) for s, r, o, _ in chain_result.trace],
                confidence=chain_result.confidence,
                rejected_reason=(
                    f"answer {chain_result.answer!r} is an intermediate node "
                    "(degenerate cycle, HRR cleanup artefact)"
                ),
            )

    # Skill-uses gate: optional confirmation that this chain shape has
    # been observed multiple times before.
    if cfg.require_skill_uses > 0 and skills is not None:
        pattern = [("O", r) for _, r, _, _ in chain_result.trace]
        skill = skills.find_matching(pattern)
        uses = (skill.success_count if skill else 0)
        if uses < cfg.require_skill_uses:
            return InducedFact(
                subject=start.lower(),
                relation=cfg.induce_relation,
                obj=chain_result.answer,
                via=[(s, r, o) for s, r, o, _ in chain_result.trace],
                confidence=chain_result.confidence,
                rejected_reason=f"skill seen {uses}x < {cfg.require_skill_uses}",
            )

    # Pick the induced relation:
    #   * If first hop's relation is a "lifting" relation (partof,
    #     isa, locatedin, ...), the source inherits the last hop's
    #     relation: (X partof Y, Y madeof Z) => (X madeof Z).
    #   * Same-relation transitive chains keep that relation.
    #   * Otherwise we fall back to the generic `implies` relation so
    #     the surface form doesn't make a false claim.
    rels = [r for _, r, _, _ in chain_result.trace]
    last_relation = rels[-1]
    first_relation = rels[0]

    def _allowed_transfers(first: str) -> frozenset[str]:
        if first in cfg.isa_family:
            return cfg.isa_transfers
        if first == "partof":
            return cfg.partof_transfers
        if first == "locatedin":
            return cfg.locatedin_transfers
        return frozenset()

    if all(r == rels[0] for r in rels) and rels[0] in cfg.transitive_relations:
        relation = rels[0]
    elif first_relation in cfg.lifting_relations:
        allowed = _allowed_transfers(first_relation)
        if last_relation in allowed:
            relation = last_relation
        elif cfg.allow_generic_implies:
            relation = cfg.induce_relation
        else:
            return InducedFact(
                subject=start.lower(),
                relation=cfg.induce_relation,
                obj=chain_result.answer,
                via=[(s, r, o) for s, r, o, _ in chain_result.trace],
                confidence=chain_result.confidence,
                rejected_reason=(
                    f"type-signature mismatch: chain [{' -> '.join(rels)}] "
                    f"forwards {last_relation!r} through first-hop "
                    f"{first_relation!r}, which only permits "
                    f"{sorted(allowed)!r}"
                ),
            )
    elif cfg.allow_generic_implies:
        relation = cfg.induce_relation
    else:
        return InducedFact(
            subject=start.lower(),
            relation=cfg.induce_relation,
            obj=chain_result.answer,
            via=[(s, r, o) for s, r, o, _ in chain_result.trace],
            confidence=chain_result.confidence,
            rejected_reason=(
                f"no meaningful induced relation for chain "
                f"[{' -> '.join(rels)}]: first hop {first_relation!r} is not "
                f"a lifting relation and chain is not same-relation transitive"
            ),
        )
    induced = InducedFact(
        subject=start.lower(),
        relation=relation.lower(),
        obj=chain_result.answer.lower(),
        via=[(s, r, o) for s, r, o, _ in chain_result.trace],
        confidence=chain_result.confidence,
    )

    # Negative-fact filter: if (subject, NOT_relation, obj) is stored
    # explicitly, the user has DENIED this fact -- never induce it.
    denied = denied_pairs_for(kb, induced.subject, induced.relation,
                              min_score=0.10)
    if any(str(sym).lower() == induced.obj for sym, _ in denied):
        induced.rejected_reason = (
            f"denied by stored negative fact "
            f"({induced.subject}, not_{induced.relation}, {induced.obj})"
        )
        return induced

    # Store the new fact and verify it.
    kb.store({"S": induced.subject, "R": induced.relation, "O": induced.obj})
    if provenance is not None:
        provenance.store(
            induced.subject, induced.relation, induced.obj,
            source="induced", confidence=induced.confidence,
            tags={"induced", f"via_{len(induced.via)}_hops"},
            derivation=induced.via,
        )

    if cfg.verify_after:
        verification = _verify_kb(
            kb, induced.subject, induced.relation, induced.obj,
            methods=["roundtrip"],
        )
        induced.verified = bool(verification["verified"])
        if not induced.verified:
            # Roll back: remove the fact we just stored.
            kb.forget({"S": induced.subject, "R": induced.relation, "O": induced.obj})
            if provenance is not None:
                provenance.forget(induced.subject, induced.relation, induced.obj)
            induced.rejected_reason = "self-verification failed"
    return induced


def induce_many(kb: ShardedKnowledgeBase,
                chains: list[tuple[str, ChainResult]],
                *, policy: InductionPolicy | None = None,
                skills: SkillLibrary | None = None,
                provenance: ProvenanceStore | None = None) -> list[InducedFact]:
    """Run induction over a batch of (start, chain_result) pairs."""
    out: list[InducedFact] = []
    for start, result in chains:
        induced = induce_from_chain(
            kb, result, start, policy=policy,
            skills=skills, provenance=provenance,
        )
        if induced is not None:
            out.append(induced)
    return out
