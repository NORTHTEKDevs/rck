"""Abductive reasoning -- given an effect, infer candidate causes.

Forward inference: cause -> effect. RCK's `infer` does this.
Abductive (backward): effect -> ? cause. THIS module.

Examples:
  observed: wetness
  candidates: rain (causes wetness), water_pipe_broke (causes wetness),
              sweat (causes wetness)

  observed: a creature has wings
  candidates: it's a bird, a bat, an insect (all isa parents that have wings)

The algorithm walks the KB in REVERSE:
  1. For an observed (?, R, V): find all S such that (S, R, V).
  2. For an observed property: find all parents/kinds that share it.
  3. Combine with confidence weighting -- many siblings sharing the
     property -> stronger candidacy.

This is what LLMs do badly (forward-trained) and humans do well
(scientific method, medical diagnosis, detective work).
"""
from __future__ import annotations

from dataclasses import dataclass

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class AbductiveCandidate:
    cause: str
    relation: str
    confidence: float
    evidence: list[tuple[str, str, str]]


def candidates_for_effect(
    kb: ShardedKnowledgeBase,
    relation: str, effect_value: str,
    *, top_k: int = 5,
) -> list[AbductiveCandidate]:
    """Find candidate causes (subjects S) such that (S, relation, effect_value).

    Direct backwards lookup. Higher cosine = stronger candidate.
    """
    relation = relation.lower(); effect_value = effect_value.lower()
    results = kb.query(
        {"R": relation, "O": effect_value}, "S", top_k=top_k,
    )
    return [
        AbductiveCandidate(
            cause=str(s),
            relation=relation,
            confidence=float(c),
            evidence=[(str(s), relation, effect_value)],
        )
        for s, c in results if c >= 0.05
    ]


def candidates_for_property(
    kb: ShardedKnowledgeBase,
    property_name: str, property_value: str,
    *, top_k: int = 8,
) -> list[AbductiveCandidate]:
    """Find candidate KINDS / PARENTS that have a property.

    Useful for "I see a creature with feathers -- what could it be?"
    The kinds (isa parents) that have (parent, has, feathers) are
    candidates.

    Combines isa-parent retrieval with property matching.
    """
    property_name = property_name.lower(); property_value = property_value.lower()
    # Step 1: who has (X, property_name, property_value) ?
    direct = candidates_for_effect(kb, property_name, property_value, top_k=top_k)
    # Step 2: any of these direct hits may themselves be kinds (parents).
    # E.g. (bird, has, feathers) -- bird is a kind. So any specific
    # instance of bird (eagle, sparrow) is also a candidate.
    enriched: list[AbductiveCandidate] = list(direct)
    for d in direct:
        # Find children of this kind via reverse isa.
        children = kb.query({"R": "isa", "O": d.cause}, "S", top_k=top_k)
        for child, score in children:
            if score < 0.10:
                continue
            enriched.append(AbductiveCandidate(
                cause=str(child),
                relation=property_name,
                confidence=min(d.confidence, float(score)) * 0.8,
                evidence=d.evidence + [(str(child), "isa", d.cause)],
            ))
    # Dedup by cause keeping highest confidence.
    by_cause: dict[str, AbductiveCandidate] = {}
    for c in enriched:
        if c.cause not in by_cause or by_cause[c.cause].confidence < c.confidence:
            by_cause[c.cause] = c
    result = sorted(by_cause.values(), key=lambda x: -x.confidence)
    return result[:top_k]


def explain(
    kb: ShardedKnowledgeBase,
    observed_value: str,
    *, max_relations: int = 5,
) -> dict:
    """Top-level abductive entry point: given an observed VALUE (e.g.
    'fur'), search across ALL relations to find what could have caused
    or what it could be a property of.

    Returns: {
        "best": <AbductiveCandidate or None>,
        "all": <list[AbductiveCandidate]>,
        "verbal": <natural-language explanation>,
    }
    """
    observed_value = observed_value.lower()
    # Sweep across all relations in the KB.
    relations: set[str] = set()
    for shard in kb._shards:
        for fact in shard._facts:
            relations.add(str(fact.get("R", "")))
    candidates: list[AbductiveCandidate] = []
    for rel in relations:
        candidates.extend(
            candidates_for_property(kb, rel, observed_value, top_k=3)
        )
    candidates.sort(key=lambda c: -c.confidence)
    candidates = candidates[:max_relations]
    if not candidates:
        return {
            "best": None, "all": [],
            "verbal": f"I cannot explain {observed_value!r} from anything in my KB.",
        }
    best = candidates[0]
    if best.confidence >= 0.20:
        verbal = (f"The most likely explanation: it is {best.cause}. "
                  f"(based on {len(best.evidence)} fact(s)).")
    else:
        names = ", ".join(c.cause for c in candidates[:3])
        verbal = (f"Possible explanations: {names}. Confidence is low; "
                  f"need more information to narrow down.")
    return {"best": best, "all": candidates, "verbal": verbal}
