"""Analogical reasoning over the sharded KB.

The classic question form is "A : B :: C : ?", e.g.:
    france : paris :: germany : ?      => berlin
    dog    : mammal :: eagle  : ?      => bird

In RCK we don't try to solve this via free HRR algebra (atoms are
random bipolar HVs without semantic prior). Instead we use the
relational memory directly:

1. Find the relation R such that (A, R, B) holds.
   This requires enumerating known relations and seeing which one
   returns B given the subject A. The strongest match wins.
2. Apply that R to C: query (C, R, ?). The cleanup result is the
   analog answer.

This is structural analogy at the KB level. It works whenever the
KB contains BOTH the source pair (A, R, B) and the target pair
(C, R, ?). For pairs where (C, R, ?) is implied by chain reasoning
but not direct, we can fall back to chain_walker.

The module exposes:
  * `find_relation(kb, A, B, ...)` -- name the relation between
    two entities.
  * `solve_analogy(kb, A, B, C, ...)` -- end-to-end analogy solving.
  * `AnalogyResult` -- structured output with R, the answer, the
    forward + reverse confidence, and a verbal form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from rck.chain_discover import Goal, discover_chains
from rck.chain_walker import Hop, walk_chain
from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class RelationCandidate:
    relation: str
    score: float


@dataclass
class AnalogyResult:
    a: str
    b: str
    c: str
    relation: Optional[str]
    relation_score: float
    answer: Optional[str]
    answer_score: float
    # Confidence-weighted alternatives: each row is
    # (relation, relation_score, answer, answer_score, joint_score)
    # where joint = relation_score * answer_score. Higher = stronger.
    alternatives: list[tuple] = None
    # For chain-fallback analogies: the compound relation as a sequence.
    chain: Optional[list[str]] = None
    via: str = "direct"  # "direct" or "chain_fallback"

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []

    def joint_score(self) -> float:
        return float(self.relation_score) * float(self.answer_score)

    def verbalize(self) -> str:
        if self.answer is None or self.relation is None:
            return f"I don't know how to relate {self.a!r} to {self.b!r}."
        return (f"{self.a} is to {self.b} as {self.c} is to {self.answer} "
                f"(relation: {self.relation}).")


def _enumerate_relations(kb: ShardedKnowledgeBase) -> list[str]:
    """Collect every R that has been stored in the KB."""
    out: set[str] = set()
    for fact in kb.all_facts():
        r = fact.get("R")
        if r is not None:
            out.add(str(r))
    return sorted(out)


def find_relation(kb: ShardedKnowledgeBase,
                  a: str, b: str,
                  *, relations: Iterable[str] | None = None,
                  min_score: float = 0.10,
                  top_k: int = 3) -> list[RelationCandidate]:
    """Find the relation R such that (a, R, b) holds.

    For each candidate R, query (a, R, ?) and check how close the
    top result is to `b`. We return up to `top_k` candidates sorted
    by score.
    """
    a_l = a.lower()
    b_l = b.lower()
    if relations is None:
        relations = _enumerate_relations(kb)
    results: list[RelationCandidate] = []
    for r in relations:
        candidates = kb.query({"S": a_l, "R": r.lower()}, "O", top_k=top_k)
        for sym, score in candidates:
            if str(sym).lower() == b_l and score >= min_score:
                results.append(RelationCandidate(
                    relation=r.lower(), score=float(score),
                ))
                break
    results.sort(key=lambda x: -x.score)
    return results[:top_k]


def solve_analogy(kb: ShardedKnowledgeBase,
                  a: str, b: str, c: str,
                  *, relations: Iterable[str] | None = None,
                  min_relation_score: float = 0.10,
                  top_k_relations: int = 3,
                  chain_fallback: bool = True,
                  chain_max_depth: int = 3,
                  scoring: str = "bayesian",
                  temperature: float = 4.0) -> AnalogyResult:
    """Solve `a : b :: c : ?` with confidence-weighted alternatives.

    Steps:
      1. Find up to `top_k_relations` candidate relations R linking A->B.
      2. For each, apply to C: query (c, R, ?). Score = R_score * answer_score.
      3. Pick the best (R, answer) pair. The selection rule depends on
         `scoring`:
           - "bayesian": softmax over joint scores; the chosen answer
             carries a calibrated probability between 0 and 1, and
             ties between similar candidates degrade gracefully.
           - "product": legacy argmax-of-product behaviour.
      4. Surface all candidates as `alternatives` for inspection.

    `temperature` controls softmax sharpness when `scoring="bayesian"`:
    higher = sharper (more peaked); lower = flatter (more uniform).
    """
    rel_candidates = find_relation(
        kb, a, b, relations=relations,
        min_score=min_relation_score, top_k=top_k_relations,
    )
    if not rel_candidates:
        # Chain fallback: A and B might be connected via a 2+ hop
        # chain rather than a single relation. Discover the chain,
        # then apply it to C with walk_chain.
        if chain_fallback:
            chains = discover_chains(
                kb, a, Goal.symbol(b), max_depth=chain_max_depth,
                beam_width=3, top_n=1,
                min_link_score=min_relation_score,
            )
            if chains:
                ch = chains[0]
                hops = [Hop(r, d) for r, d in zip(ch.relations, ch.directions)]
                # Apply the same chain shape starting at C.
                walk = walk_chain(kb, c, hops)
                if walk.answer is not None:
                    return AnalogyResult(
                        a=a.lower(), b=b.lower(), c=c.lower(),
                        relation=" -> ".join(ch.relations),
                        relation_score=float(ch.confidence),
                        answer=walk.answer,
                        answer_score=float(walk.confidence),
                        alternatives=[],
                        chain=list(ch.relations),
                        via="chain_fallback",
                    )
        return AnalogyResult(
            a=a.lower(), b=b.lower(), c=c.lower(),
            relation=None, relation_score=0.0,
            answer=None, answer_score=0.0,
            alternatives=[],
        )

    alternatives: list[tuple] = []
    for rc in rel_candidates:
        ans, ans_score = kb.answer({"S": c.lower(), "R": rc.relation}, "O")
        if ans is None:
            continue
        joint = float(rc.score) * float(ans_score)
        alternatives.append((
            rc.relation, float(rc.score), str(ans), float(ans_score), joint,
        ))

    if not alternatives:
        # We had relation candidates but none applied to C.
        chosen = rel_candidates[0]
        return AnalogyResult(
            a=a.lower(), b=b.lower(), c=c.lower(),
            relation=chosen.relation, relation_score=chosen.score,
            answer=None, answer_score=0.0,
            alternatives=[],
        )

    if scoring == "bayesian":
        # Softmax over joint scores -> calibrated probabilities.
        import math
        joints = [a[4] for a in alternatives]
        scaled = [j * temperature for j in joints]
        m = max(scaled) if scaled else 0.0
        exps = [math.exp(s - m) for s in scaled]
        z = sum(exps) or 1.0
        probs = [e / z for e in exps]
        alternatives = [
            (a[0], a[1], a[2], a[3], probs[i])
            for i, a in enumerate(alternatives)
        ]
    alternatives.sort(key=lambda x: -x[4])  # by joint or probability
    best = alternatives[0]
    return AnalogyResult(
        a=a.lower(), b=b.lower(), c=c.lower(),
        relation=best[0], relation_score=best[1],
        answer=best[2], answer_score=best[3],
        alternatives=alternatives,
    )
