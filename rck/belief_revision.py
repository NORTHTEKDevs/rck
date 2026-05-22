"""Belief revision: resolve detected contradictions.

`contradiction.detect_conflicts` finds (S, R) pairs that have
multiple high-confidence answers when R is functional. This module
decides which to KEEP and which to FORGET, then applies the decision.

Resolution priority (most important first):
  1. **Provenance source**: user > multi > external > induced > unknown.
     User-provided facts beat induced ones.
  2. **Recency**: more recently reinforced (`last_seen`) wins.
  3. **Reinforce count**: higher count wins (more evidence).
  4. **Confidence**: higher provenance confidence wins.
  5. **HRR score**: tiebreak.

The output is a structured ResolutionPlan -- the action is NOT
applied automatically. Callers (or `agent.resolve_conflicts(apply=True)`)
must explicitly commit it. This keeps belief revision auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rck.contradiction import Conflict
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


SOURCE_PRIORITY: dict[str, int] = {
    "user": 100,
    "multi": 80,
    "external": 60,
    "induced": 40,
    "rule": 30,
    "unknown": 10,
}


@dataclass
class CandidateFact:
    subject: str
    relation: str
    obj: str
    score: float
    source: str = "unknown"
    last_seen: float = 0.0
    count: int = 1
    confidence: float = 1.0

    def priority_tuple(self) -> tuple:
        return (
            SOURCE_PRIORITY.get(self.source, 0),
            self.last_seen,
            self.count,
            self.confidence,
            self.score,
        )


@dataclass
class ResolutionPlan:
    """A decision about which facts in a conflict to keep / drop."""
    conflict: Conflict
    keep: CandidateFact
    drop: list[CandidateFact] = field(default_factory=list)
    rationale: str = ""

    def verbalize(self) -> str:
        ks = self.keep
        return (f"For ({ks.subject}, {ks.relation}, ?) keep {ks.obj!r} "
                f"(source={ks.source}). Drop "
                f"{[f.obj for f in self.drop]}. {self.rationale}")


def _enrich(c: Conflict, prov: ProvenanceStore | None) -> list[CandidateFact]:
    """Turn a Conflict's (object, score) pairs into CandidateFacts with
    provenance metadata."""
    out: list[CandidateFact] = []
    for obj, score in c.candidates:
        if prov is not None:
            rec = prov.get(c.subject, c.relation, obj)
        else:
            rec = None
        cf = CandidateFact(
            subject=c.subject,
            relation=c.relation,
            obj=obj,
            score=float(score),
        )
        if rec is not None:
            cf.source = rec.source
            cf.last_seen = float(rec.last_seen)
            cf.count = int(rec.count)
            cf.confidence = float(rec.confidence)
        out.append(cf)
    return out


def plan_resolution(conflict: Conflict,
                    provenance: ProvenanceStore | None = None
                    ) -> ResolutionPlan:
    """Return a ResolutionPlan choosing which candidate to keep."""
    candidates = _enrich(conflict, provenance)
    if not candidates:
        # Shouldn't happen for a real Conflict, but guard anyway.
        empty = CandidateFact(subject=conflict.subject,
                               relation=conflict.relation,
                               obj="", score=0.0)
        return ResolutionPlan(conflict=conflict, keep=empty,
                              drop=[], rationale="no candidates")
    # Highest priority tuple wins.
    candidates.sort(key=lambda c: c.priority_tuple(), reverse=True)
    keep = candidates[0]
    drop = candidates[1:]
    rationale = _explain_choice(keep, drop)
    return ResolutionPlan(conflict=conflict, keep=keep,
                          drop=drop, rationale=rationale)


def _explain_choice(keep: CandidateFact, drop: list[CandidateFact]) -> str:
    if not drop:
        return "single-candidate resolution (no competitors)"
    other = drop[0]
    if SOURCE_PRIORITY.get(keep.source, 0) > SOURCE_PRIORITY.get(other.source, 0):
        return (f"kept source={keep.source!r} over {other.source!r} "
                f"(provenance priority)")
    if keep.last_seen > other.last_seen:
        return f"kept by recency (last_seen {keep.last_seen:.1f} > {other.last_seen:.1f})"
    if keep.count > other.count:
        return f"kept by reinforcement count ({keep.count} > {other.count})"
    if keep.confidence > other.confidence:
        return (f"kept by provenance confidence "
                f"({keep.confidence:.2f} > {other.confidence:.2f})")
    return f"kept by HRR score ({keep.score:.3f} > {other.score:.3f})"


def apply_resolution(kb: ShardedKnowledgeBase, plan: ResolutionPlan,
                     provenance: ProvenanceStore | None = None) -> int:
    """Forget every dropped fact from the KB (and provenance store).
    Returns the number of facts removed."""
    removed = 0
    for cf in plan.drop:
        kb.forget({"S": cf.subject, "R": cf.relation, "O": cf.obj})
        if provenance is not None:
            provenance.forget(cf.subject, cf.relation, cf.obj)
        removed += 1
    return removed


def resolve_all(kb: ShardedKnowledgeBase,
                conflicts: list[Conflict],
                *, provenance: ProvenanceStore | None = None,
                apply: bool = False) -> list[ResolutionPlan]:
    """Resolve every conflict in `conflicts`. If `apply=True`, drop the
    losing facts from the KB."""
    plans: list[ResolutionPlan] = []
    for c in conflicts:
        plan = plan_resolution(c, provenance=provenance)
        plans.append(plan)
        if apply:
            apply_resolution(kb, plan, provenance=provenance)
    return plans
