"""Contradiction detection.

Functional relations -- relations expected to have a SINGLE
value per subject -- are: `capital`, `isa`, `bornin`, `gender`,
`color`, `population`, etc. When the KB has multiple high-scoring
answers for `(S, R, ?)` and R is functional, that's a contradiction.

This module:
  * Maintains a default set of functional relations (FUNCTIONAL).
  * Scans the KB for (S, R) pairs with >1 high-score candidate
    where R is functional.
  * Returns structured `Conflict` objects describing each
    contradiction.
  * Supports per-fact severity scoring (the gap between the two
    competing scores).

Resolution policy is intentionally NOT in this module -- that's
belief-revision policy (likely separate) and depends on
provenance, source authority, recency, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from rck.knowledge_base import ShardedKnowledgeBase
from rck.negative_facts import (
    NEGATION_PREFIX, denegate, denied_pairs_for, is_negative,
)


# Default set of relations where each subject is expected to have
# a single value. Conservative -- err on the side of fewer false
# contradictions.
FUNCTIONAL_RELATIONS: frozenset[str] = frozenset({
    "isa", "capital", "bornin", "gender", "color", "population",
    "currency", "language", "continent", "country", "founded_in",
    "located_in", "locatedin", "birthyear", "deathyear",
    "kind", "category", "type", "father", "mother",
})


@dataclass
class Conflict:
    subject: str
    relation: str
    candidates: list[tuple[str, float]]  # (object, score), sorted desc
    severity: float                      # 0..1, higher = more severe

    def verbalize(self) -> str:
        cand_str = ", ".join(f"{o} ({s:.2f})" for o, s in self.candidates[:3])
        return (f"Conflict for ({self.subject}, {self.relation}, ?): "
                f"{cand_str}  severity={self.severity:.2f}")


@dataclass
class ContradictionPolicy:
    functional_relations: frozenset[str] = FUNCTIONAL_RELATIONS
    min_candidate_score: float = 0.10
    min_severity: float = 0.05
    top_k: int = 5

    def is_functional(self, relation: str) -> bool:
        return relation.lower() in self.functional_relations


def detect_conflicts(kb: ShardedKnowledgeBase,
                     *, subjects: Iterable[str] | None = None,
                     policy: ContradictionPolicy | None = None
                     ) -> list[Conflict]:
    """Scan the KB and return contradictions.

    Two kinds of conflict are surfaced:
      (a) Functional relation has multiple high-confidence answers:
          `(X, R, A)` and `(X, R, B)` both score high when R is in
          FUNCTIONAL_RELATIONS.
      (b) Direct negation conflict:
          `(X, R, Y)` is stored AND `(X, NOT_R, Y)` is also stored
          at non-trivial score.

    If `subjects` is None we walk every stored fact. Otherwise only
    the listed subjects are checked.
    """
    cfg = policy or ContradictionPolicy()
    # Build a list of (S, R) pairs to probe. We track POSITIVE relations
    # only; the negation check handles the not_* counterpart.
    sr_pairs: set[tuple[str, str]] = set()
    if subjects is None:
        for shard in kb._shards:
            for fact in shard.facts():
                s = str(fact.get("S", "")).lower()
                r = str(fact.get("R", "")).lower()
                if is_negative(r):
                    # For a stored (X, not_R, Y) we still need to test
                    # the POSITIVE side, so add (s, denegate(r)).
                    positive = denegate(r)
                    sr_pairs.add((s, positive))
                elif cfg.is_functional(r):
                    sr_pairs.add((s, r))
                else:
                    # Always check for direct negation conflicts even on
                    # non-functional relations.
                    sr_pairs.add((s, r))
    else:
        for s in subjects:
            s_l = str(s).lower()
            for shard in kb._shards:
                for fact in shard.facts():
                    if str(fact.get("S", "")).lower() != s_l:
                        continue
                    r = str(fact.get("R", "")).lower()
                    if is_negative(r):
                        sr_pairs.add((s_l, denegate(r)))
                    else:
                        sr_pairs.add((s_l, r))

    conflicts: list[Conflict] = []
    for s, r in sr_pairs:
        # (a) Functional-relation conflict.
        if cfg.is_functional(r):
            candidates = kb.query({"S": s, "R": r}, "O", top_k=cfg.top_k)
            strong = [(str(sym).lower(), float(score))
                      for sym, score in candidates
                      if float(score) >= cfg.min_candidate_score]
            if len(strong) >= 2:
                top_score = strong[0][1]
                second_score = strong[1][1]
                severity = max(
                    0.0,
                    1.0 - abs(top_score - second_score) / max(top_score, 1e-9),
                )
                if severity >= cfg.min_severity:
                    conflicts.append(Conflict(
                        subject=s, relation=r, candidates=strong,
                        severity=float(severity),
                    ))
        # (b) Direct negation conflict: (s, r, ?) and (s, not_r, ?) share an object.
        positives = kb.query({"S": s, "R": r}, "O", top_k=cfg.top_k)
        positives_strong = {
            str(sym).lower(): float(score)
            for sym, score in positives
            if float(score) >= cfg.min_candidate_score
        }
        if not positives_strong:
            continue
        negatives = denied_pairs_for(kb, s, r,
                                      min_score=cfg.min_candidate_score)
        for neg_sym, neg_score in negatives:
            obj = str(neg_sym).lower()
            if obj in positives_strong:
                pos_score = positives_strong[obj]
                conflicts.append(Conflict(
                    subject=s, relation=r,
                    candidates=[(obj, pos_score), (obj, float(neg_score))],
                    severity=min(1.0, (pos_score + float(neg_score)) / 2.0),
                ))
    conflicts.sort(key=lambda c: -c.severity)
    return conflicts


def report_conflicts(conflicts: list[Conflict]) -> str:
    """Human-readable summary of detected conflicts."""
    if not conflicts:
        return "No conflicts detected."
    lines = [f"Detected {len(conflicts)} conflict(s):"]
    for c in conflicts[:10]:
        lines.append(f"  {c.verbalize()}")
    if len(conflicts) > 10:
        lines.append(f"  ...({len(conflicts) - 10} more)")
    return "\n".join(lines)
