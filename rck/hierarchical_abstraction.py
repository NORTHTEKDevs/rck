"""Hierarchical fact abstraction.

When many siblings of an isa parent share the SAME (R, O) fact,
the right abstraction is to lift the fact to the parent. E.g. if
dog, cat, horse all have fur and they're all mammals, the right
generalisation is (mammal, has, fur).

This module:
  1. Scans every (R, O) pair the KB knows about.
  2. For each, finds the set of subjects that assert it.
  3. Looks up the isa parent of each subject.
  4. Whenever >= `min_support` subjects with the SAME parent assert
     the same (R, O), proposes the abstracted fact (parent, R, O).
  5. Optionally COMMITS the abstraction:
     - stores (parent, R, O) with provenance source="abstracted"
     - leaves the child facts in place (this is enrichment, not
       replacement -- removing them would be lossy if the user
       later wanted to override per-child)

Why this matters: as the agent grows, redundant per-instance facts
about a class become eligible for promotion to the class itself.
Future inductions and rules can then use the shorter (class, R, O)
edge instead of N per-instance edges.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


@dataclass
class Abstraction:
    parent: str
    relation: str
    obj: str
    sibling_support: int
    siblings: list[str] = field(default_factory=list)
    stored: bool = False
    already_existed: bool = False

    def verbalize(self) -> str:
        return (f"({self.parent}, {self.relation}, {self.obj}) -- "
                f"{self.sibling_support} siblings share this "
                f"({', '.join(self.siblings[:3])}"
                + ("..." if len(self.siblings) > 3 else "")
                + ")")


def find_abstractions(kb: ShardedKnowledgeBase,
                       *,
                       min_support: int = 3,
                       min_link_score: float = 0.10) -> list[Abstraction]:
    """Find (parent, R, O) edges where enough siblings of `parent`
    assert (sibling, R, O)."""
    # 1. Collect all (S, R, O) facts grouped by (R, O).
    by_ro: dict[tuple[str, str], set[str]] = defaultdict(set)
    isa_parents: dict[str, set[str]] = defaultdict(set)
    for fact in kb.all_facts():
        s = str(fact.get("S", "")).lower()
        r = str(fact.get("R", "")).lower()
        o = str(fact.get("O", "")).lower()
        if r.startswith("not_"):
            continue
        if r == "isa":
            isa_parents[s].add(o)
        else:
            by_ro[(r, o)].add(s)

    abstractions: list[Abstraction] = []
    # 2. For each (R, O), group subjects by their isa parents.
    for (r, o), subjects in by_ro.items():
        # parent -> [subjects who share that parent and assert (r, o)]
        parent_subjects: dict[str, list[str]] = defaultdict(list)
        for s in subjects:
            for p in isa_parents.get(s, ()):
                parent_subjects[p].append(s)
        for parent, sibs in parent_subjects.items():
            if len(sibs) < min_support:
                continue
            # Check whether (parent, R, O) is already a direct fact.
            existing = kb.query({"S": parent, "R": r}, "O", top_k=5)
            already = any(
                str(sym).lower() == o and float(sc) >= min_link_score
                for sym, sc in existing
            )
            abstractions.append(Abstraction(
                parent=parent, relation=r, obj=o,
                sibling_support=len(sibs),
                siblings=sorted(sibs),
                already_existed=already,
            ))
    # Sort by support descending.
    abstractions.sort(key=lambda a: -a.sibling_support)
    return abstractions


def commit_abstractions(kb: ShardedKnowledgeBase,
                         abstractions: list[Abstraction],
                         *,
                         provenance: ProvenanceStore | None = None
                         ) -> int:
    """Store every abstraction that isn't already a direct fact.
    Returns the number of newly stored facts."""
    n = 0
    for a in abstractions:
        if a.already_existed:
            continue
        kb.store({"S": a.parent, "R": a.relation, "O": a.obj})
        if provenance is not None:
            provenance.store(
                a.parent, a.relation, a.obj,
                source="abstracted",
                tags={"abstracted", f"from_{a.sibling_support}_siblings"},
                derivation=[(s, a.relation, a.obj) for s in a.siblings[:3]],
            )
        a.stored = True
        n += 1
    return n
