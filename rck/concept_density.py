"""Concept density map.

Per-subject fact-count histogram for discovering "well-known" vs
"stub" entities in the KB. Helps the user / agent identify:

  * The 10 best-described entities (where to ask deeper questions)
  * The 10 most under-described entities (stubs that need filling)
  * The overall distribution shape (long tail vs balanced)

Output is a `DensityMap` with structured counts and a small set of
top/bottom snippets for human inspection.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class DensityMap:
    total_subjects: int = 0
    total_facts: int = 0
    histogram: dict[int, int] = field(default_factory=dict)
    top_subjects: list[tuple[str, int]] = field(default_factory=list)
    stub_subjects: list[tuple[str, int]] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.total_facts / max(1, self.total_subjects)

    def verbalize(self) -> str:
        lines = [f"Concept density map:"]
        lines.append(f"  subjects: {self.total_subjects}")
        lines.append(f"  facts:    {self.total_facts}")
        lines.append(f"  mean:     {self.mean:.2f} facts/subject")
        if self.top_subjects:
            tops = ", ".join(f"{s}({n})" for s, n in self.top_subjects[:5])
            lines.append(f"  top:      {tops}")
        if self.stub_subjects:
            stubs = ", ".join(f"{s}({n})" for s, n in self.stub_subjects[:5])
            lines.append(f"  stubs:    {stubs}")
        return "\n".join(lines)


def density_map(kb: ShardedKnowledgeBase,
                 *, top_k: int = 10, stub_max: int = 1) -> DensityMap:
    """Compute the density map for the KB.

    Stub threshold: a subject is a "stub" if it has <= `stub_max`
    facts (default 1: only the isa fact, nothing else known).
    """
    counts: Counter = Counter()
    for fact in kb.all_facts():
        s = str(fact.get("S", "")).lower()
        r = str(fact.get("R", "")).lower()
        if r.startswith("not_"):
            continue
        counts[s] += 1

    histogram: dict[int, int] = {}
    for cnt in counts.values():
        histogram[cnt] = histogram.get(cnt, 0) + 1

    sorted_by_count = counts.most_common()
    top = sorted_by_count[:top_k]
    stubs = sorted(
        ((s, c) for s, c in sorted_by_count if c <= stub_max),
        key=lambda x: (x[1], x[0]),
    )[:top_k]
    return DensityMap(
        total_subjects=len(counts),
        total_facts=sum(counts.values()),
        histogram=dict(sorted(histogram.items())),
        top_subjects=top,
        stub_subjects=stubs,
    )
