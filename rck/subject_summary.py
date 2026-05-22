"""Summarise everything the agent knows about a subject.

Walks the KB (facts where the subject is the S), provenance store
(per-fact metadata), and the agent's caches/skills/rules touched.
Returns both a structured dict and a rendered text summary.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rck.conscious_agent import ConsciousAgent

from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


@dataclass
class SubjectSummary:
    subject: str
    n_facts: int = 0
    facts: list[tuple[str, str]] = field(default_factory=list)  # (r, o)
    negative_facts: list[tuple[str, str]] = field(default_factory=list)
    parents_via_isa: list[str] = field(default_factory=list)
    siblings_via_isa: list[str] = field(default_factory=list)
    induced_count: int = 0
    sources: dict[str, int] = field(default_factory=dict)
    n_query_hits: int = 0

    def render(self) -> str:
        lines = [f"Summary for {self.subject!r}:"]
        if not self.n_facts and not self.parents_via_isa:
            return lines[0] + " (no facts stored)"
        if self.parents_via_isa:
            lines.append(f"  is a: {', '.join(self.parents_via_isa)}")
        if self.facts:
            sample = ", ".join(f"{r}={o}" for r, o in self.facts[:5])
            lines.append(f"  facts ({self.n_facts}): {sample}"
                          + (" ..." if self.n_facts > 5 else ""))
        if self.negative_facts:
            sample = ", ".join(f"NOT {r}={o}" for r, o in self.negative_facts[:3])
            lines.append(f"  denials: {sample}")
        if self.siblings_via_isa:
            lines.append(f"  siblings: "
                          f"{', '.join(self.siblings_via_isa[:5])}")
        if self.sources:
            lines.append(f"  sources: {dict(self.sources)}")
        if self.induced_count:
            lines.append(f"  induced facts about this subject: {self.induced_count}")
        if self.n_query_hits:
            lines.append(f"  queried {self.n_query_hits}x recently")
        return "\n".join(lines)


def summarize_subject(kb: ShardedKnowledgeBase,
                       provenance: ProvenanceStore | None,
                       subject: str,
                       *,
                       query_memory=None,
                       max_facts: int = 50) -> SubjectSummary:
    subj = subject.lower()
    s = SubjectSummary(subject=subj)
    # Walk shards for facts where S == subject.
    for shard in kb._shards:
        for fact in shard.facts():
            if str(fact.get("S", "")).lower() != subj:
                continue
            r = str(fact.get("R", "")).lower()
            o = str(fact.get("O", "")).lower()
            if r.startswith("not_"):
                s.negative_facts.append((r[len("not_"):], o))
            else:
                s.facts.append((r, o))
                if r == "isa":
                    s.parents_via_isa.append(o)
            if len(s.facts) + len(s.negative_facts) >= max_facts:
                break
    s.n_facts = len(s.facts)

    # Siblings via isa (peers sharing the same parent).
    sibling_set: set[str] = set()
    for parent in s.parents_via_isa:
        peers = kb.query({"R": "isa", "O": parent}, "S", top_k=20)
        for sym, score in peers:
            if float(score) >= 0.10:
                cand = str(sym).lower()
                if cand != subj:
                    sibling_set.add(cand)
    s.siblings_via_isa = sorted(sibling_set)[:10]

    # Provenance source breakdown for subject's facts.
    if provenance is not None:
        for (sx, rx, ox), rec in provenance._records.items():
            if sx == subj:
                s.sources[rec.source] = s.sources.get(rec.source, 0) + 1
                if rec.source in {"induced", "rule"}:
                    s.induced_count += 1

    # Query memory: how often did the user ask about this subject?
    if query_memory is not None:
        for e in query_memory.all():
            if str(e.known.get("S", "")).lower() == subj:
                s.n_query_hits += 1

    return s
