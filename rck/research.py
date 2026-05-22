"""Research mode -- gather + synthesize + cite.

Given a topic, RCK:
  1. Identifies the topic entity in the KB.
  2. Pulls all directly-related facts.
  3. For each related ENTITY, pulls a 2-3 fact summary.
  4. Walks one level of `isa` parents/children for context.
  5. Assembles the result as a multi-section research brief with
     provenance citations where available.

The output looks like a Wikipedia article summary or a research memo,
with proper structure and citations. Everything is grounded in
specific triples; the inverted-architecture hallucination invariant
holds.
"""
from __future__ import annotations

from dataclasses import dataclass

from rck.knowledge_base import ShardedKnowledgeBase
from rck.longform import _facts_about, compose_overview
from rck.provenance import ProvenanceStore


@dataclass
class ResearchSection:
    heading: str
    text: str
    facts: list[tuple[str, str, str]]


def _related_entities(kb: ShardedKnowledgeBase, topic: str,
                      max_each: int = 5) -> dict:
    """Find entities related to `topic` via outgoing/incoming relations."""
    topic = topic.lower()
    # Topic's own facts (outgoing).
    outgoing = _facts_about(kb, topic)
    # Entities that have `topic` as their object on any relation (incoming).
    incoming: list[tuple[str, str]] = []
    for shard in kb._shards:
        for fact in shard._facts:
            if str(fact.get("O", "")).lower() == topic:
                incoming.append((str(fact.get("S", "")),
                                str(fact.get("R", ""))))
            if len(incoming) >= max_each * 10:
                break
    # Children via isa (entities whose isa points to topic).
    children_results = kb.query({"R": "isa", "O": topic}, "S", top_k=max_each)
    children = [(str(s), float(c)) for s, c in children_results if c >= 0.10]
    # Parents via isa (entities topic is an instance of).
    parent_results = kb.query({"S": topic, "R": "isa"}, "O", top_k=max_each)
    parents = [(str(s), float(c)) for s, c in parent_results if c >= 0.10]
    return {
        "outgoing": outgoing[:max_each * 4],
        "incoming": incoming[:max_each * 4],
        "children": children,
        "parents": parents,
    }


def research(kb: ShardedKnowledgeBase, topic: str,
             provenance: ProvenanceStore | None = None,
             *, max_sections: int = 5,
             include_citations: bool = True) -> str:
    """Produce a research-style brief on `topic`."""
    topic = topic.lower()
    rel = _related_entities(kb, topic)
    if not rel["outgoing"] and not rel["incoming"]:
        return f"I have no information about {topic.replace('_', ' ')}."

    sections: list[ResearchSection] = []

    # Section 1: Overview.
    overview = compose_overview(kb, topic)
    sections.append(ResearchSection(
        heading="Overview", text=overview,
        facts=[(topic, r, o) for r, o, _ in rel["outgoing"][:5]],
    ))

    # Section 2: Categorisation (what is it, what isa).
    if rel["parents"]:
        lines = [f"{topic.replace('_', ' ')} is classified as: "
                 + ", ".join(p.replace("_", " ") for p, _ in rel["parents"])]
        sections.append(ResearchSection(
            heading="Classification", text=" ".join(lines),
            facts=[(topic, "isa", p) for p, _ in rel["parents"]],
        ))

    # Section 3: Subtypes / instances (children).
    if rel["children"]:
        names = [c.replace("_", " ") for c, _ in rel["children"][:8]]
        text = (f"Known instances or subtypes of {topic.replace('_', ' ')} "
                f"include: {', '.join(names)}.")
        sections.append(ResearchSection(
            heading="Instances", text=text,
            facts=[(c, "isa", topic) for c, _ in rel["children"]],
        ))

    # Section 4: Relations to other entities (incoming).
    if rel["incoming"]:
        # Group by relation type.
        from collections import defaultdict
        by_rel: dict[str, list[str]] = defaultdict(list)
        for s, r in rel["incoming"][:12]:
            by_rel[r].append(s)
        lines = []
        for r, subs in by_rel.items():
            names = [s.replace("_", " ") for s in subs[:5]]
            lines.append(
                f"{topic.replace('_', ' ')} is the {r.replace('_', ' ')} "
                f"of {', '.join(names)}.")
        if lines:
            sections.append(ResearchSection(
                heading="Relations", text=" ".join(lines),
                facts=[(s, r, topic) for s, r in rel["incoming"]],
            ))

    # Section 5: Provenance (citations).
    if include_citations and provenance is not None:
        cited = []
        for sec in sections:
            for s, r, o in sec.facts:
                prov = provenance.get(s, r, o)
                if prov is not None:
                    cited.append((s, r, o, prov.source))
        if cited:
            lines = ["Source provenance for facts cited above:"]
            for s, r, o, src in cited[:12]:
                lines.append(f"  - ({s}, {r}, {o}) -- source: {src}")
            sections.append(ResearchSection(
                heading="Citations", text="\n".join(lines),
                facts=[(s, r, o) for s, r, o, _ in cited],
            ))

    # Render.
    out_lines = [f"# Research brief: {topic.replace('_', ' ').title()}", ""]
    for sec in sections[:max_sections + 1]:  # +1 because Citations is bonus
        out_lines.append(f"## {sec.heading}")
        out_lines.append("")
        out_lines.append(sec.text.replace("_", " "))
        out_lines.append("")
    return "\n".join(out_lines).strip()
