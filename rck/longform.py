"""Long-form, multi-paragraph response composer.

LLMs produce flowing prose by next-token prediction. RCK produces it by
composing many retrieved facts into a structured document with topic
sentences, transitions, and citations.

The composer is template-driven (deterministic) for v6. v7 will add the
distilled LM polisher (50-100M params) trained to take this template
draft and produce more natural prose. For now, the template draft is
already coherent enough to read like a Wikipedia summary or research
brief.

Public API:
  compose_overview(kb, entity) -> str
    Produces a 2-5 paragraph overview of `entity` using all the facts
    we know about it, grouped by topic.
  compose_essay(kb, topic, max_paragraphs=5) -> str
    Produces a longer essay walking through related entities.
  compose_comparison(kb, a, b) -> str
    Compare two entities along their shared relations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rck.knowledge_base import ShardedKnowledgeBase
from rck.nlg import render


# Relations grouped into thematic sections so paragraphs have coherent topics.
SECTIONS: dict[str, list[str]] = {
    "Identity": ["isa", "kind", "category", "role"],
    "Location": ["locatedin", "lives_in", "continent", "capital",
                 "capital_of", "country", "near", "habitat", "origin"],
    "Properties": ["color", "size", "height_m", "length_km", "atomic_number",
                   "symbol", "category", "kind", "state_at_room_temperature",
                   "group", "period", "diet", "lifespan"],
    "Composition": ["madeof", "haspart", "partof", "has", "contains"],
    "Origin and history": ["founder", "founded", "founded_year",
                           "wrote", "author", "composed", "painted",
                           "invented", "directed", "century",
                           "origin_country", "country"],
    "Activity": ["usedfor", "causes", "field", "works_at", "spoken_in",
                 "language", "currency", "population_tier"],
    "Beliefs and culture": ["sacred_text", "religion", "cuisine"],
}


def _facts_about(kb: ShardedKnowledgeBase, entity: str, min_conf: float = 0.10):
    """Return all (relation, object, confidence) for the entity."""
    facts: list[tuple[str, str, float]] = []
    entity = entity.lower()
    for shard in kb._shards:
        for fact in shard._facts:
            if str(fact.get("S", "")).lower() != entity:
                continue
            r = str(fact.get("R", "")); o = str(fact.get("O", ""))
            facts.append((r, o, 1.0))
    return facts


def _group_by_section(facts: list[tuple[str, str, float]]) -> dict[str, list[tuple[str, str, float]]]:
    """Bucket facts into thematic sections."""
    by_section: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    relation_to_section: dict[str, str] = {}
    for section, rels in SECTIONS.items():
        for r in rels:
            relation_to_section[r] = section
    misc = "Other"
    for r, o, c in facts:
        section = relation_to_section.get(r, misc)
        by_section[section].append((r, o, c))
    return dict(by_section)


def _humanize(token: str) -> str:
    return token.replace("_", " ")


def compose_overview(kb: ShardedKnowledgeBase, entity: str,
                     *, max_sentences_per_section: int = 3) -> str:
    """A 2-5 paragraph overview of `entity` grouped by topic."""
    facts = _facts_about(kb, entity)
    if not facts:
        return f"I don't have any information about {_humanize(entity)} in my knowledge base."

    sections = _group_by_section(facts)
    paragraphs: list[str] = []

    # Lead paragraph: identity / what is X
    intro_bits: list[str] = []
    for r, o, _ in sections.get("Identity", [])[:max_sentences_per_section]:
        intro_bits.append(render(entity, r, o))
    if intro_bits:
        paragraphs.append(" ".join(intro_bits))

    # Order the rest by section.
    for sec_name in ("Location", "Properties", "Composition",
                     "Origin and history", "Activity", "Beliefs and culture",
                     "Other"):
        items = sections.get(sec_name, [])
        if not items:
            continue
        body = []
        for r, o, _ in items[:max_sentences_per_section]:
            body.append(render(entity, r, o))
        if body:
            paragraphs.append(" ".join(body))

    # Join with double newlines for paragraph break.
    text = "\n\n".join(paragraphs)
    # Light surface cleanup.
    text = text.replace("_", " ")
    return text


@dataclass
class EssaySection:
    heading: str
    entities: list[str]
    text: str


def compose_essay(kb: ShardedKnowledgeBase, topic: str,
                  *, max_entities: int = 6,
                  max_sentences_per_entity: int = 2) -> str:
    """Compose an essay-style multi-section response.

    Strategy: treat `topic` as the IDENTITY of a category. Enumerate
    entities that are `isa topic`. For each, generate a 1-2 sentence
    description.
    """
    topic = topic.lower()
    # Find children of this topic.
    children = kb.query({"R": "isa", "O": topic}, "S", top_k=max_entities)
    children = [(str(s), float(c)) for s, c in children if c >= 0.10]

    if not children:
        # Fallback: just compose an overview of the topic itself.
        return compose_overview(kb, topic)

    sections: list[EssaySection] = []
    # Intro.
    intro = (f"# {_humanize(topic).capitalize()}\n\n"
             f"The following are notable {_humanize(topic)} entries from "
             f"my knowledge base.")
    sections.append(EssaySection(
        heading=topic, entities=[], text=intro,
    ))
    for ent, score in children:
        body_facts = _facts_about(kb, ent)[:max_sentences_per_entity * 2]
        sentences = [render(ent, r, o) for r, o, _ in body_facts]
        if not sentences:
            continue
        block = (f"## {_humanize(ent).capitalize()}\n\n"
                 + " ".join(sentences[:max_sentences_per_entity * 2]))
        sections.append(EssaySection(
            heading=ent, entities=[ent], text=block.replace("_", " "),
        ))

    return "\n\n".join(s.text for s in sections)


def compose_comparison(kb: ShardedKnowledgeBase, a: str, b: str) -> str:
    """Compare two entities side-by-side along their shared relations."""
    a, b = a.lower(), b.lower()
    facts_a = {r: o for r, o, _ in _facts_about(kb, a)}
    facts_b = {r: o for r, o, _ in _facts_about(kb, b)}
    shared = sorted(set(facts_a) & set(facts_b))
    if not shared:
        return (f"I have facts about both {_humanize(a)} and {_humanize(b)}, "
                f"but they share no common attributes in my KB.")
    lines = [f"# Comparison: {_humanize(a).capitalize()} vs {_humanize(b).capitalize()}",
             "",
             f"| Attribute | {_humanize(a).capitalize()} | {_humanize(b).capitalize()} |",
             "|---|---|---|"]
    for rel in shared[:15]:
        va, vb = _humanize(facts_a[rel]), _humanize(facts_b[rel])
        lines.append(f"| {rel.replace('_', ' ')} | {va} | {vb} |")
    return "\n".join(lines).replace("_", " ")
