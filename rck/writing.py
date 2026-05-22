"""Writing assistance -- draft / edit / rewrite / summarize.

LLMs handle writing tasks by generating from a prompt. RCK handles
them by:
  - DRAFT: compose a structured draft from the KB on the topic.
  - EDIT:  apply a finite set of edit operations (shorten, expand,
           rephrase) to a user-provided text.
  - REWRITE: change tone or style while preserving meaning.
  - SUMMARIZE: extract key facts via Open IE + render with templates.

This is deliberately deterministic at v6. v7's distilled LM polisher
will add the fluency layer. v8+ adds creative writing modes.
"""
from __future__ import annotations

import re

from rck.knowledge_base import ShardedKnowledgeBase
from rck.longform import compose_overview, compose_essay
from rck.open_ie import extract_triples_from_text
from rck.personality import Personality
from rck.tokenizer import sentences


# ---------------------------------------------------------------------------
#  DRAFT
# ---------------------------------------------------------------------------

def draft_about(kb: ShardedKnowledgeBase, topic: str,
                *, length: str = "medium") -> str:
    """Compose a draft on `topic`.

    `length` is one of:
      - short:  1 paragraph (overview only)
      - medium: 3-4 paragraphs (overview + key sections)
      - long:   essay form
    """
    if length == "short":
        return compose_overview(kb, topic, max_sentences_per_section=2)
    if length == "long":
        return compose_essay(kb, topic, max_entities=8)
    return compose_overview(kb, topic, max_sentences_per_section=3)


# ---------------------------------------------------------------------------
#  EDIT
# ---------------------------------------------------------------------------

def edit_shorten(text: str, *, ratio: float = 0.5) -> str:
    """Drop sentences to reduce length by approximately `ratio`."""
    sents = sentences(text)
    keep = max(1, int(len(sents) * ratio))
    return " ".join(sents[:keep])


def edit_expand(text: str, kb: ShardedKnowledgeBase) -> str:
    """For each sentence, extract triples and append related KB facts.

    Adds facts from OTHER relations the subject participates in, not
    just alternative values of the same relation.
    """
    out = []
    expansion_relations = ("color", "size", "has", "locatedin", "country",
                           "field", "isa", "category", "diet", "habitat")
    for sent in sentences(text):
        out.append(sent)
        triples = extract_triples_from_text(sent)
        added = 0
        for s, r, o in triples[:2]:
            for rel in expansion_relations:
                if rel == r:
                    continue  # already in the sentence
                results = kb.query({"S": s, "R": rel}, "O", top_k=1)
                if not results or results[0][1] < 0.10:
                    continue
                sym = str(results[0][0])
                if sym == o:
                    continue
                related = f"the {rel} of {s} is {sym}".replace("_", " ")
                out.append(f"Note also: {related}.")
                added += 1
                if added >= 2:
                    break
            if added >= 2:
                break
    return " ".join(out)


def edit_rephrase(text: str, *, personality: Personality | None = None) -> str:
    """Re-render the text sentence-by-sentence with personality styling.

    Splits on sentence boundaries, applies the personality's `render_know`
    to the first matched "X is Y" claim in each sentence, leaves the rest
    intact. Avoids the ugly substring-rewriting that the regex variant
    produced.
    """
    if personality is None:
        personality = Personality(tone="formal")
    output_sentences: list[str] = []
    for sent in sentences(text):
        m = re.match(r"^\s*(?:the\s+)?([A-Za-z_]+)\s+is\s+([A-Za-z_]+)\s*\.?\s*$",
                     sent, re.IGNORECASE)
        if m:
            subj, val = m.group(1), m.group(2)
            output_sentences.append(personality.render_know(f"{subj} is {val}"))
        else:
            output_sentences.append(sent)
    return " ".join(output_sentences)


# ---------------------------------------------------------------------------
#  SUMMARIZE
# ---------------------------------------------------------------------------

def summarize(text: str, *, max_sentences: int = 5) -> str:
    """Lift a structured summary by running Open IE and rendering."""
    triples = extract_triples_from_text(text)
    if not triples:
        # Fall back to taking the first few sentences.
        return " ".join(sentences(text)[:max_sentences])
    lines = []
    for s, r, o in triples[:max_sentences]:
        lines.append(f"- {s} {r} {o}".replace("_", " "))
    return "Summary:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
#  REWRITE
# ---------------------------------------------------------------------------

def rewrite_for_audience(text: str, *, audience: str = "general") -> str:
    """Change phrasing for different audiences. Tiny ruleset for v6."""
    tone_map = {
        "formal":     Personality(tone="formal"),
        "casual":     Personality(tone="casual"),
        "curious":    Personality(tone="curious"),
        "concise":    Personality(tone="concise"),
        "general":    Personality(tone="default"),
        "technical":  Personality(tone="formal"),
        "child":      Personality(tone="curious"),
    }
    p = tone_map.get(audience, Personality(tone="default"))
    return edit_rephrase(text, personality=p)
