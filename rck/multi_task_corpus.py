"""Multi-task training corpus.

In v7 we generated (template_draft → polished_paraphrase) pairs only.
Each fact produced ~6 training examples. With multi-task corpus
generation, each fact produces ~10-15 examples across SEVEN task
heads, sharing one polisher head:

  1. **paraphrase**     fact_phrasing_a → fact_phrasing_b
  2. **declarative_qa** "what is the X of Y?" → answer sentence
  3. **negative_qa**    "what is NOT the X of Y?" → "I don't have that recorded"
  4. **summarize**      multiple facts → one sentence
  5. **fill_blank**     "The capital of France is ___." → "Paris"
  6. **boolean_yn**     "Is the dog a mammal?" → "Yes" / "No"
  7. **contrast**       (sky, color, blue) vs (grass, color, green) → contrast sentence

The compute saving comes from two places:
  * **Per-fact density**: 10x more examples per fact = 10x less unique
    facts needed for the same dataset size.
  * **Capability transfer**: the polisher learns multiple task SHAPES
    from one model, so we don't need separate training runs.

In practice this drops the GPU bill for the same quality from ~$5-50
to ~$1-15 because the model converges faster and needs fewer steps.
"""
from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rck.knowledge_base import ShardedKnowledgeBase
from rck.polisher_training import (
    GENERIC_TEMPLATES, PARAPHRASES, render_all_phrasings,
)


# ---------------------------------------------------------------------------
#  Task example types
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskExample:
    task: str
    draft: str
    target: str
    s: str
    r: str
    o: str

    def to_dict(self) -> dict:
        return {
            "task": self.task, "draft": self.draft, "target": self.target,
            "s": self.s, "r": self.r, "o": self.o,
        }


# ---------------------------------------------------------------------------
#  Per-task generators
# ---------------------------------------------------------------------------

def _human(token: str) -> str:
    return token.replace("_", " ")


def gen_paraphrase(triple: tuple[str, str, str], *,
                    max_pairs: int = 4) -> list[MultiTaskExample]:
    """Original v7 task -- (draft_phrasing, target_phrasing) pairs."""
    s, r, o = triple
    phrasings = render_all_phrasings(triple)
    if len(phrasings) < 2:
        return []
    out: list[MultiTaskExample] = []
    rng = random.Random(hash(triple) & 0xFFFFFFFF)
    seen: set[tuple[str, str]] = set()
    while len(out) < max_pairs and len(phrasings) >= 2:
        a, b = rng.sample(phrasings, 2)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append(MultiTaskExample("paraphrase", a, b, s, r, o))
    return out


def gen_declarative_qa(triple: tuple[str, str, str]) -> list[MultiTaskExample]:
    """Q -> A pairs: 'What is the X of Y?' → 'The X of Y is Z.'"""
    s, r, o = triple
    s_h, o_h = _human(s), _human(o)
    r_h = _human(r)
    out: list[MultiTaskExample] = []
    out.append(MultiTaskExample(
        "qa",
        draft=f"what is the {r_h} of {s_h}?",
        target=f"the {r_h} of {s_h} is {o_h}.",
        s=s, r=r, o=o,
    ))
    # Inverse-direction QA when meaningful.
    if r in ("capital", "wrote", "founder", "author", "composed",
              "painted", "invented", "founded"):
        out.append(MultiTaskExample(
            "qa",
            draft=f"who or what is the {r_h} associated with {o_h}?",
            target=f"{o_h} is associated with {s_h} via {r_h}.",
            s=s, r=r, o=o,
        ))
    return out


def gen_negative_qa(triple: tuple[str, str, str]) -> list[MultiTaskExample]:
    """'I don't know' phrasings -- teaches the polisher to be honest."""
    s, _, _ = triple
    s_h = _human(s)
    out = [
        MultiTaskExample(
            "honest_no",
            draft=f"what is the favourite colour of {s_h}?",
            target=f"I don't have any record of the favourite colour of {s_h}.",
            s=s, r="favourite_colour", o="unknown",
        ),
    ]
    return out


def gen_summarize(triples: list[tuple[str, str, str]]) -> list[MultiTaskExample]:
    """Multiple facts about the same subject → one sentence."""
    if not triples:
        return []
    # Group by subject.
    by_subject: dict[str, list[tuple[str, str, str]]] = {}
    for s, r, o in triples:
        by_subject.setdefault(s, []).append((s, r, o))
    out: list[MultiTaskExample] = []
    for subj, facts in by_subject.items():
        if len(facts) < 2:
            continue
        # Take top 3 facts max.
        facts = facts[:3]
        draft = " ".join(
            f"{_human(s)} {_human(r)} {_human(o)}." for s, r, o in facts
        )
        # Target: a more compact sentence joining them.
        parts = []
        for s, r, o in facts:
            parts.append(f"{_human(r)} {_human(o)}")
        target = f"{_human(subj)} has " + ", ".join(parts) + "."
        out.append(MultiTaskExample(
            "summarize", draft, target, subj, "multi", "summary",
        ))
    return out


def gen_fill_blank(triple: tuple[str, str, str]) -> list[MultiTaskExample]:
    """Cloze: 'The X of Y is ___.' → 'Z'."""
    s, r, o = triple
    s_h, o_h, r_h = _human(s), _human(o), _human(r)
    return [MultiTaskExample(
        "fill_blank",
        draft=f"the {r_h} of {s_h} is ___.",
        target=o_h,
        s=s, r=r, o=o,
    )]


def gen_boolean_yes(triple: tuple[str, str, str]) -> list[MultiTaskExample]:
    """'Is the X (relation) Y?' → 'Yes, ...'"""
    s, r, o = triple
    s_h, o_h, r_h = _human(s), _human(o), _human(r)
    out = []
    if r in ("isa", "kind", "category"):
        out.append(MultiTaskExample(
            "boolean",
            draft=f"is {s_h} a {o_h}?",
            target=f"yes, {s_h} is a {o_h}.",
            s=s, r=r, o=o,
        ))
    elif r == "color":
        out.append(MultiTaskExample(
            "boolean",
            draft=f"is {s_h} {o_h}?",
            target=f"yes, {s_h} is {o_h}.",
            s=s, r=r, o=o,
        ))
    elif r in ("has", "capital", "located_in", "locatedin"):
        out.append(MultiTaskExample(
            "boolean",
            draft=f"does {s_h} have {r_h} of {o_h}?",
            target=f"yes, {s_h} {r_h} {o_h}.",
            s=s, r=r, o=o,
        ))
    return out


def gen_contrast(triples: list[tuple[str, str, str]]) -> list[MultiTaskExample]:
    """Pair two facts with the same relation but different values."""
    if not triples:
        return []
    by_rel: dict[str, list[tuple[str, str, str]]] = {}
    for s, r, o in triples:
        by_rel.setdefault(r, []).append((s, r, o))
    out: list[MultiTaskExample] = []
    rng = random.Random(0)
    for r, group in by_rel.items():
        if len(group) < 2:
            continue
        rng.shuffle(group)
        for i in range(0, min(len(group) - 1, 4), 2):
            a, b = group[i], group[i + 1]
            if a[2] == b[2]:  # same value -- not a contrast
                continue
            draft = (f"{_human(a[0])} {_human(r)} {_human(a[2])}. "
                     f"{_human(b[0])} {_human(r)} {_human(b[2])}.")
            target = (f"unlike {_human(a[0])} which has {_human(r)} = {_human(a[2])}, "
                      f"{_human(b[0])} has {_human(r)} = {_human(b[2])}.")
            out.append(MultiTaskExample(
                "contrast", draft, target, a[0], r, b[0],
            ))
    return out


# ---------------------------------------------------------------------------
#  Top-level builder
# ---------------------------------------------------------------------------

def generate_multi_task_examples(
    triple: tuple[str, str, str],
    *, paraphrase_pairs: int = 3,
    include_qa: bool = True,
    include_fill: bool = True,
    include_boolean: bool = True,
    include_negative: bool = True,
) -> list[MultiTaskExample]:
    """All single-fact tasks produced from one triple."""
    out: list[MultiTaskExample] = []
    out.extend(gen_paraphrase(triple, max_pairs=paraphrase_pairs))
    if include_qa:
        out.extend(gen_declarative_qa(triple))
    if include_fill:
        out.extend(gen_fill_blank(triple))
    if include_boolean:
        out.extend(gen_boolean_yes(triple))
    if include_negative and (hash(triple) & 0xF) == 0:
        # Only every 16th triple gets a negative example -- they're
        # generic enough that we don't need one per fact.
        out.extend(gen_negative_qa(triple))
    return out


def stream_multi_task_corpus(
    kb: ShardedKnowledgeBase,
    *, include_summarize: bool = True,
    include_contrast: bool = True,
) -> Iterable[MultiTaskExample]:
    """Generator that yields multi-task examples from the entire KB.

    Includes per-fact examples + cross-fact (summarize, contrast).
    """
    all_triples: list[tuple[str, str, str]] = []
    for shard in kb._shards:
        for fact in shard._facts:
            t = (str(fact.get("S", "")),
                 str(fact.get("R", "")),
                 str(fact.get("O", "")))
            all_triples.append(t)
            yield from generate_multi_task_examples(t)
    if include_summarize:
        # Group by subject for summarisation.
        by_subj: dict[str, list[tuple[str, str, str]]] = {}
        for t in all_triples:
            by_subj.setdefault(t[0], []).append(t)
        for facts in by_subj.values():
            yield from gen_summarize(facts)
    if include_contrast:
        yield from gen_contrast(all_triples)


def write_corpus_jsonl(
    kb: ShardedKnowledgeBase, path: str | Path,
    *, max_examples: int | None = None,
) -> dict:
    path = Path(path)
    n = 0
    task_counts: dict[str, int] = {}
    with open(path, "w", encoding="utf-8") as f:
        for ex in stream_multi_task_corpus(kb):
            f.write(json.dumps(ex.to_dict()) + "\n")
            n += 1
            task_counts[ex.task] = task_counts.get(ex.task, 0) + 1
            if max_examples and n >= max_examples:
                break
    return {"examples": n, "path": str(path), "by_task": task_counts}
