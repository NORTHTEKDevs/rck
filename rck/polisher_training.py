"""Training infrastructure for the v7 distilled language polisher.

The v4 inverted architecture established that RCK separates KNOWLEDGE
(in HRR) from FLUENCY (in a small LM). v7's deliverable is to actually
train that small LM. This module:

  1. Generates a synthetic training corpus by rendering RCK's templates
     against the existing KB with deliberate phrasing variation.
  2. Provides the TrainingExample data class and a Dataset that loads
     into PyTorch / HuggingFace pipelines.
  3. Stubs a NeuralPolisher class with the polisher interface so the
     architecture works today and only the model needs to be trained
     and loaded.

The actual model training happens via scripts/train_polisher.py (an
external script that imports this module). On a single A100 with a
1B-token synthetic corpus the full v7 training run is ~24 hours of
compute, ~$50-100 USD on rented GPU.

The key insight: training data is FREE because we generate it from
templates. The LM never sees a fact -- it only sees pairs of
(template_draft, polished_paraphrase). The polisher learns to phrase
ANY triple fluently without learning ANY specific triple.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from rck.knowledge_base import ShardedKnowledgeBase
from rck.nlg import render


# ---------------------------------------------------------------------------
#  Synthetic phrasing variations per relation
# ---------------------------------------------------------------------------

# Each entry maps a relation to a LIST of paraphrases. The same fact can be
# rendered in any of these ways. The LM training task is: given any one,
# produce any other.
PARAPHRASES: dict[str, list[str]] = {
    "is": [
        "{s} is {o}.",
        "{s} happens to be {o}.",
        "We can say {s} is {o}.",
        "It is true that {s} is {o}.",
    ],
    "isa": [
        "{s} is a {o}.",
        "{s} is a kind of {o}.",
        "{s} belongs to the {o} category.",
        "The {s} is classified as a {o}.",
    ],
    "color": [
        "{s} is {o} in color.",
        "The {s} has a {o} color.",
        "The color of {s} is {o}.",
        "{s} is coloured {o}.",
    ],
    "has": [
        "{s} has {o}.",
        "{s} possesses {o}.",
        "A {s} is equipped with {o}.",
        "{o} is part of {s}.",
    ],
    "wrote": [
        "{s} wrote {o}.",
        "{o} was written by {s}.",
        "The author of {o} is {s}.",
        "{s} is known for writing {o}.",
    ],
    "capital": [
        "The capital of {s} is {o}.",
        "{o} is the capital of {s}.",
        "{s} has its capital at {o}.",
        "{o} serves as the capital of {s}.",
    ],
    "locatedin": [
        "{s} is in {o}.",
        "{s} is located in {o}.",
        "You can find {s} in {o}.",
        "{s} is found in {o}.",
    ],
    "founder": [
        "{o} founded {s}.",
        "{s} was founded by {o}.",
        "The founder of {s} is {o}.",
        "{o} is the founder of {s}.",
    ],
    "madeof": [
        "{s} is made of {o}.",
        "The {s} is made out of {o}.",
        "{s} consists of {o}.",
        "{s}'s material is {o}.",
    ],
    "usedfor": [
        "{s} is used for {o}.",
        "{s} is used to {o}.",
        "The purpose of {s} is {o}.",
        "{s} serves the purpose of {o}.",
    ],
    "causes": [
        "{s} causes {o}.",
        "{s} leads to {o}.",
        "{o} is caused by {s}.",
        "When there is {s}, {o} follows.",
    ],
}

# Fallback templates for any relation we haven't paraphrased above.
GENERIC_TEMPLATES = [
    "The {r} of {s} is {o}.",
    "{s}'s {r} is {o}.",
    "We know that the {r} of {s} is {o}.",
    "Regarding {s}, its {r} is {o}.",
]


@dataclass
class TrainingExample:
    """One (draft, target) pair for polisher training."""

    draft: str        # canonical template rendering (input)
    target: str       # alternative phrasing (output)
    triple: tuple[str, str, str]  # source triple


def render_all_phrasings(triple: tuple[str, str, str]) -> list[str]:
    """Return every phrasing of a triple we know how to produce."""
    s, r, o = triple
    s_h = s.replace("_", " ")
    o_h = o.replace("_", " ")
    paraphrases = PARAPHRASES.get(r, GENERIC_TEMPLATES)
    out = []
    for tmpl in paraphrases:
        try:
            out.append(tmpl.format(s=s_h, r=r.replace("_", " "), o=o_h))
        except (IndexError, KeyError):
            continue
    return out


def generate_examples_for_triple(
    triple: tuple[str, str, str],
    *, max_pairs: int = 6,
) -> list[TrainingExample]:
    """Create (draft, target) pairs by picking 2 of the N phrasings."""
    phrasings = render_all_phrasings(triple)
    if len(phrasings) < 2:
        return []
    examples: list[TrainingExample] = []
    rng = random.Random(hash(triple) & 0xFFFFFFFF)
    pairs_seen: set[tuple[str, str]] = set()
    while len(examples) < max_pairs and len(phrasings) >= 2:
        a = rng.choice(phrasings)
        b = rng.choice(phrasings)
        if a == b:
            continue
        if (a, b) in pairs_seen:
            continue
        pairs_seen.add((a, b))
        examples.append(TrainingExample(draft=a, target=b, triple=triple))
    return examples


def generate_corpus(
    kb: ShardedKnowledgeBase,
    *, examples_per_triple: int = 3,
    max_triples: int | None = None,
) -> Iterable[TrainingExample]:
    """Stream training examples generated from every fact in the KB."""
    count = 0
    for shard in kb._shards:
        for fact in shard._facts:
            triple = (str(fact.get("S", "")),
                      str(fact.get("R", "")),
                      str(fact.get("O", "")))
            for ex in generate_examples_for_triple(triple, max_pairs=examples_per_triple):
                yield ex
                count += 1
                if max_triples is not None and count >= max_triples:
                    return


def write_corpus_jsonl(
    kb: ShardedKnowledgeBase, path: str | Path,
    *, examples_per_triple: int = 3,
    max_triples: int | None = None,
) -> dict:
    """Materialise the synthetic corpus to disk as JSONL."""
    path = Path(path)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in generate_corpus(kb,
                                   examples_per_triple=examples_per_triple,
                                   max_triples=max_triples):
            f.write(json.dumps({
                "draft": ex.draft, "target": ex.target,
                "s": ex.triple[0], "r": ex.triple[1], "o": ex.triple[2],
            }) + "\n")
            n += 1
    return {"path": str(path), "examples": n}


# ---------------------------------------------------------------------------
#  Neural polisher interface (stub)
# ---------------------------------------------------------------------------

@dataclass
class NeuralPolisher:
    """v7 placeholder for a real trained polisher.

    When `weights_path` is set and the file exists, the polisher loads
    a small transformer (50-100M params) trained on the synthetic corpus.
    Until that's available, the constructor raises a clear error so the
    user knows to either train one or fall back to the RuleBasedPolisher.

    Once trained, the model is loaded once and reused across calls.
    """
    weights_path: str | None = None
    _model: object = field(default=None, init=False)
    _tokenizer: object = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.weights_path is None:
            return
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(
                f"polisher weights not found at {self.weights_path}. "
                f"Train one with scripts/train_polisher.py."
            )
        # Lazy import torch / transformers only when we actually load.
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "NeuralPolisher requires torch + transformers. "
                "pip install torch transformers"
            ) from exc
        # Real loading happens here in v7+.
        raise NotImplementedError(
            "NeuralPolisher.load is the v7 training deliverable. "
            "See scripts/train_polisher.py."
        )

    def polish(self, draft: str, context: dict | None = None) -> str:
        if self._model is None:
            raise RuntimeError(
                "NeuralPolisher is not loaded. Either train weights and "
                "pass weights_path, or use RuleBasedPolisher instead."
            )
        # Real inference path goes here.
        raise NotImplementedError("v7 inference loop not yet implemented")
