"""Rule-based Open Information Extraction.

Converts natural-language sentences into structured (subject, relation,
object) triples without any training. This is the bootstrap path that
lets RCK ingest arbitrary text corpora and grow its own KB.

The extractor is a CONSERVATIVE rule engine: it only emits triples where
the syntactic pattern is unambiguous. False positives corrupt the KB,
so we'd rather miss a fact than invent one. Coverage gaps can be filled
by adding more patterns; precision is the priority.

Supported patterns (case-insensitive):

  X is a Y.                         -> (X, isa, Y)
  X is Y.                           -> (X, is, Y)
  X is the Y of Z.                  -> (Z, Y, X)         e.g. "Paris is the capital of France"
  X has Y.                          -> (X, has, Y)
  X has a Y.                        -> (X, has, Y)
  X wrote Y / X composed Y / ...    -> (X, <verb>, Y)
  X lives in Y / X is located in Y. -> (X, locatedin, Y)
  X is made of Y.                   -> (X, madeof, Y)
  X is used for Y.                  -> (X, usedfor, Y)
  X causes Y.                       -> (X, causes, Y)
  The Y of X is Z.                  -> (X, Y, Z)         e.g. "The capital of France is Paris"
  X is in Y.                        -> (X, locatedin, Y)

The output is a list of (s, r, o) tuples ready to feed into
ShardedKnowledgeBase.store_many or bulk_load_triples.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from rck.tokenizer import sentences as split_sentences


# ---------------------------------------------------------------------------
#  Regex patterns. Each is anchored to a full clause; order MATTERS.
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern[str], callable]] = []


def _add(rx: str, mapper):
    _PATTERNS.append((re.compile(rx, re.IGNORECASE), mapper))


# "The Y of X is Z."  ->  (x, y, z)
_add(
    r"^(?:the\s+)?(\w+)\s+of\s+(?:the\s+)?(\w+)\s+is\s+(\w+)\s*\.?$",
    lambda m: [(m.group(2).lower(), m.group(1).lower(), m.group(3).lower())],
)

# "X is the Y of Z."  ->  (z, y, x)
_add(
    r"^(?:the\s+)?(\w+)\s+is\s+the\s+(\w+)\s+of\s+(\w+)\s*\.?$",
    lambda m: [(m.group(3).lower(), m.group(2).lower(), m.group(1).lower())],
)

# "X is a/an Y."  ->  (x, isa, y)
_add(
    r"^(?:the\s+)?(\w+)\s+is\s+(?:an?\s+)(\w+)\s*\.?$",
    lambda m: [(m.group(1).lower(), "isa", m.group(2).lower())],
)

# "X is in Y." -> (x, locatedin, y)
_add(
    r"^(?:the\s+)?(\w+)\s+is\s+in\s+(?:the\s+)?(\w+)\s*\.?$",
    lambda m: [(m.group(1).lower(), "locatedin", m.group(2).lower())],
)

# "X is located in Y."  -> (x, locatedin, y)
_add(
    r"^(?:the\s+)?(\w+)\s+is\s+located\s+in\s+(?:the\s+)?(\w+)\s*\.?$",
    lambda m: [(m.group(1).lower(), "locatedin", m.group(2).lower())],
)

# "X lives in Y." -> (x, lives_in, y)
_add(
    r"^(\w+)\s+lives\s+in\s+(?:the\s+)?(\w+)\s*\.?$",
    lambda m: [(m.group(1).lower(), "lives_in", m.group(2).lower())],
)

# "X is made of Y."  -> (x, madeof, y)
_add(
    r"^(?:the\s+)?(\w+)\s+is\s+made\s+of\s+(\w+)\s*\.?$",
    lambda m: [(m.group(1).lower(), "madeof", m.group(2).lower())],
)

# "X is used for Y." -> (x, usedfor, y)
_add(
    r"^(?:the\s+)?(\w+)\s+is\s+used\s+for\s+(\w+)\s*\.?$",
    lambda m: [(m.group(1).lower(), "usedfor", m.group(2).lower())],
)

# "X causes Y."  -> (x, causes, y)
_add(
    r"^(?:the\s+)?(\w+)\s+causes\s+(?:the\s+)?(\w+)\s*\.?$",
    lambda m: [(m.group(1).lower(), "causes", m.group(2).lower())],
)

# "X is Y." (bare predicate)  ->  (x, is, y)
_add(
    r"^(?:the\s+)?(\w+)\s+is\s+(\w+)\s*\.?$",
    lambda m: [(m.group(1).lower(), "is", m.group(2).lower())],
)

# "X has a Y." / "X has Y."  ->  (x, has, y)
_add(
    r"^(?:the\s+)?(\w+)\s+has\s+(?:an?\s+)?(\w+)\s*\.?$",
    lambda m: [(m.group(1).lower(), "has", m.group(2).lower())],
)

# Authored: "X wrote Y." / "X composed Y." / "X painted Y." / "X directed Y."
_VERB_PATTERNS = [
    "wrote", "composed", "painted", "directed", "invented",
    "founded", "discovered", "designed",
]
for verb in _VERB_PATTERNS:
    _add(
        rf"^(\w+)\s+{verb}\s+(?:the\s+)?(\w+)\s*\.?$",
        lambda m, v=verb: [(m.group(1).lower(), v, m.group(2).lower())],
    )


# ---------------------------------------------------------------------------
#  Extractor
# ---------------------------------------------------------------------------

def extract_triples_from_sentence(sentence: str) -> list[tuple[str, str, str]]:
    """Apply patterns to one sentence; return all triples produced.

    A sentence may produce multiple triples (e.g. "X is a Y" -> 1 triple,
    but "the capital of france is paris" matches "The Y of X is Z").
    """
    text = sentence.strip().rstrip(".")
    if not text:
        return []
    for pattern, mapper in _PATTERNS:
        m = pattern.match(text)
        if m is not None:
            return mapper(m)
    return []


def extract_triples_from_text(text: str) -> list[tuple[str, str, str]]:
    """Walk every sentence in a text, return the union of extractions."""
    out: list[tuple[str, str, str]] = []
    for sent in split_sentences(text):
        out.extend(extract_triples_from_sentence(sent))
    return out


def extract_triples_from_sentences(sents: Iterable[str]) -> list[tuple[str, str, str]]:
    """Extract from an iterable of pre-split sentences (e.g. a corpus file)."""
    out: list[tuple[str, str, str]] = []
    for s in sents:
        out.extend(extract_triples_from_sentence(s))
    return out
