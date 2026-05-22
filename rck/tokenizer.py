"""Word-level tokenization.

The minimum that lets RCK move from char-level fragments to word-level
sequences. We split on whitespace, keep punctuation as separate tokens,
and lowercase letters. Round-trips cleanly back to readable English.

Why not BPE? BPE requires training a vocabulary on a large corpus. At
RCK's CPU scale, word-level is the right tradeoff: a vocabulary of
a few thousand atoms covers most general text and each token carries
much more meaning than a single character.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
_PUNCT_GLUE = set(".,;:!?")
_OPEN_QUOTE = {"(", "[", "{", '"', "'"}
_CLOSE_QUOTE = {")", "]", "}"}


def tokenize(text: str, lower: bool = True) -> list[str]:
    """Split text into a list of word + punctuation tokens."""
    if lower:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    """Recombine tokens into a human-readable string."""
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
            continue
        if tok in _PUNCT_GLUE:
            out.append(tok)
        elif tok in _CLOSE_QUOTE:
            out.append(tok)
        elif tokens[i - 1] in _OPEN_QUOTE:
            out.append(tok)
        else:
            out.append(" " + tok)
    return "".join(out)


def sentences(text: str) -> list[str]:
    """Crude sentence splitter for QA-style ingestion."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]
