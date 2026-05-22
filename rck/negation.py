"""Negation-aware question handling.

"What is NOT a fruit?" / "Which animals do NOT have fur?" / "Is the sky
NOT blue?"

Strategy:
  1. Detect negation in the question (NOT / isn't / aren't / never).
  2. If a value is given (e.g. fruit), enumerate things that ARE fruit,
     then return any item the user might expect (typically restricted to
     entities of the same broad kind).
  3. For boolean negation, flip the boolean result.

Coverage is intentionally narrow; without it negation defaults to the
ordinary query path which would silently mis-handle it.
"""
from __future__ import annotations

import re


_NEG_TOKENS = {"not", "isnt", "isn't", "arent", "aren't", "never", "no"}


def has_negation(question: str) -> bool:
    q = question.lower()
    return bool(re.search(r"\b(not|isn'?t|aren'?t|never|nothing)\b", q))


def strip_negation(question: str) -> str:
    """Remove negation tokens so the underlying parser can handle the rest."""
    return re.sub(r"\b(not|isn'?t|aren'?t|never|nothing)\b\s*", "", question,
                  flags=re.IGNORECASE).strip()


def negate_boolean(question: str) -> tuple[str, bool]:
    """Return (positive_question, was_negated). Caller flips the resulting
    boolean answer."""
    if has_negation(question):
        return strip_negation(question), True
    return question, False
