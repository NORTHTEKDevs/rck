"""Continual self-improvement via user corrections.

When the user says "Actually the sky is blue, not red", RCK should
remove the wrong fact and store the right one. Without this, no AI
truly learns from interaction.

Supported correction patterns (case-insensitive):

  "Actually, X is Y."             -> overwrite (X, is, Y)
  "No, X is Y."                   -> overwrite (X, is, Y)
  "X is actually Y."              -> overwrite (X, is, Y)
  "Actually, X is Y, not Z."      -> store (X, is, Y), forget (X, is, Z)
  "The Y of X is Z, not W."       -> store (X, Y, Z), forget (X, Y, W)

After applying, returns a summary so the dialog layer can confirm.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_CORRECTION_PATTERNS = [
    # "Actually, X is Y, not Z."
    (re.compile(
        r"^(?:actually|no),?\s+(?:the\s+)?(\w+)\s+is\s+(\w+),?\s+not\s+(\w+)\s*\.?$",
        re.IGNORECASE,
    ), "is_not"),
    # "The Y of X is Z, not W."
    (re.compile(
        r"^(?:the\s+)?(\w+)\s+of\s+(\w+)\s+is\s+(\w+),?\s+not\s+(\w+)\s*\.?$",
        re.IGNORECASE,
    ), "attr_of_not"),
    # "Actually, X is Y."   (no "not")
    (re.compile(
        r"^(?:actually|no),?\s+(?:the\s+)?(\w+)\s+is\s+(\w+)\s*\.?$",
        re.IGNORECASE,
    ), "is"),
    # "X is actually Y."
    (re.compile(
        r"^(?:the\s+)?(\w+)\s+is\s+actually\s+(?:an?\s+)?(\w+)\s*\.?$",
        re.IGNORECASE,
    ), "is"),
]


@dataclass
class CorrectionResult:
    matched: bool
    stored: tuple[str, str, str] | None = None
    forgot: tuple[str, str, str] | None = None
    summary: str = ""


def detect_correction(text: str) -> CorrectionResult:
    """Try to parse `text` as a correction. Returns a CorrectionResult."""
    text = text.strip()
    for pattern, kind in _CORRECTION_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        if kind == "is_not":
            entity, new_val, old_val = m.group(1), m.group(2), m.group(3)
            return CorrectionResult(
                matched=True,
                stored=(entity.lower(), "is", new_val.lower()),
                forgot=(entity.lower(), "is", old_val.lower()),
                summary=f"Got it -- updated: {entity} is {new_val} (was {old_val}).",
            )
        if kind == "attr_of_not":
            attr, entity, new_val, old_val = m.groups()
            return CorrectionResult(
                matched=True,
                stored=(entity.lower(), attr.lower(), new_val.lower()),
                forgot=(entity.lower(), attr.lower(), old_val.lower()),
                summary=f"Got it -- updated: the {attr} of {entity} is {new_val} (was {old_val}).",
            )
        if kind == "is":
            entity, val = m.group(1), m.group(2)
            return CorrectionResult(
                matched=True,
                stored=(entity.lower(), "is", val.lower()),
                forgot=None,
                summary=f"Got it -- I'll remember that {entity} is {val}.",
            )
    return CorrectionResult(matched=False)
