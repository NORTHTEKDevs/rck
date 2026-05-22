"""Temporal reasoning: months, seasons, days of week, before/after queries.

Anchored to a small canonical sequence vocabulary (months, days, seasons).
For richer temporal reasoning we'd need explicit date arithmetic; this
module covers the common-sense cases that come up in dialogue.
"""
from __future__ import annotations

from typing import Optional


MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday"]
SEASONS_NORTHERN = ["winter", "spring", "summer", "autumn"]


def _ring_lookup(seq: list[str], item: str) -> int:
    item = item.lower()
    if item not in seq:
        return -1
    return seq.index(item)


def previous(seq: list[str], item: str) -> Optional[str]:
    idx = _ring_lookup(seq, item)
    if idx < 0:
        return None
    return seq[(idx - 1) % len(seq)]


def following(seq: list[str], item: str) -> Optional[str]:
    idx = _ring_lookup(seq, item)
    if idx < 0:
        return None
    return seq[(idx + 1) % len(seq)]


def temporal_answer(question_or_item: str) -> Optional[dict]:
    """Try to answer a temporal common-sense question."""
    q = question_or_item.lower().strip().rstrip("?")
    # "what comes before X" / "what is before X"
    import re
    for direction, fn in (("before", previous), ("after", following)):
        m = re.match(rf".*\b{direction}\s+(?:the\s+)?(\w+)\s*$", q)
        if m:
            item = m.group(1)
            for seq, kind in ((MONTHS, "month"),
                              (DAYS, "day"),
                              (SEASONS_NORTHERN, "season")):
                if item in seq:
                    answer = fn(seq, item)
                    if answer:
                        return {
                            "answer": answer,
                            "kind": kind,
                            "verbal": f"The {kind} {direction} {item} is {answer}.",
                        }
    return None
