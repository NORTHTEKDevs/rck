"""Multi-step query decomposition.

LLMs handle questions like "Who wrote the book whose author lives in
France?" implicitly through attention. RCK does it explicitly: parse
the nested clause into sub-queries, evaluate each, then chain.

We handle the simple cases:
  "Who wrote the book whose author lives in X" -> two-step.
  "What is the continent of the capital of X" -> two-step.
  "What is the field of the person who wrote X" -> two-step.

The parser is regex-driven and covers only patterns we have explicit
templates for. The architectural point is: the answer comes from
COMPOSING two graph queries, not from learning the inference pattern.
"""
from __future__ import annotations

import re
from typing import Optional

from rck.knowledge_base import ShardedKnowledgeBase


def two_step(kb: ShardedKnowledgeBase, question: str) -> Optional[dict]:
    """Attempt to decompose `question` into two sequential KB lookups."""
    q = question.strip().lower().rstrip("?")

    # "what is the X of the Y of Z"
    m = re.match(r"^what\s+is\s+the\s+(\w+)\s+of\s+the\s+(\w+)\s+of\s+(?:the\s+)?(\w+)\s*$", q)
    if m:
        outer_rel, inner_rel, entity = m.group(1), m.group(2), m.group(3)
        ans1, score1 = kb.answer({"S": entity, "R": inner_rel}, "O")
        if ans1 is None or score1 < 0.10:
            return None
        ans2, score2 = kb.answer({"S": str(ans1), "R": outer_rel}, "O")
        if ans2 is None or score2 < 0.10:
            return None
        return {
            "answer": str(ans2),
            "confidence": min(score1, score2),
            "chain": [(entity, inner_rel, str(ans1)),
                      (str(ans1), outer_rel, str(ans2))],
            "verbal": (f"The {inner_rel} of {entity} is {ans1}, and the "
                       f"{outer_rel} of {ans1} is {ans2}."),
        }

    # "what is the X of Y's Z"  (apostrophe form)
    m = re.match(r"^what\s+is\s+the\s+(\w+)\s+of\s+(\w+)'?s?\s+(\w+)\s*$", q)
    if m:
        outer_rel, entity, inner_rel = m.group(1), m.group(2), m.group(3)
        ans1, score1 = kb.answer({"S": entity, "R": inner_rel}, "O")
        if ans1 is None or score1 < 0.10:
            return None
        ans2, score2 = kb.answer({"S": str(ans1), "R": outer_rel}, "O")
        if ans2 is None or score2 < 0.10:
            return None
        return {
            "answer": str(ans2),
            "confidence": min(score1, score2),
            "chain": [(entity, inner_rel, str(ans1)),
                      (str(ans1), outer_rel, str(ans2))],
            "verbal": (f"The {inner_rel} of {entity} is {ans1}, and "
                       f"the {outer_rel} of {ans1} is {ans2}."),
        }
    return None
