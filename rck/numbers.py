"""Numerical reasoning -- something LLMs are unreliable at.

RCK reads numeric values stored as strings in the KB ("8849" for
everest's height) and performs deterministic arithmetic + comparisons.
No ML, no token-by-token "thinking" -- just `int()`.
"""
from __future__ import annotations

import re
from typing import Optional

from rck.knowledge_base import ShardedKnowledgeBase


_NUMERIC_WORD = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
}


def parse_number(s: str) -> Optional[float]:
    """Parse a numeric string -- handles digits + a tiny English word set."""
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in _NUMERIC_WORD:
        return float(_NUMERIC_WORD[s])
    # Try direct.
    try:
        return float(s.replace(",", ""))
    except ValueError:
        pass
    # Try compound: "two_hundred", "twenty_five".
    parts = re.split(r"[_\s\-]+", s)
    if all(p in _NUMERIC_WORD for p in parts):
        total = 0
        current = 0
        for p in parts:
            v = _NUMERIC_WORD[p]
            if v >= 100:
                current = max(current, 1) * v
                total += current; current = 0
            else:
                current += v
        return float(total + current)
    return None


def get_numeric_attribute(kb: ShardedKnowledgeBase,
                          subject: str, relation: str) -> Optional[float]:
    """Look up (S, R, O) where O is expected to be numeric. Returns float
    or None."""
    ans, score = kb.answer({"S": subject.lower(), "R": relation.lower()}, "O")
    if ans is None or score < 0.10:
        return None
    return parse_number(str(ans))


def compare_numeric(kb: ShardedKnowledgeBase,
                    subject_a: str, subject_b: str,
                    relation: str = "height") -> dict:
    """Compare two entities along a numeric relation."""
    a = get_numeric_attribute(kb, subject_a, relation)
    b = get_numeric_attribute(kb, subject_b, relation)
    if a is None or b is None:
        return {
            "winner": None,
            "verbal": f"I don't have {relation} data for both.",
            "values": (a, b),
        }
    if a > b:
        return {
            "winner": subject_a,
            "verbal": f"{subject_a} ({a:g}) is greater than {subject_b} ({b:g}) on {relation}.",
            "values": (a, b),
        }
    if b > a:
        return {
            "winner": subject_b,
            "verbal": f"{subject_b} ({b:g}) is greater than {subject_a} ({a:g}) on {relation}.",
            "values": (a, b),
        }
    return {
        "winner": None,
        "verbal": f"{subject_a} and {subject_b} are equal on {relation} ({a:g}).",
        "values": (a, b),
    }


def threshold_query(kb: ShardedKnowledgeBase, subject: str, relation: str,
                    threshold: float, op: str = ">") -> dict:
    """Is X's <relation> > / < / >= / <= threshold?"""
    val = get_numeric_attribute(kb, subject, relation)
    if val is None:
        return {"answer": None, "verbal": f"I don't have {relation} data for {subject}."}
    ops = {">": lambda a, b: a > b, "<": lambda a, b: a < b,
           ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
           "==": lambda a, b: a == b}
    if op not in ops:
        return {"answer": None, "verbal": f"unsupported op {op!r}"}
    result = ops[op](val, threshold)
    rel_word = {">": "greater than", "<": "less than",
                ">=": "at least", "<=": "at most", "==": "exactly"}[op]
    if result:
        return {"answer": True,
                "verbal": f"Yes, {subject}'s {relation} ({val:g}) is {rel_word} {threshold:g}."}
    return {"answer": False,
            "verbal": f"No, {subject}'s {relation} ({val:g}) is not {rel_word} {threshold:g}."}


# ---------------------------------------------------------------------------
#  Pure-arithmetic helper (no KB)
# ---------------------------------------------------------------------------

_ARITH_RE = re.compile(
    r"^\s*(?:what\s+is\s+)?"
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*\??\s*$",
    re.IGNORECASE,
)


def evaluate_arithmetic(question: str) -> Optional[dict]:
    """Parse + evaluate 'what is 5 + 3?' / '12 * 4'."""
    m = _ARITH_RE.match(question)
    if not m:
        return None
    a = float(m.group(1))
    op = m.group(2)
    b = float(m.group(3))
    try:
        if op == "+": v = a + b
        elif op == "-": v = a - b
        elif op == "*": v = a * b
        elif op == "/": v = a / b if b != 0 else float("nan")
        else: return None
    except Exception:
        return None
    # Format nicely (integer if whole).
    if v == int(v):
        v_str = str(int(v))
    else:
        v_str = f"{v:g}"
    return {"answer": v, "verbal": f"{a:g} {op} {b:g} = {v_str}."}
