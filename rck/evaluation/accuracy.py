"""Factual accuracy: top-K hit rate on a labelled eval set.

Eval set format: list of (question, expected_answer_or_set).
We tolerate any of a set of valid answers (multi-valued relations).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AccuracyResult:
    total: int
    correct_top1: int
    correct_top3: int
    structured: int
    fell_back: int

    @property
    def top1_acc(self) -> float:
        return self.correct_top1 / max(1, self.total)

    @property
    def top3_acc(self) -> float:
        return self.correct_top3 / max(1, self.total)

    @property
    def structured_rate(self) -> float:
        return self.structured / max(1, self.total)


def _normalise(s) -> str:
    return str(s).lower().strip()


def measure_accuracy(agent, eval_set: list[tuple[str, object]]) -> AccuracyResult:
    """Run the agent on every (question, expected) pair; tally metrics."""
    correct_top1 = correct_top3 = structured = fell_back = 0
    for q, expected in eval_set:
        res = agent.ask(q)
        source = str(res.get("source", ""))
        if "structured" in source or "boolean" in source or "multistep" in source:
            structured += 1
        else:
            fell_back += 1

        ans = _normalise(res.get("answer"))
        valid_set = (
            {_normalise(x) for x in expected}
            if isinstance(expected, (set, list, tuple))
            else {_normalise(expected)}
        )

        if ans in valid_set:
            correct_top1 += 1
            correct_top3 += 1
            continue
        # Check candidates if present.
        cand = res.get("candidates", [])
        top3 = {_normalise(s) for s, _ in cand[:3]}
        if valid_set & top3:
            correct_top3 += 1

    return AccuracyResult(
        total=len(eval_set),
        correct_top1=correct_top1,
        correct_top3=correct_top3,
        structured=structured,
        fell_back=fell_back,
    )
