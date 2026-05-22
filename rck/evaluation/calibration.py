"""Confidence calibration -- are the model's stated confidences honest?

The Brier score is the standard metric. For each prediction we compare
the stated confidence p in [0, 1] to the ground-truth indicator
(1 if correct, 0 if wrong). Brier = mean((p - y)^2).

  Brier=0 → perfect calibration + perfect accuracy
  Brier=0.25 → uniformly random predictions at p=0.5
  Brier=1 → maximally wrong (confidently incorrect)

We also produce a calibration TABLE: bin predictions by confidence and
compare bin-mean confidence to bin-mean accuracy. A well-calibrated
model has bin-mean confidence ~= bin-mean accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrationResult:
    brier: float
    n: int
    bins: list[dict] = field(default_factory=list)


def _normalise(s) -> str:
    return str(s).lower().strip()


def brier_score(agent, eval_set: list[tuple[str, object]]) -> float:
    """Compute Brier score across the eval set."""
    if not eval_set:
        return 0.0
    sq = 0.0
    for q, expected in eval_set:
        res = agent.ask(q)
        valid_set = (
            {_normalise(x) for x in expected}
            if isinstance(expected, (set, list, tuple))
            else {_normalise(expected)}
        )
        ans = _normalise(res.get("answer"))
        y = 1 if ans in valid_set else 0
        # Take the agent's stated confidence in its top-1 prediction.
        p = float(res.get("confidence", 0.0))
        # Cap p to a probability-shaped range. The agent's cosine scores
        # are typically 0..0.5 even for correct answers, so we rescale:
        # treat any cosine >= 0.20 as ~0.9, any cosine >= 0.30 as ~0.95.
        if p >= 0.30:
            p_scaled = 0.95
        elif p >= 0.20:
            p_scaled = 0.90
        elif p >= 0.10:
            p_scaled = 0.70
        else:
            p_scaled = max(0.0, p * 3.0)  # mild scaling for low cosines
        sq += (p_scaled - y) ** 2
    return sq / len(eval_set)


def calibration_table(agent, eval_set: list[tuple[str, object]],
                      n_bins: int = 5) -> CalibrationResult:
    """Bucketed calibration table."""
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    buckets = [[] for _ in range(n_bins)]
    for q, expected in eval_set:
        res = agent.ask(q)
        valid_set = (
            {_normalise(x) for x in expected}
            if isinstance(expected, (set, list, tuple))
            else {_normalise(expected)}
        )
        ans = _normalise(res.get("answer"))
        y = 1 if ans in valid_set else 0
        p = float(res.get("confidence", 0.0))
        # Same scaling as brier_score for consistency.
        if p >= 0.30:
            p_scaled = 0.95
        elif p >= 0.20:
            p_scaled = 0.90
        elif p >= 0.10:
            p_scaled = 0.70
        else:
            p_scaled = max(0.0, p * 3.0)
        idx = min(n_bins - 1, int(p_scaled * n_bins))
        buckets[idx].append((p_scaled, y))
    bins = []
    sq_sum = 0.0; n = 0
    for k, items in enumerate(buckets):
        if not items:
            bins.append({
                "range": (bin_edges[k], bin_edges[k + 1]),
                "n": 0,
                "mean_p": None,
                "mean_acc": None,
                "gap": None,
            })
            continue
        mean_p = sum(p for p, _ in items) / len(items)
        mean_acc = sum(y for _, y in items) / len(items)
        bins.append({
            "range": (bin_edges[k], bin_edges[k + 1]),
            "n": len(items),
            "mean_p": mean_p,
            "mean_acc": mean_acc,
            "gap": abs(mean_p - mean_acc),
        })
        for p, y in items:
            sq_sum += (p - y) ** 2
            n += 1
    brier = sq_sum / max(1, n)
    return CalibrationResult(brier=brier, n=n, bins=bins)
