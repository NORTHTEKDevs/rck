"""Run the full evaluation suite + pretty print."""
from __future__ import annotations

from dataclasses import asdict

from rck.evaluation.accuracy import measure_accuracy
from rck.evaluation.calibration import calibration_table
from rck.evaluation.hallucination import measure_hallucination
from rck.evaluation.latency import measure_latency


def run_full_suite(agent, eval_set: list[tuple[str, object]],
                   *, nonsense_questions: list[str] | None = None,
                   ) -> dict:
    """Run accuracy + calibration + hallucination + latency."""
    acc = measure_accuracy(agent, eval_set)
    cal = calibration_table(agent, eval_set)
    hall = measure_hallucination(agent, nonsense_questions)
    queries_only = [q for q, _ in eval_set]
    lat = measure_latency(agent, queries_only)
    return {
        "accuracy": asdict(acc),
        "calibration": {"brier": cal.brier, "n": cal.n, "bins": cal.bins},
        "hallucination": asdict(hall),
        "latency": asdict(lat),
    }
