"""RCK evaluation harness.

Implements the standard metrics any ChatGPT-class system should be
measured on. Each metric is a separate module so they can be run in
isolation or composed into a benchmark suite.

Modules:
  * accuracy       -- factual QA top-1, top-3, top-5
  * calibration    -- Brier score: are confidences honest?
  * hallucination  -- rate of confident answers to unknowable questions
  * latency        -- p50 / p95 / p99 query latency
  * coverage       -- how many ground-truth facts can the system actually answer

Public:
    from rck.evaluation import run_full_suite
"""
from rck.evaluation.accuracy import measure_accuracy
from rck.evaluation.calibration import brier_score, calibration_table
from rck.evaluation.hallucination import measure_hallucination
from rck.evaluation.latency import measure_latency
from rck.evaluation.runner import run_full_suite

__all__ = [
    "measure_accuracy",
    "brier_score", "calibration_table",
    "measure_hallucination",
    "measure_latency",
    "run_full_suite",
]
