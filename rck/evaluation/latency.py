"""Latency benchmarking -- p50 / p95 / p99 per query."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class LatencyResult:
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float


def measure_latency(agent, queries: list[str]) -> LatencyResult:
    durations: list[float] = []
    for q in queries:
        t0 = time.time()
        agent.ask(q)
        durations.append((time.time() - t0) * 1000.0)
    durations.sort()
    n = len(durations)
    def _pct(p):
        idx = min(n - 1, int(p * n))
        return durations[idx]
    return LatencyResult(
        n=n,
        p50_ms=_pct(0.50),
        p95_ms=_pct(0.95),
        p99_ms=_pct(0.99),
        max_ms=durations[-1] if durations else 0.0,
        mean_ms=sum(durations) / max(1, n),
    )
