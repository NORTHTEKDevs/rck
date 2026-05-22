"""Auto-shard sizing -- avoid the capacity cliff at runtime.

Last session's capacity study found a sharp recall cliff when
max-shard-fill > ~80 (at D=4096). This module provides the helper
that picks n_shards from n_facts so the cliff is avoided.

Operational rules of thumb (from the measured data):

  D=4096:  target_max_fill = 80  -> n_shards >= n_facts / 80
  D=8192:  target_max_fill = 160 -> n_shards >= n_facts / 160 (capacity ~2x)

We also clamp n_shards into a sensible band (>=8 because tiny shard
counts waste lookup time; <=4096 because we don't want pathological
shard counts at the high end).
"""
from __future__ import annotations

from dataclasses import dataclass


# Per-D operational constants from the capacity study (data/capacity_study.json).
TARGET_MAX_FILL_BY_DIM = {
    512:   30,
    1024:  45,
    2048:  60,
    4096:  80,
    8192:  160,
    16384: 320,
}


@dataclass
class ShardRecommendation:
    n_shards: int
    target_max_fill: int
    expected_recall_at_1: float
    reasoning: str


def recommend_shards(n_facts: int, dim: int = 4096,
                     safety_factor: float = 1.25,
                     min_shards: int = 8,
                     max_shards: int = 4096) -> ShardRecommendation:
    """Recommend n_shards to keep max-shard-fill under the cliff."""
    target_fill = TARGET_MAX_FILL_BY_DIM.get(
        dim,
        # Interpolate for unusual D: target ~ D/50 by hand-fit.
        max(20, dim // 50),
    )
    # Bake in safety factor (load not perfectly even).
    needed = int((n_facts * safety_factor) / target_fill)
    # Round up to next power of two for clean sharding.
    n_shards = max(min_shards, _next_power_of_two(needed))
    n_shards = min(max_shards, n_shards)
    reasoning = (
        f"At D={dim}, capacity study indicates target_max_fill={target_fill}. "
        f"For {n_facts:,} facts with {safety_factor:.2f}x safety, "
        f"need >={needed} shards. Rounded to power-of-two: {n_shards}."
    )
    # Expected recall is roughly 95%+ when within the band.
    expected_recall = 0.95 if needed <= n_shards else 0.60
    return ShardRecommendation(
        n_shards=n_shards, target_max_fill=target_fill,
        expected_recall_at_1=expected_recall, reasoning=reasoning,
    )


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def auto_shard_for_kb(n_facts: int, dim: int = 4096) -> int:
    """One-line convenience used by ConsciousAgent at construction time."""
    return recommend_shards(n_facts, dim=dim).n_shards
