"""HRR capacity profiler -- measure recall as a function of D, shards, facts.

The fundamental scaling questions for RCK's substrate:

  Q1. At fixed D, how does recall degrade as we add facts?
  Q2. At fixed total facts, how does recall improve as we add shards?
  Q3. What's the optimal (D, shards) for a given target capacity?

This module provides a reusable profiler so anyone can re-run these
measurements on their hardware. The experiment results that ship with
v12 are in docs/design/RCK-SESSION-LOG-2026-05-21.md.

Methodology:
  - Generate N synthetic (S, R, O) triples with unique (S, R) keys.
  - Insert them into a ShardedKnowledgeBase at the target (D, shards).
  - Query each stored fact and check whether the top-1 match is the
    expected O with cosine >= confidence threshold.
  - Report: recall@1, recall@3, mean top-1 cosine, false-positive rate.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class CapacityResult:
    n_facts: int
    dim: int
    n_shards: int
    recall_at_1: float
    recall_at_3: float
    mean_top1_cos: float
    median_top1_cos: float
    p10_top1_cos: float
    fill_per_shard_max: int
    fill_per_shard_mean: float
    elapsed_load_s: float
    elapsed_query_s: float


def _synthetic_triples(n: int, seed: int = 0) -> list[tuple[str, str, str]]:
    """Generate n unique (S, R, O) triples that look like real facts."""
    rng = random.Random(seed)
    out: list[tuple[str, str, str]] = []
    relations = ["isa", "color", "has", "locatedin", "size", "diet"]
    for i in range(n):
        s = f"entity_{i}"
        r = rng.choice(relations)
        o = f"value_{i}"
        out.append((s, r, o))
    return out


def profile(n_facts: int, dim: int, n_shards: int,
            seed: int = 0,
            min_conf: float = 0.10) -> CapacityResult:
    """Run one capacity experiment at the given config."""
    kb = ShardedKnowledgeBase(dim=dim, n_shards=n_shards, seed=seed)
    triples = _synthetic_triples(n_facts, seed=seed)

    t0 = time.time()
    for s, r, o in triples:
        kb.store({"S": s, "R": r, "O": o})
    elapsed_load = time.time() - t0

    hits_1 = 0; hits_3 = 0; cos_top1: list[float] = []
    t0 = time.time()
    for s, r, o in triples:
        results = kb.query({"S": s, "R": r}, "O", top_k=3)
        if not results:
            continue
        top1_sym, top1_score = results[0]
        if str(top1_sym) == o and top1_score >= min_conf:
            hits_1 += 1
        top3 = {str(sym) for sym, score in results if score >= min_conf}
        if o in top3:
            hits_3 += 1
        cos_top1.append(float(top1_score))
    elapsed_query = time.time() - t0

    cos_top1.sort()
    mean_cos = sum(cos_top1) / max(1, len(cos_top1))
    median_cos = cos_top1[len(cos_top1) // 2] if cos_top1 else 0.0
    p10 = cos_top1[len(cos_top1) // 10] if cos_top1 else 0.0

    sizes = kb.shard_sizes()
    return CapacityResult(
        n_facts=n_facts, dim=dim, n_shards=n_shards,
        recall_at_1=hits_1 / max(1, len(triples)),
        recall_at_3=hits_3 / max(1, len(triples)),
        mean_top1_cos=mean_cos, median_top1_cos=median_cos,
        p10_top1_cos=p10,
        fill_per_shard_max=max(sizes) if sizes else 0,
        fill_per_shard_mean=sum(sizes) / max(1, len(sizes)),
        elapsed_load_s=elapsed_load,
        elapsed_query_s=elapsed_query,
    )


def sweep(n_facts_list: list[int], dim: int, n_shards: int,
          seed: int = 0, min_conf: float = 0.10) -> list[CapacityResult]:
    """Sweep n_facts at fixed (D, shards)."""
    return [profile(n, dim, n_shards, seed=seed, min_conf=min_conf)
            for n in n_facts_list]


def shard_sweep(n_facts: int, dim: int, shards_list: list[int],
                seed: int = 0, min_conf: float = 0.10) -> list[CapacityResult]:
    """Sweep n_shards at fixed (D, n_facts)."""
    return [profile(n_facts, dim, s, seed=seed, min_conf=min_conf)
            for s in shards_list]


def dim_sweep(n_facts: int, n_shards: int, dim_list: list[int],
              seed: int = 0, min_conf: float = 0.10) -> list[CapacityResult]:
    """Sweep D at fixed (n_facts, shards)."""
    return [profile(n_facts, d, n_shards, seed=seed, min_conf=min_conf)
            for d in dim_list]


def find_capacity(dim: int, n_shards: int,
                  target_recall: float = 0.90,
                  step: int = 500,
                  max_facts: int = 50_000,
                  seed: int = 0) -> dict:
    """Binary-ish search for the largest n_facts at which recall@1
    stays above `target_recall`. Returns the boundary + curve.

    Coarser scan first; we just walk upward until recall drops.
    """
    last_above = 0
    curve: list[CapacityResult] = []
    n = step
    while n <= max_facts:
        r = profile(n, dim, n_shards, seed=seed)
        curve.append(r)
        if r.recall_at_1 < target_recall:
            break
        last_above = n
        n += step
    return {
        "dim": dim, "n_shards": n_shards,
        "target_recall": target_recall,
        "capacity_facts": last_above,
        "boundary_at_facts": n if n <= max_facts else None,
        "curve": curve,
    }
