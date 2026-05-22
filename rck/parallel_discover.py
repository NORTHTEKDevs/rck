"""Parallel batch chain discovery.

Single chain discoveries are too quick (~15ms each) to benefit much
from intra-discovery parallelism. The bigger opportunity is BATCH:
when we have many (start, target) probes to process, they're
independent and can run concurrently.

This module exposes:
  * `batch_discover` -- run `discover_chains` over a list of probes
    using a ThreadPoolExecutor. Returns one DiscoveredChain (or None)
    per input row.

Each probe still hits the shared KB, which is read-only during
discovery. The GIL is released inside numpy ops in HRR cleanup,
so the speedup is bounded by how much wall time discoveries
actually spend in numpy vs Python overhead.

Use this for batch maintenance passes (cascade_induct over many
candidates, mass evaluation of a benchmark, etc).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Optional

from rck.chain_discover import DiscoveredChain, Goal, discover_chains
from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class BatchDiscoveryResult:
    start: str
    target: str
    chain: Optional[DiscoveredChain]


# Empirical cap from v14-parallel-discover study: speedup peaks at
# ~4 workers and degrades by 8. Beyond that the GIL + scheduler
# overhead dominates.
DEFAULT_MAX_WORKERS_CAP: int = 4


def auto_worker_count(n_probes: int, *, cap: int = DEFAULT_MAX_WORKERS_CAP
                       ) -> int:
    """Pick a sensible default worker count.

    Rules:
      * Never more workers than probes (no point launching extras).
      * Never more than `cap` (study showed gains plateau).
      * Use `min(os.cpu_count() or 1, cap)` as the ceiling.
      * Always at least 1.
    """
    cpu = os.cpu_count() or 1
    return max(1, min(n_probes, cap, cpu))


def batch_discover(kb: ShardedKnowledgeBase,
                   probes: Iterable[tuple[str, str]],
                   *, max_depth: int = 3, beam_width: int = 3,
                   top_n: int = 1, min_link_score: float = 0.10,
                   max_workers: int | None = None
                   ) -> list[BatchDiscoveryResult]:
    """Run discover_chains in parallel over a list of (start, target)
    probes. Order of results matches input order.

    `max_workers=None` (default) auto-tunes via `auto_worker_count`
    based on the probe count and the host CPU count.
    """
    probe_list = list(probes)
    if max_workers is None:
        max_workers = auto_worker_count(len(probe_list))
    if not probe_list:
        return []
    results: list[Optional[BatchDiscoveryResult]] = [None] * len(probe_list)

    def _one(i: int, start: str, target: str) -> BatchDiscoveryResult:
        chains = discover_chains(
            kb, start, Goal.symbol(target),
            max_depth=max_depth, beam_width=beam_width,
            top_n=top_n, min_link_score=min_link_score,
        )
        return BatchDiscoveryResult(
            start=start.lower(), target=target.lower(),
            chain=chains[0] if chains else None,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_one, i, s, t): i
            for i, (s, t) in enumerate(probe_list)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()

    return [r for r in results if r is not None]
