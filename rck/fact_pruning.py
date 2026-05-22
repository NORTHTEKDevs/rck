"""Fact pruning by provenance confidence.

When the KB grows large, low-confidence facts (perhaps induced
under noisy chains long ago) accumulate noise. This module scans
the provenance store, identifies records below a threshold, and
removes them from both the KB and the provenance store.

Policies decide WHICH facts to prune:
  * `low_confidence`: provenance.confidence < threshold
  * `stale`: last_seen older than `max_age_seconds`
  * `low_use`: count < min_count AND source not in user/multi

The combination is conservative -- only facts that meet ALL the
active criteria get dropped. User-asserted facts are excluded by
default unless `prune_user_facts=True`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


@dataclass
class PruningPolicy:
    min_confidence: float = 0.1
    max_age_seconds: float | None = None  # None = no age cap
    min_count: int = 1
    prune_user_facts: bool = False
    prune_negative_facts: bool = False


@dataclass
class PruningReport:
    examined: int = 0
    dropped: int = 0
    by_source: dict[str, int] = None

    def __post_init__(self):
        if self.by_source is None:
            self.by_source = {}


def prune(kb: ShardedKnowledgeBase, provenance: ProvenanceStore,
          *, policy: PruningPolicy | None = None) -> PruningReport:
    """Drop facts that match every active criterion in `policy`."""
    cfg = policy or PruningPolicy()
    now = time.time()
    report = PruningReport()
    keys_to_drop: list[tuple[str, str, str]] = []
    for key, rec in provenance._records.items():
        report.examined += 1
        if rec.source == "user" and not cfg.prune_user_facts:
            continue
        if key[1].startswith("not_") and not cfg.prune_negative_facts:
            continue
        if rec.confidence >= cfg.min_confidence:
            continue
        if rec.count >= cfg.min_count + 1:
            # Highly reinforced facts get a pass even at low confidence.
            continue
        if (cfg.max_age_seconds is not None
                and (now - rec.last_seen) < cfg.max_age_seconds):
            continue
        keys_to_drop.append(key)
    for s, r, o in keys_to_drop:
        kb.forget({"S": s, "R": r, "O": o})
        rec = provenance._records.pop((s, r, o), None)
        if rec is not None:
            report.dropped += 1
            report.by_source[rec.source] = (
                report.by_source.get(rec.source, 0) + 1
            )
    return report
