"""Rank subjects by importance.

Importance is a weighted score combining:
  * fact_count -- how many facts the subject is the SUBJECT of
  * object_count -- how many facts the subject appears as OBJECT
  * query_count -- how many times the subject appeared in query_memory
  * derivation_count -- how many derived facts cite the subject in
    their derivation chain

Higher importance = the agent has invested more in / been asked
more about this entity. Useful for surface UIs (show top entities),
maintenance (prioritise relations for these subjects in gap_detection),
and culling (low-importance subjects with few facts may be stale).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore
from rck.query_memory import QueryMemory


@dataclass
class ImportanceScore:
    subject: str
    fact_count: int = 0
    object_count: int = 0
    query_count: int = 0
    derivation_count: int = 0

    def score(self, *,
              w_fact: float = 1.0,
              w_object: float = 0.5,
              w_query: float = 2.0,
              w_derivation: float = 1.0) -> float:
        return (self.fact_count * w_fact
                + self.object_count * w_object
                + self.query_count * w_query
                + self.derivation_count * w_derivation)


def rank_subjects(kb: ShardedKnowledgeBase,
                  *,
                  provenance: ProvenanceStore | None = None,
                  query_memory: QueryMemory | None = None,
                  top_k: int = 20) -> list[ImportanceScore]:
    """Return the top-K subjects by composite importance."""
    fact_counts: defaultdict[str, int] = defaultdict(int)
    object_counts: defaultdict[str, int] = defaultdict(int)
    derivation_counts: defaultdict[str, int] = defaultdict(int)

    for shard in kb._shards:
        for fact in shard.facts():
            s = str(fact.get("S", "")).lower()
            o = str(fact.get("O", "")).lower()
            if s:
                fact_counts[s] += 1
            if o:
                object_counts[o] += 1

    if provenance is not None:
        for _key, rec in provenance._records.items():
            for s_d, _r_d, o_d in rec.derivation:
                derivation_counts[str(s_d).lower()] += 1
                derivation_counts[str(o_d).lower()] += 1

    query_counts: defaultdict[str, int] = defaultdict(int)
    if query_memory is not None:
        for e in query_memory.all():
            s = str(e.known.get("S", "")).lower()
            if s:
                query_counts[s] += 1

    all_subjects = (
        set(fact_counts) | set(object_counts)
        | set(query_counts) | set(derivation_counts)
    )
    scores = [
        ImportanceScore(
            subject=s,
            fact_count=fact_counts[s],
            object_count=object_counts[s],
            query_count=query_counts[s],
            derivation_count=derivation_counts[s],
        )
        for s in all_subjects if s
    ]
    scores.sort(key=lambda x: -x.score())
    return scores[:top_k]
