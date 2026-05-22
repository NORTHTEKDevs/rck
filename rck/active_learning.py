"""Active learning -- identify what RCK should ask about next.

Three sources of "ask about this" signal:

  1. Gap detection (already in rck/curiosity.py): siblings share a
     relation that this entity is missing.
  2. Low-confidence facts: facts the agent has stored but with weak
     evidence. Re-asking refreshes them.
  3. Provenance-deprived facts: facts that weren't sourced from a
     reliable origin. Worth corroborating.

Each candidate carries an *expected information gain* (EIG) score so
the agent can prioritise. Returns a ranked list of questions.
"""
from __future__ import annotations

from dataclasses import dataclass

from rck.curiosity import detect_global_gaps
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


@dataclass
class ActiveLearningCandidate:
    question: str
    reason: str
    expected_info_gain: float
    subject: str
    relation: str


def _eig_from_gap(agreement: float, sibling_count: int) -> float:
    """High agreement + many siblings = high information gain."""
    return agreement * (1 - 1 / (sibling_count + 1))


def _eig_from_confidence(confidence: float) -> float:
    """Lower confidence = higher value in refreshing."""
    # Peaks around 0.1-0.2 -- not so weak we'd reject, not so strong
    # that confirmation is redundant.
    return max(0.0, 1.0 - abs(confidence - 0.15) * 4)


def find_low_confidence_facts(provenance: ProvenanceStore,
                              *, low_max: float = 0.4,
                              top_n: int = 20) -> list[ActiveLearningCandidate]:
    """Facts whose stored confidence has dropped into the uncertain zone."""
    out: list[ActiveLearningCandidate] = []
    for (s, r, o), rec in provenance._records.items():
        if rec.confidence > low_max:
            continue
        eig = _eig_from_confidence(rec.confidence)
        out.append(ActiveLearningCandidate(
            question=f"Is it still true that the {r.replace('_', ' ')} of "
                     f"{s.replace('_', ' ')} is {o.replace('_', ' ')}?",
            reason="low-confidence stored fact",
            expected_info_gain=eig,
            subject=s, relation=r,
        ))
    out.sort(key=lambda c: -c.expected_info_gain)
    return out[:top_n]


def find_provenance_gaps(provenance: ProvenanceStore,
                         kb: ShardedKnowledgeBase,
                         top_n: int = 20) -> list[ActiveLearningCandidate]:
    """Facts in the KB that have no provenance record at all."""
    out: list[ActiveLearningCandidate] = []
    for shard in kb._shards:
        for fact in shard._facts:
            s = str(fact.get("S", "")); r = str(fact.get("R", ""))
            o = str(fact.get("O", ""))
            if provenance.get(s, r, o) is None:
                out.append(ActiveLearningCandidate(
                    question=f"Can you confirm: is the {r.replace('_', ' ')} "
                             f"of {s.replace('_', ' ')} equal to "
                             f"{o.replace('_', ' ')}?",
                    reason="no recorded provenance",
                    expected_info_gain=0.5,
                    subject=s, relation=r,
                ))
            if len(out) >= top_n * 2:
                break
        if len(out) >= top_n * 2:
            break
    return out[:top_n]


def find_gap_candidates(kb: ShardedKnowledgeBase,
                        *, sample_size: int = 30,
                        top_n: int = 20) -> list[ActiveLearningCandidate]:
    """Wrap rck.curiosity.detect_global_gaps in the ActiveLearningCandidate API."""
    gaps = detect_global_gaps(kb, sample_size=sample_size,
                                min_agreement=0.4, min_siblings=3)
    out = [
        ActiveLearningCandidate(
            question=g.question,
            reason="sibling-based knowledge gap",
            expected_info_gain=_eig_from_gap(g.agreement, g.sibling_count),
            subject=g.entity, relation=g.relation,
        )
        for g in gaps
    ]
    out.sort(key=lambda c: -c.expected_info_gain)
    return out[:top_n]


def prioritize_questions(
    kb: ShardedKnowledgeBase,
    provenance: ProvenanceStore,
    *,
    n_each: int = 10,
    n_total: int = 15,
) -> list[ActiveLearningCandidate]:
    """Return a deduplicated, EIG-ranked list of things to ask about."""
    candidates: list[ActiveLearningCandidate] = []
    candidates.extend(find_gap_candidates(kb, top_n=n_each))
    candidates.extend(find_low_confidence_facts(provenance, top_n=n_each))
    candidates.extend(find_provenance_gaps(provenance, kb, top_n=n_each))

    # Dedup by (subject, relation).
    seen: set[tuple[str, str]] = set()
    out: list[ActiveLearningCandidate] = []
    for c in sorted(candidates, key=lambda x: -x.expected_info_gain):
        key = (c.subject, c.relation)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= n_total:
            break
    return out
