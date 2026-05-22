"""Cascading-induction study: does iterated chain induction yield
new knowledge that a single pass would miss?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rck.bulk_ingest import bulk_load_triples
from rck.cascading_induction import cascade_induct, top_pattern_signatures
from rck.chain_induction import InductionPolicy
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore
from rck.shard_sizing import recommend_shards
from rck.skills import SkillLibrary


def main() -> int:
    print("=" * 70)
    print(" CASCADING INDUCTION STUDY on commonsense KB")
    print("=" * 70)

    triples = []
    with open("data/commonsense_kb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            triples.append((r["s"], r["r"], r["o"]))
    n_shards = recommend_shards(len(triples) * 2, dim=4096).n_shards

    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    bulk_load_triples(kb, triples)
    skills = SkillLibrary()
    prov = ProvenanceStore()
    initial = kb.size()

    print(f"\nInitial KB: {initial} facts in {n_shards} shards")

    policy = InductionPolicy(
        min_confidence=0.15,
        min_chain_length=2,
        verify_after=True,
    )

    t0 = time.perf_counter()
    res = cascade_induct(
        kb, max_rounds=5, probes_per_round=80,
        policy=policy, skills=skills, provenance=prov,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"Saturated: {res.saturated}")
    print(f"\n{'round':>5}  {'probes':>7}  {'chains':>7}  "
          f"{'induced':>8}  {'verified':>9}  {'kb_size':>8}")
    for r in res.rounds:
        print(f"{r.round:>5}  {r.probes_tried:>7}  {r.chains_discovered:>7}  "
              f"{r.facts_induced:>8}  {r.facts_verified:>9}  {r.facts_after:>8}")

    print(f"\nKB growth: {initial} -> {res.final_facts} (+{res.final_facts - initial})")
    print(f"Total verified inductions: {res.total_verified}")

    print(f"\nTop induced-chain patterns:")
    for sig, count in top_pattern_signatures(res, top_k=10):
        print(f"  {count:>3}x  {' -> '.join(sig)}")

    print(f"\nSample induced facts (first 10):")
    for f in res.induced_facts[:10]:
        print(f"  ({f.subject}, {f.relation}, {f.obj})  via {len(f.via)} hops")

    out = {
        "initial_facts": initial,
        "final_facts": res.final_facts,
        "growth": res.final_facts - initial,
        "saturated": res.saturated,
        "elapsed_s": elapsed,
        "rounds": [
            {
                "round": r.round,
                "probes_tried": r.probes_tried,
                "chains_discovered": r.chains_discovered,
                "facts_induced": r.facts_induced,
                "facts_verified": r.facts_verified,
                "facts_after": r.facts_after,
            }
            for r in res.rounds
        ],
        "top_patterns": [
            {"signature": list(sig), "count": count}
            for sig, count in top_pattern_signatures(res, top_k=10)
        ],
    }
    out_path = Path("data/cascade_induction_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
