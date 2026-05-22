"""Skill-prior speedup study.

Does using a SkillLibrary as a relation-ordering prior speed up
chain discovery on the commonsense KB? We compare:
  * Cold:  discover_chains with no prior.
  * Warm:  discover_chains with a SkillLibrary pre-populated from
           a previous cascade-induction pass.
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
from rck.cascading_induction import cascade_induct
from rck.chain_discover import Goal, discover_chains
from rck.chain_induction import InductionPolicy
from rck.knowledge_base import ShardedKnowledgeBase
from rck.shard_sizing import recommend_shards
from rck.skills import SkillLibrary


def _build_probes(triples, limit=40):
    by_subject = {}
    for s, r, o in triples:
        by_subject.setdefault(s, []).append((r, o))
    probes = []
    for s, edges in by_subject.items():
        direct = {o for _, o in edges}
        for _, mid in edges:
            if mid in by_subject:
                for _, target in by_subject[mid]:
                    if target != s and target not in direct:
                        probes.append((s, target))
                        break
        if len(probes) >= limit:
            break
    return probes


def main() -> int:
    print("=" * 70)
    print(" SKILL-PRIOR SPEEDUP STUDY")
    print("=" * 70)

    triples = []
    with open("data/commonsense_kb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            triples.append((r["s"], r["r"], r["o"]))
    n_shards = recommend_shards(len(triples) * 2, dim=4096).n_shards

    # Two fresh KBs from the same triples (deterministic seeds).
    kb_cold = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    bulk_load_triples(kb_cold, triples)
    kb_warm = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    bulk_load_triples(kb_warm, triples)

    # Pre-populate the warm KB's skill library via cascade induction.
    warm_skills = SkillLibrary()
    print("Pre-warming skills via cascade induction ...")
    res = cascade_induct(
        kb_warm, max_rounds=2, probes_per_round=80,
        policy=InductionPolicy(min_confidence=0.15),
        skills=warm_skills,
    )
    print(f"  Skills collected: {warm_skills.stats()}")

    probes = _build_probes(triples, limit=40)
    print(f"\nBenchmarking {len(probes)} probes (cold vs warm)...")

    def run(kb, skills_prior):
        hits = 0
        total_time = 0.0
        for start, target in probes:
            t0 = time.perf_counter()
            chains = discover_chains(
                kb, start, Goal.symbol(target),
                max_depth=3, beam_width=3, top_n=1,
                min_link_score=0.10, skills_prior=skills_prior,
            )
            total_time += time.perf_counter() - t0
            if chains:
                hits += 1
        return hits, total_time

    print("\n  Cold (no prior) ...")
    hits_cold, t_cold = run(kb_cold, skills_prior=None)
    print("  Warm (with skill prior) ...")
    hits_warm, t_warm = run(kb_warm, skills_prior=warm_skills)

    print(f"\nResults:")
    print(f"  Cold: {hits_cold}/{len(probes)} hits in {t_cold:.2f}s "
          f"({t_cold/len(probes)*1000:.1f}ms avg)")
    print(f"  Warm: {hits_warm}/{len(probes)} hits in {t_warm:.2f}s "
          f"({t_warm/len(probes)*1000:.1f}ms avg)")
    speedup = t_cold / max(1e-6, t_warm)
    print(f"  Speedup: {speedup:.2f}x")

    out = {
        "n_probes": len(probes),
        "hits_cold": hits_cold,
        "hits_warm": hits_warm,
        "time_cold_s": t_cold,
        "time_warm_s": t_warm,
        "speedup": speedup,
        "warm_skill_stats": warm_skills.stats(),
    }
    out_path = Path("data/skill_prior_speedup_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
