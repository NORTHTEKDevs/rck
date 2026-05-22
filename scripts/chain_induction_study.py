"""Empirical chain-induction study on the commonsense KB.

For each (start, target) pair where chain discovery succeeds, we
induce the direct shortcut fact. We measure:
  * how many shortcut facts are created
  * what fraction pass self-verification
  * how much faster repeat queries become
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
from rck.chain_discover import Goal, discover_chains
from rck.chain_induction import InductionPolicy, induce_from_chain
from rck.chain_walker import Hop, walk_chain
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore
from rck.shard_sizing import recommend_shards


def main() -> int:
    print("=" * 70)
    print(" CHAIN INDUCTION STUDY on commonsense KB")
    print("=" * 70)

    triples = []
    with open("data/commonsense_kb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            triples.append((r["s"], r["r"], r["o"]))

    n_shards = recommend_shards(len(triples) * 2, dim=4096).n_shards
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    bulk_load_triples(kb, triples)
    prov = ProvenanceStore()
    pre_facts = kb.size()
    print(f"\nLoaded {pre_facts} facts into {n_shards} shards")

    # Probe pairs (2-hop transitive).
    by_subject: dict[str, list[tuple[str, str]]] = {}
    for s, r, o in triples:
        by_subject.setdefault(s, []).append((r, o))
    probes: list[tuple[str, str]] = []
    for s, edges in by_subject.items():
        for _, mid in edges:
            if mid in by_subject:
                for _, target in by_subject[mid]:
                    if target != s and target not in {o for _, o in edges}:
                        probes.append((s, target))
                        break
        if len(probes) >= 80:
            break

    policy = InductionPolicy(
        min_confidence=0.15,
        min_chain_length=2,
        verify_after=True,
    )

    induced_total = 0
    verified = 0
    rejected = 0
    new_relation_pairs = set()
    induced_pairs: list[tuple[str, str, str]] = []

    t0 = time.perf_counter()
    for start, target in probes:
        chains = discover_chains(
            kb, start, Goal.symbol(target), max_depth=3,
            beam_width=3, top_n=1, min_link_score=0.10,
        )
        if not chains:
            continue
        ch = chains[0]
        hops = [Hop(r, d) for r, d in zip(ch.relations, ch.directions)]
        # Re-walk to get the full ChainResult with confidence propagation.
        result = walk_chain(kb, start, hops)
        if result.answer is None:
            continue
        induced = induce_from_chain(kb, result, start,
                                     policy=policy, provenance=prov)
        if induced is None:
            continue
        induced_total += 1
        if induced.verified:
            verified += 1
            new_relation_pairs.add(induced.relation)
            induced_pairs.append((induced.subject, induced.relation, induced.obj))
        else:
            rejected += 1

    elapsed = time.perf_counter() - t0
    post_facts = kb.size()
    growth = post_facts - pre_facts

    print(f"\nProbed {len(probes)} 2-hop pairs in {elapsed:.2f}s")
    print(f"Induction attempts:      {induced_total}")
    print(f"  verified + stored:     {verified}")
    print(f"  rolled back (failed):  {rejected}")
    print(f"KB facts:                {pre_facts} -> {post_facts} (+{growth})")
    print(f"Distinct relations used: {len(new_relation_pairs)}")
    print(f"\nSample induced facts:")
    for s, r, o in induced_pairs[:10]:
        print(f"  ({s}, {r}, {o})")

    # Latency benefit: re-query one of the induced facts directly.
    if induced_pairs:
        s, r, o = induced_pairs[0]
        t0 = time.perf_counter()
        ans, score = kb.answer({"S": s, "R": r}, "O")
        direct_ms = (time.perf_counter() - t0) * 1000
        print(f"\nDirect lookup of induced ({s},{r},?): {ans!r} score={score:.3f} in {direct_ms:.2f}ms")

    out = {
        "n_probes": len(probes),
        "n_induced": induced_total,
        "n_verified": verified,
        "n_rejected": rejected,
        "facts_before": pre_facts,
        "facts_after": post_facts,
        "growth": growth,
        "relations_seen": sorted(new_relation_pairs),
        "elapsed_s": elapsed,
    }
    out_path = Path("data/chain_induction_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
