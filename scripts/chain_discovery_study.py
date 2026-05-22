"""Empirical chain-discovery study on the real commonsense KB.

Questions:
  1. Can chain_discover find multi-hop chains in a realistic KB?
  2. What's the distribution of useful chain depths?
  3. Which relations show up most often in successful chains?
  4. How does success rate change with allow_reverse?
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rck.bulk_ingest import bulk_load_triples
from rck.chain_discover import Goal, discover_chains
from rck.knowledge_base import ShardedKnowledgeBase


def load_commonsense() -> tuple[ShardedKnowledgeBase, list[tuple[str, str, str]]]:
    triples = []
    with open("data/commonsense_kb.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            triples.append((rec["s"], rec["r"], rec["o"]))
    kb = ShardedKnowledgeBase(dim=4096, n_shards=64, seed=0)
    bulk_load_triples(kb, triples)
    return kb, triples


def main() -> int:
    print("=" * 70)
    print(" CHAIN DISCOVERY STUDY on commonsense KB")
    print("=" * 70)

    kb, triples = load_commonsense()
    print(f"\nLoaded {len(triples)} input triples ({kb.size()} after auto-inverse)")
    relations = sorted({t[1] for t in triples})
    print(f"Relations: {relations}")

    # Build a set of (start, target) probe pairs. We want pairs where
    # the target is more than 1 hop away in the KB graph.
    by_subject: dict[str, list[tuple[str, str]]] = {}
    for s, r, o in triples:
        by_subject.setdefault(s, []).append((r, o))

    # Two-hop transitive probes: (start) -[r1]-> X -[r2]-> (target).
    probes: list[tuple[str, str]] = []
    for s, edges in by_subject.items():
        if len(probes) >= 20:
            break
        for _, mid in edges:
            if mid in by_subject:
                for _, target in by_subject[mid]:
                    if target != s and target not in {o for _, o in edges}:
                        probes.append((s, target))
                        break
            if len(probes) >= 20:
                break

    print(f"\nProbing chain discovery on {len(probes)} 2-hop transitive pairs ...")
    depths: list[int] = []
    relation_in_chain: Counter = Counter()
    discovered = 0
    timings: list[float] = []

    for start, target in probes:
        goal = Goal.symbol(target)
        t0 = time.perf_counter()
        chains = discover_chains(
            kb, start, goal, max_depth=4,
            beam_width=3, top_n=1, min_link_score=0.10,
        )
        timings.append(time.perf_counter() - t0)
        if not chains:
            print(f"  {start:>15} -> {target!r:>15}  MISS")
            continue
        chain = chains[0]
        depths.append(len(chain.relations))
        for r in chain.relations:
            relation_in_chain[r] += 1
        discovered += 1
        print(f"  {start:>15} -> {target!r:>15}  via {' -> '.join(chain.relations)}  "
              f"conf={chain.confidence:.3f}")

    print(f"\nDiscovery rate: {discovered}/{len(probes)} = {discovered/max(1, len(probes)):.1%}")
    print(f"Depth distribution: {Counter(depths)}")
    print(f"Top relations in successful chains: {relation_in_chain.most_common(5)}")
    if timings:
        print(f"Avg discovery time: {sum(timings)/len(timings)*1000:.1f}ms")
        print(f"Total time: {sum(timings):.2f}s")

    out_path = Path("data/chain_discovery_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "n_input_triples": len(triples),
        "n_facts_after_symmetrise": kb.size(),
        "n_probes": len(probes),
        "n_chains_discovered": discovered,
        "discovery_rate": discovered / max(1, len(probes)),
        "depth_distribution": dict(Counter(depths)),
        "top_relations": dict(relation_in_chain.most_common(10)),
        "avg_discovery_ms": sum(timings) / max(1, len(timings)) * 1000 if timings else 0.0,
    }, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
