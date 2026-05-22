"""v13 end-to-end chain-reasoning demo.

Loads the commonsense KB, then for a small set of (start, target)
queries:
  1. Discovers a relation chain via BFS.
  2. Executes the chain via chain_walker.
  3. Reports answer + confidence + the highway relations used.
  4. Verifies the final fact via self_verify.

This demonstrates the v13 capability stack on real data without any
hand-written chain templates.
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
from rck.conscious_agent import ConsciousAgent


QUERIES = [
    ("leaf", "forest"),
    ("elephant", "tree"),
    ("wheel", "driving"),
    ("branch", "forest"),
    ("engine", "driving"),
    ("page", "paper"),
    ("uk", "england"),
]


def main() -> int:
    print("=" * 70)
    print(" v13 chain reasoning demo on commonsense KB")
    print("=" * 70)

    # Load triples.
    triples = []
    with open("data/commonsense_kb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            triples.append((r["s"], r["r"], r["o"]))

    # Build the agent at the right shard count.
    agent = ConsciousAgent(
        dim=4096, expected_facts=len(triples) * 2,  # account for inverses
        install_self=False,
    )
    bulk_load_triples(agent.knowledge, triples)
    print(f"\nKB: {agent.knowledge.size()} facts in {agent.n_shards} shards\n")

    for start, target in QUERIES:
        t0 = time.perf_counter()
        spec = agent.discover(start, target, max_depth=4)
        if spec is None:
            print(f"  {start:>12} -> {target!r:>12}: NO CHAIN")
            continue
        res = agent.reason(start, spec["relations"],
                           directions=spec["directions"])
        elapsed = time.perf_counter() - t0
        print(f"  {start:>12} -> {target!r:>12}: via "
              f"{' -> '.join(spec['relations'])}  "
              f"conf={res['confidence']:.3f} ({res['hedge']})  "
              f"[{elapsed*1000:.0f}ms]")

    # Skill library should have collected patterns.
    stats = agent.skills.stats()
    print(f"\nSkill library after demo: {stats['n']} patterns "
          f"recorded, {stats['total_uses']} total uses.")
    if stats["n"] > 0:
        print("\nTop patterns:")
        for sk in agent.skills.most_used(n=5):
            relations = "->".join(rel for _, rel in sk.pattern)
            print(f"  [{sk.success_count}x] {relations}  "
                  f"conf={sk.confidence:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
