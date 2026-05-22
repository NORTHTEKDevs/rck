"""Analogy benchmark on the commonsense KB.

Build (a, b, c, expected_d) probes where (a, R, b) and (c, R, d) are
both stored in the KB. Then run solve_analogy(a, b, c) and check
whether the returned d matches.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rck.analogy import solve_analogy
from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase
from rck.shard_sizing import recommend_shards


def main() -> int:
    print("=" * 70)
    print(" ANALOGY BENCHMARK on commonsense KB")
    print("=" * 70)

    triples = []
    with open("data/commonsense_kb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            triples.append((r["s"], r["r"], r["o"]))
    n_shards = recommend_shards(len(triples) * 2, dim=4096).n_shards
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    bulk_load_triples(kb, triples)
    print(f"\nKB: {kb.size()} facts in {n_shards} shards")

    # Group triples by relation. For each relation with >=2 distinct
    # (s, o) pairs, build analogy probes.
    by_relation: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for s, r, o in triples:
        by_relation[r].append((s, o))

    probes: list[tuple[str, str, str, str, str]] = []
    for r, pairs in by_relation.items():
        # Deduplicate.
        unique = list(dict.fromkeys(pairs))
        if len(unique) < 2:
            continue
        for i, (a, b) in enumerate(unique[:6]):
            for c, d in unique[i + 1: i + 4]:
                if a == c:
                    continue
                probes.append((a, b, c, d, r))
        if len(probes) >= 100:
            break

    print(f"\nBuilt {len(probes)} analogy probes")
    correct = 0
    rel_correct = 0
    answer_failures: list[tuple] = []
    rel_failures: list[tuple] = []
    for a, b, c, d, r in probes:
        res = solve_analogy(kb, a, b, c)
        if res.relation == r:
            rel_correct += 1
        else:
            rel_failures.append((a, b, c, d, r, res.relation))
        if res.answer == d:
            correct += 1
        else:
            answer_failures.append((a, b, c, d, r, res.answer))

    print(f"\nResults:")
    print(f"  Correct R inferred:  {rel_correct}/{len(probes)} = "
          f"{rel_correct/len(probes):.1%}")
    print(f"  Correct D inferred:  {correct}/{len(probes)} = "
          f"{correct/len(probes):.1%}")

    print(f"\nSample correct analogies (first 5):")
    shown = 0
    for a, b, c, d, r in probes:
        res = solve_analogy(kb, a, b, c)
        if res.answer == d:
            print(f"  {a} : {b} :: {c} : {res.answer}   (via {r})")
            shown += 1
            if shown >= 5:
                break

    print(f"\nSample WRONG answers (first 5):")
    for a, b, c, d, r, got in answer_failures[:5]:
        print(f"  {a} : {b} :: {c} : expected {d!r}, got {got!r}  (via {r})")

    out = {
        "n_probes": len(probes),
        "relation_correct": rel_correct,
        "answer_correct": correct,
        "rel_accuracy": rel_correct / max(1, len(probes)),
        "answer_accuracy": correct / max(1, len(probes)),
        "sample_failures": [
            {"a": a, "b": b, "c": c, "expected": d,
             "got": got, "relation": r}
            for a, b, c, d, r, got in answer_failures[:20]
        ],
    }
    out_path = Path("data/analogy_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
