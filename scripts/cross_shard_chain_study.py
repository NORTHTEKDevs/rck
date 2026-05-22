"""Cross-shard chain study.

Each hop in a walked chain resolves at the shard chosen by
hash(subject || relation) % n_shards. Successive hops usually land
on DIFFERENT shards. We measure:
  * Avg shards crossed per chain.
  * How often the start-shard and end-shard differ.
  * Correlation with shard count (more shards -> more crossings?).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rck.bulk_ingest import bulk_load_triples
from rck.chain_discover import Goal, discover_chains
from rck.knowledge_base import ShardedKnowledgeBase
from rck.shard_sizing import recommend_shards


def _shard_of(subject: str, relation: str, n: int) -> int:
    key = f"{subject.lower()}\x00{relation.lower()}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "little") % n


def main() -> int:
    print("=" * 70)
    print(" CROSS-SHARD CHAIN STUDY")
    print("=" * 70)

    triples = []
    with open("data/commonsense_kb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            triples.append((r["s"], r["r"], r["o"]))

    summary = []
    for shard_count in (16, 64, 128, 256):
        kb = ShardedKnowledgeBase(dim=4096, n_shards=shard_count, seed=0)
        bulk_load_triples(kb, triples)

        # 2-hop transitive probes (same shape as previous studies).
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
            if len(probes) >= 60:
                break

        shards_crossed_per_chain: list[int] = []
        start_end_diff = 0
        successes = 0
        for start, target in probes:
            chains = discover_chains(
                kb, start, Goal.symbol(target), max_depth=3,
                beam_width=3, top_n=1, min_link_score=0.10,
            )
            if not chains:
                continue
            successes += 1
            shards_used: list[int] = []
            for s, r, o, _ in chains[0].trace:
                shards_used.append(_shard_of(s, r, shard_count))
            distinct = len(set(shards_used))
            shards_crossed_per_chain.append(distinct)
            if shards_used[0] != shards_used[-1]:
                start_end_diff += 1

        if successes == 0:
            continue
        avg_crossings = sum(shards_crossed_per_chain) / successes
        row = {
            "n_shards": shard_count,
            "n_probes": len(probes),
            "successes": successes,
            "avg_distinct_shards_per_chain": avg_crossings,
            "fraction_endpoints_differ": start_end_diff / successes,
            "distribution": dict(Counter(shards_crossed_per_chain)),
        }
        summary.append(row)
        print(f"\n[n_shards={shard_count}] probes={len(probes)} "
              f"successes={successes}")
        print(f"  avg distinct shards per chain: {avg_crossings:.2f}")
        print(f"  endpoints differ: {start_end_diff}/{successes} = "
              f"{start_end_diff/successes:.1%}")
        print(f"  distribution: {row['distribution']}")

    out_path = Path("data/cross_shard_chain_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
