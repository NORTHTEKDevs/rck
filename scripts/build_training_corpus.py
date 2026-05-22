"""Build the v7 training corpus -- 100k+ (draft, target) pairs.

Strategy:
  1. Stream every (S, R, O) fact from all loaded KBs.
  2. For each fact, generate every available paraphrase.
  3. Create cross-paraphrase pairs (draft, target) so the model learns
     that any phrasing implies any other.
  4. Apply ENTITY SUBSTITUTION: pick another entity with the same isa
     parent and swap into the template -- this multiplies the corpus
     and helps the model generalise to novel entities.

Run:
    python scripts/build_training_corpus.py \\
        --out data/training_corpus.jsonl \\
        --examples-per-triple 8 \\
        --substitutions-per-pair 2

Output: JSONL with {"draft": ..., "target": ..., "s": ..., "r": ..., "o": ...}.
At default settings on the v6 KB you get ~150k-400k examples.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from rck.bulk_ingest import bulk_load_jsonl
from rck.knowledge_base import ShardedKnowledgeBase
from rck.polisher_training import generate_examples_for_triple, render_all_phrasings


def _entities_by_parent(kb: ShardedKnowledgeBase) -> dict[str, list[str]]:
    """Group entities by their isa parent. Used for entity substitution."""
    out: dict[str, list[str]] = defaultdict(list)
    for shard in kb._shards:
        for fact in shard._facts:
            r = str(fact.get("R", ""))
            if r in ("isa", "kind", "category"):
                parent = str(fact.get("O", ""))
                child = str(fact.get("S", ""))
                out[parent].append(child)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_training_corpus")
    p.add_argument("--out", required=True)
    p.add_argument("--kb", nargs="+", default=[
        "data/commonsense_kb.jsonl",
        "data/extended_kb.jsonl",
        "data/massive_kb.jsonl",
    ])
    p.add_argument("--examples-per-triple", type=int, default=6)
    p.add_argument("--substitutions-per-pair", type=int, default=2)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)

    print(f"Loading KBs from: {args.kb}")
    kb = ShardedKnowledgeBase(dim=4096, n_shards=128, seed=0)
    for fp in args.kb:
        if Path(fp).exists():
            stats = bulk_load_jsonl(kb, fp, symmetrize=True)
            print(f"  + {fp}: {stats['facts']:,} facts loaded")
        else:
            print(f"  ! {fp} not found, skipping")
    print(f"Total in KB: {kb.size():,}")

    entities_by_parent = _entities_by_parent(kb)
    print(f"Found {len(entities_by_parent)} isa parents for substitution.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as outf:
        # Iterate every fact and emit examples + substitutions.
        for shard in kb._shards:
            for fact in shard._facts:
                if args.max_examples and written >= args.max_examples:
                    break
                triple = (
                    str(fact.get("S", "")),
                    str(fact.get("R", "")),
                    str(fact.get("O", "")),
                )

                # Base examples.
                for ex in generate_examples_for_triple(
                    triple, max_pairs=args.examples_per_triple,
                ):
                    outf.write(json.dumps({
                        "draft": ex.draft, "target": ex.target,
                        "s": triple[0], "r": triple[1], "o": triple[2],
                    }) + "\n")
                    written += 1

                # Entity substitutions: find another entity of the same
                # parent and emit the same paraphrase pairs.
                s, r, o = triple
                # Find the parent of `s`.
                parent_results = kb.query({"S": s, "R": "isa"}, "O", top_k=1)
                if not parent_results or parent_results[0][1] < 0.10:
                    continue
                parent = str(parent_results[0][0])
                siblings = [e for e in entities_by_parent.get(parent, [])
                            if e != s and e != o]
                if not siblings:
                    continue
                for _ in range(args.substitutions_per_pair):
                    if not siblings:
                        break
                    sub = rng.choice(siblings)
                    sub_triple = (sub, r, o)
                    for ex in generate_examples_for_triple(
                        sub_triple,
                        max_pairs=max(1, args.examples_per_triple // 2),
                    ):
                        outf.write(json.dumps({
                            "draft": ex.draft, "target": ex.target,
                            "s": sub, "r": r, "o": o,
                        }) + "\n")
                        written += 1
            if args.max_examples and written >= args.max_examples:
                break

    print(f"\nWrote {written:,} training examples -> {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
