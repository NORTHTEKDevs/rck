"""How deep can RCK reason? Empirical chain-depth vs confidence study.

We build a long synthetic chain of facts:
    a_0 -> a_1 -> a_2 -> ... -> a_N
(every edge uses the relation `next`).
Then we walk the chain at depths 1..N and record:
  * whether the final answer is correct
  * the propagated confidence
  * the hedge ("strong" / "moderate" / "weak" / "uncertain")

This pins down the practical reasoning horizon of the v13 KB.

Output:
  data/chain_depth_study.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rck.bulk_ingest import bulk_load_triples
from rck.chain_walker import Hop, walk_chain
from rck.confidence_propagation import PropagationConfig
from rck.knowledge_base import ShardedKnowledgeBase


def build_chain_kb(length: int, dim: int = 4096, n_shards: int = 64,
                   seed: int = 0) -> tuple[ShardedKnowledgeBase, list[str]]:
    kb = ShardedKnowledgeBase(dim=dim, n_shards=n_shards, seed=seed)
    nodes = [f"a_{i:04d}" for i in range(length + 1)]
    triples = [(nodes[i], "next", nodes[i + 1]) for i in range(length)]
    bulk_load_triples(kb, triples)
    return kb, nodes


def run_study(depths: list[int], *, dim: int = 4096, n_shards: int = 64,
              rules: tuple[str, ...] = ("product", "min", "geometric_mean"),
              ) -> list[dict]:
    max_depth = max(depths)
    kb, nodes = build_chain_kb(length=max_depth, dim=dim, n_shards=n_shards)

    results: list[dict] = []
    for rule in rules:
        cfg = PropagationConfig(rule=rule, chain_decay=0.95)
        for depth in depths:
            chain = [Hop("next") for _ in range(depth)]
            res = walk_chain(kb, nodes[0], chain, config=cfg)
            correct = (res.answer == nodes[depth])
            results.append({
                "depth": depth,
                "rule": rule,
                "answer": res.answer,
                "expected": nodes[depth],
                "correct": bool(correct),
                "confidence": float(res.confidence),
                "hedge": res.hedge,
                "aborted_at": int(res.aborted_at),
                "trace_len": len(res.trace),
            })
    return results


def main() -> int:
    depths = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    print("=" * 70)
    print(" CHAIN-DEPTH STUDY (relation `next`, single linear chain)")
    print("=" * 70)
    rows = run_study(depths)

    # Pretty-print.
    print(f"\n{'rule':>15}  {'depth':>5}  {'correct':>7}  "
          f"{'conf':>6}  {'hedge':>10}")
    for r in rows:
        print(f"{r['rule']:>15}  {r['depth']:>5}  {str(r['correct']):>7}  "
              f"{r['confidence']:>6.3f}  {r['hedge']:>10}")

    # Reasoning horizon per rule: largest depth where correct AND hedge >= 'weak'.
    by_rule: dict[str, int] = {}
    for r in rows:
        if r["correct"] and r["hedge"] in {"strong", "moderate", "weak"}:
            by_rule[r["rule"]] = max(by_rule.get(r["rule"], 0), r["depth"])
    print()
    for rule, horizon in by_rule.items():
        print(f"Reasoning horizon ({rule}): {horizon} hops")

    out_path = Path("data/chain_depth_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "depths": depths,
        "reasoning_horizon_by_rule": by_rule,
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
