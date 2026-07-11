"""Empirical chain-discovery study across all bundled KB tiers.

Questions:
  1. Can chain_discover find multi-hop chains in realistic KBs of
     increasing size and relation count?
  2. How does discovery latency scale with facts and relations?
  3. How much does the per-shard live-relation index (v15.2) save?

Each KB tier is probed with 30 two-hop transitive pairs. Every probe is
discovered twice (with and without the relation index), 3 repeats each,
and we report the per-probe MEDIAN across repeats to damp scheduler
noise. Absolute wall-clock numbers are machine-dependent; the paper
quotes them as indicative, alongside the machine-independent query
counts.
"""
from __future__ import annotations

import json
import statistics
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
from rck.shard_sizing import auto_shard_for_kb

TIERS: list[tuple[str, list[str]]] = [
    ("commonsense", ["commonsense_kb.jsonl"]),
    ("+extended", ["commonsense_kb.jsonl", "extended_kb.jsonl"]),
    ("+ultra", ["commonsense_kb.jsonl", "extended_kb.jsonl",
                "ultra_kb.jsonl"]),
    ("+massive", ["commonsense_kb.jsonl", "extended_kb.jsonl",
                  "ultra_kb.jsonl", "massive_kb.jsonl"]),
]
N_PROBES = 30
N_REPEATS = 3


def load_tier(files: list[str]) -> tuple[ShardedKnowledgeBase,
                                         list[tuple[str, str, str]], int]:
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for fn in files:
        with open(f"data/{fn}", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                t = (rec["s"], rec["r"], rec["o"])
                if t not in seen:
                    seen.add(t)
                    triples.append(t)
    # Size shards the same way the agent does for a KB this big.
    probe_kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(probe_kb, triples)
    n_shards = auto_shard_for_kb(probe_kb.size())
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    bulk_load_triples(kb, triples)
    return kb, triples, n_shards


def make_probes(triples: list[tuple[str, str, str]],
                n: int) -> list[tuple[str, str]]:
    """Two-hop transitive probes: (start) -[r1]-> X -[r2]-> (target)."""
    by_subject: dict[str, list[tuple[str, str]]] = {}
    for s, r, o in triples:
        by_subject.setdefault(s, []).append((r, o))
    probes: list[tuple[str, str]] = []
    for s, edges in by_subject.items():
        if len(probes) >= n:
            break
        for _, mid in edges:
            if mid in by_subject:
                for _, target in by_subject[mid]:
                    if target != s and target not in {o for _, o in edges}:
                        probes.append((s, target))
                        break
            if len(probes) >= n:
                break
    return probes


def run_tier(name: str, files: list[str]) -> dict:
    kb, triples, n_shards = load_tier(files)
    relations = sorted({t[1] for t in triples})
    probes = make_probes(triples, N_PROBES)
    print(f"\n--- {name}: {kb.size()} facts, {len(relations)} relations, "
          f"{n_shards} shards, {len(probes)} probes")

    # Count HRR queries issued per mode (machine-independent metric).
    query_counts = {"n": 0}
    original_query = kb.query

    def counting_query(*args, **kwargs):
        query_counts["n"] += 1
        return original_query(*args, **kwargs)

    kb.query = counting_query

    index = kb.relation_index()
    # Warmup (JIT-ish caches, page-in) before any timing.
    if probes:
        discover_chains(kb, probes[0][0], Goal.symbol(probes[0][1]),
                        max_depth=4, beam_width=3, top_n=1,
                        relation_index=index)

    results: dict[str, dict] = {}
    for mode in ("indexed", "no_index"):
        per_probe_ms: list[float] = []
        discovered = 0
        depths: list[int] = []
        relation_in_chain: Counter = Counter()
        query_counts["n"] = 0
        for start, target in probes:
            goal = Goal.symbol(target)
            reps: list[float] = []
            chains = []
            for _ in range(N_REPEATS):
                t0 = time.perf_counter()
                if mode == "indexed":
                    chains = discover_chains(
                        kb, start, goal, max_depth=4, beam_width=3,
                        top_n=1, min_link_score=0.10,
                        relation_index=index)
                else:
                    chains = discover_chains(
                        kb, start, goal, max_depth=4, beam_width=3,
                        top_n=1, min_link_score=0.10,
                        use_relation_index=False)
                reps.append((time.perf_counter() - t0) * 1000.0)
            per_probe_ms.append(statistics.median(reps))
            if chains:
                discovered += 1
                depths.append(len(chains[0].relations))
                for r in chains[0].relations:
                    relation_in_chain[r] += 1
        results[mode] = {
            "avg_ms": statistics.mean(per_probe_ms) if per_probe_ms else 0.0,
            "p95_ms": (statistics.quantiles(per_probe_ms, n=20)[18]
                       if len(per_probe_ms) >= 20 else
                       max(per_probe_ms, default=0.0)),
            "discovered": discovered,
            "discovery_rate": discovered / max(1, len(probes)),
            "depth_distribution": dict(Counter(depths)),
            "top_relations": dict(relation_in_chain.most_common(5)),
            "hrr_queries_total": query_counts["n"] // max(1, N_REPEATS),
        }
        print(f"  {mode:>9}: avg {results[mode]['avg_ms']:7.1f}ms  "
              f"p95 {results[mode]['p95_ms']:7.1f}ms  "
              f"rate {results[mode]['discovery_rate']:.0%}  "
              f"HRR queries/pass ~{results[mode]['hrr_queries_total']}")

    idx, no = results["indexed"], results["no_index"]
    speedup = no["avg_ms"] / idx["avg_ms"] if idx["avg_ms"] else 0.0
    query_cut = 1.0 - (idx["hrr_queries_total"] /
                       max(1, no["hrr_queries_total"]))
    print(f"  speedup {speedup:.1f}x, HRR queries cut {query_cut:.0%}")
    if idx["discovery_rate"] != no["discovery_rate"]:
        print("  NOTE: discovery rate differs between modes "
              "(index pruned a noise-only edge)")

    return {
        "tier": name,
        "kb_files": files,
        "n_facts": kb.size(),
        "n_relations": len(relations),
        "n_shards": n_shards,
        "n_probes": len(probes),
        "indexed": idx,
        "no_index": no,
        "speedup": speedup,
        "hrr_query_cut": query_cut,
    }


def main() -> int:
    print("=" * 70)
    print(" CHAIN DISCOVERY STUDY -- all KB tiers, "
          "with/without live-relation index")
    print("=" * 70)
    tiers = [run_tier(name, files) for name, files in TIERS]

    print("\nPaper-ready table (indexed mode):")
    print(f"{'KB':>14} {'facts':>6} {'rels':>5} {'shards':>6} "
          f"{'rate':>5} {'avg ms':>8} {'speedup':>8}")
    for t in tiers:
        print(f"{t['tier']:>14} {t['n_facts']:>6} {t['n_relations']:>5} "
              f"{t['n_shards']:>6} {t['indexed']['discovery_rate']:>5.0%} "
              f"{t['indexed']['avg_ms']:>8.1f} {t['speedup']:>7.1f}x")

    out_path = Path("data/chain_discovery_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "n_probes_per_tier": N_PROBES,
        "n_repeats": N_REPEATS,
        "timing_note": ("per-probe median of repeats; absolute ms is "
                        "machine-dependent, hrr_queries_total is not"),
        "tiers": tiers,
    }, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
