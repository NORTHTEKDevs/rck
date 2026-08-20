"""Comparative baselines: what does the HRR substrate actually cost, and buy?

Every study in paper section 5 measures RCK against RCK. This one measures it
against the obvious alternatives on identical data with identical protocols,
because the first question any reviewer asks is "why not just use a dict / a
graph library?" and the paper had no answer.

Two axes, matching the two paper sections that are most exposed:

  Axis A (paper 5.6, storage + retrieval): RCK vs a plain Python dict index.
  Axis B (paper 5.5, chain discovery):     RCK vs networkx BFS.

What the comparators are and are not
------------------------------------
They are *lower bounds on a specialised index*, not competing systems. Neither
a dict nor a networkx DiGraph does provenance, calibrated confidence, IDK
detection, chain induction, negative facts, contradiction resolution, or
federated merge. RCK's value, if it has one, lives in that layer -- not in the
substrate. The point of this study is to say exactly what the substrate costs
so the rest of the paper's claims can be read honestly.

Reproduce:
    python scripts/baseline_study.py            # full, writes data/baseline_study.json
    python scripts/baseline_study.py --quick    # 10k tier only
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
CONCEPTNET = ROOT / "data" / "conceptnet_scale_100k.jsonl"
OUT = ROOT / "data" / "baseline_study.json"

SEED = 0
N_RECALL_PROBES = 1000
N_IDK_PROBES = 300
N_CHAIN_PROBES = 30


def load_facts(limit: int) -> list[tuple[str, str, str]]:
    facts = []
    with open(CONCEPTNET, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            d = json.loads(line)
            facts.append((d["s"], d["r"], d["o"]))
    return facts


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


# --------------------------------------------------------------------------
# Axis A: storage + retrieval
# --------------------------------------------------------------------------

def bench_dict(facts, sample, valid) -> dict:
    """Exact (S,R) -> [O] index. The floor: what a hash map gives you."""
    base = rss_mb()
    t0 = time.perf_counter()
    idx: dict[tuple[str, str], list[str]] = {}
    for s, r, o in facts:
        idx.setdefault((s, r), []).append(o)
    ingest = time.perf_counter() - t0
    mem = rss_mb() - base

    lat, hits = [], 0
    for s, r, o in sample:
        t = time.perf_counter_ns()
        got = idx.get((s, r))
        lat.append((time.perf_counter_ns() - t) / 1e6)
        if got and got[0] in valid[(s, r)]:
            hits += 1
    lat.sort()

    idk_ok = sum(1 for i in range(N_IDK_PROBES)
                 if idx.get((f"__absent_{i}__", "isa")) is None)

    return {
        "system": "dict",
        "ingest_s": round(ingest, 3),
        "rss_mb": round(mem, 1),
        "recall_at_1": round(hits / len(sample), 4),
        "query_median_ms": round(statistics.median(lat), 6),
        "query_p95_ms": round(lat[int(0.95 * len(lat))], 6),
        "idk_rate": round(idk_ok / N_IDK_PROBES, 4),
    }


def bench_rck(facts, sample, valid) -> dict:
    from rck.conscious_agent import ConsciousAgent

    base = rss_mb()
    t0 = time.perf_counter()
    agent = ConsciousAgent(expected_facts=len(facts) * 2)
    for s, r, o in facts:
        agent.tell(s, r, o)
    ingest = time.perf_counter() - t0
    mem = rss_mb() - base

    lat, hits = [], 0
    for s, r, o in sample:
        t = time.perf_counter_ns()
        res = agent.ask_with_idk({"S": s, "R": r}, "O")
        lat.append((time.perf_counter_ns() - t) / 1e6)
        if res.top_symbol in valid[(s, r)]:
            hits += 1
    lat.sort()

    idk_ok = 0
    for i in range(N_IDK_PROBES):
        res = agent.ask_with_idk({"S": f"__absent_{i}__", "R": "isa"}, "O")
        if res.state.name == "IDK":
            idk_ok += 1

    return {
        "system": "rck",
        "ingest_s": round(ingest, 3),
        "rss_mb": round(mem, 1),
        "recall_at_1": round(hits / len(sample), 4),
        "query_median_ms": round(statistics.median(lat), 6),
        "query_p95_ms": round(lat[int(0.95 * len(lat))], 6),
        "idk_rate": round(idk_ok / N_IDK_PROBES, 4),
        "n_shards": agent.knowledge.n_shards,
    }


# --------------------------------------------------------------------------
# Axis B: two-hop chain discovery
# --------------------------------------------------------------------------

def make_chain_probes(facts, n):
    """(start, end) pairs with a known two-hop path, so both systems are
    asked to find something that provably exists."""
    fwd = defaultdict(list)
    for s, r, o in facts:
        fwd[s].append((r, o))
    rng = random.Random(SEED)
    probes, seen = [], set()
    for s in rng.sample(sorted(fwd), min(len(fwd), 4000)):
        for _, mid in fwd[s]:
            for _, end in fwd.get(mid, []):
                if end != s and (s, end) not in seen:
                    seen.add((s, end))
                    probes.append((s, end))
                    break
            if len(probes) >= n:
                break
        if len(probes) >= n:
            break
    return probes[:n]


def bench_networkx_chains(facts, probes) -> dict:
    import networkx as nx

    base = rss_mb()
    t0 = time.perf_counter()
    g = nx.DiGraph()
    for s, r, o in facts:
        g.add_edge(s, o, relation=r)
    build = time.perf_counter() - t0
    mem = rss_mb() - base

    lat, found = [], 0
    for start, end in probes:
        t = time.perf_counter_ns()
        try:
            path = nx.shortest_path(g, start, end)
            ok = len(path) - 1 <= 4
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            ok = False
        lat.append((time.perf_counter_ns() - t) / 1e6)
        found += ok
    lat.sort()
    return {
        "system": "networkx",
        "build_s": round(build, 3),
        "rss_mb": round(mem, 1),
        "discovery_rate": round(found / len(probes), 4),
        "median_ms": round(statistics.median(lat), 4),
    }


def bench_rck_chains(facts, probes) -> dict:
    from rck.conscious_agent import ConsciousAgent

    base = rss_mb()
    t0 = time.perf_counter()
    agent = ConsciousAgent(expected_facts=len(facts) * 2)
    for s, r, o in facts:
        agent.tell(s, r, o)
    build = time.perf_counter() - t0
    mem = rss_mb() - base

    lat, found = [], 0
    for start, end in probes:
        t = time.perf_counter_ns()
        spec = agent.discover(start, end, max_depth=4)
        lat.append((time.perf_counter_ns() - t) / 1e6)
        found += bool(spec and spec.get("relations"))
    lat.sort()
    return {
        "system": "rck",
        "build_s": round(build, 3),
        "rss_mb": round(mem, 1),
        "discovery_rate": round(found / len(probes), 4),
        "median_ms": round(statistics.median(lat), 4),
    }


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="10k tier only")
    args = ap.parse_args()

    if not CONCEPTNET.exists():
        raise SystemExit(f"missing {CONCEPTNET}; see scripts/import_conceptnet.py")

    tiers = [10_000] if args.quick else [10_000, 30_000, 100_000]
    results = {"seed": SEED, "storage": [], "chains": []}

    for n in tiers:
        facts = load_facts(n)
        valid = defaultdict(set)
        for s, r, o in facts:
            valid[(s, r)].add(o)
        sample = random.Random(1).sample(facts, min(N_RECALL_PROBES, len(facts)))

        print(f"\n=== {n:,} facts: storage + retrieval ===")
        for fn in (bench_dict, bench_rck):
            row = fn(facts, sample, valid)
            row["facts"] = n
            results["storage"].append(row)
            print(f"  {row['system']:9s} ingest={row['ingest_s']:7.3f}s "
                  f"rss={row['rss_mb']:7.1f}MB recall@1={row['recall_at_1']:.4f} "
                  f"q50={row['query_median_ms']:.6f}ms")

    chain_facts = load_facts(10_000)
    probes = make_chain_probes(chain_facts, N_CHAIN_PROBES)
    print(f"\n=== two-hop chain discovery, {len(probes)} probes on 10,000 facts ===")
    for fn in (bench_networkx_chains, bench_rck_chains):
        row = fn(chain_facts, probes)
        row["facts"] = 10_000
        row["n_probes"] = len(probes)
        results["chains"].append(row)
        print(f"  {row['system']:9s} build={row['build_s']:6.3f}s "
              f"rss={row['rss_mb']:7.1f}MB found={row['discovery_rate']:.4f} "
              f"median={row['median_ms']:.4f}ms")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
