"""Scale study: RCK on six-figure real-world knowledge (ConceptNet).

The v15.2 paper's honest ceiling was 7,080 benchmarked facts. This
study measures the substrate at 10k / 30k / 100k real ConceptNet
assertions on a laptop CPU, made possible by shard-local cleanup
(v15.3): per-query cost depends on shard fill, not vocabulary size.

Data pipeline (fully reproducible):
  1. conceptnet-assertions-5.7.0.csv.gz (public download)
  2. scripts/import_conceptnet.py --language en --min-weight 2.0
  3. This script deterministically dedupes and takes the first
     100,000 unique (s, r, o) triples, writing them to
     data/conceptnet_scale_100k.jsonl (committed to the repo, so
     steps 1-2 are only needed to regenerate that file from source).

Per tier we measure:
  - ingest wall-clock and facts/second (bulk load, auto-sharded,
    symmetrize=False so the fact count is exact)
  - resident memory after ingest
  - recall@1 / recall@3 on 1,000 sampled stored facts, labeled
    against the valid-object set (ConceptNet is heavily multi-valued)
  - direct-query latency mean / median / p95
  - BFS chain discovery on 30 two-hop probes (live-relation index)
  - IDK safety: 300 never-stored (S, R) probes -- top-1 score
    distribution must sit far below present-fact scores

Output: data/scale_study.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from rck.bulk_ingest import bulk_load_triples
from rck.chain_discover import Goal, discover_chains
from rck.knowledge_base import ShardedKnowledgeBase
from rck.shard_sizing import auto_shard_for_kb

SUBSET_PATH = Path("data/conceptnet_scale_100k.jsonl")
SUBSET_SIZE = 100_000
TIERS = [10_000, 30_000, 100_000]
N_RECALL_SAMPLES = 1_000
N_ABSENT_PROBES = 300
N_DISCOVERY_PROBES = 30


def rss_mb() -> float | None:
    """Resident set size in MB (Windows + POSIX; None if unavailable)."""
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes as wt

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wt.DWORD),
                    ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            fn = k32.K32GetProcessMemoryInfo
            fn.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
            fn.restype = wt.BOOL
            k32.GetCurrentProcess.restype = wt.HANDLE
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return None
            return pmc.WorkingSetSize / (1024 * 1024)
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024.0
    except Exception:
        return None


def ensure_subset(source: Path | None) -> list[tuple[str, str, str]]:
    if SUBSET_PATH.exists():
        triples = []
        with open(SUBSET_PATH, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                triples.append((rec["s"], rec["r"], rec["o"]))
        return triples
    if source is None or not source.exists():
        raise SystemExit(
            f"{SUBSET_PATH} not found and no --source given. Regenerate "
            "via scripts/import_conceptnet.py (see module docstring).")
    triples = []
    seen: set[tuple[str, str, str]] = set()
    with open(source, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            t = (rec["s"], rec["r"], rec["o"])
            if t not in seen:
                seen.add(t)
                triples.append(t)
            if len(triples) >= SUBSET_SIZE:
                break
    if len(triples) < SUBSET_SIZE:
        print(f"warning: source yielded only {len(triples)} unique triples")
    with open(SUBSET_PATH, "w", encoding="utf-8") as f:
        for s, r, o in triples:
            f.write(json.dumps({"s": s, "r": r, "o": o}) + "\n")
    print(f"Wrote {SUBSET_PATH} ({len(triples)} triples)")
    return triples


def two_hop_probes(triples: list[tuple[str, str, str]],
                   n: int) -> list[tuple[str, str]]:
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


def run_tier(triples: list[tuple[str, str, str]], n_facts: int) -> dict:
    tier = triples[:n_facts]
    n_shards = auto_shard_for_kb(n_facts)
    print(f"\n--- {n_facts:,} facts, {n_shards} shards")

    rss_before = rss_mb()
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    t0 = time.perf_counter()
    bulk_load_triples(kb, tier, symmetrize=False)
    ingest_s = time.perf_counter() - t0
    rss_after = rss_mb()
    print(f"  ingest: {ingest_s:.1f}s ({n_facts / ingest_s:,.0f} facts/s), "
          f"RSS {rss_after:.0f} MB" if rss_after else
          f"  ingest: {ingest_s:.1f}s")

    # Ground truth: valid objects per (S, R).
    valid: dict[tuple[str, str], set[str]] = defaultdict(set)
    for s, r, o in tier:
        valid[(s, r)].add(o)
    keys = sorted(valid.keys())

    rng = np.random.default_rng(0)
    idx = rng.choice(len(keys), size=min(N_RECALL_SAMPLES, len(keys)),
                     replace=False)
    sample = [keys[i] for i in sorted(idx)]

    hits1 = hits3 = 0
    latencies: list[float] = []
    present_scores: list[float] = []
    for s, r in sample:
        t0 = time.perf_counter()
        results = kb.query({"S": s, "R": r}, "O", top_k=3)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if results and str(results[0][0]) in valid[(s, r)]:
            hits1 += 1
            present_scores.append(float(results[0][1]))
        if any(str(sym) in valid[(s, r)] for sym, _ in results):
            hits3 += 1

    recall1 = hits1 / len(sample)
    recall3 = hits3 / len(sample)
    lat_med = statistics.median(latencies)
    lat_p95 = statistics.quantiles(latencies, n=20)[18]
    print(f"  recall@1 {recall1:.1%}  recall@3 {recall3:.1%}  "
          f"query median {lat_med:.2f}ms p95 {lat_p95:.2f}ms")

    # IDK safety: never-stored (S, R) pairs.
    subjects = sorted({s for s, _ in keys})
    absent_scores: list[float] = []
    for i in range(N_ABSENT_PROBES):
        s = subjects[int(rng.integers(0, len(subjects)))]
        results = kb.query({"S": s, "R": f"neverstored_{i}"}, "O", top_k=1)
        absent_scores.append(float(results[0][1]) if results else 0.0)
    absent_p95 = statistics.quantiles(absent_scores, n=20)[18] \
        if len(absent_scores) >= 20 else max(absent_scores, default=0.0)
    present_p05 = statistics.quantiles(present_scores, n=20)[0] \
        if len(present_scores) >= 20 else min(present_scores, default=0.0)
    print(f"  IDK separation: absent p95 {absent_p95:.3f} vs "
          f"present p5 {present_p05:.3f}")

    # Discovery.
    probes = two_hop_probes(tier, N_DISCOVERY_PROBES)
    index = kb.relation_index()
    found = 0
    disc_ms: list[float] = []
    for start, target in probes:
        t0 = time.perf_counter()
        chains = discover_chains(kb, start, Goal.symbol(target),
                                 max_depth=4, beam_width=3, top_n=1,
                                 min_link_score=0.10,
                                 relation_index=index)
        disc_ms.append((time.perf_counter() - t0) * 1000.0)
        if chains:
            found += 1
    disc_med = statistics.median(disc_ms) if disc_ms else 0.0
    print(f"  discovery: {found}/{len(probes)} "
          f"({found / max(1, len(probes)):.0%}), median {disc_med:.0f}ms")

    return {
        "n_facts": n_facts,
        "n_shards": n_shards,
        "n_relations": len({r for _, r, _ in tier}),
        "n_symbols": kb.codebook.size(),
        "ingest_s": ingest_s,
        "ingest_facts_per_s": n_facts / ingest_s,
        "rss_mb_after_ingest": rss_after,
        "rss_mb_delta": (rss_after - rss_before)
        if rss_after and rss_before else None,
        "recall_at_1": recall1,
        "recall_at_3": recall3,
        "n_recall_samples": len(sample),
        "query_ms_mean": statistics.mean(latencies),
        "query_ms_median": lat_med,
        "query_ms_p95": lat_p95,
        "absent_score_p95": absent_p95,
        "absent_score_max": max(absent_scores, default=0.0),
        "present_score_p05": present_p05,
        "discovery_rate": found / max(1, len(probes)),
        "discovery_ms_median": disc_med,
        "n_discovery_probes": len(probes),
    }


def main() -> int:
    p = argparse.ArgumentParser(prog="scale_study")
    p.add_argument("--source", type=Path, default=None,
                   help="full ConceptNet import JSONL (only needed to "
                        "regenerate data/conceptnet_scale_100k.jsonl)")
    args = p.parse_args()

    print("=" * 70)
    print(" SCALE STUDY -- ConceptNet English, single CPU thread, D=4096")
    print("=" * 70)
    triples = ensure_subset(args.source)
    print(f"Subset: {len(triples):,} unique triples, "
          f"{len({r for _, r, _ in triples})} relations, "
          f"{len({x for s, _, o in triples for x in (s, o)}):,} entities")

    tiers = [run_tier(triples, n) for n in TIERS if n <= len(triples)]

    print("\nPaper-ready table:")
    print(f"{'facts':>8} {'shards':>6} {'ingest':>8} {'r@1':>6} {'r@3':>6} "
          f"{'q med':>7} {'q p95':>7} {'disc':>5}")
    for t in tiers:
        print(f"{t['n_facts']:>8,} {t['n_shards']:>6} "
              f"{t['ingest_s']:>7.1f}s {t['recall_at_1']:>6.1%} "
              f"{t['recall_at_3']:>6.1%} {t['query_ms_median']:>6.2f}ms "
              f"{t['query_ms_p95']:>6.2f}ms {t['discovery_rate']:>5.0%}")

    out = Path("data/scale_study.json")
    out.write_text(json.dumps({
        "source": "ConceptNet 5.7 English, min-weight 2.0, first "
                  f"{SUBSET_SIZE:,} unique triples "
                  "(data/conceptnet_scale_100k.jsonl)",
        "environment": "single process, D=4096, seed=0, "
                       "symmetrize=False, shard-local cleanup (default)",
        "tiers": tiers,
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
