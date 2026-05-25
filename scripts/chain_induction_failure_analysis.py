"""Diagnostic: where does chain induction fail?

Re-runs the same 400-probe study used in the expanded chain-induction
result, but captures EVERY induction outcome with full provenance so we
can categorize failures. Output feeds the v15.1 precision-push work.

Buckets:
  * verified           -- stored fact passed self-verify roundtrip
  * rejected_confidence-- chain confidence < min_confidence
  * rejected_inverse   -- gate 1: consecutive inverse relations
  * rejected_same_rel  -- gate 2: non-transitive same-relation chain
  * rejected_cycle     -- gate 3: degenerate cycle (answer = intermediate)
  * rejected_denied    -- gate 4: negative-fact match
  * rejected_selfverify-- chain passed gates, stored, but roundtrip failed

The first six gates fire BEFORE storage. The seventh ("selfverify")
fires AFTER storage and represents HRR retrieval noise rather than
semantic error -- it's the failure mode a 5th semantic gate cannot
catch, and the one that responds to shard / dimension tuning.

Outputs:
  data/chain_induction_failures.json   -- machine-readable detail
  Prints summary table to stdout.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
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


KB_FILES = [
    "data/commonsense_kb.jsonl",
    "data/extended_kb.jsonl",
    "data/massive_kb.jsonl",
    "data/ultra_kb.jsonl",
]
TARGET_PROBES = 400
MAX_DEPTH = 3
BEAM_WIDTH = 3
MIN_LINK_SCORE = 0.10


def categorize(reason: str | None) -> str:
    if reason is None:
        return "verified"
    if reason.startswith("confidence"):
        return "rejected_confidence"
    if "phantom shortcut" in reason or "inverse pair" in reason:
        return "rejected_inverse"
    if "non-transitive same-relation" in reason:
        return "rejected_same_rel"
    if "degenerate cycle" in reason or "intermediate node" in reason:
        return "rejected_cycle"
    if "denied by stored negative fact" in reason:
        return "rejected_denied"
    if "self-verification failed" in reason:
        return "rejected_selfverify"
    if "skill seen" in reason:
        return "rejected_skill"
    if "type-signature mismatch" in reason:
        return "rejected_typesig"
    if "no meaningful induced relation" in reason:
        return "rejected_no_relation"
    return "rejected_other"


def load_kb() -> tuple[ShardedKnowledgeBase, list[tuple[str, str, str]]]:
    seen: set[tuple[str, str, str]] = set()
    triples: list[tuple[str, str, str]] = []
    for path in KB_FILES:
        p = Path(path)
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                t = (r["s"], r["r"], r["o"])
                if t in seen:
                    continue
                seen.add(t)
                triples.append(t)
    n_shards = recommend_shards(len(triples) * 2, dim=4096).n_shards
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    bulk_load_triples(kb, triples)
    return kb, triples


def generate_probes(triples: list[tuple[str, str, str]],
                    target: int) -> list[tuple[str, str]]:
    by_subject: dict[str, list[tuple[str, str]]] = {}
    direct: dict[str, set[str]] = {}
    for s, r, o in triples:
        by_subject.setdefault(s, []).append((r, o))
        direct.setdefault(s, set()).add(o)
    probes: list[tuple[str, str]] = []
    for s, edges in by_subject.items():
        for _, mid in edges:
            if mid in by_subject:
                for _, target_node in by_subject[mid]:
                    if target_node == s:
                        continue
                    if target_node in direct.get(s, set()):
                        continue
                    probes.append((s, target_node))
                    if len(probes) >= target:
                        return probes
                    break
    return probes


def main() -> int:
    print("=" * 78)
    print(" CHAIN INDUCTION FAILURE ANALYSIS (v15 diagnostic)")
    print("=" * 78)

    kb, triples = load_kb()
    print(f"\nLoaded {kb.size()} unique facts from {len(KB_FILES)} KB files "
          f"into {kb.n_shards} shards")

    probes = generate_probes(triples, TARGET_PROBES)
    print(f"Generated {len(probes)} 2-hop probe pairs (target {TARGET_PROBES})")

    policy = InductionPolicy(
        min_confidence=0.15,
        min_chain_length=2,
        verify_after=True,
    )
    prov = ProvenanceStore()

    chains_found = 0
    attempts = 0
    records: list[dict] = []

    t0 = time.perf_counter()
    for start, target in probes:
        chains = discover_chains(
            kb, start, Goal.symbol(target),
            max_depth=MAX_DEPTH, beam_width=BEAM_WIDTH, top_n=1,
            min_link_score=MIN_LINK_SCORE,
        )
        if not chains:
            continue
        chains_found += 1
        ch = chains[0]
        hops = [Hop(r, d) for r, d in zip(ch.relations, ch.directions)]
        result = walk_chain(kb, start, hops)
        if result.answer is None:
            continue
        induced = induce_from_chain(
            kb, result, start, policy=policy, provenance=prov,
        )
        if induced is None:
            continue
        attempts += 1
        bucket = categorize(induced.rejected_reason)
        records.append({
            "probe_start": start,
            "probe_target": target,
            "induced_subject": induced.subject,
            "induced_relation": induced.relation,
            "induced_obj": induced.obj,
            "chain_relations": [r for _, r, _, _ in result.trace],
            "chain_trace": [
                {"s": s, "r": r, "o": o, "score": round(sc, 4)}
                for s, r, o, sc in result.trace
            ],
            "confidence": round(induced.confidence, 4),
            "verified": induced.verified,
            "bucket": bucket,
            "reason": induced.rejected_reason,
        })
    elapsed = time.perf_counter() - t0

    counts = Counter(r["bucket"] for r in records)
    verified_n = counts.get("verified", 0)
    precision = verified_n / attempts if attempts else 0.0

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Probes:           {len(probes)}")
    print(f"Chains found:     {chains_found}")
    print(f"Induction attempts: {attempts}")
    print(f"Verified (stored): {verified_n}")
    print(f"Precision = verified/attempts = {precision:.3f}")

    print(f"\n--- Bucket breakdown ---")
    bucket_order = [
        "verified",
        "rejected_confidence",
        "rejected_inverse",
        "rejected_same_rel",
        "rejected_cycle",
        "rejected_denied",
        "rejected_typesig",
        "rejected_no_relation",
        "rejected_selfverify",
        "rejected_skill",
        "rejected_other",
    ]
    for b in bucket_order:
        n = counts.get(b, 0)
        if n == 0:
            continue
        pct = (n / attempts * 100) if attempts else 0.0
        print(f"  {b:25s} {n:4d}  ({pct:5.1f}%)")

    sv_records = [r for r in records if r["bucket"] == "rejected_selfverify"]
    print(f"\n--- Self-verify failures (HRR retrieval noise candidates) ---")
    print(f"Count: {len(sv_records)}")
    if sv_records:
        rel_pairs = Counter(tuple(r["chain_relations"]) for r in sv_records)
        print(f"Top chain-relation patterns:")
        for pat, n in rel_pairs.most_common(8):
            print(f"  {' -> '.join(pat):40s} {n}")
        induced_rels = Counter(r["induced_relation"] for r in sv_records)
        print(f"\nInduced relation distribution:")
        for rel, n in induced_rels.most_common(8):
            print(f"  {rel:25s} {n}")
        print(f"\nFirst 10 failed inductions:")
        for r in sv_records[:10]:
            chain_str = " -> ".join(r["chain_relations"])
            print(f"  ({r['induced_subject']}, {r['induced_relation']}, "
                  f"{r['induced_obj']})  via [{chain_str}]  "
                  f"conf={r['confidence']:.3f}")

    verified_records = [r for r in records if r["bucket"] == "verified"]
    print(f"\n--- Verified-but-suspicious heuristics ---")
    # Heuristic 1: "implies" relation = gate fell back to generic; weakest semantically
    implies_n = sum(1 for r in verified_records if r["induced_relation"] == "implies")
    print(f"Used generic 'implies' fallback relation: {implies_n}/{verified_n} "
          f"({implies_n / verified_n * 100:.1f}%)")
    # Heuristic 2: induced obj appears in subject as substring (e.g. (catfish, x, cat))
    substr_n = sum(1 for r in verified_records
                   if r["induced_obj"] in r["induced_subject"]
                   or r["induced_subject"] in r["induced_obj"])
    print(f"Subject/object share substring (suspect):  {substr_n}")
    # Heuristic 3: induced relation differs from any first-hop lifting relation
    # but isn't "implies" -- means the lifting-relation gate FORWARDED a relation
    # that may not transfer. Flag for manual review.
    lifting = {"isa", "partof", "locatedin", "memberof", "instanceof", "kind"}
    forwarded_n = sum(
        1 for r in verified_records
        if r["chain_relations"]
        and r["chain_relations"][0] in lifting
        and r["induced_relation"] == r["chain_relations"][-1]
        and r["induced_relation"] not in lifting
        and r["induced_relation"] != "implies"
    )
    print(f"Lifting-rel forwarded last hop:            {forwarded_n}")

    out = {
        "n_probes": len(probes),
        "n_chains_found": chains_found,
        "n_attempts": attempts,
        "n_verified": verified_n,
        "precision": round(precision, 4),
        "elapsed_s": round(elapsed, 2),
        "counts_by_bucket": dict(counts),
        "records": records,
    }
    out_path = Path("data/chain_induction_failures.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path} ({out_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
