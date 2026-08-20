"""Does the HRR substrate earn its place?

Paper section 5.0 established that on storage/retrieval and chain discovery the
substrate is dominated by a plain dict and by networkx. Two properties were
named there as the remaining candidates that could still justify it:

  * federated merge without an entity-alignment step (section 8.4)
  * analogy as native vector algebra (section 5.7)

Neither had ever been compared against a non-VSA alternative. This study does
that, on identical data, with the same information available to both sides.

The baselines are deliberately plain -- a dict of indices and about twenty
lines of set logic -- because the question is not "can something beat RCK" but
"does the vector substrate buy anything a trivial symbolic implementation does
not already have".

Reproduce:
    python scripts/substrate_justification_study.py
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "data" / "commonsense_kb.jsonl"
OUT = ROOT / "data" / "substrate_justification_study.json"


def load_triples() -> list[tuple[str, str, str]]:
    out = []
    with open(KB_PATH, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            s = d.get("s") or d.get("subject")
            r = d.get("r") or d.get("relation")
            o = d.get("o") or d.get("object")
            if s and r and o:
                out.append((str(s), str(r), str(o)))
    return out


# ---------------------------------------------------------------------------
# Experiment A: analogy
# ---------------------------------------------------------------------------

def build_analogy_probes(triples, limit=100):
    """Same construction as scripts/analogy_study.py: (a,R,b) and (c,R,d)
    both stored, ask a:b::c:? and expect d."""
    by_rel = defaultdict(list)
    for s, r, o in triples:
        by_rel[r].append((s, o))
    probes = []
    for r, pairs in by_rel.items():
        unique = list(dict.fromkeys(pairs))
        if len(unique) < 2:
            continue
        for i, (a, b) in enumerate(unique[:6]):
            for c, d in unique[i + 1:i + 4]:
                if len({a, b, c, d}) == 4:
                    probes.append((a, b, c, d, r))
        if len(probes) >= limit:
            break
    return probes[:limit]


def symbolic_analogy(triples, probes) -> dict:
    """The whole non-VSA analogy solver: two indices and a loop.

    Infer the relation from (a, ?, b), then answer (c, r, ?). Prefers a
    relation that actually yields an answer for c, which is the same
    disambiguation the vector solver gets from its confidence weighting.
    """
    sr_to_o = defaultdict(list)
    so_to_r = defaultdict(list)
    for s, r, o in triples:
        sr_to_o[(s, r)].append(o)
        so_to_r[(s, o)].append(r)

    valid = defaultdict(set)
    for s, r, o in triples:
        valid[(s, r)].add(o)

    t0 = time.perf_counter()
    exact = relation_ok = in_valid_set = 0
    for a, b, c, d, r_true in probes:
        rels = so_to_r.get((a, b), [])
        answer = inferred = None
        for r in rels:
            got = sr_to_o.get((c, r))
            if got:
                inferred, answer = r, got[0]
                break
        if inferred == r_true:
            relation_ok += 1
        if answer == d:
            exact += 1
        if answer is not None and answer in valid[(c, r_true)]:
            in_valid_set += 1
    elapsed = time.perf_counter() - t0

    n = len(probes)
    return {
        "system": "symbolic (dict + set logic)",
        "relation_accuracy": round(relation_ok / n, 4),
        "exact_answer_accuracy": round(exact / n, 4),
        "valid_set_accuracy": round(in_valid_set / n, 4),
        "total_s": round(elapsed, 4),
        "ms_per_probe": round(elapsed / n * 1000, 4),
    }


def rck_analogy(triples, probes) -> dict:
    from rck.analogy import solve_analogy
    from rck.bulk_ingest import bulk_load_triples
    from rck.knowledge_base import ShardedKnowledgeBase

    kb = ShardedKnowledgeBase(dim=4096, n_shards=64, seed=0)
    bulk_load_triples(kb, triples)

    valid = defaultdict(set)
    for s, r, o in triples:
        valid[(s, r)].add(o)

    t0 = time.perf_counter()
    exact = relation_ok = in_valid_set = 0
    for a, b, c, d, r_true in probes:
        res = solve_analogy(kb, a, b, c)
        answer = getattr(res, "answer", None)
        inferred = getattr(res, "relation", None)
        if inferred == r_true:
            relation_ok += 1
        if answer == d:
            exact += 1
        if answer is not None and answer in valid[(c, r_true)]:
            in_valid_set += 1
    elapsed = time.perf_counter() - t0

    n = len(probes)
    return {
        "system": "rck (HRR vector algebra)",
        "relation_accuracy": round(relation_ok / n, 4),
        "exact_answer_accuracy": round(exact / n, 4),
        "valid_set_accuracy": round(in_valid_set / n, 4),
        "total_s": round(elapsed, 4),
        "ms_per_probe": round(elapsed / n * 1000, 4),
    }


# ---------------------------------------------------------------------------
# Experiment B: federated merge
# ---------------------------------------------------------------------------

def split_triples(triples):
    """Party A gets the even-indexed facts, party B the odd."""
    return triples[0::2], triples[1::2]


def rename(triples, suffix):
    """Same entities, different identifiers -- the case entity alignment
    actually exists to solve."""
    return [(f"{s}{suffix}", r, f"{o}{suffix}") for s, r, o in triples]


def merge_dict(a_triples, b_triples, probes) -> dict:
    t0 = time.perf_counter()
    idx = defaultdict(list)
    for s, r, o in a_triples:
        idx[(s, r)].append(o)
    for s, r, o in b_triples:          # the entire merge
        idx[(s, r)].append(o)
    elapsed = time.perf_counter() - t0

    hits = sum(1 for s, r, o in probes if o in idx.get((s, r), []))
    return {
        "system": "dict merge",
        "merge_s": round(elapsed, 5),
        "post_merge_recall": round(hits / len(probes), 4),
    }


def merge_rck(a_triples, b_triples, probes) -> dict:
    from rck.conscious_agent import ConsciousAgent

    a = ConsciousAgent(expected_facts=len(a_triples) * 3)
    for s, r, o in a_triples:
        a.tell(s, r, o)
    b = ConsciousAgent(expected_facts=len(b_triples) * 3)
    for s, r, o in b_triples:
        b.tell(s, r, o)

    t0 = time.perf_counter()
    a.merge_from(b)
    elapsed = time.perf_counter() - t0

    hits = 0
    for s, r, o in probes:
        res = a.ask_with_idk({"S": s, "R": r}, "O")
        alts = {sym for sym, _ in getattr(res, "alternatives", [])}
        if res.top_symbol == o or o in alts:
            hits += 1
    return {
        "system": "rck bundle-sum merge",
        "merge_s": round(elapsed, 5),
        "post_merge_recall": round(hits / len(probes), 4),
    }


# ---------------------------------------------------------------------------

def main() -> None:
    if not KB_PATH.exists():
        raise SystemExit(f"missing {KB_PATH}")
    triples = load_triples()
    results = {"kb_triples": len(triples), "analogy": [], "merge": {}}

    print(f"KB: {len(triples)} triples\n")

    print("=== Experiment A: analogy, a:b::c:? ===")
    probes = build_analogy_probes(triples)
    print(f"  {len(probes)} probes, identical construction to analogy_study.py")
    for fn in (symbolic_analogy, rck_analogy):
        row = fn(triples, probes)
        row["n_probes"] = len(probes)
        results["analogy"].append(row)
        print(f"  {row['system']:28s} relation={row['relation_accuracy']:.4f} "
              f"exact={row['exact_answer_accuracy']:.4f} "
              f"valid-set={row['valid_set_accuracy']:.4f} "
              f"{row['ms_per_probe']:.3f}ms/probe")

    print("\n=== Experiment B: federated merge ===")
    a_tri, b_tri = split_triples(triples)

    # B1: shared naming. Both parties use the same identifiers.
    probes_shared = b_tri[:200]
    print(f"  B1 shared naming, {len(probes_shared)} probes from party B:")
    b1 = [merge_dict(a_tri, b_tri, probes_shared),
          merge_rck(a_tri, b_tri, probes_shared)]
    for row in b1:
        print(f"     {row['system']:24s} merge={row['merge_s']:.5f}s "
              f"recall={row['post_merge_recall']:.4f}")

    # B2: divergent naming -- the case entity alignment exists for.
    b_renamed = rename(b_tri, "_v2")
    probes_renamed = b_renamed[:200]
    print(f"  B2 divergent naming (party B uses '<entity>_v2'), "
          f"{len(probes_renamed)} probes:")
    b2 = [merge_dict(a_tri, b_renamed, probes_renamed),
          merge_rck(a_tri, b_renamed, probes_renamed)]
    for row in b2:
        print(f"     {row['system']:24s} merge={row['merge_s']:.5f}s "
              f"recall={row['post_merge_recall']:.4f}")

    # B3: does either side RESOLVE the divergence? Probe party A's names
    # after merging B's renamed copy -- an aligner would unify them.
    probes_cross = [(f"{s}_v2", r, o) for s, r, o in a_tri[:200]]
    print(f"  B3 cross-name resolution (ask for A's facts under B's names), "
          f"{len(probes_cross)} probes:")
    b3 = [merge_dict(a_tri, b_renamed, probes_cross),
          merge_rck(a_tri, b_renamed, probes_cross)]
    for row in b3:
        print(f"     {row['system']:24s} recall={row['post_merge_recall']:.4f}")

    results["merge"] = {"shared_naming": b1, "divergent_naming": b2,
                        "cross_name_resolution": b3}
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
