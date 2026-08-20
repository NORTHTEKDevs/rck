"""The analogy benchmark that our own protocol never ran.

Section 5.10 found a twenty-line symbolic solver beats RCK's vector algebra on
the existing analogy benchmark -- but flagged the reason that result is not
conclusive: the protocol constructs every probe so that `(a, R, b)` is
*already stored*, which makes exact indexing sufficient by design.

This study removes exactly that crutch. For each probe the edge `(a, R, b)` is
**held out of the knowledge base**, so the relation cannot be looked up and must
be *generalised* from the rest of the graph. That is the regime where vector
algebra is supposed to have an advantage, and it is the last measured chance
the HRR substrate has to justify itself.

Both sides get the same held-out KB. Neither can cheat.

Baselines:
  * type-based symbolic -- infer R by which relation's argument types best fit
    (a, b), the obvious non-VSA generalisation strategy.
  * majority-relation    -- always guess the most common relation. The floor.
    If a system cannot beat this, it has learned nothing.

Reproduce:
    python scripts/analogy_generalization_study.py
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "data" / "commonsense_kb.jsonl"
OUT = ROOT / "data" / "analogy_generalization_study.json"


def load_triples():
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


def build_probes(triples, limit=100):
    """(a, b, c, d, R) where (a,R,b) and (c,R,d) are stored, and R has at
    least 3 other pairs so generalisation is actually possible."""
    by_rel = defaultdict(list)
    for s, r, o in triples:
        by_rel[r].append((s, o))
    probes = []
    for r, pairs in by_rel.items():
        unique = list(dict.fromkeys(pairs))
        if len(unique) < 5:            # need enough support to generalise from
            continue
        for i, (a, b) in enumerate(unique[:8]):
            for c, d in unique[i + 1:i + 3]:
                if len({a, b, c, d}) == 4:
                    probes.append((a, b, c, d, r))
        if len(probes) >= limit:
            break
    return probes[:limit]


def majority_baseline(triples, probes) -> dict:
    """The floor: always guess the most common relation, then its first
    object for c. Anything that cannot beat this has learned nothing."""
    common = Counter(r for _, r, _ in triples).most_common(1)[0][0]
    sr_to_o = defaultdict(list)
    for s, r, o in triples:
        sr_to_o[(s, r)].append(o)

    rel_ok = ans_ok = 0
    for a, b, c, d, r_true in probes:
        if common == r_true:
            rel_ok += 1
        got = sr_to_o.get((c, common))
        if got and got[0] == d:
            ans_ok += 1
    n = len(probes)
    return {"system": "majority-relation (floor)",
            "relation_accuracy": round(rel_ok / n, 4),
            "answer_accuracy": round(ans_ok / n, 4),
            "ms_per_probe": 0.0}


def type_symbolic(triples, probes) -> dict:
    """Non-VSA generalisation: score each relation by how well (a, b) fits
    the argument types it is observed with, using the HELD-OUT graph."""
    n = len(probes)
    rel_ok = ans_ok = 0
    t0 = time.perf_counter()

    for a, b, c, d, r_true in probes:
        held = [(s, r, o) for s, r, o in triples
                if not (s == a and r == r_true and o == b)]
        subj_of = defaultdict(set)
        obj_of = defaultdict(set)
        sr_to_o = defaultdict(list)
        for s, r, o in held:
            subj_of[r].add(s)
            obj_of[r].add(o)
            sr_to_o[(s, r)].append(o)

        # Type fit: does `a` appear as a subject of R, and `b` as an object?
        # Break ties toward relations that actually yield an answer for c.
        best, best_score = None, -1.0
        for r in subj_of:
            score = (1.0 if a in subj_of[r] else 0.0) + (1.0 if b in obj_of[r] else 0.0)
            if sr_to_o.get((c, r)):
                score += 0.5
            if score > best_score:
                best, best_score = r, score
        if best == r_true:
            rel_ok += 1
        got = sr_to_o.get((c, best or ""))
        if got and got[0] == d:
            ans_ok += 1

    elapsed = time.perf_counter() - t0
    return {"system": "type-based symbolic",
            "relation_accuracy": round(rel_ok / n, 4),
            "answer_accuracy": round(ans_ok / n, 4),
            "ms_per_probe": round(elapsed / n * 1000, 4)}


def rck_vector(triples, probes) -> dict:
    """RCK's vector algebra, on the same held-out KB."""
    from rck.analogy import solve_analogy
    from rck.bulk_ingest import bulk_load_triples
    from rck.knowledge_base import ShardedKnowledgeBase

    n = len(probes)
    rel_ok = ans_ok = 0
    t0 = time.perf_counter()

    for a, b, c, d, r_true in probes:
        held = [(s, r, o) for s, r, o in triples
                if not (s == a and r == r_true and o == b)]
        kb = ShardedKnowledgeBase(dim=4096, n_shards=64, seed=0)
        bulk_load_triples(kb, held)
        res = solve_analogy(kb, a, b, c)
        if getattr(res, "relation", None) == r_true:
            rel_ok += 1
        if getattr(res, "answer", None) == d:
            ans_ok += 1

    elapsed = time.perf_counter() - t0
    return {"system": "rck (HRR vector algebra)",
            "relation_accuracy": round(rel_ok / n, 4),
            "answer_accuracy": round(ans_ok / n, 4),
            "ms_per_probe": round(elapsed / n * 1000, 4)}


def main() -> None:
    triples = load_triples()
    probes = build_probes(triples)
    print(f"KB: {len(triples)} triples")
    print(f"Probes: {len(probes)}, each with its (a,R,b) edge HELD OUT")
    print("The relation must be generalised, not looked up.\n")

    rows = []
    for fn in (majority_baseline, type_symbolic, rck_vector):
        row = fn(triples, probes)
        row["n_probes"] = len(probes)
        rows.append(row)
        print(f"  {row['system']:28s} relation={row['relation_accuracy']:.4f} "
              f"answer={row['answer_accuracy']:.4f} "
              f"{row['ms_per_probe']:.3f}ms/probe")

    floor = rows[0]["relation_accuracy"]
    print(f"\n  floor (majority relation) = {floor:.4f}")
    for row in rows[1:]:
        verdict = "ABOVE floor" if row["relation_accuracy"] > floor else "AT OR BELOW floor"
        print(f"  {row['system']:28s} -> {verdict}")

    OUT.write_text(json.dumps({"kb_triples": len(triples), "results": rows},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
