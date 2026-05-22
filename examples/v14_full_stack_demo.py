"""End-to-end v14 demo: load a real KB and exercise every major
reasoning capability built in this session.

Shows in order:
  1. Direct retrieval with IDK detection (ask_with_idk)
  2. Calibrated retrieval with provenance discount (calibrated_ask)
  3. Chain discovery + walking (discover + reason)
  4. Cascading chain induction
  5. Rule extraction + composition
  6. Cascading rule instantiation
  7. Analogy with chain fallback
  8. Set reasoning (intersect)
  9. Contradiction detection + belief revision
 10. Explain-why on a derived fact

Run:
    python examples/v14_full_stack_demo.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rck.bulk_ingest import bulk_load_triples
from rck.conscious_agent import ConsciousAgent


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main() -> int:
    print("RCK v14 full-stack demo on the commonsense KB")
    print("=" * 70)

    triples = []
    with open("data/commonsense_kb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            triples.append((r["s"], r["r"], r["o"]))

    agent = ConsciousAgent(
        dim=4096, expected_facts=len(triples) * 2,
        install_self=False,
    )
    bulk_load_triples(agent.knowledge, triples)
    # Mark these as user-provided in provenance.
    for s, r, o in triples:
        agent.provenance.store(s, r, o, source="user")
    print(f"Loaded {agent.knowledge.size()} facts in {agent.n_shards} shards")

    # ---- 1. ask_with_idk ----
    section("1. Direct retrieval with IDK detection")
    for s, r in [("dog", "isa"), ("dog", "has"), ("qux", "isa")]:
        res = agent.ask_with_idk({"S": s, "R": r}, "O")
        print(f"  ({s},{r},?) -> {res.state.value:10s} {res.top_symbol!r:15s} "
              f"score={res.top_score:.3f}")

    # ---- 2. Calibrated retrieval ----
    section("2. Calibrated retrieval with provenance discount")
    rows = agent.calibrated_ask({"S": "dog", "R": "isa"}, "O", top_k=3)
    for r in rows:
        print(f"  {r.symbol!r:15s} raw={r.raw_score:.3f}  "
              f"calibrated={r.calibrated_score:.3f}  source={r.source}")

    # ---- 3. Chain discovery + walk ----
    section("3. Chain discovery + walking")
    t0 = time.perf_counter()
    spec = agent.discover("leaf", "forest")
    if spec:
        ans = agent.reason("leaf", spec["relations"], directions=spec["directions"])
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  leaf -> forest via {' -> '.join(spec['relations'])}  "
              f"answer={ans['answer']}  conf={ans['confidence']:.3f}  "
              f"[{elapsed:.0f}ms]")

    # ---- 4. Cascading chain induction ----
    section("4. Cascading chain induction")
    from rck.cascading_induction import cascade_induct
    from rck.chain_induction import InductionPolicy
    pre = agent.knowledge.size()
    casc = cascade_induct(
        agent.knowledge, max_rounds=2, probes_per_round=40,
        policy=InductionPolicy(min_confidence=0.15),
        skills=agent.skills, provenance=agent.provenance,
    )
    print(f"  rounds={len(casc.rounds)}  verified={casc.total_verified}  "
          f"kb {pre} -> {agent.knowledge.size()}")

    # ---- 5. Rule extraction + composition ----
    section("5. Rule extraction + composition")
    rules = agent.extract_rules()
    print(f"  Extracted {rules.size()} rules. Top 3:")
    for rule in rules.top_rules(n=3):
        print(f"    [{rule.support}x conf={rule.confidence:.2f}] {rule.verbalize()}")
    composed = agent.compose_rules()
    print(f"  Composed {len(composed)} new rules from pairwise composition.")
    for c in composed[:3]:
        print(f"    -> {c.verbalize()}")

    # ---- 6. Cascading rule instantiation ----
    section("6. Cascading rule instantiation")
    pre = agent.knowledge.size()
    rc = agent.cascade_instantiate_rules(max_rounds=3)
    print(f"  rounds={len(rc.rounds)}  verified={rc.total_verified()}  "
          f"kb {pre} -> {agent.knowledge.size()}")

    # ---- 7. Analogy ----
    section("7. Analogy with chain fallback")
    pairs = [
        ("dog", "mammal", "cat"),
        ("france", "paris", "germany"),
        ("leaf", "forest", "branch"),
    ]
    for a, b, c in pairs:
        res = agent.analogy(a, b, c)
        print(f"  {a}:{b}::{c}:{res.answer}  via={res.via}  "
              f"relation={res.relation}  joint={res.joint_score():.3f}")

    # ---- 8. Set reasoning ----
    section("8. Set reasoning (intersect)")
    # Find subjects that are mammals AND have fur.
    rows = agent.intersect([
        {"R": "isa", "O": "mammal"},
        {"R": "has", "O": "fur"},
    ], "S", top_k=10)
    print(f"  Mammals that have fur: {[str(r.symbol) for r in rows[:5]]}")

    # ---- 9. Contradiction + revision ----
    section("9. Contradiction detection + belief revision")
    # Inject a contradiction.
    agent.tell("fish", "isa", "animal")  # user fact
    agent.knowledge.store({"S": "fish", "R": "isa", "O": "vegetable"})
    agent.provenance.store("fish", "isa", "vegetable", source="induced",
                            tags={"induced", "via_3_hops"})
    conflicts = agent.detect_conflicts(subjects=["fish"])
    print(f"  Detected {len(conflicts)} conflict(s) on fish.")
    for c in conflicts[:3]:
        print(f"    {c.verbalize()}")
    plans = agent.resolve_conflicts(subjects=["fish"], apply=True)
    for p in plans[:3]:
        print(f"    -> {p.verbalize()}")

    # ---- 10. Explain-why ----
    section("10. Explain-why on a derived fact")
    if rc.induced_facts:
        f = rc.induced_facts[0]
        node = agent.explain_why(f.subject, f.relation, f.obj)
        print(node.verbalize())

    print("\n" + "=" * 70)
    print(" Done.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
