"""v5.0 -- the 'beyond LLMs' demo.

Demonstrates five capabilities no LLM can match at any scale:
  1. Knowledge provenance (every fact has source / timestamp / confidence)
  2. Memory hierarchies (working / episodic / procedural)
  3. Counterfactual universes (branch / modify / discard)
  4. Active curiosity (detect knowledge gaps + ask)
  5. Abductive reasoning (effect -> candidate causes)

Run:
    python -m examples.beyond_llm_demo
"""
from __future__ import annotations

from rck.abduction import candidates_for_effect, candidates_for_property, explain
from rck.bulk_ingest import bulk_load_jsonl, bulk_load_triples
from rck.conscious_agent import ConsciousAgent
from rck.curiosity import detect_gaps, detect_global_gaps
from rck.memory_hierarchy import (
    EpisodicMemory, ProceduralMemory, WorkingMemory,
    consolidate_episodic_to_semantic,
)
from rck.provenance import ProvenanceStore
from rck.universes import UniverseManager


def banner(s: str) -> None:
    print(f"\n{'=' * 64}\n {s}\n{'=' * 64}")


def main() -> int:
    banner("RCK v5.0 -- BEYOND LLMs")
    print(" Five structural capabilities no transformer can match.")

    agent = ConsciousAgent(dim=4096, n_shards=64, seed=0)
    bulk_load_jsonl(agent.knowledge, "data/commonsense_kb.jsonl",
                    symmetrize=True)
    bulk_load_jsonl(agent.knowledge, "data/extended_kb.jsonl",
                    symmetrize=True)
    print(f"\nKB loaded: {agent.knowledge.size():,} facts.")

    # ---- 1. Provenance ----
    banner("1. Knowledge provenance -- every fact has audit trail")
    prov = ProvenanceStore()
    prov.store("sky", "color", "blue",
               source="wikipedia.org/sky", tags={"verified"})
    prov.store("sky", "color", "blue",
               source="user_K")  # reinforce
    prov.store("alien", "color", "green",
               source="user_unverified", confidence=0.3)

    for triple in (("sky", "color", "blue"), ("alien", "color", "green")):
        rec = prov.get(*triple)
        print(f"  '{triple[0]} {triple[1]} {triple[2]}' -> {prov.summarize(*triple)}")

    print("\n  Filter by low confidence (could be forgotten):")
    for s, r, o in prov.low_confidence_facts(threshold=0.5):
        print(f"    candidate for cleanup: ({s}, {r}, {o})")

    # ---- 2. Memory hierarchies ----
    banner("2. Memory hierarchies -- working / episodic / procedural")
    wm = WorkingMemory(capacity=4)
    em = EpisodicMemory()
    pm = ProceduralMemory()

    for thought in ["sky is blue", "user asked about colors",
                    "retrieved 'blue'", "responded"]:
        wm.push(thought, salience=1.0)
    print(f"\n  working memory (last {wm.size()}):")
    for it in wm.all():
        print(f"    - {it.content}")

    # Simulate a few conversations.
    em.record("user", "asked", "What is the capital of France?")
    em.record("system", "answered", "Paris")
    em.record("user", "asked", "What is the capital of France?")
    em.record("user", "asked", "What is the capital of France?")
    em.record("user", "told", "I like geography")

    print(f"\n  episodic memory: {em.size()} events")
    pattern = consolidate_episodic_to_semantic(em, threshold=3, kind="asked")
    print("  consolidation (recurring questions worth promoting to semantic):")
    for content, n in pattern:
        print(f"    {n}x: {content!r}")

    pm.store("fact_lookup", "retrieve a single fact from KB",
             ["parse_question", "query_kb", "render_nl"])
    pm.store("multi_hop",   "chain inference across isa parents",
             ["lookup_parent", "lookup_property", "compose"])
    pm.get("fact_lookup").record_use(succeeded=True)
    pm.get("fact_lookup").record_use(succeeded=True)
    pm.get("multi_hop").record_use(succeeded=False)

    print("\n  procedural memory (named programs):")
    for proc in pm.all():
        print(f"    {proc.name:>12} -- used {proc.usage_count}x, "
              f"success {proc.success_rate():.1%}")

    # ---- 3. Counterfactual universes ----
    banner("3. Counterfactual reasoning -- 'what if?' without affecting truth")
    mgr = UniverseManager(kb=agent.knowledge)
    print(f"  ground truth: capital of france = "
          f"{mgr.root().answer('france', 'capital')[0]}")

    branch = mgr.branch("what_if_germany")
    branch.tell("germany", "capital", "berlin_alternate")
    branch.forget("germany", "capital", "berlin")
    print(f"  in branch:    capital of germany = "
          f"{branch.answer('germany', 'capital')[0]}")
    branch.discard()
    print(f"  after discard: capital of germany = "
          f"{mgr.root().answer('germany', 'capital')[0]}")
    print("  -> ground truth was never modified.")

    # ---- 4. Curiosity / gap detection ----
    banner("4. Active curiosity -- detect what RCK doesn't know that it should")
    # We need a small ad-hoc KB where some entity has a gap.
    small_kb = agent.knowledge
    # Find global gaps across the loaded KB.
    gaps = detect_global_gaps(small_kb, sample_size=30,
                              min_agreement=0.4, min_siblings=3)
    print(f"\n  found {len(gaps)} candidate gaps. Top 5:")
    for g in gaps[:5]:
        print(f"    [{g.agreement:.0%} of {g.sibling_count} siblings have it] "
              f"-> {g.question}")

    # ---- 5. Abductive reasoning ----
    banner("5. Abduction -- given an effect, find candidate causes")
    # Add a tiny causal subgraph.
    bulk_load_triples(small_kb, [
        ("rain", "causes", "wetness"),
        ("sweat", "causes", "wetness"),
        ("spill", "causes", "wetness"),
    ], symmetrize=False)

    print("  observed: wetness")
    for c in candidates_for_effect(small_kb, "causes", "wetness", top_k=5):
        print(f"    candidate cause: {c.cause}  (cosine {c.confidence:.2f})")

    res = explain(small_kb, "wetness")
    print(f"\n  explain('wetness') -> {res['verbal']}")

    print("\n  observed: 'feathers' (what kind of creature could have these?)")
    for c in candidates_for_property(small_kb, "has", "feathers", top_k=5):
        print(f"    candidate kind: {c.cause}  (cosine {c.confidence:.2f})")

    banner("DONE")
    print(" Each of these 5 capabilities is structurally inaccessible to")
    print(" LLMs. Together they make RCK a different category of system --")
    print(" not just a cheaper LLM, but a cognitive OS where LLMs are")
    print(" single-shot generative functions.")
    print()
    print(" See docs/design/RCK-v5-BEYOND-LLM.md for the full argument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
