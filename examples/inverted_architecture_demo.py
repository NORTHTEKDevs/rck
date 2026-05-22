"""v4.0 -- the Inverted Architecture demo.

Demonstrates the headline v4 claim: knowledge lives in the HRR store,
the language layer ONLY polishes surface form. Every claim in the output
is traceable to a structured retrieval.

Run:
    python -m examples.inverted_architecture_demo
"""
from __future__ import annotations

from pathlib import Path

from rck.bulk_ingest import bulk_load_jsonl
from rck.conscious_agent import ConsciousAgent
from rck.inverted_lm import InvertedLM, RuleBasedPolisher


def banner(s: str) -> None:
    print(f"\n{'=' * 64}\n {s}\n{'=' * 64}")


def main() -> int:
    banner("RCK v4.0 -- INVERTED ARCHITECTURE")
    print(" Knowledge: HRR store (99% of the work).")
    print(" Fluency:   small polisher (1% of the work).")
    print(" Total compute to build: ~$0 (no LM training yet).")

    agent = ConsciousAgent(dim=4096, n_shards=128, seed=0)
    stats1 = bulk_load_jsonl(agent.knowledge, "data/commonsense_kb.jsonl",
                              symmetrize=True)
    stats2 = bulk_load_jsonl(agent.knowledge, "data/extended_kb.jsonl",
                              symmetrize=True)
    print(f"\nLoaded {stats1['facts'] + stats2['facts']:,} facts. "
          f"Total KB size: {agent.knowledge.size():,}")

    inv = InvertedLM(agent=agent, polisher=RuleBasedPolisher())

    # ---- side-by-side: draft (template) vs polished (final) ---------------
    banner("Draft vs polished (the polisher's job is surface only)")
    for q in [
        "What color is the sky?",
        "What is the capital of France?",
        "What is the symbol of gold?",
        "Is a dog a mammal?",
    ]:
        d_only = inv.generate(q, polish=False)
        polished = inv.generate(q, polish=True)
        print(f"\n  > {q}")
        print(f"    draft:    {d_only['response']}")
        print(f"    polished: {polished['response']}")

    # ---- multi-fact compositional output ---------------------------------
    banner("Multi-sentence describe() through the polisher")
    for entity in ("elephant", "earth", "gold", "shakespeare"):
        res = inv.describe_entity(entity)
        print(f"\n  describe('{entity}'):")
        print(f"    draft:    {res['draft']}")
        print(f"    polished: {res['response']}")

    # ---- think-aloud through the polisher --------------------------------
    banner("Think-aloud chain-of-thought, polished")
    for q in [
        "What is the continent of paris?",
        "What does the dog have?",
    ]:
        res = inv.think_aloud(q)
        print(f"\n  > {q}")
        print(f"    response: {res['response']}")

    # ---- the hallucination invariant -------------------------------------
    banner("Hallucination invariant -- no fact in the answer that is not in the KB")
    for q in [
        "What color is the alien?",
        "Who wrote BookThatDoesNotExist?",
        "What is the capital of Atlantis?",
    ]:
        res = inv.generate(q)
        print(f"\n  > {q}")
        print(f"    response: {res['response']}")
        print(f"    source:   {res.get('source', 'unknown')}")

    banner("DONE")
    print(" The polisher in v4 is a rule-based stub. v7 will swap in a")
    print(" 50-100M-param distilled transformer trained ONLY on")
    print(" template-rendered drafts. Same architecture, fluent prose.")
    print(" See docs/design/RCK-v4-DEEP-RESEARCH.md for the v4 -> v10 plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
