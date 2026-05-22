"""RCK v3.0 -- comprehensive demo of every capability.

Exercises:
  1. Bulk KB load (1000+ facts across 12 domains)
  2. Open IE bootstrap from raw text
  3. Factual / boolean / enumeration / comparison / multi-step
  4. Multi-hop chain inference
  5. Numerical reasoning + arithmetic tools
  6. Temporal reasoning
  7. Spatial reasoning (containment chains)
  8. Synonym normalization
  9. Multi-turn dialogue with pronouns + topic
 10. Theory of mind (separate belief KB)
 11. Self-model + introspection
 12. Multi-sentence describe()
 13. Think-aloud chain-of-thought
 14. Personality / tone styling
 15. Tool/action registry (calculator, time, length)
 16. User-correction-driven self-improvement
 17. Session persistence
 18. Calibrated confidence verbalisation

Run:
    python -m examples.v3_full_demo
"""
from __future__ import annotations

import time
from pathlib import Path

from rck.bulk_ingest import bulk_load_jsonl
from rck.conscious_agent import ConsciousAgent


def banner(s: str) -> None:
    print(f"\n{'=' * 64}\n {s}\n{'=' * 64}")


def Q(ai: ConsciousAgent, q: str) -> dict:
    res = ai.ask(q)
    verb = res.get("verbal", "(empty)")
    src = res.get("source", "?")
    print(f"  > {q}")
    print(f"    {verb}")
    print(f"    [src={src}]")
    return res


def main() -> int:
    banner("RCK v3.0 -- COMPREHENSIVE DEMO")

    # ---- bulk load --------------------------------------------------------
    ai = ConsciousAgent(dim=4096, n_shards=128, seed=0)
    t0 = time.time()
    s1 = bulk_load_jsonl(ai.knowledge, "data/commonsense_kb.jsonl", symmetrize=True)
    s2 = bulk_load_jsonl(ai.knowledge, "data/extended_kb.jsonl", symmetrize=True)
    print(f"\nLoaded {s1['facts'] + s2['facts']} facts "
          f"(+ {s1['symmetrized'] + s2['symmetrized']} symmetrized) in {time.time() - t0:.2f}s.")
    print(f"Total facts: {ai.knowledge.size():,}")

    # ---- 1. Direct factual ----
    banner("1. Direct factual + multi-hop chain inference")
    Q(ai, "What color is the sky?")
    Q(ai, "What is the capital of France?")
    Q(ai, "What is the symbol of gold?")
    Q(ai, "What is the continent of paris?")          # multi-hop
    Q(ai, "What does saturn orbit?")

    # ---- 2. Boolean / enum / comparison ----
    banner("2. Boolean / enumeration / comparison")
    Q(ai, "Is a dog a mammal?")
    Q(ai, "Is a dog an animal?")                       # transitive isa
    Q(ai, "Is the sky red?")                           # contradicted
    Q(ai, "What are mammals?")
    Q(ai, "What are planets?")
    Q(ai, "Is an elephant bigger than a mouse?")

    # ---- 3. Numerical reasoning ----
    banner("3. Numerical reasoning + arithmetic")
    Q(ai, "What is 7 + 3")
    Q(ai, "what is 100 / 4")
    Q(ai, "what is 25 * 4")

    # ---- 4. Temporal ----
    banner("4. Temporal reasoning")
    Q(ai, "what comes before march")
    Q(ai, "what comes after december")
    Q(ai, "what comes before monday")

    # ---- 5. Tool use ----
    banner("5. Tool / action registry")
    Q(ai, "how long is hello world")
    res = ai.ask("what time is it")
    print(f"  > what time is it\n    {res.get('verbal') or res}\n    [src={res.get('source')}]")

    # ---- 6. Synonyms ----
    banner("6. Synonym normalization (hue, writer, ...)")
    Q(ai, "What hue is the sky?")
    Q(ai, "What is the writer of hamlet?")

    # ---- 7. Multi-turn dialogue ----
    banner("7. Multi-turn dialogue with topic inheritance")
    Q(ai, "What color is the sky?")
    Q(ai, "What about the grass?")
    Q(ai, "What about the rose?")

    # ---- 8. Theory of mind ----
    banner("8. Theory of mind")
    ai.tell_belief("bob",   "france", "capital", "lyon")
    ai.tell_belief("alice", "france", "capital", "paris")
    print(ai.what_does_x_think("bob",   "france", "capital").get("verbal"))
    print(ai.what_does_x_think("alice", "france", "capital").get("verbal"))

    # ---- 9. Self-awareness ----
    banner("9. Self-awareness")
    print("  > who are you?")
    print(f"    {ai.who_am_i()}")

    # ---- 10. Multi-sentence describe ----
    banner("10. Multi-sentence describe()")
    for ent in ("elephant", "earth", "gold"):
        print(f"\n  describe('{ent}'):")
        print("    " + ai.describe(ent).replace(". ", ".\n    "))

    # ---- 11. Think-aloud chain-of-thought ----
    banner("11. Think-aloud chain-of-thought")
    res = ai.ask("What is the continent of paris?", think_aloud=True)
    print(f"  > {ai._last_query}\n    {res['verbal']}")
    if res.get("think_aloud"):
        print(f"    thought: {res['think_aloud']}")

    # ---- 12. User correction ----
    banner("12. User-correction-driven self-improvement")
    # Pretend the user wants to teach us "the alien is green"
    res = ai.ask("Actually the alien is green.")
    print(f"  > Actually the alien is green.\n    {res.get('verbal')}\n    [src={res.get('source')}]")
    # Now ask what color the alien is.
    Q(ai, "What is the alien?")

    # Now correct a wrong belief.
    Q(ai, "What color is the rose?")
    res = ai.ask("Actually the rose is white, not red.")
    print(f"  > Actually the rose is white, not red.\n    {res.get('verbal')}\n    [src={res.get('source')}]")
    Q(ai, "What color is the rose?")

    # ---- 13. Open IE bootstrap on the fly ----
    banner("13. Open IE: feed natural-language text, then query it")
    info = ai.ingest_text(
        "The dragon is a creature. The dragon has scales. "
        "The dragon breathes fire. The dragon lives in a cave."
    )
    print(f"  ingested {info['facts']} facts from 4 sentences.")
    Q(ai, "What is the dragon?")
    Q(ai, "What does the dragon have?")

    # ---- 14. Personality styling ----
    banner("14. Personality / tone")
    for tone in ("formal", "casual", "concise", "curious"):
        ai.personality.tone = tone
        out = ai.personality.render_know("blue")
        print(f"  tone={tone:>8s}: {out}")
    ai.personality.tone = "default"

    # ---- 15. State summary ----
    banner("15. Final state")
    s = ai.state()
    for k in ("version", "facts", "beliefs", "n_shards", "dialogue_turns"):
        print(f"  {k:>16s}: {s.get(k)}")

    banner("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
