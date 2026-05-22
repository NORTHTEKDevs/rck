"""Full v1.4 conversational demo.

Demonstrates everything an LLM is asked to do, evaluated against
ground-truth retrieval + chain inference:

  1. Factual QA over a 450+ fact KB.
  2. Boolean questions: "is X a Y?"
  3. Enumeration: "list all mammals"
  4. Comparison: "is X bigger than Y?"
  5. Multi-hop inference: (dog isa mammal) + (mammal has fur) -> dog has fur.
  6. Multi-turn dialogue with pronoun resolution: "What about it?"
  7. Theory of mind (Bob believes X).
  8. Self-awareness ("who are you?", "what do you know?").
  9. Calibrated confidence ("I know" / "I think" / "I don't know").

Run:
    python -m examples.conversational_demo
"""
from __future__ import annotations

from pathlib import Path

from rck.conscious_agent import ConsciousAgent


SECTION = "=" * 64


def banner(s: str) -> None:
    print(f"\n{SECTION}\n {s}\n{SECTION}")


def Q(ai: ConsciousAgent, q: str) -> str:
    res = ai.ask(q)
    verb = res.get("verbal", "(no answer)")
    src = res.get("source", "?")
    print(f"  you> {q}")
    print(f"  rck> {verb}")
    print(f"       [source={src}]")
    if res.get("reasoning"):
        print(f"       chain: {res['reasoning']}")
    return verb


def main() -> int:
    banner("RCK v1.4 -- CONVERSATIONAL DEMO")
    ai = ConsciousAgent(dim=4096, n_shards=64, seed=0)
    n = ai.load_jsonl(Path("data/commonsense_kb.jsonl"))
    print(f"\nLoaded {n} facts. Total (incl self-model): {ai.knowledge.size()}")

    banner("(1) Direct factual QA")
    Q(ai, "What color is the sky?")
    Q(ai, "What is the capital of Japan?")
    Q(ai, "Who wrote Hamlet?")
    Q(ai, "What is the field of Einstein?")

    banner("(2) Multi-hop chain inference (NOT directly stored)")
    Q(ai, "What does the dog have?")           # via dog isa mammal, mammal has fur
    Q(ai, "What does the cat have?")
    Q(ai, "What is the locatedin of paris?")   # via paris locatedin france
    Q(ai, "What is the continent of paris?")   # multi-hop: paris -> france -> europe

    banner("(3) Boolean questions")
    Q(ai, "Is a dog a mammal?")
    Q(ai, "Is a snake a mammal?")
    Q(ai, "Does the elephant have tusks?")
    Q(ai, "Does the bee have stinger?")
    Q(ai, "Is the sky red?")

    banner("(4) Enumeration questions")
    Q(ai, "What are mammals?")
    Q(ai, "What are fruits?")
    Q(ai, "list all insect")

    banner("(5) Comparison questions")
    Q(ai, "Is an elephant bigger than a mouse?")
    Q(ai, "Is a cat bigger than a whale?")
    Q(ai, "Is a tree bigger than a house?")

    banner("(6) Multi-turn dialogue with pronoun + topic inheritance")
    Q(ai, "What color is the sky?")
    Q(ai, "What about the grass?")            # topic = color (last relation)
    Q(ai, "What about the rose?")
    Q(ai, "Where does it live?")               # ambiguous; resolves to last entity
    Q(ai, "What does the dog have?")
    Q(ai, "What about it?")                    # 'it' should resolve to dog

    banner("(7) Theory of mind")
    ai.tell_belief("bob",   "france", "capital", "lyon")
    ai.tell_belief("alice", "france", "capital", "paris")
    ai.tell_belief("carol", "sky",    "color",   "green")
    for believer in ("bob", "alice", "carol"):
        for s, r in (("france", "capital"), ("sky", "color")):
            res = ai.what_does_x_think(believer, s, r)
            if res.get("answer"):
                print(f"  {res['verbal']}  truth={res.get('ground_truth')}  match={res.get('matches_truth')}")

    banner("(8) Self-awareness")
    print(f"  you> who are you?")
    print(f"  rck> {ai.who_am_i()}")
    Q(ai, "What is the version of the rck?")

    banner("(9) Calibrated 'I don't know'")
    Q(ai, "What color is the alien?")
    Q(ai, "Who invented teleportation?")

    banner("(10) Introspection")
    # Take a couple of LM steps so think() has something to summarise.
    for c in "hello":
        ai.introspect_buf.record(ai.lm.step(c, learn=True))
    print(ai.think())

    banner("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
