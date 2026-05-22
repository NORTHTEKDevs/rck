"""Headline demo for v1.3: scale, self-model, theory-of-mind, introspection.

Loads ~370 ConceptNet-style common-sense facts, installs RCK's self-model,
then runs a 60-question evaluation across the full common-sense domain.
Demonstrates:

  - Wide-domain QA across colors, capitals, animals, materials, uses,
    locations, causes, authors, scientists, foods, parts.
  - Self-knowledge: "What are you?" / "What do you know?" / "What can you do?"
  - Theory of mind: "Bob thinks Lyon is the capital of France" stored
    separately from ground truth.
  - Meta-cognitive verbalisation: "I know X" / "I'm not sure" / "I don't know"
    depending on confidence category.
  - Introspection: think() reports current internal state.

Run:
    python -m examples.conscious_demo
"""
from __future__ import annotations

from pathlib import Path

from rck.conscious_agent import ConsciousAgent


# ---- evaluation set --------------------------------------------------------

QA = [
    # Colors
    ("What color is the sky?",            "blue"),
    ("What color is the grass?",          "green"),
    ("What color is the rose?",           "red"),
    ("What color is the lemon?",          "yellow"),
    ("What color is the carrot?",         "orange"),
    ("What color is the strawberry?",     "red"),
    # Capitals
    ("What is the capital of France?",    "paris"),
    ("What is the capital of Japan?",     "tokyo"),
    ("What is the capital of Brazil?",    "brasilia"),
    ("What is the capital of Egypt?",     "cairo"),
    ("What is the capital of Iran?",      "tehran"),
    ("What is the capital of Thailand?",  "bangkok"),
    ("What is the capital of Norway?",    "oslo"),
    ("What is the capital of Greece?",    "athens"),
    # Has-parts: multi-valued so we accept any of these as correct.
    ("What does the cat have?",           {"fur", "tail", "whiskers", "claws"}),
    ("What does the elephant have?",      {"tusks", "trunk", "ears"}),
    ("What does the zebra have?",         {"stripes", "hooves"}),
    ("What does the spider have?",        {"legs", "webs"}),
    ("What does the bee have?",           {"wings", "stinger"}),
    # Categories / isa
    ("What is the kind of the dog?",      "mammal"),
    ("What is the kind of the snake?",    "reptile"),
    ("What is the kind of the ant?",      "insect"),
    # Authors
    ("Who wrote Hamlet?",                 "shakespeare"),
    ("Who wrote 1984?",                   "orwell"),
    ("Who wrote Iliad?",                  "homer"),
    ("Who wrote Ulysses?",                "joyce"),
    ("Who wrote Metamorphosis?",          "kafka"),
    # Scientists / fields
    ("What is the field of Einstein?",    "physics"),
    ("What is the field of Darwin?",      "biology"),
    ("What is the field of Turing?",      "computerscience"),
    # Causes
    ("What is the causes of the rain?",   "wetness"),
    ("What is the causes of the fire?",   "heat"),
    # Materials
    ("What is the madeof of the book?",   "paper"),
    ("What is the madeof of the table?",  "wood"),
    ("What is the madeof of the window?", "glass"),
    # Uses
    ("What is the usedfor of the knife?", "cutting"),
    ("What is the usedfor of the pen?",   "writing"),
    ("What is the usedfor of the car?",   "driving"),
    # Foods
    ("What is the category of the apple?",   "fruit"),
    ("What is the category of the carrot?",  "vegetable"),
    ("What is the category of the rice?",    "grain"),
    ("What is the category of the chicken?", "meat"),
    # Continents
    ("What is the continent of France?",  "europe"),
    ("What is the continent of Japan?",   "asia"),
    ("What is the continent of Egypt?",   "africa"),
    ("What is the continent of Brazil?",  "southamerica"),
    ("What is the continent of Australia?", "oceania"),
    # Parts
    ("What is the partof of the wheel?",  "car"),
    ("What is the partof of the leaf?",   "tree"),
    ("What is the partof of the page?",   "book"),
    # Locations
    ("What is the locatedin of the kitchen?", "house"),
    ("What is the locatedin of the fish?",    "water"),
    ("What is the locatedin of the star?",    "sky"),
    # Numbers
    ("What is the value of the three?",   "3"),
    ("What is the value of the seven?",   "7"),
    # Unknowns -- model should soft-reject
    ("What color is the alien?",          None),
    ("Who wrote the moonfish?",           None),
    ("What is the capital of mars?",      None),
]


def main() -> int:
    print("=" * 64)
    print(" RCK v1.3 -- common-sense + self-model + introspection")
    print("=" * 64)

    ai = ConsciousAgent(dim=4096, n_shards=64, seed=0)

    # Bulk load the canned common-sense KB.
    path = Path("data/commonsense_kb.jsonl")
    n_loaded = ai.load_jsonl(path)
    print(f"\nLoaded {n_loaded} common-sense facts into {ai.n_shards} shards.")
    util = ai.knowledge.utilization()
    print(f"  shards: max={util['max_shard']} avg={util['avg_shard']:.1f} top4={util['histogram_top4']}")
    s = ai.state()
    print(f"  total facts (incl self-model): {s['facts']}")

    # ---- self awareness ---------------------------------------------------
    print("\n[1] Self-awareness:")
    print(f"  Q: who are you?")
    print(f"  A: {ai.who_am_i()}")
    print()
    for q in ["What is the version of the rck?",
              "What does the rck use?",
              "What does the rck cannot?"]:
        res = ai.ask(q)
        print(f"  Q: {q}")
        print(f"  A: {res['verbal']}  ({res['source']}, conf={res['confidence']:.2f})")

    # ---- common-sense QA -------------------------------------------------
    print("\n[2] Common-sense QA evaluation:")
    known_correct = 0; known_total = 0
    unknown_correct = 0; unknown_total = 0
    misses = []
    for q, expected in QA:
        res = ai.ask(q)
        if expected is None:
            unknown_total += 1
            # Soft-reject means category is unknown / guess.
            confident_wrong = (res["category"] in {"know", "think"})
            ok = not confident_wrong
            unknown_correct += ok
            if not ok and len(misses) < 5:
                misses.append((q, "<unknown expected>", res))
            continue
        known_total += 1
        ans = (res["answer"] or "").lower().strip()
        if isinstance(expected, set):
            valid = {e.lower() for e in expected}
            ok = ans in valid
        else:
            ok = ans == expected.lower()
        known_correct += ok
        if not ok and len(misses) < 8:
            misses.append((q, str(expected), res))

    print(f"  known questions:  {known_correct}/{known_total}  ({known_correct / known_total:.1%})")
    print(f"  unknown soft-no:  {unknown_correct}/{unknown_total}  ({unknown_correct / unknown_total:.1%})")
    total = known_total + unknown_total
    overall = known_correct + unknown_correct
    print(f"  overall:          {overall}/{total}  ({overall / total:.1%})")
    if misses:
        print("\n  misses (first 8):")
        for q, exp, res in misses[:8]:
            print(f"    '{q}' exp={exp!r} got='{res['answer']}' "
                  f"(source={res['source']}, conf={res['confidence']:.2f})")

    # ---- theory of mind ---------------------------------------------------
    print("\n[3] Theory of mind:")
    ai.tell_belief("bob",   "france", "capital", "lyon")
    ai.tell_belief("alice", "france", "capital", "paris")
    ai.tell_belief("carol", "sky",    "color",   "green")
    for believer, subject, relation in [
        ("bob", "france", "capital"),
        ("alice", "france", "capital"),
        ("carol", "sky", "color"),
    ]:
        res = ai.what_does_x_think(believer, subject, relation)
        truth = res.get("ground_truth")
        print(f"  {res['verbal']}  "
              f"(ground truth: {truth}, matches: {res.get('matches_truth')})")

    # ---- meta-cognitive verbalisation ------------------------------------
    print("\n[4] Meta-cognitive verbalisation:")
    for q in [
        "What color is the sky?",
        "What is the capital of France?",
        "What color is the alien?",
        "Who invented teleportation?",
    ]:
        res = ai.ask(q)
        print(f"  Q: {q}")
        print(f"  A: {res['verbal']}  [{res['category']}, conf={res['confidence']:.2f}]")

    # ---- introspection ---------------------------------------------------
    print("\n[5] Introspection:")
    # Drive a couple of LM steps so think() has something to report on.
    for ch in "hello":
        tr = ai.lm.step(ch, learn=True)
        ai.introspect_buf.record(tr)
    print(ai.think())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
