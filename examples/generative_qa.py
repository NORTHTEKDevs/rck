"""The real generative-AI demo: tell RCK facts, then ask questions.

Trains a GenerativeRCK on a small world-knowledge corpus, then asks 25+
questions and reports per-question correctness. Demonstrates:

  - Knowledge ingestion from natural-language "The X is Y." sentences.
  - Compositional retrieval ("what color is the sky" -> "blue").
  - Generalisation over unseen relations.
  - Free-form fallback when the question has no stored fact.

Run:
    python -m examples.generative_qa
"""
from __future__ import annotations

from pathlib import Path

from rck.generative import GenerativeRCK


# Questions + expected answers. We grade against the structured-retrieval
# answer; the generative fallback is graded as "soft" (not counted toward
# accuracy, but shown for transparency).
EVAL = [
    ("What color is the sky?",            "blue"),
    ("What color is the grass?",          "green"),
    ("What color is the sun?",            "yellow"),
    ("What color is the rose?",           "red"),
    ("What color is the snow?",           "white"),
    ("What color is the lemon?",          "yellow"),
    ("What is the capital of France?",    "paris"),
    ("What is the capital of Germany?",   "berlin"),
    ("What is the capital of Japan?",     "tokyo"),
    ("What is the capital of Russia?",    "moscow"),
    ("What is the capital of Brazil?",    "brasilia"),
    ("What is the capital of Canada?",    "ottawa"),
    ("Where does Alice live?",            "paris"),
    ("Where does Bob live?",              "berlin"),
    ("Where does Eve live?",              "cairo"),
    ("Who wrote Hamlet?",                 "shakespeare"),
    ("Who wrote LordOfTheRings?",         "tolkien"),
    ("Who wrote 1984?",                   "orwell"),
    ("What does the cat have?",           "fur"),
    ("What does the bird have?",          "feathers"),
    ("What does the elephant have?",      "tusks"),
    ("What does the zebra have?",         "stripes"),
    ("What is the size of the elephant?", "huge"),
    ("What is the size of the mouse?",    "tiny"),
    ("What is the age of Alice?",         "30"),
    ("What is the age of Carol?",         "40"),
    # Unknown -- should NOT confidently answer:
    ("What color is the elephant?",       None),
    ("Who wrote nothing?",                None),
]


def main() -> int:
    print("=" * 64)
    print(" RCK GENERATIVE QA")
    print("=" * 64)
    g = GenerativeRCK(dim=4096, seed=0)

    text = Path("data/world_knowledge.txt").read_text(encoding="utf-8")
    info = g.ingest(text)
    print(f"\ningested: tokens={info['tokens_seen']:,}  "
          f"facts_extracted={info['new_facts']}  "
          f"codebook={g.codebook.size()}  memory={g.memory.size()}")

    print("\n--- evaluation ----")
    structured_correct = 0; structured_total = 0
    unknown_correct = 0; unknown_total = 0
    soft_examples = []

    for q, expected in EVAL:
        res = g.ask(q)
        if expected is None:
            # The model must NOT confidently answer.
            unknown_total += 1
            confident = res["source"].startswith("structured") and res["confidence"] > 0.2
            ok = not confident
            unknown_correct += ok
            mark = "OK  " if ok else "WRONG"
            print(f"  [{mark}] '{q}'")
            print(f"          source={res['source']}  conf={res['confidence']:.2f}  ans='{res['answer']}'")
        else:
            structured_total += 1
            answered = (res["answer"] or "").lower().strip()
            ok = answered == expected.lower()
            structured_correct += ok
            mark = "OK  " if ok else "MISS"
            print(f"  [{mark}] '{q}' -> '{res['answer']}'   (expected '{expected}', "
                  f"source={res['source']}, conf={res['confidence']:.2f})")
            if not ok:
                soft_examples.append((q, expected, res))

    print("\n--- summary ----")
    if structured_total:
        print(f"  known questions:     {structured_correct}/{structured_total}  "
              f"({structured_correct / structured_total:.1%})")
    if unknown_total:
        print(f"  unknown 'soft no':   {unknown_correct}/{unknown_total}  "
              f"({unknown_correct / unknown_total:.1%})")
    total_q = structured_total + unknown_total
    total_c = structured_correct + unknown_correct
    print(f"  overall:             {total_c}/{total_q}  ({total_c / total_q:.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
