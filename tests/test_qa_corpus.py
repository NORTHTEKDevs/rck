"""End-to-end QA test: ingest world_knowledge.txt + ask 26 questions."""
from pathlib import Path

from rck.generative import GenerativeRCK


KNOWN = [
    ("What color is the sky?",            "blue"),
    ("What color is the grass?",          "green"),
    ("What color is the sun?",            "yellow"),
    ("What is the capital of France?",    "paris"),
    ("What is the capital of Germany?",   "berlin"),
    ("What is the capital of Japan?",     "tokyo"),
    ("Where does Alice live?",            "paris"),
    ("Where does Bob live?",              "berlin"),
    ("Who wrote Hamlet?",                 "shakespeare"),
    ("Who wrote 1984?",                   "orwell"),
    ("What does the cat have?",           "fur"),
    ("What does the bird have?",          "feathers"),
    ("What does the elephant have?",      "tusks"),
    ("What is the size of the elephant?", "huge"),
    ("What is the size of the mouse?",    "tiny"),
    ("What is the age of Alice?",         "30"),
]

UNKNOWN = [
    "What color is the alien?",
    "Who invented teleportation?",
]


def test_world_knowledge_qa_full_accuracy():
    g = GenerativeRCK(dim=4096, seed=0)
    text = Path("data/world_knowledge.txt").read_text(encoding="utf-8")
    g.ingest(text)

    correct = 0
    for q, expected in KNOWN:
        res = g.ask(q)
        if (res["answer"] or "").lower().strip() == expected:
            correct += 1
    assert correct == len(KNOWN), f"{correct}/{len(KNOWN)} -- some known questions regressed"


def test_unknown_questions_do_not_confidently_answer():
    g = GenerativeRCK(dim=4096, seed=0)
    g.ingest(Path("data/world_knowledge.txt").read_text(encoding="utf-8"))
    for q in UNKNOWN:
        res = g.ask(q)
        # Either no structured route, or a structured route with low confidence.
        if res["source"].startswith("structured"):
            assert res["confidence"] < 0.30, f"too confident on unknown: {q} -> {res}"
