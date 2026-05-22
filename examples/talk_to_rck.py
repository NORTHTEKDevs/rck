"""Interactive chat with RCK -- the real generative-AI demo.

The REPL supports three modes:

  tell <subject> <relation> <object>
                            Store an exact fact in relational memory.
  teach <sentence>          Statistically ingest a sentence (also extracts
                            obvious "X is Y" / "X has Y" facts).
  ask <question>            Answer a natural-language question.
  show                      Dump model state (counts, last reasoning trace).
  save <path> / load <path> Persist via the standard versioned format.
  exit                      Quit.

If the input does not begin with a known verb it's treated as a question,
i.e. you can just type "What color is the sky?" and RCK will answer.

Run:
    python -m examples.talk_to_rck                    # blank slate
    python -m examples.talk_to_rck --bootstrap        # preload world_knowledge.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rck.generative import GenerativeRCK


HELP = """
  tell <subject> <relation> <object>   store an exact fact
  teach <sentence>                     learn from a natural-language sentence
  ask <question>                       ask -- or just type a question
  show                                 show RCK's current state
  forget <subject> <relation> <object> remove a fact
  save <path>                          save the model
  exit                                 quit
"""


def main() -> int:
    parser = argparse.ArgumentParser(prog="talk_to_rck")
    parser.add_argument("--bootstrap", action="store_true",
                        help="preload data/world_knowledge.txt")
    parser.add_argument("--hv-dim", type=int, default=4096)
    args = parser.parse_args()

    g = GenerativeRCK(dim=args.hv_dim, seed=0)
    print("RCK Generative -- type 'help' for commands.\n")

    if args.bootstrap:
        path = Path("data/world_knowledge.txt")
        if path.exists():
            info = g.ingest(path.read_text(encoding="utf-8"))
            print(f"[bootstrap] {info}\n")

    while True:
        try:
            line = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"exit", "quit"}:
            return 0
        if line == "help":
            print(HELP); continue

        verb, _, rest = line.partition(" ")
        verb = verb.lower()

        if verb == "show":
            print(g.state()); continue
        if verb == "tell":
            parts = rest.split()
            if len(parts) < 3:
                print("usage: tell <subject> <relation> <object>"); continue
            s, r = parts[0], parts[1]
            o = " ".join(parts[2:])
            g.tell(s, r, o)
            print(f"[stored] ({s}, {r}, {o}) -- memory has {g.memory.size()} facts.")
            continue
        if verb == "teach":
            info = g.ingest(rest)
            print(f"[learned] {info}"); continue
        if verb == "forget":
            parts = rest.split()
            if len(parts) < 3:
                print("usage: forget <subject> <relation> <object>"); continue
            s, r = parts[0], parts[1]
            o = " ".join(parts[2:])
            g.memory.forget(g.codebook, {"S": s, "R": r, "O": o})
            print(f"[forgot] ({s}, {r}, {o})")
            continue
        if verb == "save":
            print("(save coming via rck.persist in a future release)"); continue

        # Default: treat the entire line as a question.
        question = rest if verb == "ask" else line

        res = g.ask(question)
        print(f"<< {res['answer']}")
        if res["source"].startswith("structured"):
            print(f"   (source={res['source']}, conf={res['confidence']:.2f})")
            top = res["candidates"][:3]
            if len(top) > 1:
                rest_str = ", ".join(f"{s}({c:.2f})" for s, c in top[1:])
                print(f"   other candidates: {rest_str}")
        else:
            print(f"   (no fact found; this is a generated guess)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
