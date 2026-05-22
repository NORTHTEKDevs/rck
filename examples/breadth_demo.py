"""v1.5 breadth demo: 1000+ facts across multiple domains.

Loads both the common-sense KB (486 facts) and the extended KB (547
facts) including chemistry, astronomy, geography, music, sports,
historical figures, languages, etc. Then asks questions spanning every
domain.

Demonstrates:
  - Knowledge breadth approaching small-LM coverage.
  - Multi-hop inheritance (city -> country -> continent).
  - Inverse-relation lookups (author -> wrote, wrote -> author).
  - Boolean, enumeration, comparison query types.
  - Multi-sentence descriptions of entities.
  - Think-aloud reasoning traces.

Run:
    python -m examples.breadth_demo
"""
from __future__ import annotations

import time
from pathlib import Path

from rck.bulk_ingest import auto_symmetrize, bulk_load_jsonl
from rck.conscious_agent import ConsciousAgent


SECTION = "=" * 64


def banner(s: str) -> None:
    print(f"\n{SECTION}\n {s}\n{SECTION}")


def Q(ai: ConsciousAgent, q: str, think: bool = False) -> tuple[str, bool]:
    res = ai.ask(q, think_aloud=think)
    verb = res.get("verbal", "")
    src = res.get("source", "?")
    print(f"  > {q}")
    print(f"    {verb}")
    if think and res.get("think_aloud"):
        print(f"      [thought: {res['think_aloud']}]")
    return verb, "structured" in src or "boolean" in src or "comparison" in src


def main() -> int:
    banner("RCK v1.5 BREADTH DEMO -- 1000+ facts across multiple domains")
    ai = ConsciousAgent(dim=4096, n_shards=128, seed=0)

    t0 = time.time()
    stats1 = bulk_load_jsonl(ai.knowledge, "data/commonsense_kb.jsonl",
                              symmetrize=True)
    stats2 = bulk_load_jsonl(ai.knowledge, "data/extended_kb.jsonl",
                              symmetrize=True)
    n_extra = auto_symmetrize(ai.knowledge)
    elapsed = time.time() - t0
    print(f"\nLoaded {stats1['facts']} commonsense + {stats2['facts']} extended "
          f"+ {n_extra} auto-symmetrized in {elapsed:.2f}s.")
    print(f"Total facts in KB: {ai.knowledge.size():,}")
    util = ai.knowledge.utilization()
    print(f"Shard distribution: max={util['max_shard']} avg={util['avg_shard']:.1f}")

    banner("(1) Chemistry")
    Q(ai, "What is the symbol of the gold?")
    Q(ai, "What is the symbol of the iron?")
    Q(ai, "What is the atomic_number of the oxygen?")

    banner("(2) Astronomy")
    Q(ai, "What is the position of the earth?")
    Q(ai, "What is the size of the jupiter?")
    Q(ai, "What does the saturn orbit?")

    banner("(3) Geography")
    Q(ai, "What is the locatedin of the everest?")
    Q(ai, "What is the height of the kilimanjaro?")
    Q(ai, "What is the locatedin of the amazon?")
    Q(ai, "What is the continent of paris?")   # multi-hop

    banner("(4) History + arts")
    Q(ai, "What is the field of mozart?")
    Q(ai, "What is the country of davinci?")
    Q(ai, "What is the field of nietzsche?")
    Q(ai, "What is the field of dirac?")

    banner("(5) Languages")
    Q(ai, "What is the spoken_in of french?")
    Q(ai, "What is the spoken_in of swahili?")

    banner("(6) Occupations + everyday")
    Q(ai, "What is the works_at of the doctor?")
    Q(ai, "What is the works_at of the chef?")
    Q(ai, "What is the family of the piano?")
    Q(ai, "What is the venue of the soccer?")
    Q(ai, "What is the travels_on of the boat?")

    banner("(7) Inverse-relation auto-lookups")
    Q(ai, "Who wrote hamlet?")
    Q(ai, "Who wrote 1984?")
    # Inverse of wrote: author->wrote means we can also ask in reverse.
    print("\n  inverse direction:")
    res = ai.ask("What is the wrote of shakespeare?")
    print(f"    {res.get('verbal')}")

    banner("(8) Boolean + comparison + enumeration on extended data")
    Q(ai, "Is gold an element?")
    Q(ai, "Is earth a planet?")
    Q(ai, "Is jupiter bigger than mars_planet?")
    Q(ai, "What are elements?")    # huge enumeration
    Q(ai, "What are planets?")

    banner("(9) Multi-sentence description (compose_answer)")
    for entity in ("elephant", "shakespeare", "earth", "gold"):
        print(f"\n  describe('{entity}'):")
        print("    " + ai.describe(entity).replace(". ", ".\n    "))

    banner("(10) Think-aloud reasoning")
    Q(ai, "What is the continent of paris?", think=True)
    Q(ai, "What does the dog have?", think=True)

    banner("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
