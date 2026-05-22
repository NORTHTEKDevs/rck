"""Rule extraction study on the commonsense KB.

Run cascading induction (which populates the SkillLibrary as a side
effect), then extract symbolic rules. Show that we can derive
universal patterns from the empirical chain successes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rck.bulk_ingest import bulk_load_triples
from rck.cascading_induction import cascade_induct
from rck.chain_induction import InductionPolicy
from rck.knowledge_base import ShardedKnowledgeBase
from rck.rule_extraction import extract_rules
from rck.shard_sizing import recommend_shards
from rck.skills import SkillLibrary


def main() -> int:
    print("=" * 70)
    print(" RULE EXTRACTION STUDY on commonsense KB")
    print("=" * 70)

    triples = []
    with open("data/commonsense_kb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            triples.append((r["s"], r["r"], r["o"]))
    n_shards = recommend_shards(len(triples) * 2, dim=4096).n_shards
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    bulk_load_triples(kb, triples)
    skills = SkillLibrary()
    policy = InductionPolicy(min_confidence=0.15)

    print(f"\nKB: {kb.size()} facts in {n_shards} shards")
    print("Running cascading induction to populate skills ...")
    res = cascade_induct(
        kb, max_rounds=4, probes_per_round=120,
        policy=policy, skills=skills,
    )
    print(f"  Skill library now: {skills.stats()}")
    print(f"  Verified inductions: {res.total_verified}")

    print("\nExtracting rules (min_support=2, min_confidence=0.5) ...")
    rules = extract_rules(skills, min_support=2, min_confidence=0.5)
    print(f"  Extracted {rules.size()} rules")

    print("\nTop 10 rules by support:")
    for rule in rules.top_rules(n=10):
        print(f"  [{rule.support}x conf={rule.confidence:.2f}]  "
              f"{rule.verbalize()}")

    out_path = Path("data/rule_extraction_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "kb_facts": kb.size(),
        "n_skills": skills.stats()["n"],
        "n_rules": rules.size(),
        "rules": [
            {
                "body": r.body,
                "head": r.head,
                "support": r.support,
                "confidence": r.confidence,
                "verbal": r.verbalize(),
            }
            for r in rules.top_rules(n=20)
        ],
    }, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
