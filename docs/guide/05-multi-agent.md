# 5. Multi-agent

RCK agents are objects. You can run many of them and combine the
results.

## Federated merge

Two agents trained on disjoint domains can fold their state together:

```python
medical = ConsciousAgent(); medical.load_state("./medical-agent")
legal   = ConsciousAgent(); legal.load_state("./legal-agent")

medical.merge_from(legal)
# Now `medical` knows both. Provenance for facts that came from `legal`
# is preserved (with source='multi' on collisions).
```

What gets merged:
* **HRR knowledge bases** — bipolar bundle sum.
* **Skill counters** — added together.
* **Provenance records** — count summed, last_seen takes the max,
  source becomes `multi` on disagreement.

What does NOT get merged:
* `query_memory` (episode logs are per-agent).
* `chain_cache` (different topologies anyway).
* `calibration` tally (per-agent metacognition).

## Consensus voting

Run a query across multiple agents and aggregate:

```python
from rck.consensus import majority

result = majority(
    [agent_a, agent_b, agent_c],
    {"S": "patient", "R": "diagnosis"},
    "O",
    mode="both",   # majority + confidence tiebreak
)
print(result.chosen, result.chosen_votes, result.chosen_score)
for v in result.candidates:
    print(v.symbol, v.votes, v.total_score, v.contributors)
```

Modes:
* `"majority"` — most votes wins.
* `"confidence"` — sum of scores wins (good when agents have very
  different confidence).
* `"both"` — majority with confidence tiebreak (default).

Agents that say IDK abstain.

## Diff

When you've merged or branched agents, see what diverged:

```python
report = agent_a.diff_with(agent_b)
print(report.summary())
# {'facts_only_in_a': ..., 'facts_only_in_b': ..., 'facts_shared': ...,
#  'skills_only_in_a': ..., 'rules_only_in_a': ..., ...}

for triple in report.only_in_a_facts[:5]:
    print(triple)
```

## A practical pattern: domain agents

```python
# Train (= ingest) several domain agents in parallel.
medical = ConsciousAgent()
medical.load_jsonl("data/pubmed_triples.jsonl")
medical.maintain(checkpoint_dir="./medical")

legal = ConsciousAgent()
legal.load_jsonl("data/case_law_triples.jsonl")
legal.maintain(checkpoint_dir="./legal")

# Query mode: vote across them.
result = majority([medical, legal], q, "O", mode="confidence")

# Or: fold into a single super-agent for offline analysis.
combined = ConsciousAgent()
combined.merge_from(medical)
combined.merge_from(legal)
combined.maintain()
```

## Next

- [06-maintenance](06-maintenance.md): the nightly pass that keeps
  an agent healthy.
