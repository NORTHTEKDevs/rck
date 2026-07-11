# 6. Maintenance

Keeping an RCK agent healthy over time.

## The one-call nightly pass

```python
summary = agent.maintain(checkpoint_dir="./agent-state")
```

Runs eight phases in order:

1. `cascade_induct` — chain-driven derivation to fixed point.
2. `cascade_instantiate_rules` — rule-driven derivation to fixed point.
3. `propagate_negations` — lift `NOT_R` through `isa` / `partof`.
4. `resolve_conflicts(apply=True)` — drop losing facts.
5. `promote_skills` — frequent skill families become rules.
6. `consolidate_episodes` — stable hot signatures get cache pre-warmed.
7. `warm_cache_from_history` — predictive prefetch from observed queries.
8. `checkpoint` — `save_state(checkpoint_dir)` if provided.

Returns a summary dict suitable for cron logging:

```python
{
  'chain_induction_verified': 11,
  'chain_induction_rounds': 4,
  'rule_cascade_verified': 121,
  'rule_cascade_rounds': 3,
  'negations_propagated': 0,
  'conflicts_resolved': 0,
  'skills_promoted': 4,
  'episodes_consolidated': 0,
  'episodes_ambiguous_flagged': 0,
  'cache_entries_warmed': 0,
  'final_kb_size': 839,
  'skill_library': {'n': 57, ...},
  'checkpoint': {'skills': 57, 'provenance': 839, 'query_memory': 0},
}
```

Every phase has a flag (all default `True`). For a lighter pass:

```python
agent.maintain(
    propagate_negations=False,
    consolidate_episodes=False,
    checkpoint_dir=None,
)
```

## When to run it

* After bulk ingestion of new facts.
* On a daily / weekly cron.
* Before persisting an agent for sharing.
* When `agent.status()` shows growing query memory but no recent
  cache entries.

## Status dashboard

```python
print(agent.status_report())
```

```
RCK agent status:
  KB facts:           839 (in 16 shards)
  Provenance records: 716
  Sources:            user=716, induced=11, rule=112
  Skills:             n=57  uses=247  high_conf=17
  Rules:              25
    top: forall X0, X1, X2. (X0 locatedin X1) and (X1 continent X2)
                            => (X0 continent X2)
  Query memory:       312 episodes
    states:           known=298, ambiguous=8, idk=6
  Chain cache:        size=14  hits=42  v=4
```

## Capacity monitoring

```python
report = agent.shard_balance()
print(report.verbalize())
```

Flags shards near the capacity cliff (target_fill ≈ 80 at D=4096)
and suggests a reshard target.

## Pruning

When the KB has accumulated low-confidence induced or rule-derived
facts:

```python
report = agent.prune_facts(min_confidence=0.1)
print(report.dropped, "facts removed")
```

User-asserted and negative facts are protected by default.

## Hot-path prefetch

`maintain()` already runs this, but you can also call it directly:

```python
warmed = agent.warm_cache_from_history(top_k=10)
```

Uses `query_memory.hot_signatures` to pick the most frequent recent
queries and pre-discover their chains.

## Calibration tracking

When the user supplies ground truth for a previous question. There
must BE a previous question — `record_truth` matches against the
episode log, so a query that was never asked returns
`{"updated": False, "reason": "no prior episode"}`:

```python
agent.tell("dog", "isa", "mammal")
agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")   # logs the episode

result = agent.record_truth(
    {"S": "dog", "R": "isa"}, "O",
    correct_answer="mammal",
)
# -> {'updated': True, 'relation': 'isa', 'predicted': 'mammal',
#     'correct_answer': 'mammal', 'was_correct': True, ...}
```

Walks `query_memory` to find the matching episode and updates
`CalibrationTally` with predicted-vs-actual.

```python
summary = agent.calibration.summary()
print(summary["isa"])
# {'know_right': 12, 'know_wrong': 1, 'calibration_score': 0.923, ...}
```

## A complete cron job

```python
#!/usr/bin/env python3
"""Daily RCK maintenance for ./my-agent"""

import sys
from rck.conscious_agent import ConsciousAgent

agent = ConsciousAgent(expected_facts=10_000)
agent.load_state("./my-agent")

summary = agent.maintain(checkpoint_dir="./my-agent")

import json
print(json.dumps(summary, indent=2))
```

## Next

- [07-faq](07-faq.md): what RCK is *not*.
