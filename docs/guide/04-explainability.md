# 4. Explainability

How to ask RCK "why do you think that?" and get a real answer.

## The provenance graph

Every fact in RCK is stored in two places:

1. **HRR memory** — the math vector that retrieval uses.
2. **Provenance store** — a dict keyed by `(s, r, o)` with:
   * `source` — `"user"` / `"induced"` / `"rule"` / `"multi"` / etc.
   * `timestamp`, `last_seen` — when we first saw it and last touched it.
   * `confidence` — current confidence (1.0 for fresh user facts;
     decays with failures, grows with reinforcement).
   * `count` — how many times the fact was reinforced.
   * `tags` — labels: `{"induced", "via_2_hops"}`, `{"abstracted",
     "from_3_siblings"}`, etc.
   * `derivation` — the chain of source facts that produced THIS fact
     (empty for user-asserted).

## explain_why

For any fact, walk back to the user-asserted facts that grounded it:

```python
node = agent.explain_why("leaf", "locatedin", "forest")
print(node.verbalize())
```

Output:

```
(leaf, locatedin, forest)  source=induced
  (leaf, partof, tree)   source=user (leaf)
  (tree, locatedin, forest)   source=user (leaf)
```

The tree terminates at:
* `source="user"` — the user told us this.
* `source="unknown"` — no provenance record (rare in well-managed KBs).
* `source="cycle"` — cycle detected during the walk.
* `max_depth` reached.

`node.depth()` gives the deepest level. `node.total_facts()` counts
all nodes in the derivation tree. `explanation_summary(node)` returns
a flat dict suitable for logging.

## Why this matters

LLMs can't do this. Their "explanations" are just more generated
text, with no causal connection to whatever produced the original
answer. RCK's explanation is the actual derivation that produced
the fact. You can audit it.

## Calibrated confidence by source

Two facts can be in the KB with the same retrieval cosine but very
different provenance. `agent.calibrated_ask` exposes this:

```python
rows = agent.calibrated_ask({"S": "fish", "R": "isa"}, "O", top_k=3)
for r in rows:
    print(r.symbol, r.raw_score, "->", r.calibrated_score,
          "source=", r.source)
```

Default discount factors (in `rck.confidence_calibration`):

| source | factor |
|---|---|
| `user` | 1.0 |
| `multi` | 1.05 (small reinforcement bonus) |
| `external` | 0.85 |
| `induced` | `0.9^via_hops`, floored at 0.5 |
| `rule` | (same as induced) |
| `unknown` | 1.0 (trust the substrate) |

A 3-hop induced fact at raw cosine 0.5 ends up with calibrated score
~0.36; a user-asserted fact at the same raw cosine keeps 0.5.

## Episodic memory + drift

Every `ask_with_idk` is logged in `agent.query_memory`. When the same
query returns a different answer than last time, the new result
flags it:

```python
res = agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
if res.drift_from_prior:
    print("WARNING:", res.drift_from_prior)
```

For aggregate views:

```python
report = agent.drift_report(last_k=200)
print(report["total_drift_events"])
print(report["by_relation"])
for row in report["by_signature"][:5]:
    print(row)
```

## Contradictions & resolution

Detect (S, R, O) facts that have multiple competing answers, or
positive-AND-negative collisions:

```python
conflicts = agent.detect_conflicts()
for c in conflicts:
    print(c.verbalize())
```

Resolve them with source priority:

```python
plans = agent.resolve_conflicts(apply=True)
for plan in plans:
    print(plan.verbalize())
# -> "For (fish, isa, ?) keep 'animal' (source='user'). Drop ['vegetable'].
#    kept source='user' over 'induced' (provenance priority)"
```

## Persistence

```python
counts = agent.save_state("./agent-checkpoint")
# {'skills': N, 'provenance': M, 'query_memory': K}

# Later, in a new process:
agent2 = ConsciousAgent()
agent2.load_state("./agent-checkpoint")
```

This preserves skills, provenance, query history. (KB tensors are
saved/loaded separately via `rck.persist`.)

## Next

- [05-multi-agent](05-multi-agent.md): merge agents, build consensus.
