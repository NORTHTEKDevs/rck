# 2. Tell and ask

The shape of everything you say to an RCK agent.

## The triple

Every fact is a `(subject, relation, object)` triple. Subjects and
objects are entities (strings — they'll be lowercased). Relations are
verbs / properties (also strings).

```python
agent.tell("dog",      "isa",       "mammal")
agent.tell("dog",      "has",       "fur")
agent.tell("paris",    "capital_of", "france")
agent.tell("water",    "boils_at",   "100c")
```

There are no fixed schemas. You invent the relation names you need.

## Symmetrisation

Some relation pairs are inverses (e.g. `capitalof` / `capital`,
`partof` / `haspart`, `wrote` / `author`). When you tell RCK one
direction, it auto-stores the other:

```python
agent.tell("rome", "capitalof", "italy")
ans, _ = agent.knowledge.answer({"S": "italy", "R": "capital"}, "O")
# -> 'rome'
```

The list of recognised inverse pairs is in
`rck.bulk_ingest.INVERSE_PAIRS`; relation names must match those
entries exactly (it's `capitalof`, not `capital_of` — an unrecognised
name is simply stored one-way, with no error). Add your own pairs to
taste.

## Bulk loading

For a list of triples:

```python
from rck.bulk_ingest import bulk_load_triples

bulk_load_triples(agent.knowledge, [
    ("paris", "capital_of", "france"),
    ("berlin", "capital_of", "germany"),
    ("madrid", "capital_of", "spain"),
])
```

For a JSONL file (one `{"s": ..., "r": ..., "o": ...}` per line):

```python
agent.load_jsonl("data/commonsense_kb.jsonl")
```

`load_jsonl` triggers a chain-cache invalidation; small per-fact
`agent.tell()` calls do not.

## Ask shapes

Every ask is a partial fact with one role unknown.

```python
# Unknown object.
agent.knowledge.answer({"S": "paris", "R": "capital_of"}, "O")

# Unknown subject (who's capital_of france?).
agent.knowledge.answer({"R": "capital_of", "O": "france"}, "S")

# Unknown relation (rare; useful for relation discovery).
agent.knowledge.answer({"S": "paris", "O": "france"}, "R")
```

## The full ask API at a glance

| Method | Returns | Use when |
|---|---|---|
| `agent.knowledge.answer(known, role)` | `(symbol, score)` | You just want top-1 |
| `agent.knowledge.query(known, role, top_k=k)` | `[(symbol, score)]` | You want top-K |
| `agent.ask_with_idk(known, role)` | `EpistemicAnswer` | You care about IDK |
| `agent.calibrated_ask(known, role)` | `[CalibratedAnswer]` | You want provenance discount |
| `agent.intersect([c1, c2], role)` | `[SetCandidate]` | Multiple constraints |
| `agent.union([c1, c2], role)` | `[SetCandidate]` | Match any constraint |

## Provenance

Every `tell` records a provenance entry:

```python
rec = agent.provenance.get("paris", "capital_of", "france")
print(rec.source)        # 'user'
print(rec.timestamp)
print(rec.confidence)
print(rec.count)
```

Provenance is your audit trail. Every derived fact (from chains,
rules, abstractions) carries the same metadata plus a `derivation`
chain pointing at the facts it came from.

## Denying

A negative fact is positive certainty:

```python
agent.deny("dog", "isa", "fish")
```

Stored as `(dog, NOT_isa, fish)`. Affects downstream derivation:
chain induction and rule instantiation will both refuse to assert
the positive form.

## Forgetting

To remove a fact directly:

```python
agent.knowledge.forget({"S": "paris", "R": "capital_of", "O": "france"})
agent.provenance.forget("paris", "capital_of", "france")
```

For broader cleanup based on confidence:

```python
report = agent.prune_facts(min_confidence=0.1)
print(report.dropped)
```

This protects user-asserted facts by default; set `prune_user_facts=True`
if you really want to nuke them.

## Next

- [03-reasoning](03-reasoning.md): now that the agent knows things,
  how it draws conclusions from them.
