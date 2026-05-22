# 3. Reasoning

How RCK gets to answers that aren't a single hop away.

## Chain walking

When you know the chain shape, walk it directly:

```python
from rck.chain_walker import Hop

res = agent.reason("france", ["capital", "locatedin", "isa"])
print(res["answer"])        # -> 'continent'
print(res["confidence"])    # propagated geometric-mean
print(res["hedge"])         # 'strong' / 'moderate' / 'weak' / 'uncertain'
print(res["trace"])         # the per-hop (s, r, o, score) trail
```

By default the propagation rule is `geometric_mean`. This is the
v13/v14 change that extends usable chain depth from ~10 hops (under
the legacy `product` rule) to 50+ hops. See
[v13-chain-depth-finding](../design/v13-chain-depth-finding.md).

## Chain discovery

When you DON'T know the chain shape, search:

```python
spec = agent.discover("france", "continent", max_depth=4)
print(spec["relations"])   # e.g. ['capital', 'locatedin', 'isa']
```

BFS over the KB. Returns the first viable chain (or `None`). Cached
by `(start, target)` so repeats are O(1). The cache auto-invalidates
on bulk writes.

## Reverse hops

If `bulk_load_triples` symmetrised your data, forward edges cover
most cases. If not, you may need `allow_reverse=True` so the search
considers `(?, R, target)` edges:

```python
spec = agent.discover("country", "paris", allow_reverse=True)
```

## Chain induction (learn by deriving)

When a chain walk succeeds, you can store the shortcut:

```python
induced = agent.induce("leaf", "forest")
# -> InducedFact(subject='leaf', relation='locatedin', obj='forest',
#                via=[('leaf', 'partof', 'tree'),
#                     ('tree', 'locatedin', 'forest')],
#                verified=True)
```

The filter stack guarantees precision:

1. **Inverse-pair filter** — rejects chains like
   `author -> wrote` (hub round-trips through Shakespeare don't
   tell us anything about plays Shakespeare didn't write).
2. **Non-transitive same-relation filter** — `wrote -> wrote`
   is HRR-cleanup noise, not transitivity.
3. **Lifting-relation gate** — only propagate the last relation
   when the first hop is a containment relation
   (`isa`, `partof`, `locatedin`, `memberof`). Otherwise emit
   the generic `implies` relation.
4. **Intermediate-cycle filter** — reject chains whose answer
   is an earlier node.

Each induced fact gets a provenance record with `source="induced"`
and a derivation chain pointing back to its source facts.

See [v13-chain-induction-finding](../design/v13-chain-induction-finding.md).

## Cascading induction

One pass produces shortcuts. Those shortcuts open NEW chains. Iterate
to fixed point:

```python
from rck.cascading_induction import cascade_induct
from rck.chain_induction import InductionPolicy

result = cascade_induct(
    agent.knowledge,
    max_rounds=4,
    probes_per_round=80,
    policy=InductionPolicy(min_confidence=0.15),
    skills=agent.skills,
    provenance=agent.provenance,
)
print(result.total_verified, "new facts in", len(result.rounds), "rounds")
```

On the bundled commonsense KB, this typically adds ~10-20 verified
facts in 3-4 rounds before saturating.

## Rule extraction & instantiation

When a chain shape succeeds many times, it's a Rule:

```python
store = agent.extract_rules(min_support=2, min_confidence=0.5)
for rule in store.top_rules(n=5):
    print(rule.verbalize())
# -> forall X0, X1, X2. (X0 locatedin X1) and (X1 continent X2)
#                       => (X0 continent X2)
```

Apply rules forward to derive more facts:

```python
new_facts = agent.instantiate_rules()
# Or with cascade to fixed point:
result = agent.cascade_instantiate_rules(max_rounds=3)
```

Rules can also be composed symbolically:

```python
composed = agent.compose_rules()
# Stitches R1; R2 when R1.head matches R2.body[0]
```

## Analogy

```python
res = agent.analogy("france", "paris", "germany")
print(res.answer)         # -> 'berlin'
print(res.relation)       # -> 'capital'
print(res.joint_score())  # calibrated probability
```

The default scoring mode is Bayesian softmax over `top_k_relations`
candidate relations. Falls back to a chain-walker if no single
relation links A and B (e.g. for "france:europe :: germany:?").

## Causal reasoning

For `causes`-typed chains specifically:

```python
effects = agent.downstream_effects("rain", max_depth=3)
causes = agent.root_causes("injury", max_depth=3)
```

Returns `CausalNode` objects with depth, score, and the chain.

## Reasoning horizon at a glance

| Capability | Depth | Latency |
|---|---|---|
| Direct retrieval | 1 hop | <1 ms |
| Chain discovery | 2-4 hops | 8-55 ms |
| Chain walking (geometric mean) | 50+ hops | ~25 ms |
| Cascading induction | until fixed point | seconds |
| Rule cascade | until fixed point | seconds |

## Next

- [04-explainability](04-explainability.md): how to ask
  "why do you think that?" and get a real answer.
