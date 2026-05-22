# v13 analogical reasoning finding

Date: 2026-05-21
Status: closed (positive; 88.7% accuracy on commonsense)

## Question

Can RCK solve `A : B :: C : ?` analogies through its existing
relational substrate, without any specially-trained analogy module?

## Approach

We don't try free HRR algebra (atoms are random bipolar HVs without
semantic prior). Instead we use two relational queries:

1. **find_relation(A, B)**: enumerate the KB's relations, query
   `(A, R, ?)` for each, score by how close the top result is to B.
   The relation with the highest score wins.
2. **apply**: query `(C, R, ?)` -> answer.

This is structural analogy: the RELATION is the analog operator,
the codebook does the cleanup.

## Result on commonsense KB

115 probes, all of the form `(A, R, B)` and `(C, R, D)` both stored:

* **Relation inferred correctly: 98/115 = 85.2%**
* **Final answer correct: 102/115 = 88.7%**

The answer can be correct even when R is wrong (a different-but-valid
relation produces the same target via chance HRR overlap).

## Failure modes

Almost all failures come from **multi-valued relations** like `has`:

```
dog : legs :: cat : expected 'fur', got 'claws'
```

Both "cats have claws" and "cats have fur" are true. The expected
answer is arbitrary; our 'claws' is equally valid. The 88.7% is
therefore a LOWER BOUND on real-world correctness.

A small number of failures come from the relation enumeration
returning a different valid R than the probe's intended one.

## Examples that worked

* `france : paris :: germany : berlin   (via capital)`
* `dog : mammal :: eagle : bird         (via isa)`
* `dog : mammal :: cat : mammal         (via isa)`

## ConsciousAgent integration

```python
res = agent.analogy("france", "paris", "germany")
# res.relation == "capital", res.answer == "berlin"
```

## Why this matters vs LLM-style analogy

LLMs do analogy through implicit embedding geometry. The answer is
ungrounded -- you can't audit why it picked one analog over another.

RCK's analogy is auditable: `res.relation` and `res.relation_score`
expose exactly which relation the system used, and the chain of
queries that produced the answer.

## Open work

* Multi-relation analogies (paired-relation patterns like
  "father : son :: queen : king").
* Confidence-weighted relation choice instead of strict argmax.
* When find_relation returns 0 candidates, try chain_walker to
  derive R via 2-hop reasoning.
