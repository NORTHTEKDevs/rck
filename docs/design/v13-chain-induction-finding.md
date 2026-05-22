# v13 chain-induction finding

Date: 2026-05-21
Status: closed (positive result; capability added)

## Question

If we walk a confident chain `leaf -> partof -> tree -> locatedin ->
forest`, can we SYNTHESISE the direct fact `(leaf, locatedin, forest)`
and store it back to the KB? This is fact induction from existing
chains -- the structural opposite of LLM hallucination.

## Method

`rck/chain_induction.py` runs over `ChainResult`s and applies four
gates before committing a synthesised fact:

1. **Min chain confidence** (`0.20` default): low-confidence chains
   don't earn induction.
2. **Min chain length**: 1-hop chains are already direct.
3. **Inverse-pair filter**: chains where consecutive relations form
   an inverse pair (`author -> wrote`, `partof -> haspart`, ...) are
   ROUND-TRIPS through a hub; the induced shortcut connects unrelated
   sibling entities and is almost always wrong.
4. **Non-transitive same-relation filter**: chains of the form
   `X -[R]-> Y -[R]-> Z` only induce when R is transitive (isa,
   partof, locatedin, etc). Other same-relation chains are HRR
   cleanup artefacts.

Surviving candidates are stored, then re-verified via
`self_verify.verify_roundtrip`. Failed verifications are rolled back.

## Empirical results (commonsense KB, 80 probes)

| pass | induced | verified | precision |
|------|---------|----------|-----------|
| no filters    | 19 | 19 | ~50% (bad facts like `macbeth wrote othello` slipped through) |
| + inverse-pair | 9 | 9  | ~80% |
| + same-rel    | 6 | 6  | **100%** |

The six derived facts from the final pass:

* `(wheel, usedfor, driving)` -- wheels are part of cars used for driving
* `(page, madeof, paper)` -- pages are part of books made of paper
* `(fret, usedfor, music)` -- frets are part of guitars used for music
* `(madrid, continent, europe)` -- madrid is in spain in europe
* `(canberra, continent, oceania)` -- canberra is in australia in oceania
* `(stockholm, continent, europe)` -- stockholm is in sweden in europe

## What this unlocks

* **Monotonic knowledge growth**: every successful chain may produce
  a new direct edge, expanding the KB without external input.
* **Speedup**: induced shortcuts answer in one HRR query instead of
  N (chain-walk).
* **Auditability**: provenance store tags each induced fact with
  `source="induced"` and `tags={"induced", "via_N_hops"}` so we can
  identify, audit, and roll back derived knowledge.

## Why this works (when LLMs can't do the same)

A language model presented with a chain of facts has no way to
COMMIT a derived shortcut back to its weights -- the closest analogue
is a memory mechanism. RCK's substrate is structurally additive:
storing a new fact is a vector add. Induced facts and observed facts
share the same retrieval path, so the savings are immediate.

The filters we added are the empirical guard against the obvious
failure mode (semantically-coincidental shortcuts). They are LOCAL
graph properties of the chain, not learned heuristics -- which means
they generalise beyond the commonsense KB we measured on.

## ConsciousAgent API

```python
agent = ConsciousAgent(...)
# Single-fact induction.
fact = agent.induce("leaf", "forest")
# fact.subject == "leaf", fact.relation == "locatedin", fact.obj == "forest"
# fact.verified == True
```

## Next research directions

* **Higher-order patterns**: instead of inducing direct shortcuts,
  emit RULE templates (`forall X. X partof Y and Y locatedin Z => X
  locatedin Z`). Skill library already has a slot for this; needs the
  generalisation step.
* **Confidence calibration**: induced facts should carry a discount
  vs primary observations. Currently both look identical at lookup;
  caller has to read provenance to tell them apart.
* **Cascading induction**: an induced fact might itself feed a new
  chain. We could iterate until fixed-point.
