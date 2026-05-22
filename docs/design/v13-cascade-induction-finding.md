# v13 cascading-induction finding

Date: 2026-05-21
Status: closed (positive result; capability added)

## Question

A single induction pass produces new direct facts. Those facts ARE
new KB edges. Do they open up second-pass chains that weren't
reachable before? If we keep iterating, does the KB approach a
fixed point of derivable knowledge?

## Method

`rck/cascading_induction.py` runs `induce_from_chain` in a loop. At
each round it:

1. Builds 2-hop transitive probe pairs from the CURRENT KB state
   (so each round sees the previously-induced edges as candidates).
2. Discovers chains for fresh probes (`seen_pairs` tracks what we've
   already tried).
3. Runs the standard filters (inverse-pair, non-transitive
   same-relation, cycle detection, lifting-relation gate) before
   committing.
4. Tags every verified induction with provenance + skill records.

The cascade terminates when no new verified facts are added in a
round (saturation) or after `max_rounds`.

## Lifting-relation gate (new filter)

A subtle failure surfaced during this study:

```
caesar -> country -> rome -> capitalof -> italy
=> (caesar, capitalof, italy)   # WRONG -- caesar isn't a capital
```

The chain confidence is fine and the cycle filter doesn't fire
(italy isn't an intermediate). The bug is in REUSING the LAST
relation as the induced shortcut's relation when the FIRST hop
isn't a containment / class-inclusion relation.

New gate: the induced relation defaults to `implies` UNLESS the
first hop's relation is in a "lifting" set
(`isa`, `partof`, `locatedin`, `memberof`, ...). Lifting relations
let the source inherit properties downstream:

```
wheel partof car, car usedfor driving
  => (wheel, usedfor, driving)        VALID (partof lifts)

caesar country rome, rome capitalof italy
  => (caesar, implies, italy)         SAFE (country doesn't lift)
```

## Results on commonsense KB

```
round   probes   chains   induced   verified   kb_size
    1       80       69        68          2       718
    2       80       75        74          6       724
    3       80       75        73          3       727
    4       11        9         9          0       727
```

* **Saturation in 4 rounds.** Round 4 found 0 new facts; cascade
  terminated.
* **Round 2 outperforms round 1** (6 vs 2 verifications): the new
  shortcuts from round 1 opened new chain possibilities, exactly as
  the cascade design predicts.
* **+11 verified new facts** added to the KB (716 -> 727), all with
  provenance source=`induced`.
* **Top productive pattern**: `locatedin -> continent` (6x), a
  classic city-country-continent transitive shortcut.

## Sample induced facts (final pass)

* `(canberra, continent, oceania)`, `(madrid, continent, europe)`,
  `(stockholm, continent, europe)` -- direct city -> continent edges
  derived from city -> country -> continent chains.
* `(caesar, implies, italy)`, `(cleopatra, implies, africa)` --
  generic "association exists" labels rather than spurious specific
  claims.
* `(steam, implies, fish)`, `(winter, implies, shivering)` -- weak
  associations preserved as `implies`, not as false specific claims.

## What this changes architecturally

The KB now has TWO modes of growth:

1. **External input**: `agent.tell()` from user / corpus / ingest.
2. **Internal derivation**: `cascade_induct(kb)` from existing
   chains.

Both look identical at retrieval time, but the provenance store
distinguishes them (`source="user"` vs `source="induced"`). A
maintenance task that runs nightly can promote chained reasoning
into direct edges, with full audit trail.

## Open work

* **Higher-order rule extraction**: instead of inducing one fact
  per chain, extract the universal rule it instantiates. The
  SkillLibrary already records the pattern signatures
  (`('locatedin', 'continent')` x 6) -- needs a step that emits
  `forall X. X locatedin Y, Y continent Z => X continent Z` to a
  rules store.
* **Cascade with `implies`**: facts inducted under `implies` could
  themselves feed weaker downstream chains. We currently treat them
  the same as any other edge.
* **Confidence-weighted retrieval**: induced facts should perhaps
  carry a small score discount vs first-hand observations.
