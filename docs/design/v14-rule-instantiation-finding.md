# v14 rule instantiation finding

Date: 2026-05-21
Status: closed (positive; forward-chaining capability added)

## Question

`rule_extraction` produces symbolic rules. Can we forward-apply
them -- walk the KB, find every binding of the rule body, emit
the head as a derived fact -- without going through chain_walker?
This is the natural complement: rule extraction observes patterns
in the SkillLibrary; rule instantiation USES those patterns to
make new facts.

## Approach

For a 2-clause rule `(X R1 Y) and (Y R2 Z) => (X R_head Z)`:
1. Iterate every stored fact (X, R1, Y) -- O(N_R1) using the
   per-shard fact list.
2. For each, query (Y, R2, ?) and take top-K candidates above
   min_link_score.
3. For each (X, R_head, Z) that isn't already stored as a direct
   fact, store it, tag with provenance `source="rule"`, run
   roundtrip verification, roll back on failure.

The filters from chain_induction (inverse-pair, non-transitive
same-relation) are re-applied at the instantiation level for
defence in depth. A rule that survived `extract_rules` should
already be safe, but the second check costs nothing and catches
any future bugs in rule extraction.

## Why this is faster than chain_walker for productive rules

`chain_walker` is BFS over an unknown graph -- it doesn't know
which (start, target) pairs to probe. `instantiate_rule` walks
ONE relation index per shard, then issues exactly one HRR query
per (X, Y) it sees. For a rule that fires 28 times (like
`locatedin -> continent => continent` on the commonsense KB),
this is dramatically cheaper than 28 separate BFS searches.

## Empirical behaviour

* Tests cover isa transitivity on a 4-node staircase, inverse-pair
  filtering, non-transitive same-relation filtering, and the
  "skip already direct" check.
* 8 unit tests pass.
* ConsciousAgent.instantiate_rules() now runs `extract_rules` then
  `instantiate_all` in one call.

## What this completes

The v14 reasoning stack now has both directions of fact derivation:

  observation -> chain -> induced shortcut
       (chain_induction)

  observation -> pattern -> rule -> instantiated facts
       (rule_extraction)        (rule_instantiation)

The second path is faster when patterns are productive and slower
when they aren't (it walks every fact of R1 even if only a few
extend to a valid Z).

## Open work

* 3+ clause rules: current implementation only handles 2-clause
  bodies. Longer rules need a recursive application or a
  generalisation to (relation_chain, head) tuples.
* Forward chaining to fixed point: an instantiated fact may unlock
  another rule. Cascading rule application would compose nicely
  with cascading_induction.
* Rule confidence decay: rules instantiated against synthetic
  facts (provenance source="induced" or "rule") could compound
  errors over multiple passes. Confidence calibration should
  penalise the depth of rule application.
