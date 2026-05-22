# RCK v14 narrative -- the reasoning stack

Date: 2026-05-21
Audience: someone returning to the project cold

## What changed from v12 to v14

v13/v14 turned RCK from a "stores facts and answers single queries"
system into a complete reasoning stack with introspection,
self-verification, fact derivation, and rule induction.

Stack, from substrate upward:

```
                +--------------------------------+
                | EpistemicAnswer (KNOWN/AMBIG/IDK)
                | -------------------------------|
                | CalibratedAnswer (per-source)  |
 surface layer  | SetCandidate (intersect/union) |
                | AnalogyResult (A:B::C:?)        |
                +----+---------------------------+
                     |
                +----v---------------------------+
                | RuleStore (forall-rules)       |
                | SkillLibrary (chain patterns)  |
   knowledge    | ProvenanceStore (per-fact)     |
                | ChainCache (memoization)       |
                +----+---------------------------+
                     |
                +----v---------------------------+
                | chain_induction (derived facts)|
                | chain_walker / chain_discover  |
   reasoning    | cascading_induction (fixed-pt) |
                | confidence_propagation (geo)   |
                +----+---------------------------+
                     |
                +----v---------------------------+
   substrate    | ShardedKnowledgeBase + HRR     |
                | RelationalMemory (Plate)       |
                | Codebook + bind/bundle/unbind  |
                +--------------------------------+
```

## The core findings, in order

### 1. Capacity cliff at max-shard-fill ~80 (D=4096)

Documented earlier (v11). Shard sizing now auto-computes
`n_shards >= n_facts / target_fill` to stay under the cliff.

### 2. Sparse-HRR substrate: NOT a drop-in replacement

`docs/design/v13-sparse-substrate-finding.md`. Sparse-binary HRR
saves 6-13x RAM per atom but per-shard capacity drops 8-16x, so
net memory is WORSE at scale. Dense bipolar stays as the production
KB substrate. Sparse stays useful for similarity-only caches.

### 3. Reasoning horizon is propagation-rule limited, not substrate

`docs/design/v13-chain-depth-finding.md`. The KB can chain 50+ hops
correctly. The OLD product-rule propagation collapsed confidence
exponentially even for correct chains. Switching default to
**geometric-mean** extends usable depth from ~10 hops to 50+.

### 4. Chain discovery is fast and accurate

`docs/design/v13-chain-discovery-finding.md`. BFS over the KB using
per-relation HRR queries hits 100% on 2-hop probes in 8-55ms across
716-4109 fact KBs. The "highway" relations (partof, locatedin, etc)
show up in most successful chains -- empirical evidence the KB has
informative shortcut structure.

### 5. Fact induction with empirical filters

`docs/design/v13-chain-induction-finding.md` and
`v13-cascade-induction-finding.md`. Confident chains can be
SYNTHESISED back into direct edges. Four gates discovered through
empirical failure analysis:

* inverse-pair (author -> wrote) -- catches hub round-trips.
* non-transitive same-relation (wrote -> wrote) -- catches HRR
  cleanup artefacts.
* intermediate-cycle -- chain answer == an earlier node.
* lifting-relation gate -- only propagate the LAST relation when
  the FIRST hop is `isa | partof | locatedin | memberof`;
  otherwise fall back to generic `implies`.

After filtering, cascading induction reaches a fixed point in
~4 rounds on the commonsense KB, adding ~11 NEW verified facts
all tagged with `source="induced"`.

### 6. Rule extraction from skill library

`docs/design/v14-narrative.md` is the integration layer for
`rck/rule_extraction.py`. The same gates that filter induced facts
filter extracted rules. On the commonsense KB after a cascade pass,
the system extracts 25 universal rules at confidence 1.0:

```
(X locatedin Y) and (Y continent Z) => (X continent Z)    [28x]
(X capitalof Y) and (Y continent Z) => (X implies Z)      [26x]
(X partof Y) and (Y locatedin Z)    => (X locatedin Z)    [3x]
```

### 7. Skill-prior-guided discovery

`docs/design/v13-skill-prior-speedup.md`. Reordering relation
expansion by skill utility gives +1 hit (40/40 vs 39/40 cold) and
modest 1.06x speedup on commonsense. Effect grows with relation
count; on a Wikipedia-scale KB the win should be larger.

### 8. Analogical reasoning at the relational layer

`docs/design/v13-analogy-finding.md`. Solve A:B::C:? by finding R
such that (A, R, B) holds, then applying R to C. Benchmark on
commonsense KB: 88.7% answer accuracy, 85.2% relation accuracy on
115 probes. Failures mostly come from multi-valued relations (`has`)
where multiple body parts are valid analogs.

### 9. Confidence calibration by provenance

`rck/confidence_calibration.py`. User-provided facts keep full
score; induced facts are discounted by `0.9^via_hops` (floor 0.5);
multi-source facts get a small bonus. CalibratedAnswer exposes
both raw and calibrated scores so callers can audit.

### 10. Explicit IDK

`rck/idk_detection.py`. EpistemicAnswer classifies each query as
KNOWN, AMBIGUOUS, or IDK. The system now REPORTS uncertainty
instead of always picking a top-1 atom.

### 11. Branching set reasoning

`rck/set_reasoning.py`. Intersection / union / difference across
multiple constraints. Solves "European countries that use the euro"
without a query language.

### 12. Chain memoization

`rck/chain_cache.py`. LRU cache (bounded 256) keyed by (start,
target). Repeat queries are O(1) instead of re-running BFS.

### 13. Cross-shard distribution confirmed

`docs/design/v13-cross-shard-chain-finding.md`. 2-hop chains touch
~2 distinct shards at all realistic shard counts; >95% of chains
have endpoints on different shards. Reasoning IS distributed
across the shard pool, not concentrated.

## What the agent now exposes

```python
agent = ConsciousAgent(dim=4096, expected_facts=N, ...)

agent.tell(s, r, o)                          # +provenance
agent.ask_with_idk({S, R}, "O")              # KNOWN/AMBIG/IDK
agent.calibrated_ask({S, R}, "O")            # per-source score
agent.intersect([{...}, {...}], "S")         # set intersection
agent.union([{...}, {...}], "S")             # set union
agent.discover(start, target)                # BFS chain (+ cache)
agent.reason(start, [r1, r2, r3])            # walk a known chain
agent.induce(start, target)                  # chain + induct
agent.analogy(A, B, C)                       # A:B::C:?
agent.extract_rules(min_support=2)           # symbolic rules
agent.verify_fact(s, r, o)                   # self-verify a triple
```

## Suite

447 tests passing.

## Open work

* Higher-order rule INSTANTIATION: given a stored Rule, apply it
  forward to derive new (s, r, o) tuples without a chain walk.
* Confidence-weighted relation choice in analogy (currently
  argmax over scored relations).
* Parallel chain walker for cross-shard speedup at deep chains.
* Cache invalidation on bulk_load_triples (currently caller must
  call agent.chain_cache.clear()).
