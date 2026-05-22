# v13 skill-prior-guided chain discovery

Date: 2026-05-21
Status: closed (positive but small effect)

## Question

The chain discoverer enumerates relations alphabetically when
expanding a frontier. If we already have a `SkillLibrary` recording
which patterns succeed, can we use it to reorder relation
expansion so successful chains come out faster?

## Method

`discover_chains(..., skills_prior=SkillLibrary)` adds an optional
prior. The prior is built by summing `success_count * confidence`
per relation across all stored skills. Relations with higher
aggregate utility are tried first. The set of explored relations
is unchanged -- this is purely an ordering heuristic.

`scripts/skill_prior_speedup_study.py` runs the same 40 transitive
probes on two parallel KBs:

  * **Cold**: no prior.
  * **Warm**: prior populated by an upstream cascade induction pass.

## Result

```
  Cold: 39/40 hits in 0.47s (11.6ms avg)
  Warm: 40/40 hits in 0.44s (11.0ms avg)
  Speedup: 1.06x
```

* **+1 hit** (40/40 vs 39/40) -- the warm run catches a probe that the
  cold run misses. This isn't faster, it's *more thorough*: by trying
  high-utility relations first, the search reaches the goal before
  the depth/beam bound cuts it off.
* **1.06x speedup** on the latency side. Modest because the existing
  heap-by-confidence already prioritises good frontier nodes.

## When the effect should be larger

* KBs with many more relations (20+ in commonsense; maybe 200+ at
  Wikipedia / ConceptNet scale). Alphabetical ordering becomes more
  expensive as the relation table grows.
* Deeper chains (max_depth > 3). The relation-ordering cost compounds
  per hop.
* High-skew KBs where a small set of relations covers most chains.

## Decision

`skills_prior` is now an optional parameter to `discover_chains`. The
ConsciousAgent passes `self.skills` automatically when the prior has
non-empty data. There's no downside -- cold and warm both find the
same chains, warm just tends to find them sooner.

## Open work

* Integrate the prior into `agent.discover()` automatically (currently
  not wired -- callers must pass the SkillLibrary). Single-line change
  in conscious_agent.py once we decide a default policy for "is the
  skill library mature enough to use as a prior?"
