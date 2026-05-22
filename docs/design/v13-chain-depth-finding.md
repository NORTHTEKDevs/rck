# v13 reasoning-horizon finding

Date: 2026-05-21
Status: closed (positive result; default changed)

## Question

How deep can RCK reason through a chain of KB lookups before it becomes
unreliable? Where is the bottleneck -- recall, or confidence math?

## Method

`scripts/chain_depth_study.py` builds a 50-fact linear chain
(`a_0 -[next]-> a_1 -[next]-> a_2 -> ...`), stored in a normal
sharded KB at D=4096, n_shards=64. We walk the chain at depths
1, 2, 3, 5, 7, 10, 15, 20, 30, 50 with each of three propagation
rules: `product`, `min`, `geometric_mean`.

## Result

| depth | product (conf, hedge) | min (conf, hedge) | geometric_mean (conf, hedge) |
|-------|-----------------------|-------------------|------------------------------|
| 1     | 0.711 strong          | 0.711 strong      | 0.711 strong                  |
| 5     | 0.204 moderate        | 0.571 strong      | 0.618 strong                  |
| 10    | 0.056 weak            | 0.442 strong      | 0.495 strong                  |
| 15    | 0.011 uncertain       | 0.236 moderate    | 0.378 strong                  |
| 20    | 0.010 uncertain       | 0.183 moderate    | 0.296 moderate                |
| 30    | 0.010 uncertain       | 0.109 moderate    | 0.172 moderate                |
| 50    | 0.010 uncertain       | 0.039 weak        | 0.061 weak                    |

**Recall is 100% at every depth tested up to 50.** The KB substrate
itself can correctly chain at least 50 hops; nothing breaks. The
"reasoning horizon" was a property of the **propagation rule**,
not the substrate.

Reasoning horizons (largest depth at correct + hedge >= weak):

* `product`:          10 hops
* `min`:              50+ hops
* `geometric_mean`:   50+ hops

## Why product over-penalises

Each clean lookup returns cosine ~0.7 -- this is "strong" for HRR
because the cosine is dominated by codebook crosstalk noise, not by
the probability of the answer being correct. Multiplying these scores
across N hops gives 0.7^N, which is exponentially small even when
every individual hop is in fact correct.

A cosine of 0.7 in HRR cleanup is NOT a probability of correctness;
it's a similarity measure that empirically corresponds to ~99%+ correct
on clean lookups. The propagation rule needs to reflect that asymmetry.

## Decision

Default propagation rule changes from `product` to `geometric_mean`
in `rck.confidence_propagation.PropagationConfig`. Existing tests
that don't pass an explicit `rule=` still pass under the new default
(verified, 363/363).

For chains where every link is independent AND scores are calibrated
to probabilities, `product` remains the correct choice -- callers
can still set `rule="product"` explicitly.

## What this unlocks

* The new `rck.chain_walker.walk_chain` becomes useful at 5-30 hops
  rather than capped at 3.
* Multi-step reasoning ("what is the X of the Y of the Z of W")
  scales to chains LLMs would struggle to follow.
* Skill discovery (`rck.skills`) can record long successful
  patterns instead of being limited to 2-3 hop templates.

## What this does NOT change

* Recall fidelity (still bounded by per-shard capacity cliff
  documented in `docs/design/v13-sparse-substrate-finding.md`).
* Calibration: a "strong" verdict at depth 50 is qualitatively
  different from a "strong" verdict at depth 1 (more error modes
  upstream). Higher-level callers should still cross-check via
  `rck.self_verify` on the final answer.
