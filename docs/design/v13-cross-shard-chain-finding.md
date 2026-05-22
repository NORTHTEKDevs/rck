# v13 cross-shard chain study

Date: 2026-05-21
Status: closed (positive observation)

## Question

Each hop in a walked chain lands on the shard determined by
`hash(subject || relation) % n_shards`. Do chains tend to STAY on
one shard (HRR cleanup noise leading to local cycles) or do they
properly TRAVERSE shards (genuine distributed reasoning)?

## Method

`scripts/cross_shard_chain_study.py` runs 60 transitive probes from
the commonsense KB at four shard counts (16, 64, 128, 256) and
records the number of distinct shards each successful chain visits.

## Results

| n_shards | successes | avg distinct shards | endpoints differ |
|----------|-----------|---------------------|------------------|
| 16       | 59/60     | 1.92                | 86.4%            |
| 64       | 60/60     | 2.00                | 95.0%            |
| 128      | 60/60     | 2.02                | 96.7%            |
| 256      | 60/60     | 2.02                | 98.3%            |

* **Chains touch ~2 distinct shards** -- matching the 2-hop depth of
  the probes. This is genuinely distributed reasoning, not a single
  shard answering the question.
* **Endpoints land on different shards >95% of the time** at
  realistic shard counts (>=64). At 256 shards, the start and end
  agree only 1.7% of the time -- by chance.
* **Success rate is 100%** at higher shard counts. Auto-sharding via
  `recommend_shards` keeps each shard well below the capacity cliff.

## Implication

Chain reasoning IS parallelizable across shards. A future optimisation
could dispatch per-hop queries to shard workers in parallel; the gain
is bounded by chain depth (no win for 1-hop queries, 2-3x for typical
chains).

## What the study does NOT show

* It doesn't measure latency at scale (we'd need 10x more facts).
* It only looks at 2-hop chains. 4-5 hop chains likely visit more
  shards; would expect avg approaching 4-5 at large `n_shards`.

## Open work

* Re-run on a 5000+ fact KB to see if avg crossings scales linearly
  with depth.
* Build a parallel walker that fires per-hop KB queries in worker
  threads.
