# v13 chain-discovery finding

Date: 2026-05-21
Status: closed (positive result; capability added)

## Question

`chain_walker` *executes* a chain when the relation sequence is known
in advance. But real questions don't come with chain specs. Can we
*discover* the relation sequence by searching the HRR-KB?

## Method

`rck/chain_discover.py` implements BFS over the sharded KB. At each
frontier node it queries every known relation (`kb.query(S=node, R=rel)`)
and treats the returned top-K objects as next-frontier nodes. The
heap is keyed by propagated geometric-mean confidence, so the most
plausible chains come out first.

A goal is a callable that says "is THIS node what I'm looking for?":
* `Goal.symbol("paris")` -- exact atom match
* `Goal.relation_value("isa", "continent")` -- (atom, isa, continent) holds
* `Goal.custom(lambda node, kb: ...)` -- arbitrary

`scripts/chain_discovery_study.py` runs the discoverer on real KBs
loaded from `data/commonsense_kb.jsonl`, `ultra_kb.jsonl`, and
`massive_kb.jsonl`, with 2-hop transitive `(start, target)` probes.

## Results

| KB          | facts | shards | probes | hits | success | avg latency |
|-------------|-------|--------|--------|------|---------|-------------|
| commonsense | 716   | 16     | 30     | 29   | 97%     | 14.5 ms     |
| ultra       | 2599  | 64     | 30     | 30   | 100%    | 7.7 ms      |
| massive     | 4109  | 128    | 30     | 30   | 100%    | 55.5 ms     |

Sub-30ms latency for chain discovery up to ~2.6k facts. Auto-shard
sizing keeps per-shard fact counts low, which lets the per-relation
KB query stay cheap; the cost grows linearly with the codebook's
relation count (massive KB has more relations).

## What chain discovery enables

* Answering questions where the chain shape isn't a known template.
  Example traces:
  * `elephant -> tree`         via `has -> partof`
  * `leaf -> forest`           via `partof -> locatedin`
  * `wing -> feathers`         via `partof -> hassubtype -> has`
* Auto-populating the `SkillLibrary` (`record_success` is called for
  every successful discovered chain via `agent.reason()`).
* Detecting "highway" relations -- those that appear in many chains
  reveal the KB's most informative reasoning links. On the
  commonsense KB the top highway relations are:
  `partof` (14x), `usedfor` (5x), `locatedin` (4x), `has` (3x),
  `color` (3x).

## Reverse hops

`allow_reverse=True` lets the search traverse `(?, R, O)` edges
in addition to `(S, R, ?)`. This matters when the auto-inverse
symmetrisation in `bulk_ingest` was NOT run -- in those cases the
backward link is the only one stored. The test
`test_discover_uses_reverse_edges_when_allowed` pins this down.

## Failure modes observed

* "Semantically coincidental" chains: `carrot -> fruit via color ->
  category` chains through carrot's color (orange), then the
  category of orange (fruit). The chain is correct in HRR scoring
  but the semantics are accidental. This is an intrinsic property
  of unconstrained graph search; a higher-level filter on
  relation-pair plausibility (skill library) is the right place to
  prevent it.
* Hard depth cliff: at `max_depth >= 5` on a 4k-fact KB the search
  begins to time out (>200ms) and rarely returns useful chains.
  In practice 3-4 hops is the sweet spot for interactive use.

## ConsciousAgent integration

```python
agent = ConsciousAgent(...)
# Discover the chain.
spec = agent.discover("france", "continent", max_depth=4)
# Execute it.
res = agent.reason("france", spec["relations"],
                   directions=spec["directions"])
```

Discovery + execution = end-to-end multi-hop reasoning for ad-hoc
queries without natural-language template parsing.
