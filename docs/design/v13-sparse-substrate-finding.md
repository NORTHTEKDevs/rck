# v13 sparse-substrate empirical finding

Date: 2026-05-21
Status: closed (negative result for substrate switch)

## Question

Can the sparse-binary HRR substrate replace dense bipolar as the
production KB substrate? The 2026-05-20 SCAN-lite validation hit 100%
on both. If the sparse bundle-and-cleanup KB path matches dense on
realistic capacity, we get ~6-13x less RAM per fact.

## Method

`scripts/sparse_capacity_study.py` measures recall@1 on N random
unique-S triples stored in a single `SparseRelationalMemory`, then
queried back. Sweep over D x k x N. Cliff = largest N at recall >= 90%.

## Result (per-shard capacity at recall >= 90%)

| D     | k   | sparse cliff | dense cliff (reference) | ratio |
|-------|-----|--------------|--------------------------|-------|
| 4096  | 80  | 0 facts      | 80 facts                 | 0.00x |
| 4096  | 160 | 10 facts     | 80 facts                 | 0.13x |
| 8192  | 160 | 10 facts     | 160 facts                | 0.06x |
| 8192  | 320 | 20 facts     | 160 facts                | 0.13x |
| 16384 | 320 | 20 facts     | 320 facts                | 0.06x |

Per-atom RAM: sparse is 6.4-12.8x smaller than dense.

## Conclusion

**Sparse substrate is not a drop-in replacement for the dense KB.**
At equal D and density, sparse bundle-and-cleanup loses ~8-16x in
per-shard capacity. Compensating with more shards (system-level
sharding) gives **net higher** total memory than dense for any
realistic KB size.

The earlier SCAN-lite validation passed because the test used a
deterministic primitive->action dict for retrieval, not the
sparse-bundle storage path. SCAN's bundle stress was minimal.

## What sparse HRR IS still useful for

- Direct similarity / codebook cleanup with very large vocabularies
  (no bundling, just per-atom matching). The sparse cosine is fast
  and gives a 6-13x RAM win for the atom table.
- Cheap secondary indexes (e.g. synonym / nearest-neighbor lookup
  alongside the dense relational memory).
- Long-tail entity caches where each entity is queried as a single
  symbol, never bundled with others.

## v13 substrate decision

Dense bipolar stays as the production KB substrate. Sparse stays
as an experimental tool, used only where bundling is not required.

`recommend_shards()` continues to size shards using the **dense**
`TARGET_MAX_FILL_BY_DIM` table (80 facts/shard at D=4096). No
`SparseShardedKnowledgeBase` is wired into `ConsciousAgent`.

## What would change this

A sparse substrate that preserves bundling capacity would need:
- Block-sparse representation (e.g. FBC, Fourier sparse) with
  analytical orthogonality guarantees.
- Or much higher D (>=65k) -- defeats the memory win.
- Or a learned cleanup network (Hopfield-style energy minimisation).

These are open research directions for v14+.
