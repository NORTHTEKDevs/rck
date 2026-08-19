# Production Core - Design

> Status: approved design, not yet implemented.
> Target: v16.0. Scope: the work that every downstream deployment shape needs.

## Problem

RCK v15.3.1 is a sound research artifact - 757 passing tests, reproducible studies,
a paper that retracts its own claims where they didn't hold. It is not yet
deployable, for four reasons that are independent of any particular use case.

Measured on v15.3.1 (commit 63fcb33), all four agents below holding the **same
5,000 ConceptNet facts**, scored with the paper's own valid-object-set labeling:

| `expected_facts` | Shards | Max shard fill | recall@1 |
|---:|---:|---:|---:|
| 50,000 | 1024 | 84 | 100.0% |
| 5,000 | 128 | 187 | 99.8% |
| 1,000 | 16 | 697 | 60.5% |
| 200 | 8 | 1346 | 24.0% |

A knowledge base that outgrows its provisioning loses recall silently - no
exception, no warning at write time, just wrong answers. `shard_balance()`
diagnoses it exactly (`suggested_action='reshard to 256 shards'`); there is no
`reshard()` to call. Every other item below is smaller, but this one is a
correctness bug for any KB that grows after construction.

## Non-goals

Explicitly out of scope for v16.0: multi-tenancy, authentication, billing, a UI,
LLM-based ingestion, and horizontal distribution. Those belong to specific
deployment shapes and are deferred until one is chosen. This design covers only
what all of them share.

---

## 1. Online resharding

**Enabler.** Each shard retains a fact log - it is what `RelationIndex` is built
from (§4.5) and what `shard_balance()` counts. Resharding is therefore a pure
re-bundle from retained data: no derivation is repeated and nothing is lost.

**API.**

```python
agent.reshard(n_shards=None)   # None -> shard_balance().suggested_n_shards
```

Allocate a new shard array, re-route every logged fact by
`blake2b(S || R) % n_new`, re-bundle, rebuild the relation index, invalidate the
chain cache (already versioned and write-invalidated).

**Auto-trigger.** `tell()` checks fill against `target_fill` (80) and reshards
synchronously on the write that crosses it. Growth is by powers of two, so a KB
reaching N facts reshards O(log N) times at O(N) each - amortized O(log N) per
fact, the standard dynamic-array bargain. Correctness-first: a latency spike on
one write is strictly better than silent recall loss on every subsequent read.

**Invariants that must survive a reshard**, each with its own test: asserted
facts, inverse-symmetrized facts, negative facts (`deny`), provenance edges,
skills, query memory, and the fitted calibrator. Provenance/skills/query-memory
are keyed independently of shard index and should pass through untouched - the
tests exist to prove that, not to assume it.

**Open question for implementation.** Whether the fact log holds pre- or
post-symmetrization triples (the paper reports 5,991 raw → 7,080 stored). If pre,
symmetrization must be re-applied during re-bundle. Resolve by reading
`knowledge_base.py` before writing code.

**Acceptance.** Construct at `expected_facts=200`, insert 5,000 facts, score the
400-probe valid-set protocol above: **≥99% recall@1, no manual intervention**.
This is the exact probe that produced the 24.0% row.

---

## 2. API subtraction

`ConsciousAgent` exposes **68 public methods**; `rck/__init__.py` exports **91
names but declares only 29 in `__all__`**, so 62 are accidentally public. Nothing
about this is adoptable - the first question any evaluator asks is "what do I
call?"

This item is pure deletion, and it is the highest value-per-hour work in the plan.

**Frozen surface (v16.0).** Fourteen calls:

```
tell  deny  ask_with_idk  explain_why  discover  induce
detect_conflicts  resolve_conflicts  merge_from  maintain
status_report  save_state  load_state  reshard
```

**Mechanism.** `__all__` becomes exactly the frozen list. A module-level
`__getattr__` on `rck` raises `DeprecationWarning` for the 62 accidental
re-exports. Genuinely experimental surfaces move under `rck.experimental`.

**Constraint.** The 757 tests import internals directly (`from rck.analogy import
...`) and must keep working. Only the top-level `rck.X` convenience re-exports are
gated. Green suite is the gate on this item.

---

## 3. Durability

`save_state` is a full dump. Products crash mid-write.

Append-only write-ahead log of `(S, R, O, source, timestamp)` with fsync on
append, plus periodic snapshots written by atomic rename. Reload is
latest-snapshot + WAL-tail replay.

**This item also produces most of item 4** - the WAL is the decision log. Build
them in this order for that reason.

**Acceptance.** `kill -9` during a write burst → reload → zero fact loss, zero
corruption, verified over 20 randomized kill points.

---

## 4. Replay format

The commercial thesis, and the only item here that is a product rather than
hygiene. Nothing on the market sells *this decision is re-executable in 2029*.

```python
DecisionRecord = {
  rck_version, kb_snapshot_hash, query, answer,
  derivation_tree, calibrator_id, seed, timestamp, signature
}
```

Replay loads the snapshot **by hash**, re-runs the query, and asserts a
byte-identical answer and derivation tree.

**Design constraint that forces this shape.** Bundling is float addition, which is
not associative - bundle order changes the result bit-for-bit. Replay therefore
must run against a stored snapshot, **never** against a re-ingest of the same
facts in a different order. Re-ingestion is not guaranteed to reproduce the
substrate; a snapshot is. Any API that implies "replay by re-feeding the facts" is
incorrect and must not be offered.

**Acceptance.** A record written on machine A replays byte-identically on a clean
machine B with only the record and the snapshot present.

---

## Sequencing

1 → 3 → 4 → 2. Resharding gates everything (it changes on-disk shard layout, so
doing it after durability would invalidate the format). Durability produces the
WAL that replay consumes. Subtraction lands last so the frozen surface includes
`reshard` and the replay calls, and is frozen only once.

## Risks

- **Reshard invalidates shard-index-keyed state.** Mitigated by the invariant test
  matrix; the audit for such state is the first implementation task.
- **Float non-associativity** makes re-ingest replay unsound. Mitigated by design:
  snapshot-hash replay only.
- **Freezing the API too early.** Mitigated by sequencing subtraction last.
- **`expected_facts` remains in the constructor** for backward compatibility, but
  becomes a hint rather than a contract once auto-reshard lands.
