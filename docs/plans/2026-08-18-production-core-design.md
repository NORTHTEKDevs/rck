# Production Core - Design

> Status: item 1 implemented and verified; items 2-4 designed, not yet built.
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

**Auto-trigger.** `store()` checks fill against a per-`dim` `target_fill` and
reshards synchronously on the write that crosses it, but only when resharding can
actually relieve that shard. Correctness-first: a latency spike on one write beats
silent recall loss on every subsequent read.

**Growth policy - the part that took three attempts.** Two naive policies both
fail, and the failures are worth recording because neither is obvious:

1. *Double on every overloaded write.* `_shard_index` routes on `(S, R)` alone, so
   every fact sharing one key is pinned to the same shard **at every `n_shards`**.
   A key past the cliff can never be split, so each subsequent write re-triggers a
   doubling: 90 facts of `(alice, authored, paper_i)` reached **8192 shards**
   (recommendation: 8), one full O(N) re-bundle each. A 200-write loop attempts
   ~2^120 shards and hangs the machine. Doubling is the dynamic-array bargain, and
   it does not apply here, because array elements are re-routable and `(S, R)`
   keys are not.
2. *Stop at `recommend_shards()`.* Bounded, but its 1.25x load-skew safety factor
   is too small for real multi-valued data: 5,000 ConceptNet facts left **11/256
   shards over the cliff** (99.8% recall) - the same silent-degradation class this
   item exists to remove.

The shipped policy: grow only when the overloaded shard holds no single `(S, R)`
key above `target_fill` (a hot key is a hard ceiling, not a sizing problem), and
overshoot `recommend_shards()` by a bounded `_GROWTH_OVERSHOOT` (8x), hard-capped
at `_MAX_SHARDS`.

**Known limitation.** Two distinct `(S, R)` keys that each hold close to
`target_fill` facts *and* collide under `blake2b(S||R) % n` cannot be separated at
any reachable `n_shards`. Chasing such a pair does not converge - it drove 2,400
facts to 4,096 shards before the overshoot bound was added. Growth therefore stops
and leaves the shard overloaded; `shard_balance()` still reports it, so the
condition is visible rather than silent. This is a substrate property, not a bug.

**Invariants that survive a reshard**, each with a test: asserted facts,
inverse-symmetrized facts, negative facts (`deny`), provenance edges, skills, and
query memory. `ConsciousAgent.n_shards` does *not* track `knowledge.n_shards`
after a reshard, so `session.py` persists the live per-KB counts instead - without
that, any session that had auto-resharded failed to reload with `IndexError`.

**Resolved open question.** `RelationalMemory._facts` holds post-symmetrization
facts as stored, so re-bundling must *not* re-symmetrize (it would double the
inverse edges).

### Status: DONE - measured, not asserted

| | v15.3.1 | shipped |
|---|---:|---:|
| recall@1 (200-provisioned, 5,000 facts) | 24.0% | **100.0%** |
| Overloaded shards | 8/8 | **0** |
| Max shard fill | 1346 | 78 |
| 200 hot-key writes | hangs the machine | `n_shards=8`, bounded |
| `target_fill` at `dim=2048` | 80 (real cliff is 60) | 60 |
| Session round-trip after reshard | `IndexError` | exact round-trip |

Suite: **769 passed**. Acceptance is the exact 400-probe valid-set protocol that
produced the 24.0% row, run with no manual `reshard()` call.

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

A durable, versioned record of a single answer that can be re-executed later and
checked for bit-identical output. This is what makes RCK's determinism usable
rather than merely true: an answer given today stays defensible years from now,
against the exact substrate state that produced it.

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

- ~~**Reshard invalidates shard-index-keyed state.**~~ Resolved. Provenance,
  skills, and query memory are keyed independently and passed through untouched;
  `RelationIndex` rebuilds per call. The one real case was
  `ConsciousAgent.n_shards` going stale, which broke session reload - fixed by
  persisting live per-KB counts.
- **Float non-associativity** makes re-ingest replay unsound. Mitigated by design:
  snapshot-hash replay only.
- **Freezing the API too early.** Mitigated by sequencing subtraction last.
- **`expected_facts` remains in the constructor** for backward compatibility, but
  is now a hint rather than a contract.

## Definition of done (corrected)

An earlier draft of this design required "zero overloaded shards after unattended
growth." That is **not achievable in general** and the criterion was wrong: a
colliding pair of near-cliff `(S, R)` keys cannot be separated at any reachable
`n_shards` (see item 1's known limitation). The achievable contract is:

- No shard is over `target_fill` **because of under-provisioning** - only ever
  because of a hot key or an unsplittable collision.
- Growth is bounded by `_GROWTH_OVERSHOOT` x `recommend_shards()`, capped at
  `_MAX_SHARDS`, so the reshard loop always terminates.
- Any residual overloaded shard is reported by `shard_balance()`, never silent.
