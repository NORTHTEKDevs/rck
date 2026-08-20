# Pluggable Backend - Design and Phase 1 Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement Phase 1 task-by-task.

**Goal:** Let RCK's reasoning layer run on an exact index instead of the HRR substrate, so the layer's value is separable from a substrate that six measured axes say does not earn its place.

**Why now:** Paper sections 5.0 and 5.10 measured the substrate against non-VSA baselines on ingest, memory, query latency, chain discovery, analogy, and federated merge. It won none. The contribution is the layer above it - provenance, calibrated confidence, IDK, six-gate induction, negative facts, contradiction resolution, replayable decisions - and the honest next step is to demonstrate that layer does not depend on holographic representations.

**This is not a deletion.** `HRRKnowledgeBase` stays, stays the default, and stays the research artifact. A second backend makes the claim testable rather than rhetorical.

---

## Measured coupling (the reason this is feasible)

`_shards` is referenced across 34 modules, which looks fatal until you count what is actually done with it:

- **Genuinely HRR-specific: 2 lines**, both persistence - `session.py:37` and `session.py:52` (`np.stack([s._memory for s in ...])`).
- **Substrate-internal by right** (they are *about* sharding, and stay HRR-only): `knowledge_base.py` (55), `shard_balance.py` (19), `shard_sizing.py` (15), `sparse_relational.py` (13), `capacity_profiler.py` (12), `snapshot_hash.py` (5), `session.py` (18).
- **Reasoning layer: 1-4 references each** across ~20 modules (`dreaming.py` 4, `federated_merge.py` 3, then `gap_detection`, `theory_of_mind`, `contradiction`, `curiosity`, `analogy`, `chain_discover`, `abduction`, `cascading_induction`, `negation_propagation`, `rule_instantiation`, `concept_density`, `entity_similarity`, `hierarchical_abstraction`, `subject_summary`, `subject_importance`, `relation_cooccurrence`, `active_learning`, `research`, `bulk_ingest` at 1-2 each). These are overwhelmingly *"enumerate every fact"* loops of the shape:

  ```python
  for shard in kb._shards:
      for fact in shard.facts():
          ...
  ```

**The reasoning layer is therefore already substrate-agnostic in substance and coupled only in syntax.** Phase 1 removes the syntax coupling; Phase 2 adds the second backend.

**Out of scope, permanently:** `agent.py`, `compose.py`, `generative.py`, `bigram.py`, `fep.py`, `server.py`, `mcp_server.py` all touch `.codebook`, but they are the char-LM / generative subsystem (`RCKAgent`, `GenerativeRCK`), not `ConsciousAgent`'s reasoning path. They stay HRR-only and this plan does not touch them.

---

## The interface

Phase 2 will define `KnowledgeBackend` as exactly what the reasoning layer uses:

```
store(fact, *, _log=True)      forget(fact)         query(known, unknown_role, ...)
query_union(...)               answer(known, unknown_role)
size()                         all_facts()          relation_index()
dim   n_shards   wal
```

`n_shards` stays on the interface as an int a dict backend reports as `1`, because `shard_balance()` and diagnostics read it. `reshard()` becomes a no-op there.

---

# Phase 1: remove the syntax coupling

Phase 1 ships alone and is independently valuable: it makes "the reasoning layer does not depend on HRR" a property enforced by a test rather than a claim in a paper.

### Task 1: `all_facts()` on the knowledge base

**Files:** `rck/knowledge_base.py`; test `tests/test_backend_interface.py` (create).

Add to `ShardedKnowledgeBase`:

```python
    def all_facts(self) -> list[dict[str, Hashable]]:
        """Every stored fact, in shard order then insertion order.

        The substrate-agnostic way to enumerate the KB. Reasoning modules
        must use this rather than reaching into `_shards`, so the layer
        can run on a non-HRR backend.
        """
        return [f for shard in self._shards for f in shard.facts()]
```

**Tests:** returns every stored fact; count matches `size()` after symmetrization; stable across a `reshard()` (same set, order may change); empty KB returns `[]`.

Commit.

### Task 2: migrate the reasoning layer off `_shards`

**Files:** the ~20 reasoning modules listed above. **Do NOT touch** `knowledge_base.py`, `shard_balance.py`, `shard_sizing.py`, `sparse_relational.py`, `capacity_profiler.py`, `snapshot_hash.py`, `session.py`, or the generative subsystem.

Find every site:

```
Select-String -Path (Get-ChildItem rck -Recurse -Filter *.py) -Pattern '_shards'
```

Replace enumeration loops with `kb.all_facts()`. Work **one module at a time**, running the full suite after each - a silent behaviour change in `contradiction.py` or `negation_propagation.py` is exactly the kind of defect this project has already shipped twice.

Some sites will not be plain enumeration (e.g. `federated_merge.py` merges shard-to-shard, `dreaming.py` may sample per shard). **Those are legitimately substrate-aware. Leave them, and list them in your report** - they become Phase 2's explicit backend-specific paths. Do not force them through `all_facts()` just to make a grep clean.

Commit per module or in small logical groups.

### Task 3: enforce it with a test

**Files:** `tests/test_backend_interface.py`.

```python
SUBSTRATE_OWNED = {
    "knowledge_base.py", "shard_balance.py", "shard_sizing.py",
    "sparse_relational.py", "capacity_profiler.py", "snapshot_hash.py",
    "session.py",
    # generative subsystem, HRR-only by design
    "agent.py", "compose.py", "generative.py", "bigram.py", "fep.py",
    "server.py", "gen_server.py", "mcp_server.py",
}
# Modules that legitimately need shard-level access; each entry needs a
# reason in the comment above it, and Phase 2 gives them a backend hook.
ALLOWED_EXCEPTIONS = { ... }   # fill from Task 2's report


def test_reasoning_layer_does_not_reach_into_shards():
    """The layer above the substrate must not depend on HRR internals.
    Paper 5.0/5.10 measure that the substrate does not earn its place;
    this test is what keeps the layer portable off it."""
    offenders = []
    for path in (ROOT / "rck").rglob("*.py"):
        if path.name in SUBSTRATE_OWNED or path.name in ALLOWED_EXCEPTIONS:
            continue
        if "_shards" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, f"reasoning modules reaching into _shards: {offenders}"
```

This is the deliverable. Without it the coupling grows back on the next feature.

Commit.

---

## Phase 1 definition of done

- [ ] `python -m pytest -q` green (baseline **833 passed**)
- [ ] `all_facts()` exists, tested, and is the documented way to enumerate
- [ ] No reasoning-layer module references `_shards` except a short, individually justified exception list
- [ ] The guard test fails if a new module reaches into `_shards`
- [ ] **No behaviour change** - this phase is pure refactor; any test needing modification is a red flag to report, not to fix quietly

## Phase 2 (separate plan, not this one)

`DictKnowledgeBase` implementing the interface; `ConsciousAgent(backend="hrr"|"dict")`; backend-aware `session.py` and `snapshot_hash.py`; a parity suite asserting both backends give identical answers on the same facts; and a re-run of `scripts/baseline_study.py` with the dict backend to quantify what the reasoning layer costs when the substrate is removed.

The parity suite is the real prize: if every reasoning test passes on both backends, "the reasoning layer does not require HRR" stops being an argument and becomes a fact.
