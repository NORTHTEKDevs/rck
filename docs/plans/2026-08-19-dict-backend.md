# Phase 2: DictKnowledgeBase and the parity suite

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run RCK's reasoning layer on an exact index, and prove with a parity suite that it behaves identically. That converts "the substrate does not earn its place" from a measured argument into a demonstrated fact.

**Architecture:** A second knowledge base implementing the surface the reasoning layer already uses (Phase 1 established what that surface is). `ConsciousAgent(backend="hrr"|"dict")` selects it. HRR stays the default and stays the research artifact; nothing is deleted.

**Tech Stack:** Python 3.11+ stdlib. No new dependencies. No numpy in the dict path.

---

## Verified state

- Phase 1 shipped: `all_facts()` exists, 16 reasoning modules were migrated off `kb._shards`, and `tests/test_backend_interface.py` enforces that with a `"._shards"` check. Suite is **847 passed**.
- **Five modules legitimately still touch `_shards`** and are in `ALLOWED_EXCEPTIONS`:
  - `federated_merge.py` - sums two shards' `_memory` HRR tensors directly.
  - `dreaming.py::compress_duplicates` - writes `shard._facts` directly.
  - `curiosity.py::detect_global_gaps`, `research.py::_related_entities`, `subject_summary.py` - each has a `break` nested inside the per-fact loop with no matching outer break, so the cap applies **per shard** and results depend on shard count.
- The surface the layer actually uses, from Phase 1's measurement: `store`, `forget`, `query`, `query_union`, `answer`, `size`, `shard_sizes`, `all_facts`, `relation_index`, `reshard`, and the fields `dim`, `n_shards`, `seed`, `wal`, `codebook`, `_fact_count`.

---

## The design decision this plan turns on

`DictKnowledgeBase` must expose **`_shards` as a single pseudo-shard** with `.facts()` and `._facts`, so the five exception modules keep working unchanged.

That has a consequence which must be **documented and tested, not hidden**: with one shard, the mis-nested `break` in `curiosity` / `research` / `subject_summary` applies globally instead of per shard. Those three functions will therefore legitimately return *different* results on the two backends. That is a pre-existing latent bug in those modules (Phase 1 found it), surfaced by the backend swap - **not** a parity failure to paper over.

**So: the parity suite asserts equality everywhere EXCEPT those three functions, which get an explicit documented-divergence test instead.** Do not force them equal. Do not "fix" the break nesting in this plan - that is a behaviour change and belongs in its own commit.

---

### Task 1: `DictKnowledgeBase`

**Files:** Create `rck/dict_knowledge_base.py`; test `tests/test_dict_backend.py`.

Implement the full surface above with an exact index. Sketch:

```python
@dataclass
class DictKnowledgeBase:
    dim: int = 4096          # accepted and reported, unused
    n_shards: int = 1        # always 1; reshard() is a no-op
    seed: int = 0
    wal: WriteAheadLog | None = None

    def store(self, fact, *, _log=True): ...
    def query(self, known, unknown_role, top_k=3, shard_subset=None, cleanup="local"): ...
```

Requirements:

- `query` returns `[(symbol, score), ...]` like the HRR path. **Exact matches score 1.0**; a miss returns `[]`. The IDK layer thresholds on score, so exact hits must clear its threshold and misses must produce no candidates.
- `query` must handle every slot pattern the HRR version does: `(S,R)->O`, `(S,O)->R`, `(R,O)->S`, and fan-out when a slot is missing. Read `ShardedKnowledgeBase.query` and match its **contract**, not its implementation.
- `answer(known, unknown_role)` returns `(symbol|None, score)`.
- `all_facts()`, `size()`, `shard_sizes()` -> `[len(facts)]`, `relation_index()` -> a `RelationIndex` over the single pseudo-shard.
- `reshard(n=None)` is a documented no-op returning `{"resharded": False, ...}`.
- `wal` support: `store`/`forget` append when a WAL is attached and `_log` is true, exactly like the HRR path, so durability works on both backends.
- `codebook`: expose a minimal stand-in only if something needs it; **if nothing does, do not invent one** - report that instead.

**Tests:** store/query round-trip on every slot pattern; multi-valued `(S,R)` returns all objects; a miss returns `[]`; `forget` removes; `size` counts; `all_facts` matches insertion order; WAL append fires.

Commit.

### Task 2: backend selection on `ConsciousAgent`

**Files:** `rck/conscious_agent.py`; `tests/test_dict_backend.py`.

Add `backend: str = "hrr"`. In `__post_init__`, build `knowledge` and `beliefs` from the chosen backend. Default unchanged, so the 847 existing tests must stay green untouched.

`shard_balance()` on the dict backend should report something honest and non-crashing (one shard, no cliff, no reshard suggestion).

Commit.

### Task 3: the parity suite - the actual deliverable

**Files:** `tests/test_backend_parity.py`.

Drive **both** backends through the same operations on the same facts and assert identical results. Cover, at minimum:

- `tell` / `deny` / `ask_with_idk` including the IDK state on unknown queries
- `explain_why` derivation trees (identical structure, sources, and leaves)
- `discover`, `reason`, `induce`
- `detect_conflicts`, `resolve_conflicts`
- `extract_rules`, `instantiate_rules`
- `maintain()` end to end
- `checkpoint` / `load_session` round-trip on the dict backend
- multi-hop chains at depth 2..6 on the CLUTRR-style generator from `scripts/clutrr_style_study.py`

Parametrize over `backend` wherever possible so one test body covers both.

Then the documented-divergence test:

```python
@pytest.mark.parametrize("fn", ["detect_global_gaps", "_related_entities",
                                "summarize_subject"])
def test_shard_dependent_functions_diverge_by_design(fn):
    """These three cap their results with a `break` nested inside the
    per-fact loop and no matching outer break, so the cap applies per
    shard. With one pseudo-shard the dict backend applies it globally.
    The divergence is a pre-existing latent bug surfaced by the backend
    swap, not a parity failure -- this test pins it so it is visible."""
```

Commit.

### Task 4: persistence and hashing across backends

**Files:** `rck/session.py`, `rck/snapshot_hash.py`; tests.

`session.py` holds the only two genuinely HRR-specific lines in the codebase (`np.stack([s._memory ...])`). Branch on backend: persist the fact list for dict, arrays for HRR. `load_session` must restore the right backend, recorded in `meta.json`.

`snapshot_hash.state_hash` hashes `_memory` bytes; for the dict backend hash the canonical fact list plus hyper instead. **Decision record:** the two backends will produce different hashes for the same logical facts. That is correct - a `DecisionRecord` pins a substrate state, and the substrates differ. Assert it explicitly rather than leaving it implied.

Tests: dict-backend session round-trip preserves facts *and* provenance (regression for the `2b3dfac` bug); `state_hash` is stable per backend; replay `VERIFIED` on the dict backend.

Commit.

### Task 5: measure what the substrate cost

**Files:** `scripts/baseline_study.py` (extend), `data/baseline_study.json`.

Add the dict backend as a third row alongside `dict` and `rck`. Now the comparison is not "RCK vs a toy index" but **"the same reasoning layer on two substrates"**, which is the honest framing and the one the paper needs.

Report ingest, RSS, recall@1, query median at the 10k/30k/100k tiers.

Commit.

---

## Definition of done

- [ ] `python -m pytest -q` green (baseline **847 passed**); default behaviour unchanged
- [ ] Every parity test passes on both backends
- [ ] The three shard-dependent divergences are pinned by an explicit test, not hidden
- [ ] Dict-backend session round-trip preserves facts and provenance
- [ ] `baseline_study.py` reports the same reasoning layer on both substrates
- [ ] Nothing deleted; `backend="hrr"` remains the default

## Report explicitly

- Any method on the surface that could not be implemented exactly, and why.
- Any parity test that required a tolerance rather than exact equality - each one is a real semantic difference and needs naming.
- Whether `codebook` was needed at all on the dict path.
