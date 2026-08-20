# Phase 2: DictKnowledgeBase and the parity suite

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
>
> **Revision 2.** Adversarially reviewed before implementation; three Critical
> findings folded in, marked **[R2]**. One review finding was checked against
> the source and **rejected** - see "Rejected finding" at the end. Do not
> re-litigate either set.

**Goal:** Run RCK's reasoning layer on an exact index, and prove with a parity suite that it behaves identically where it should - and document precisely where it should not. That converts "the substrate does not earn its place" from a measured argument into a demonstrated fact.

**Architecture:** A second knowledge base implementing the surface Phase 1 established. `ConsciousAgent(backend="hrr"|"dict")`. HRR stays the default and stays the research artifact; nothing is deleted.

**Tech Stack:** Python 3.11+ stdlib. No new dependencies. No numpy in the dict path.

---

## Verified state

- Phase 1 shipped: `all_facts()` exists, 16 reasoning modules migrated off `kb._shards`, `tests/test_backend_interface.py` enforces it via a `"._shards"` check. Suite **847 passed**.
- Five modules legitimately still touch `_shards` (`ALLOWED_EXCEPTIONS`): `federated_merge.py`, `dreaming.py::compress_duplicates`, and `curiosity.py::detect_global_gaps` / `research.py::_related_entities` / `subject_summary.py::summarize_subject`.
- Surface in use: `store`, `forget`, `query`, `query_union`, `answer`, `size`, `shard_sizes`, `all_facts`, `relation_index`, `reshard`, plus fields `dim`, `n_shards`, `seed`, `wal`, `codebook`, `_fact_count`.
- `InductionPolicy.min_confidence = 0.20`; `IDKPolicy.idk_threshold = 0.08`. Both matter below.

---

## The three things this plan must get right

### [R2] A. Parity is NOT "identical everywhere". Three divergence classes are expected.

Asserting blanket equality will produce failures that are correct behaviour, and an implementer will then "fix" them by weakening tests. Name them up front:

1. **Shard-partition-dependent functions.** The three exception modules above each cap results with a `break` nested inside the per-fact loop, with no matching break on the outer shard loop - verified in source: `curiosity.detect_global_gaps` has `for shard` at indent 4, `for fact` at 8, `break` at 12. So the cap applies **per shard**. A one-shard dict backend applies it globally. Different results, by design. Pin with an explicit divergence test.
2. **[R2] Density-dependent epistemic state.** HRR crosstalk under load can push a *true, stored* answer below `idk_threshold` or into a near-tie with noise, flipping `ask_with_idk` to IDK/AMBIGUOUS for a query with one unambiguous answer. Measured on a deliberately overloaded single shard: the stored answer scored 0.0466 and ranked *below* an unrelated entity at 0.0496. The dict backend correctly returns KNOWN. **Do not assert IDK-state equality on KBs dense enough to cross the capacity cliff.** Either keep parity KBs under the cliff and say so, or carve the divergence out explicitly.
3. **[R2] Induction Gate 1 is substrate-relative, not identical-by-construction.** `cascade_induct` calls `walk_chain(...)` with no `config=`, so it always uses the default `geometric_mean` rule. On the dict backend every hop is exactly 1.0, so chain confidence is `1.0 * chain_decay**(n-1)` (0.95^5 ~ 0.77 at depth 6) - **always above `min_confidence=0.20`, so Gate 1 can never reject.** On HRR, real crosstalk legitimately pushes some true chains below the floor. The dict backend will therefore commit inductions HRR rejects. That is a genuine behavioural difference and must be **tested at realistic shard load**, not assumed away. Task 3 owns this.

### [R2] B. `federated_merge` will crash on the dict backend unless this plan says otherwise.

`federated_merge.py` does `target.knowledge._shards[i].merge(src_shard)`, which sums `RelationalMemory._memory` numpy tensors. A dict pseudo-shard has no `_memory` and no `merge()`. Today that is an `AttributeError` the moment either side is a dict-backend agent, and `merge_from` appears nowhere in the original Task 3 list, so it would have shipped untested.

**Decision:** give the pseudo-shard a `merge(other)` that dedup-unions `_facts` (the exact-index equivalent of a bundle sum), and **raise a clear `TypeError` on mixed-backend merges** - summing an HRR tensor into a dict index is meaningless. Task 3 must test both paths.

### C. The pseudo-shard, and the index invariant

`DictKnowledgeBase` exposes `_shards` as a single pseudo-shard with `.facts()`, `._facts`, and `.merge()`, so the exception modules keep working.

**[R2] Index invariant.** `dreaming.compress_duplicates` reassigns `shard._facts = keep` directly, bypassing `store`/`forget`. If `query()` reads a separately-maintained index, it will keep returning removed duplicates at score 1.0 after a dedup. **Either derive the query index fresh from `_facts` on every call, or rebuild it whenever `_facts` is reassigned.** State which in the docstring. Task 3 must **re-query a deduped fact**, not merely count facts.

---

### Task 1: `DictKnowledgeBase`

**Files:** Create `rck/dict_knowledge_base.py`; test `tests/test_dict_backend.py`.

Implement the surface with an exact index.

- `query` returns `[(symbol, score), ...]`; exact matches score **1.0**, misses return `[]`. Handle every slot pattern `ShardedKnowledgeBase.query` supports - `(S,R)->O`, `(S,O)->R`, `(R,O)->S`, and fan-out with a missing slot. Match its **contract**, not its implementation.
- **[R2] Multi-valued tie-break must be explicit and deterministic: insertion order**, matching `all_facts()`. `chain_walker.walk_chain` and `answer()` take `results[0]` with no ambiguity handling, so a multi-valued intermediate hop can silently walk a different chain on each backend. Document this as a known non-parity point.
- `answer(known, unknown_role)` -> `(symbol|None, score)`.
- `all_facts()`, `size()`, `shard_sizes()` -> `[n]`, `relation_index()` over the pseudo-shard.
- `reshard(n=None)` is a no-op returning `{"resharded": False, ...}`. **[R2]** `session.load_session` calls `agent.beliefs.reshard(...)` - confirm the no-op return shape does not break it.
- `wal` support on `store`/`forget` exactly as the HRR path, so durability works on both.
- `codebook`: provide a stand-in **only if something actually needs it**. If nothing does, do not invent one - report that.

**Tests:** every slot pattern; multi-valued returns all objects in insertion order; miss returns `[]`; `forget` removes; `size`; `all_facts` order; WAL append fires; `merge` dedup-unions; mixed-backend merge raises `TypeError`.

Commit.

### Task 2: backend selection

**Files:** `rck/conscious_agent.py`; `tests/test_dict_backend.py`.

Add `backend: str = "hrr"`; build `knowledge` and `beliefs` from it in `__post_init__`. Default unchanged - the 847 existing tests must stay green untouched. `shard_balance()` on dict must report honestly and not crash (one shard, no cliff, no suggestion).

Commit.

### Task 3: the parity suite - the deliverable

**Files:** `tests/test_backend_parity.py`.

Parametrize over `backend` and assert equality for: `tell`/`deny`/`ask_with_idk`, `explain_why` trees, `discover`, `reason`, `induce`, `detect_conflicts`, `resolve_conflicts`, `extract_rules`, `instantiate_rules`, `maintain()`, `checkpoint`/`load_session`, and multi-hop chains at depth 2..6.

**[R2] Required additions:**
- **Anchor to ground truth, not just to each other.** Cross-backend equality can pass while both are wrong. Use `scripts/clutrr_style_study.py`'s `symbolic_infer` / `example.end` as an independent oracle and assert **each backend against it**.
- **Induction-gate parity at realistic load** (near/over the HRR cliff), per divergence class 3. Expect and document that dict commits more.
- **Re-query a deduped fact after `maintain()`**, per the index invariant.
- **A chain walk through a genuinely multi-valued intermediate hop**, to see whether the backends diverge.
- **`merge_from` on dict, and mixed-backend raising.**
- Keep parity KBs **under the HRR capacity cliff** for IDK-state equality assertions, or carve out the divergence explicitly.

Then the divergence test, using the **accurate per-function language already in `tests/test_backend_interface.py` lines 95-113** rather than one blanket sentence.

Commit.

### Task 4: persistence and hashing

**Files:** `rck/session.py`, `rck/snapshot_hash.py`; tests.

**[R2] There are FOUR `_memory` call sites in `session.py`, not two:** lines 37 and 145 (knowledge save/load) and 52 and 159 (beliefs save/load). Task 2 puts both KBs on the backend switch, so branching only the knowledge path crashes `checkpoint()` on a dict-backend agent. Handle all four.

**[R2] Thread the backend through explicitly:** `save_session`'s `meta` has no `"backend"` key today, and `load_session`'s `ConsciousAgent(...)` call has no `backend=`. Add both, and set it **before** any backend-dependent branching in `load_session` runs.

`snapshot_hash.state_hash`: hash the canonical fact list plus hyper for dict. **The two backends will hash differently for the same logical facts. That is correct** - a `DecisionRecord` pins a substrate state and the substrates differ. Assert it explicitly.

Tests: dict session round-trip preserves facts **and provenance** (regression for `2b3dfac`); `state_hash` stable per backend; replay `VERIFIED` on dict.

Commit.

### Task 5: measure what the substrate cost

**Files:** `scripts/baseline_study.py`, `data/baseline_study.json`.

Add the dict backend as a third row. The comparison becomes **the same reasoning layer on two substrates**, which is the framing the paper needs. Report ingest, RSS, recall@1, query median at 10k/30k/100k.

Commit.

---

## Definition of done

- [ ] `python -m pytest -q` green (baseline **847 passed**); default behaviour unchanged
- [ ] Parity holds everywhere except the three named divergence classes
- [ ] Each divergence is pinned by an explicit test, not hidden
- [ ] At least one parity assertion is anchored to an independent oracle
- [ ] Dict session round-trip preserves facts and provenance
- [ ] `merge_from` works on dict; mixed-backend raises
- [ ] `baseline_study.py` reports the same layer on both substrates
- [ ] Nothing deleted; `backend="hrr"` stays default

## Report explicitly

- Any surface method not implementable exactly, and why.
- Any parity test needing a tolerance rather than exact equality - each is a real semantic difference and must be named.
- Whether `codebook` was needed on the dict path at all.
- Whether the isotonic `ScoreCalibrator` (only reachable via `PropagationConfig(rule="calibrated_product")`, never wired into `maintain()` by default) turned out to matter.

## Rejected finding - do not act on it

The review claimed `curiosity.detect_global_gaps`'s `break` sits at the same indent as its `for fact` loop and therefore exits the outer loop correctly, making it a different mechanism from the other two. **Checked against source and rejected:** `for shard` is at indent 4, `for fact` at 8, `break` at 12. The break is inside the fact loop and exits only that loop, exactly like `research.py` and `subject_summary.py`. All three share one mechanism.
