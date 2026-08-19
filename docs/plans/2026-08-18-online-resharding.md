# Online Resharding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make a knowledge base that outgrows its provisioning reshard itself, so recall never degrades silently.

**Architecture:** Every shard already retains its raw facts in `RelationalMemory._facts` (exposed by `.facts()`), so resharding is a pure re-bundle from retained data - no derivation is repeated and nothing is lost. Add `ShardedKnowledgeBase.reshard()`, then trigger it automatically from `store()` when a shard crosses the capacity cliff, growing by powers of two for amortized O(log N) cost per fact.

**Tech Stack:** Python 3.11+, numpy, pytest. No new dependencies.

---

## Context the implementer needs

Read these before starting:

- `rck/knowledge_base.py:36` - `_shard_index(subject, relation, n_shards)`, the routing function. Already parameterized by `n_shards`, so it needs no change.
- `rck/knowledge_base.py:94-142` - `ShardedKnowledgeBase`. A dataclass. Fields `dim`, `n_shards`, `seed`; `init=False` fields `codebook`, `_shards`, `_fact_count`.
- `rck/relational.py:86-94` - `RelationalMemory.store()`. Appends to `self._facts` **and** adds to `self._memory`.
- `rck/relational.py:160` - `.facts()` returns `list(self._facts)`.
- `rck/conscious_agent.py:138-139` - the existing `recommend_shards(expected_facts, dim=...)` sizing helper. Reuse it; do not invent a new formula. Check the import line at the top of that file for its module.
- `rck/conscious_agent.py:214` - `shard_balance()`, which already returns `suggested_n_shards` and `target_fill=80`.

**Three traps, each of which will silently corrupt state:**

1. **Do not call `self.store()` inside `reshard()`.** It increments `_fact_count`, which would double the count on every reshard. Write into the new shards directly.
2. **Do not rebuild `self.codebook`.** It holds the cached cleanup matrix. Reuse the existing instance - resharding changes fact *routing*, never symbol encoding.
3. **Do not re-symmetrize.** `_facts` holds facts as they were stored, i.e. already post-symmetrization (symmetrization happens upstream in `ConsciousAgent`). Re-applying it would double the inverse edges.

`relation_index()` builds a fresh snapshot per call (`rck/knowledge_base.py:140-142`), so it needs no invalidation.

---

### Task 1: `reshard()` - explicit, manual

**Files:**
- Modify: `rck/knowledge_base.py` (add method to `ShardedKnowledgeBase`)
- Test: `tests/test_reshard.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_reshard.py
import pytest
from rck.knowledge_base import ShardedKnowledgeBase, _shard_index


def _kb_with(n_facts, n_shards=8):
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    for i in range(n_facts):
        kb.store({"S": f"s{i}", "R": "isa", "O": f"o{i}"})
    return kb


def test_reshard_preserves_every_fact_and_routes_correctly():
    kb = _kb_with(500, n_shards=8)
    before = {tuple(sorted(f.items())) for sh in kb._shards for f in sh.facts()}

    kb.reshard(64)

    assert kb.n_shards == 64
    assert len(kb._shards) == 64
    after = {tuple(sorted(f.items())) for sh in kb._shards for f in sh.facts()}
    assert after == before, "reshard lost or duplicated facts"

    # Every fact must now live in the shard its (S, R) routes to at n=64.
    for idx, sh in enumerate(kb._shards):
        for f in sh.facts():
            assert _shard_index(str(f["S"]), str(f["R"]), 64) == idx


def test_reshard_does_not_inflate_fact_count():
    kb = _kb_with(500, n_shards=8)
    assert kb.size() == 500
    kb.reshard(64)
    assert kb.size() == 500, "reshard double-counted _fact_count"


def test_reshard_is_a_noop_when_target_equals_current():
    kb = _kb_with(100, n_shards=16)
    codebook_before = kb.codebook
    kb.reshard(16)
    assert kb.n_shards == 16
    assert kb.codebook is codebook_before, "reshard must not rebuild the codebook"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reshard.py -v`
Expected: FAIL with `AttributeError: 'ShardedKnowledgeBase' object has no attribute 'reshard'`

**Step 3: Write minimal implementation**

Add to `ShardedKnowledgeBase` in `rck/knowledge_base.py`, immediately after `store_many`:

```python
    def reshard(self, n_shards: int | None = None) -> dict:
        """Re-bundle every stored fact into a new shard array.

        Facts are retained per shard (`RelationalMemory._facts`), so this is a
        pure re-route: nothing is re-derived and nothing is lost. The codebook
        is reused -- resharding changes routing, never symbol encoding.
        """
        if n_shards is None:
            n_shards = max(self.n_shards * 2,
                           _recommend_shards(self._fact_count, dim=self.dim))
        if n_shards == self.n_shards:
            return {"n_shards": self.n_shards, "facts": self._fact_count,
                    "resharded": False}

        facts = [f for shard in self._shards for f in shard.facts()]
        new_shards = [
            RelationalMemory(dim=self.dim, seed=self.seed,
                             role_names=("S", "R", "O", "B"))
            for _ in range(n_shards)
        ]
        for f in facts:
            idx = _shard_index(str(f.get("S", "")), str(f.get("R", "")), n_shards)
            # Direct write: self.store() would re-increment _fact_count.
            new_shards[idx].store(self.codebook, f)

        self._shards = new_shards
        self.n_shards = n_shards
        return {"n_shards": n_shards, "facts": len(facts), "resharded": True}
```

Add the sizing import at the top of `rck/knowledge_base.py`. Use the same module `conscious_agent.py:139` imports it from, aliased to avoid a name clash:

```python
from rck.<sizing_module> import recommend_shards as _recommend_shards
```

If that import is circular, inline a local import inside `reshard()` instead.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reshard.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add rck/knowledge_base.py tests/test_reshard.py
git commit -m "feat: ShardedKnowledgeBase.reshard() re-bundles facts into a new shard array"
```

---

### Task 2: Prove resharding restores recall

This is the acceptance criterion from the design doc, and it is the exact probe that measured 24.0%.

**Files:**
- Test: `tests/test_reshard.py` (append)

**Step 1: Write the failing test**

```python
import json, random
from collections import defaultdict
from pathlib import Path
from rck.conscious_agent import ConsciousAgent

DATA = Path(__file__).parent.parent / "data" / "conceptnet_scale_100k.jsonl"


def _load(n):
    facts = []
    with open(DATA, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            d = json.loads(line)
            facts.append((d["s"], d["r"], d["o"]))
    return facts


@pytest.mark.skipif(not DATA.exists(), reason="ConceptNet subset not present")
def test_underprovisioned_agent_recovers_recall_after_reshard():
    """Regression for the measured v15.3.1 blocker: a 200-provisioned agent
    holding 5,000 facts scored 24.0% recall@1 with no warning."""
    facts = _load(5000)
    valid = defaultdict(set)
    for s, r, o in facts:
        valid[(s, r)].add(o)
    sample = random.Random(1).sample(facts, 400)

    agent = ConsciousAgent(expected_facts=200)
    for s, r, o in facts:
        agent.tell(s, r, o)

    agent.kb.reshard()   # explicit for now; Task 3 makes this automatic

    hits = sum(1 for s, r, o in sample
               if agent.ask_with_idk({"S": s, "R": r}, "O").top_symbol in valid[(s, r)])
    recall = hits / len(sample)
    assert recall >= 0.99, f"recall {recall:.1%} after reshard, expected >=99%"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reshard.py::test_underprovisioned_agent_recovers_recall_after_reshard -v`
Expected: FAIL. One reshard doubles 8 → 16 shards, which is still far under the ~1024 needed for 5,000 facts, so recall stays low.

If `agent.kb` is not the attribute name, find it: `grep -n "self\.kb\b\|self\._kb\b" rck/conscious_agent.py`.

**Step 3: Write minimal implementation**

The single-step doubling in Task 1 is insufficient. Make `reshard(None)` size to the *current* fact count in one shot rather than merely doubling - `_recommend_shards(self._fact_count, ...)` already returns the right target; the `max(self.n_shards * 2, ...)` guard exists only to guarantee forward progress. Verify the guard is not clamping the recommendation downward:

```python
            n_shards = max(self.n_shards * 2,
                           _recommend_shards(self._fact_count, dim=self.dim))
```

is correct - `max` takes the larger. If the test still fails, print `_recommend_shards(5000, dim=4096)` and confirm it returns ≥128; if it does not, the sizing helper itself is the bug and must be fixed here rather than worked around.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reshard.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add tests/test_reshard.py rck/knowledge_base.py
git commit -m "test: reshard restores recall on an under-provisioned KB (was 24.0%)"
```

---

### Task 3: Automatic reshard on write

**Files:**
- Modify: `rck/knowledge_base.py` - add two dataclass fields, extend `store()`
- Test: `tests/test_reshard.py` (append)

**Step 1: Write the failing test**

```python
def test_growth_past_provisioning_auto_reshards():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    for i in range(2000):
        kb.store({"S": f"s{i}", "R": "isa", "O": f"o{i}"})

    assert kb.n_shards > 8, "KB never resharded while growing 250x past capacity"
    assert max(kb.shard_sizes()) <= kb.target_fill, \
        f"a shard is still over the cliff: {max(kb.shard_sizes())} > {kb.target_fill}"
    assert kb.size() == 2000


def test_auto_reshard_can_be_disabled():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    kb.auto_reshard = False
    for i in range(1000):
        kb.store({"S": f"s{i}", "R": "isa", "O": f"o{i}"})
    assert kb.n_shards == 8
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reshard.py::test_growth_past_provisioning_auto_reshards -v`
Expected: FAIL - `AttributeError: ... 'target_fill'`, then `assert 8 > 8`.

**Step 3: Write minimal implementation**

Add two fields to `ShardedKnowledgeBase` alongside `seed`:

```python
    auto_reshard: bool = True
    target_fill: int = 80   # measured capacity cliff at D=4096 (paper 5.4)
```

Extend `store()`:

```python
    def store(self, fact: dict[str, Hashable]) -> None:
        """Store a fact in the shard determined by (S, R)."""
        s, r = str(fact.get("S", "")), str(fact.get("R", ""))
        idx = _shard_index(s, r, self.n_shards)
        self._shards[idx].store(self.codebook, fact)
        self._fact_count += 1
        if self.auto_reshard and self._shards[idx].size() > self.target_fill:
            self.reshard()
```

The check is `len()` on a list - O(1), so it costs nothing on the common path. Growth is by powers of two, so a KB reaching N facts reshards O(log N) times at O(N) each.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reshard.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add rck/knowledge_base.py tests/test_reshard.py
git commit -m "feat: auto-reshard on write when a shard crosses the capacity cliff"
```

---

### Task 4: Invariants must survive a reshard

Resharding rebuilds `_shards`. Anything keyed by shard index is at risk. These tests prove the claim rather than assuming it.

**Files:**
- Test: `tests/test_reshard.py` (append)

**Step 1: Write the failing test**

```python
def test_reshard_preserves_negatives_provenance_and_derivation():
    agent = ConsciousAgent(expected_facts=100)
    agent.tell("dog", "isa", "mammal")
    agent.tell("mammal", "isa", "animal")
    agent.deny("dog", "isa", "fish")
    agent.induce("dog", "animal")

    why_before = agent.explain_why("dog", "isa", "animal").verbalize()

    for i in range(1500):                    # force at least one reshard
        agent.tell(f"filler{i}", "isa", f"thing{i}")

    assert agent.ask_with_idk({"S": "dog", "R": "isa"}, "O").top_symbol == "mammal"
    assert agent.explain_why("dog", "isa", "animal").verbalize() == why_before, \
        "derivation tree changed across reshard"

    neg = agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    assert neg.top_symbol != "fish", "denied fact resurfaced after reshard"


def test_relation_index_is_correct_after_reshard():
    kb = _kb_with(500, n_shards=8)
    kb.reshard(64)
    idx = kb.relation_index()
    assert idx.n_shards == 64, "stale relation index after reshard"
    for shard_id in idx.shards_with("isa"):
        assert any(f["R"] == "isa" for f in kb._shards[shard_id].facts())
```

**Step 2: Run test to verify it fails or passes**

Run: `python -m pytest tests/test_reshard.py -v`

Expected: these may pass immediately - provenance, skills, and query memory are keyed independently of shard index, and `relation_index()` is built fresh per call. **If they pass on the first run, that is the intended outcome and the tests stay as regression guards.** If any fails, that failure is a real defect uncovered by this task; fix it before moving on, and do not weaken the assertion to make it pass.

**Step 3: Commit**

```bash
git add tests/test_reshard.py
git commit -m "test: reshard preserves negatives, provenance, derivations, relation index"
```

---

### Task 5: Full regression

**Step 1: Run the whole suite**

Run: `python -m pytest -q`
Expected: **763 passed** (757 existing + 6 new), zero failures.

The 757 baseline was captured on this machine at commit 63fcb33 in 71.26 s. Any pre-existing test that now fails is a regression introduced by this work - fix it, do not skip it.

**Step 2: Re-measure the blocker end to end**

Confirm the original measurement is genuinely closed, with auto-reshard doing the work (no manual `reshard()` call):

```python
# scratch, do not commit
agent = ConsciousAgent(expected_facts=200)
for s, r, o in facts_5000:
    agent.tell(s, r, o)
print(agent.shard_balance())        # expect: no overloaded shards
# then score the 400-probe valid-set protocol -> expect >=99%
```

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: online resharding (v16.0 production core, item 1)"
```

---

## Definition of done

- [ ] `python -m pytest -q` → 763 passed
- [ ] A 200-provisioned agent holding 5,000 facts scores ≥99% recall@1 with **no manual intervention**
- [ ] `shard_balance()` reports zero overloaded shards after unattended growth
- [ ] `size()` is exact after repeated reshards (no `_fact_count` inflation)
- [ ] Negatives, provenance, and derivation trees are byte-identical across a reshard

## Not in this plan

Items 2-4 of the production core (API subtraction, durability, replay format) each get their own plan. Do not start them here - resharding changes on-disk shard layout, so durability must be designed against the post-reshard shape.
