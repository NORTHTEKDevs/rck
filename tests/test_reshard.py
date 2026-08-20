import pytest
from rck.knowledge_base import (
    _GROWTH_OVERSHOOT, _MAX_SHARDS, ShardedKnowledgeBase, _shard_index,
)
from rck.shard_sizing import recommend_shards


def _kb_with(n_facts, n_shards=8):
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    for i in range(n_facts):
        kb.store({"S": f"s{i}", "R": "isa", "O": f"o{i}"})
    return kb


def test_skewed_load_growth_is_bounded_and_overshoot_is_capped():
    """Skewed load must overshoot recommend_shards() but stay bounded.

    recommend_shards() assumes a 1.25x load-skew safety factor that real
    multi-valued data exceeds, so stopping at the recommendation leaves
    shards over the cliff (measured: 11/256 on 5,000 ConceptNet facts).
    But chasing every last overloaded shard does not converge either: two
    distinct (S, R) keys that each hold ~target_fill facts and collide
    under blake2b cannot be separated at any n_shards, and chasing that
    pair drove 2,400 facts to 4,096 shards (recommendation: 64) at one
    full O(N) re-bundle per doubling.

    So the contract is a BOUNDED overshoot: grow past the recommendation,
    stop at _GROWTH_OVERSHOOT x it, and leave any residual collision
    visible via shard_balance() rather than silent.
    """
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    # 40 subjects x 60 objects: each (S, R) key holds 60 facts (under the
    # 80 cliff, individually splittable) but they collide at low counts.
    for s in range(40):
        for o in range(60):
            kb.store({"S": f"subj{s}", "R": "has", "O": f"obj{s}_{o}"})

    assert kb.size() == 2400
    recommended = recommend_shards(kb.size(), dim=4096).n_shards
    assert kb.n_shards <= _GROWTH_OVERSHOOT * recommended, (
        f"growth ran away: n_shards={kb.n_shards} vs "
        f"{_GROWTH_OVERSHOOT}x recommendation ({recommended})"
    )
    assert kb.n_shards <= _MAX_SHARDS
    # Any shard left over the cliff must be an unsplittable key collision,
    # never mere under-provisioning.
    for sh in kb._shards:
        if sh.size() > kb.target_fill:
            keys = {(str(f["S"]), str(f["R"])) for f in sh.facts()}
            assert len(keys) < sh.size() / 2, (
                f"shard over cliff with {len(keys)} well-spread keys is "
                f"under-provisioning, not a collision"
            )


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

    agent.knowledge.reshard()   # explicit for now; Task 3 makes this automatic

    hits = sum(1 for s, r, o in sample
               if agent.ask_with_idk({"S": s, "R": r}, "O").top_symbol in valid[(s, r)])
    recall = hits / len(sample)
    assert recall >= 0.99, f"recall {recall:.1%} after reshard, expected >=99%"


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


def test_reshard_preserves_negatives_provenance_and_derivation():
    agent = ConsciousAgent(expected_facts=100)
    agent.tell("dog", "isa", "mammal")
    agent.tell("mammal", "isa", "animal")
    agent.deny("dog", "isa", "fish")
    agent.induce("dog", "animal")

    why_before = agent.explain_why("dog", "isa", "animal").verbalize()

    for i in range(1500):                    # force at least one reshard
        agent.tell(f"filler{i}", "isa", f"thing{i}")

    # "dog isa mammal" (asserted) and "dog isa animal" (induced shortcut) are
    # both true and score an exact HRR tie pre-reshard (0.7124... ==
    # 0.7124...), correctly flagged AMBIGUOUS by the epistemic layer; which
    # one sorts first as `top_symbol` is a tie-break-order accident, not a
    # resharding guarantee (reproduces identically with auto_reshard=False).
    # The real invariant reshard owes us is that the asserted fact is not
    # LOST -- it must remain recoverable as top_symbol or a surfaced
    # alternative.
    ans = agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    recovered = {ans.top_symbol} | {sym for sym, _ in ans.alternatives}
    assert "mammal" in recovered, "asserted fact lost after reshard"
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


def test_hot_key_does_not_storm():
    """A single (S, R) key pins every fact to one shard no matter how many
    shards exist -- resharding can never split it. Before the fix,
    reshard()'s default was max(n_shards * 2, recommend_shards(...)), and
    the n_shards*2 term always won once the hot shard crossed target_fill,
    so n_shards doubled on every subsequent write, unbounded.

    Bounded to 90 facts, NOT the ~120 originally suggested in the ticket:
    an explicit safety directive for this task states that on the unfixed
    code this loop is an exponential resource storm that has already hung
    the machine once, and to never exceed ~90 hot-key writes until Bug 1
    is fixed. 90 facts already reproduces the storm decisively (confirmed:
    n_shards reaches 8192 while recommend_shards(90) says 8), so it fully
    exercises the failure without the runaway risk that ~120 would carry
    on the unfixed code (each fact past the trigger point doubles again).
    """
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    for i in range(90):
        kb.store({"S": "alice", "R": "authored", "O": f"paper{i}"})

    limit = recommend_shards(kb.size(), dim=4096).n_shards
    assert kb.n_shards <= limit, (
        f"n_shards={kb.n_shards} blew past recommend_shards()={limit} "
        "-- hot-key reshard storm"
    )
    assert kb.size() == 90

    stored = {tuple(sorted(f.items())) for sh in kb._shards for f in sh.facts()}
    assert len(stored) == 90, "hot-key facts lost across reshard(s)"

    res = kb.query({"S": "alice", "R": "authored"}, "O", top_k=1)
    assert res, "hot key unretrievable after storm-capped reshard"


def test_session_round_trip_survives_reshard():
    """Bug 3 regression: ConsciousAgent.n_shards is set once in
    __post_init__ and never updated when reshard() mutates
    agent.knowledge.n_shards (or agent.beliefs.n_shards) in place. Before
    the fix, session.py serialized the stale agent.n_shards into
    meta['hyper']['n_shards'] while shard_mems/knowledge_facts were built
    from the real, larger agent.knowledge._shards -- load_session then
    rebuilt with the stale (smaller) shard count and indexed into it with
    the larger saved list.

    Confirmed via a live repro before this fix: 1000 distinct-subject
    facts (NOT a hot key -- a legitimate reshard) pushed
    knowledge.n_shards from 8 to 32 while agent.n_shards stayed 8;
    load_session raised `IndexError: list index out of range` at
    `agent.knowledge._shards[i]._facts = list(facts)`.

    Beliefs have the same exposure: ConsciousAgent derives the belief
    shard count as `n_shards // 2` at construction time, which does not
    track independent belief-KB reshards either.
    """
    import tempfile
    from rck import session as _session

    agent = ConsciousAgent(n_shards=8, dim=4096)
    for i in range(1000):
        agent.tell(f"subject{i}", "isa", f"category{i % 37}")
    for i in range(300):
        agent.tell_belief("bob", f"believer_subject{i}", "isa",
                           f"believer_cat{i % 20}")

    assert agent.n_shards != agent.knowledge.n_shards, (
        "test setup must actually desync agent.n_shards from "
        "knowledge.n_shards, or this test proves nothing"
    )

    d = tempfile.mkdtemp()
    _session.save_session(agent, d)
    loaded = _session.load_session(d)

    assert loaded.knowledge.n_shards == agent.knowledge.n_shards
    assert loaded.beliefs.n_shards == agent.beliefs.n_shards
    assert loaded.knowledge.size() == agent.knowledge.size()
    assert loaded.beliefs.size() == agent.beliefs.size()

    # Facts must be usable after reload, not just present.
    res = loaded.knowledge.query({"S": "subject5", "R": "isa"}, "O", top_k=1)
    assert res, "reloaded KB cannot answer a query after a reshard round-trip"
    bres = loaded.what_does_x_think("bob", "believer_subject5", "isa")
    assert bres["answer"] is not None, \
        "reloaded belief KB cannot answer a query after a reshard round-trip"


def test_target_fill_resolves_per_dim():
    """target_fill must come from the per-dim cliff table
    (shard_sizing.TARGET_MAX_FILL_BY_DIM), not a flat dataclass default --
    a flat 80 at dim=2048 (real cliff 60) lets auto-reshard silently miss
    the window it exists to guard."""
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    assert kb.target_fill == 60

    kb_override = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0, target_fill=99)
    assert kb_override.target_fill == 99
