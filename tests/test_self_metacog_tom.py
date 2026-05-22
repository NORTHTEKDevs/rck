"""Tests for self-model, meta-cognition, and theory of mind."""
from rck.knowledge_base import ShardedKnowledgeBase
from rck.metacog import (
    CalibrationTally, epistemic_category, verbalize,
)
from rck.self_model import install_self_model, self_describe
from rck.theory_of_mind import (
    make_belief_kb, store_belief, what_does_x_think, believers_of,
)


# ---- self-model ------------------------------------------------------------

def test_self_model_installs_and_retrieves():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    n = install_self_model(kb)
    assert n >= 30
    ans, score = kb.answer({"S": "rck", "R": "version"}, "O")
    assert ans == "15.0.0"
    assert score > 0.1


def test_self_describe_returns_grounded_description():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    install_self_model(kb)
    desc = self_describe(kb)
    assert "RCK" in desc or "rck" in desc
    assert "1.3.0" in desc or "neuro" in desc.lower()


def test_self_describe_includes_limits():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    install_self_model(kb)
    desc = self_describe(kb)
    assert any(word in desc.lower() for word in [
        "cannot", "limit", "no_internet", "match_gpt"
    ])


# ---- meta-cognition --------------------------------------------------------

def test_epistemic_category_thresholds():
    assert epistemic_category(0.30) == "know"
    assert epistemic_category(0.15) == "think"
    assert epistemic_category(0.07) == "guess"
    assert epistemic_category(0.01) == "unknown"


def test_verbalize_strong_confidence():
    out = verbalize("blue", 0.30, source="structured")
    assert "know" in out.lower()
    assert "blue" in out


def test_verbalize_low_confidence():
    out = verbalize("blue", 0.03)
    assert "don't know" in out.lower()


def test_verbalize_uses_via_relation_hint():
    out = verbalize("blue", 0.25, source="structured-via-is")
    assert "is" in out


def test_calibration_tally_tracks_correctness():
    tally = CalibrationTally()
    tally.record("color", 0.30, correct=True)
    tally.record("color", 0.30, correct=True)
    tally.record("color", 0.30, correct=False)
    score = tally.calibration_score("color")
    assert 0.6 < score < 0.7  # 2/3


# ---- theory of mind --------------------------------------------------------

def test_belief_distinct_from_ground_truth():
    kb_world = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    kb_world.store({"S": "france", "R": "capital", "O": "paris"})
    kb_beliefs = make_belief_kb(dim=4096, n_shards=8, seed=0)
    store_belief(kb_beliefs, "bob", "france", "capital", "lyon")

    truth, _ = kb_world.answer({"S": "france", "R": "capital"}, "O")
    bob_thinks = what_does_x_think(kb_beliefs, "bob", "france", "capital")

    assert truth == "paris"
    assert any(s == "lyon" for s, _ in bob_thinks)


def test_multiple_believers_separate():
    kb = make_belief_kb(dim=4096, n_shards=16, seed=0)
    store_belief(kb, "alice", "sky", "color", "blue")
    store_belief(kb, "bob", "sky", "color", "green")
    alice = what_does_x_think(kb, "alice", "sky", "color")
    bob = what_does_x_think(kb, "bob", "sky", "color")
    assert any(s == "blue" for s, _ in alice)
    assert any(s == "green" for s, _ in bob)


def test_believers_of_lookup():
    kb = make_belief_kb(dim=4096, n_shards=16, seed=0)
    store_belief(kb, "alice", "sky", "color", "blue")
    store_belief(kb, "carol", "sky", "color", "blue")
    store_belief(kb, "bob", "sky", "color", "green")
    res = believers_of(kb, "sky", "color", "blue", top_k=5)
    top_syms = [s for s, _ in res]
    assert "alice" in top_syms and "carol" in top_syms


# ---- introspection (smoke test) -------------------------------------------

def test_introspect_records_steps_and_summarises():
    from rck.agent import RCKAgent
    from rck.introspect import IntrospectionBuffer, think

    agent = RCKAgent(hv_dim=256, n_columns=2, reservoir_dim=16, n_clauses=4,
                     vocab_size=16, fep_rank=8, bigram_order=1, seed=0)
    buf = IntrospectionBuffer(max_history=16)
    for c in "hello":
        tr = agent.step(c, learn=True, teacher_next=None)
        buf.record(tr)
    s = buf.stats()
    assert s["steps"] == 5
    text = think(agent, buf, last_query="say hello")
    assert "internal state" in text.lower()
    assert "steps" in text.lower()
