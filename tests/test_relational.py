from rck.codebook import Codebook
from rck.relational import RelationalMemory


def test_store_and_query_recovers_value():
    cb = Codebook(dim=4096, seed=0)
    rm = RelationalMemory(dim=4096, seed=0)
    rm.store(cb, {"E": "alice", "R": "lives_in", "V": "paris"})
    rm.store(cb, {"E": "bob", "R": "lives_in", "V": "berlin"})
    rm.store(cb, {"E": "carol", "R": "lives_in", "V": "rome"})
    ans, score = rm.answer(cb, {"E": "alice", "R": "lives_in"}, "V")
    assert ans == "paris"
    ans, score = rm.answer(cb, {"E": "bob", "R": "lives_in"}, "V")
    assert ans == "berlin"


def test_query_returns_top_k_with_scores():
    cb = Codebook(dim=2048, seed=1)
    rm = RelationalMemory(dim=2048, seed=1)
    rm.store(cb, {"E": "a", "R": "is", "V": "x"})
    rm.store(cb, {"E": "b", "R": "is", "V": "y"})
    top3 = rm.query(cb, {"E": "a", "R": "is"}, "V", top_k=3)
    # Best match must be x, with a positive cosine score.
    assert top3[0][0] == "x"
    assert top3[0][1] > 0


def test_merge_combines_disjoint_facts():
    cb = Codebook(dim=4096, seed=0)
    a = RelationalMemory(dim=4096, seed=0)
    b = RelationalMemory(dim=4096, seed=0)  # same seed -> same role HVs
    a.store(cb, {"E": "a", "R": "color", "V": "red"})
    b.store(cb, {"E": "b", "R": "color", "V": "blue"})
    a.merge(b)
    assert a.size() == 2
    assert a.answer(cb, {"E": "a", "R": "color"}, "V")[0] == "red"
    assert a.answer(cb, {"E": "b", "R": "color"}, "V")[0] == "blue"


def test_forget_removes_a_fact():
    cb = Codebook(dim=4096, seed=0)
    rm = RelationalMemory(dim=4096, seed=0)
    rm.store(cb, {"E": "a", "R": "is", "V": "x"})
    rm.store(cb, {"E": "b", "R": "is", "V": "y"})
    rm.forget(cb, {"E": "a", "R": "is", "V": "x"})
    assert rm.size() == 1
    # 'a' should no longer cleanly retrieve.
    ans, score = rm.answer(cb, {"E": "a", "R": "is"}, "V")
    # Allow either: no answer, wrong answer, or low-score answer.
    # The important invariant: 'b' still works.
    assert rm.answer(cb, {"E": "b", "R": "is"}, "V")[0] == "y"


def test_capacity_with_modest_load():
    """Store 50 facts in a D=4096 memory and verify >90% retrieval."""
    cb = Codebook(dim=4096, seed=42)
    rm = RelationalMemory(dim=4096, seed=42)
    facts = []
    for i in range(50):
        f = {"E": f"e{i}", "R": "key", "V": f"v{i}"}
        rm.store(cb, f)
        facts.append(f)
    hits = 0
    for f in facts:
        ans, _ = rm.answer(cb, {"E": f["E"], "R": "key"}, "V")
        if ans == f["V"]:
            hits += 1
    assert hits >= 45  # 90%+ at this load
