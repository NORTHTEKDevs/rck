import pytest

from rck.compose import CompositionalReasoner


def test_compositional_3slot_unseen_combination_passes():
    cr = CompositionalReasoner(dim=4096, seed=0)
    for n, s in [("red", "R"), ("blue", "L")]:
        cr.teach_pair("color", n, s)
    for n, s in [("ball", "o"), ("cube", "c")]:
        cr.teach_pair("shape", n, s)
    for n, s in [("one", "1"), ("two", "2")]:
        cr.teach_pair("quantity", n, s)
    # None of the 8 combos were trained.
    correct = 0; total = 0
    for q in ("one", "two"):
        for c in ("red", "blue"):
            for s in ("ball", "cube"):
                slots = {"quantity": q, "color": c, "shape": s}
                _, rendered, _ = cr.compose(slots)
                expected = f"{('1' if q == 'one' else '2')} {('R' if c == 'red' else 'L')} {('o' if s == 'ball' else 'c')}"
                correct += rendered == expected; total += 1
    assert correct == total, f"compositional gen failed: {correct}/{total}"


def test_merge_brings_disjoint_knowledge():
    a = CompositionalReasoner(dim=4096, seed=0)
    b = CompositionalReasoner(dim=4096, seed=0)
    a.teach_pair("color", "red", "R")
    a.teach_pair("color", "blue", "L")
    b.teach_pair("shape", "ball", "o")
    b.teach_pair("shape", "cube", "c")
    a.merge(b)
    for slot, val, expected in [("color", "red", "R"), ("color", "blue", "L"),
                                ("shape", "ball", "o"), ("shape", "cube", "c")]:
        ans, _ = a._single_slot_answer(slot, val)
        assert ans == expected, f"merge lost {slot}={val}"


def test_rename_atom_keeps_recall():
    cr = CompositionalReasoner(dim=4096, seed=0)
    cr.teach_pair("color", "red", "R")
    cr.rename("R", "crimson")
    ans, _ = cr._single_slot_answer("color", "red")
    assert ans == "crimson"


def test_forget_fact_breaks_recall():
    cr = CompositionalReasoner(dim=4096, seed=0)
    cr.teach_pair("color", "red", "R")
    cr.teach_pair("color", "blue", "L")
    cr.forget_fact({"color": "red"}, "R")
    ans, score = cr._single_slot_answer("color", "red")
    # Either no result or low confidence.
    assert score < 0.10
    # Blue still works.
    ans2, _ = cr._single_slot_answer("color", "blue")
    assert ans2 == "L"
