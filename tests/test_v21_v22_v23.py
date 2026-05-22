"""Tests for v2.1 (personality), v2.2 (actions), v2.3 (corrections)."""
from rck.actions import make_default_registry
from rck.corrections import detect_correction
from rck.personality import Personality


# ---- personality ----------------------------------------------------------

def test_personality_formal_tone():
    p = Personality(tone="formal")
    out = p.render_know("blue")
    assert "established" in out


def test_personality_casual_tone():
    p = Personality(tone="casual")
    out = p.render_know("blue")
    assert "Yeah" in out or "yeah" in out


def test_personality_concise():
    p = Personality(tone="concise")
    assert p.render_know("blue").endswith(".")
    assert len(p.render_know("blue")) <= 10


def test_personality_calibrated_hedging():
    p = Personality(hedging="calibrated")
    # High confidence -> "know"
    out = p.render_verbal("blue", 0.35)
    assert "know" in out.lower() or "blue" in out.lower()
    # Medium -> "think"
    out = p.render_verbal("blue", 0.15)
    assert "blue" in out.lower()
    # Low -> "don't know"
    out = p.render_verbal("blue", 0.02)
    assert "don't" in out.lower() or "no record" in out.lower()


def test_personality_strict_refuses_low_confidence():
    p = Personality(hedging="strict")
    out = p.render_verbal("blue", 0.05)
    assert "don't" in out.lower() or "no record" in out.lower()


# ---- actions --------------------------------------------------------------

def test_action_registry_lists_default_tools():
    r = make_default_registry()
    names = {t["name"] for t in r.list_tools()}
    assert {"calculator", "time_now", "length_of"} <= names


def test_action_calculator_via_registry():
    r = make_default_registry()
    match = r.match("what is 7 + 3")
    assert match is not None
    tool, _ = match
    assert tool.name == "calculator"
    res = r.invoke("calculator", "what is 7 + 3")
    assert res["ok"] is True
    assert res["answer"] == 10.0


def test_action_length_extracts_string():
    r = make_default_registry()
    match = r.match("how long is hello world")
    assert match is not None
    tool, groups = match
    res = r.invoke("length_of", groups["s"])
    assert res["length"] == len("hello world")


def test_action_time_now_returns_iso():
    r = make_default_registry()
    res = r.invoke("time_now")
    assert "epoch" in res and "iso" in res
    assert "T" in res["iso"]


# ---- corrections ---------------------------------------------------------

def test_correction_actually_x_is_y_not_z():
    res = detect_correction("Actually, the sky is blue, not red.")
    assert res.matched is True
    assert res.stored == ("sky", "is", "blue")
    assert res.forgot == ("sky", "is", "red")


def test_correction_no_x_is_y():
    res = detect_correction("No, the sky is blue.")
    assert res.matched is True
    assert res.stored == ("sky", "is", "blue")
    assert res.forgot is None


def test_correction_attribute_form():
    res = detect_correction("The capital of France is Paris, not Lyon.")
    assert res.matched is True
    assert res.stored == ("france", "capital", "paris")
    assert res.forgot == ("france", "capital", "lyon")


def test_correction_x_is_actually_y():
    res = detect_correction("The dog is actually a mammal.")
    assert res.matched is True
    assert res.stored == ("dog", "is", "mammal")


def test_correction_not_a_correction():
    res = detect_correction("What color is the sky?")
    assert res.matched is False
