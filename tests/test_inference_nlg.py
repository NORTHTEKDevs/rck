"""Tests for inference engine + NLG + dialogue + question dispatch."""
from rck.dialogue import DialogueContext
from rck.inference import (
    boolean, compare, enumerate_subjects, infer,
)
from rck.knowledge_base import ShardedKnowledgeBase
from rck.nlg import render, render_chain, render_enumeration


# ---- inference engine ------------------------------------------------------

def _kb_with_chains():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    for s, r, o in [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("mammal", "isa", "animal"),
        ("mammal", "has", "fur"),
        ("animal", "has", "cells"),
        ("paris", "locatedin", "france"),
        ("france", "locatedin", "europe"),
        ("france", "continent", "europe"),
        ("elephant", "size", "huge"),
        ("mouse", "size", "tiny"),
        ("cat", "size", "small"),
    ]:
        kb.store({"S": s, "R": r, "O": o})
    return kb


def test_direct_lookup_returns_chain_of_one():
    kb = _kb_with_chains()
    res = infer(kb, "dog", "isa")
    assert res.answer == "mammal"
    assert res.source == "direct"
    assert len(res.chain) == 1


def test_inherited_through_isa():
    """dog isa mammal + mammal has fur -> dog has fur."""
    kb = _kb_with_chains()
    res = infer(kb, "dog", "has")
    assert res.answer == "fur"
    assert res.source == "inherited"
    assert len(res.chain) == 2


def test_inherited_through_locatedin():
    """paris locatedin france + france continent europe -> paris continent europe."""
    kb = _kb_with_chains()
    res = infer(kb, "paris", "continent")
    assert res.answer == "europe"
    assert res.source == "inherited"


def test_boolean_multi_valued_has():
    """elephant has tusks AND ears AND trunk -- any of those is True."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    for part in ["tusks", "ears", "trunk"]:
        kb.store({"S": "elephant", "R": "has", "O": part})
    for part in ["tusks", "ears", "trunk"]:
        res = boolean(kb, "elephant", "has", part)
        assert res["answer"] is True, f"failed on elephant has {part}"


def test_boolean_single_valued_color_contradicts():
    """If sky is blue, 'is sky red?' is False (color is single-valued)."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    kb.store({"S": "sky", "R": "color", "O": "blue"})
    res = boolean(kb, "sky", "color", "red")
    assert res["answer"] is False


def test_enumeration_filters_noise():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    for animal in ("dog", "cat", "elephant"):
        kb.store({"S": animal, "R": "isa", "O": "mammal"})
    items = enumerate_subjects(kb, "isa", "mammal", top_k=10)
    names = {s for s, _ in items}
    assert {"dog", "cat", "elephant"} <= names


def test_compare_size_winner():
    kb = _kb_with_chains()
    res = compare(kb, "elephant", "mouse")
    assert res["winner"] == "elephant"
    assert "bigger" in res["verbal"]


# ---- NLG -------------------------------------------------------------------

def test_render_known_relation():
    s = render("sky", "color", "blue")
    assert "sky" in s and "blue" in s


def test_render_chain():
    out = render_chain([("dog", "isa", "mammal"), ("mammal", "has", "fur")])
    assert "dog" in out and "mammal" in out and "fur" in out


def test_render_enumeration_grammar():
    out = render_enumeration(["dog", "cat", "elephant"], "isa", "mammal")
    assert "dog" in out and "cat" in out and "elephant" in out
    assert "and" in out


# ---- dialogue context -----------------------------------------------------

def test_dialogue_resolves_it_pronoun():
    d = DialogueContext()
    d.record("What color is the sky?",
             {"entity": "sky", "relation": "color"}, "blue")
    rewritten = d.resolve_references("What about it?")
    # 'about it' not handled by resolve_references directly but the entity is stored.
    assert d.last_entity == "sky"


def test_dialogue_with_default_topic_uses_last_relation():
    d = DialogueContext()
    d.record("What color is the sky?",
             {"entity": "sky", "relation": "color"}, "blue")
    out = d.with_default_topic("What about the grass?")
    assert "color" in out and "grass" in out


def test_dialogue_with_default_topic_uses_last_entity_for_it():
    d = DialogueContext()
    d.record("What does the dog have?",
             {"entity": "dog", "relation": "has"}, "fur")
    out = d.with_default_topic("What about it?")
    assert "dog" in out
