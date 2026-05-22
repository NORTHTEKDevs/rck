"""Tests for v1.6-v1.9: open IE, numbers, temporal, spatial, negation, multistep."""
from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase
from rck.multistep import two_step
from rck.negation import has_negation, strip_negation, negate_boolean
from rck.numbers import (
    compare_numeric, evaluate_arithmetic, get_numeric_attribute,
    parse_number, threshold_query,
)
from rck.spatial import direction_between, is_inside
from rck.temporal import following, previous, temporal_answer


# ---- numbers ---------------------------------------------------------------

def test_parse_number_digits():
    assert parse_number("8849") == 8849.0
    assert parse_number("3.14") == 3.14
    assert parse_number("1,000") == 1000.0


def test_parse_number_words():
    assert parse_number("seven") == 7.0
    assert parse_number("twenty") == 20.0


def test_parse_number_compound():
    assert parse_number("twenty_five") == 25.0
    assert parse_number("three hundred") == 300.0


def test_parse_number_unparseable():
    assert parse_number("blue") is None
    assert parse_number("") is None


def test_numeric_attribute_lookup():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [("everest", "height", "8849"),
                            ("k2", "height", "8611")], symmetrize=False)
    assert get_numeric_attribute(kb, "everest", "height") == 8849.0


def test_compare_numeric():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [("everest", "height", "8849"),
                            ("k2", "height", "8611")], symmetrize=False)
    result = compare_numeric(kb, "everest", "k2", relation="height")
    assert result["winner"] == "everest"
    assert "greater" in result["verbal"]


def test_threshold_query():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [("everest", "height", "8849")], symmetrize=False)
    result = threshold_query(kb, "everest", "height", 6000.0, op=">")
    assert result["answer"] is True
    result = threshold_query(kb, "everest", "height", 10000.0, op=">")
    assert result["answer"] is False


def test_evaluate_arithmetic_addition():
    res = evaluate_arithmetic("what is 5 + 3")
    assert res["answer"] == 8


def test_evaluate_arithmetic_multiplication():
    res = evaluate_arithmetic("12 * 4")
    assert res["answer"] == 48


def test_evaluate_arithmetic_division():
    res = evaluate_arithmetic("what is 100 / 4?")
    assert res["answer"] == 25


def test_evaluate_arithmetic_not_arithmetic():
    assert evaluate_arithmetic("hello world") is None


# ---- temporal --------------------------------------------------------------

def test_temporal_previous_following():
    from rck.temporal import MONTHS, DAYS
    assert previous(MONTHS, "march") == "february"
    assert following(MONTHS, "december") == "january"  # wraps
    assert previous(DAYS, "monday") == "sunday"


def test_temporal_question_before():
    res = temporal_answer("what comes before march")
    assert res["answer"] == "february"
    assert "february" in res["verbal"]


def test_temporal_question_after():
    res = temporal_answer("what comes after december")
    assert res["answer"] == "january"


def test_temporal_unknown_returns_none():
    assert temporal_answer("what color is the sky") is None


# ---- spatial ---------------------------------------------------------------

def test_is_inside_chain():
    # Bigger D + more shards so HRR cross-binding artifact doesn't dominate
    # at low fact counts.
    kb = ShardedKnowledgeBase(dim=4096, n_shards=32, seed=0)
    bulk_load_triples(kb, [("paris", "locatedin", "france"),
                            ("france", "locatedin", "europe")],
                       symmetrize=False)
    res = is_inside(kb, "europe", "paris")
    assert res["answer"] is True
    assert res["depth"] == 2


def test_is_inside_negative():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [("paris", "locatedin", "france")], symmetrize=False)
    res = is_inside(kb, "asia", "paris")
    assert res["answer"] is False


def test_direction_between():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [("germany", "north_of", "italy")], symmetrize=False)
    res = direction_between(kb, "germany", "italy")
    assert res["answer"] == "north"


# ---- negation --------------------------------------------------------------

def test_has_negation_detection():
    assert has_negation("Is the sky not blue?")
    assert has_negation("Is the dog not a mammal?")
    assert has_negation("Isn't the cat a mammal?")
    assert not has_negation("Is the sky blue?")


def test_strip_negation():
    out = strip_negation("Is the sky not blue?")
    assert "not" not in out.lower()
    assert "sky" in out and "blue" in out


def test_negate_boolean_signals_flip():
    q, neg = negate_boolean("Is the sky not blue?")
    assert neg is True
    assert "not" not in q.lower()


# ---- multistep -------------------------------------------------------------

def test_two_step_query():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("france", "capital", "paris"),
        ("paris", "locatedin", "europe"),
    ], symmetrize=False)
    res = two_step(kb, "What is the locatedin of the capital of France?")
    assert res is not None
    assert res["answer"] == "europe"
    assert len(res["chain"]) == 2


def test_two_step_returns_none_when_pattern_doesnt_match():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    res = two_step(kb, "What color is the sky?")
    assert res is None
