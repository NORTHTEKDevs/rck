"""Tests for the Open IE extractor."""
from rck.open_ie import (
    extract_triples_from_sentence,
    extract_triples_from_text,
)


def test_extract_isa():
    assert extract_triples_from_sentence("The dog is a mammal.") == [
        ("dog", "isa", "mammal"),
    ]


def test_extract_bare_is():
    # Bare predicate without "a/an" -> `is` relation, not `isa`.
    assert extract_triples_from_sentence("The sky is blue.") == [
        ("sky", "is", "blue"),
    ]


def test_extract_has():
    assert extract_triples_from_sentence("The elephant has tusks.") == [
        ("elephant", "has", "tusks"),
    ]


def test_extract_capital_pattern():
    # "X is the Y of Z" -> (Z, Y, X)
    assert extract_triples_from_sentence("Paris is the capital of France.") == [
        ("france", "capital", "paris"),
    ]


def test_extract_capital_alt_pattern():
    # "The Y of X is Z" -> (X, Y, Z)
    assert extract_triples_from_sentence("The capital of France is Paris.") == [
        ("france", "capital", "paris"),
    ]


def test_extract_located_in():
    assert extract_triples_from_sentence("Everest is in Nepal.") == [
        ("everest", "locatedin", "nepal"),
    ]


def test_extract_lives_in():
    assert extract_triples_from_sentence("Alice lives in Paris.") == [
        ("alice", "lives_in", "paris"),
    ]


def test_extract_made_of():
    assert extract_triples_from_sentence("The window is made of glass.") == [
        ("window", "madeof", "glass"),
    ]


def test_extract_used_for():
    assert extract_triples_from_sentence("The knife is used for cutting.") == [
        ("knife", "usedfor", "cutting"),
    ]


def test_extract_wrote():
    assert extract_triples_from_sentence("Shakespeare wrote Hamlet.") == [
        ("shakespeare", "wrote", "hamlet"),
    ]


def test_extract_composed():
    assert extract_triples_from_sentence("Mozart composed Requiem.") == [
        ("mozart", "composed", "requiem"),
    ]


def test_extract_causes():
    assert extract_triples_from_sentence("Rain causes wetness.") == [
        ("rain", "causes", "wetness"),
    ]


def test_extract_returns_empty_when_no_pattern_matches():
    assert extract_triples_from_sentence("This sentence has no pattern!") == []


def test_extract_from_paragraph():
    text = (
        "The sky is blue. The grass is green. The dog is a mammal. "
        "Paris is the capital of France. The knife is used for cutting. "
        "Shakespeare wrote Hamlet."
    )
    triples = extract_triples_from_text(text)
    assert ("sky", "is", "blue") in triples
    assert ("grass", "is", "green") in triples
    assert ("dog", "isa", "mammal") in triples
    assert ("france", "capital", "paris") in triples
    assert ("knife", "usedfor", "cutting") in triples
    assert ("shakespeare", "wrote", "hamlet") in triples


def test_extract_bootstrap_volume():
    """Feed 20 simple sentences and verify >18 extract correctly."""
    sentences = [
        "The dog is a mammal.",
        "The cat is a mammal.",
        "The fish is an animal.",
        "The rose is red.",
        "The leaf is green.",
        "The window is made of glass.",
        "The pen is used for writing.",
        "The car is used for driving.",
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
        "Alice lives in Paris.",
        "Bob lives in Berlin.",
        "Mozart composed Requiem.",
        "Shakespeare wrote Hamlet.",
        "Picasso painted Guernica.",
        "Edison invented LightBulb.",
        "The dog has fur.",
        "The bird has feathers.",
        "Rain causes wetness.",
    ]
    triples = []
    for s in sentences:
        triples.extend(extract_triples_from_sentence(s))
    assert len(triples) >= 18, f"only {len(triples)} extracted from 20 sentences"
