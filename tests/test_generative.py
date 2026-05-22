from rck.generative import GenerativeRCK, parse_question, _extract_simple_facts


def test_parse_what_color_is():
    parsed = parse_question("What color is the sky?")
    assert parsed == ("sky", "color", None)


def test_parse_what_is_the_attr_of_entity():
    parsed = parse_question("What is the capital of France?")
    assert parsed == ("france", "capital", None)


def test_parse_who_wrote():
    parsed = parse_question("Who wrote Hamlet?")
    assert parsed == ("hamlet", "wrote", None)


def test_parse_where_does_x_live():
    parsed = parse_question("Where does Alice live?")
    assert parsed is not None
    assert parsed[0] == "alice"


def test_extract_be_fact():
    facts = _extract_simple_facts("The sky is blue.")
    assert ("sky", "is", "blue") in facts


def test_extract_of_fact():
    facts = _extract_simple_facts("The capital of France is Paris.")
    assert ("france", "capital", "paris") in facts


def test_tell_and_ask_roundtrip():
    g = GenerativeRCK(dim=4096, seed=0)
    g.tell("sky", "color", "blue")
    g.tell("grass", "color", "green")
    g.tell("apple", "color", "red")

    res = g.ask("What color is the sky?")
    assert res["answer"] == "blue"
    assert res["source"] == "structured"
    assert res["confidence"] > 0.1


def test_ingest_extracts_facts_from_be_sentences():
    g = GenerativeRCK(dim=4096, seed=0)
    g.ingest("The sky is blue. The grass is green. The fox is red.")
    res = g.ask("What is the sky?")
    # 'what is X' uses 'is' relation -- our store maps to it.
    assert res["source"] == "structured"
    assert res["answer"] == "blue"


def test_unknown_question_falls_back_to_generation():
    g = GenerativeRCK(dim=2048, seed=0)
    g.tell("sky", "color", "blue")
    res = g.ask("What color is the elephant?")
    # No fact about elephants -- must NOT confidently answer 'blue'.
    assert res["source"] != "structured" or res["confidence"] < 0.1


def test_state_reports_fact_and_token_counts():
    g = GenerativeRCK(dim=2048, seed=0)
    g.tell("a", "is", "b")
    g.ingest("the cat is small.")
    s = g.state()
    assert s["fact_count"] >= 2
    assert s["tokens_ingested"] > 0
