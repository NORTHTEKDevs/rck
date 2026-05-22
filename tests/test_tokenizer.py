from rck.tokenizer import tokenize, detokenize, sentences


def test_tokenize_basic_word_split():
    assert tokenize("The quick brown fox.") == ["the", "quick", "brown", "fox", "."]


def test_tokenize_preserves_punctuation():
    assert tokenize("Hello, world! How are you?") == [
        "hello", ",", "world", "!", "how", "are", "you", "?",
    ]


def test_tokenize_handles_numbers_and_underscores():
    assert tokenize("agent_42 saw 3 dogs") == ["agent_42", "saw", "3", "dogs"]


def test_roundtrip_simple_sentence():
    toks = tokenize("The cat sat on the mat.")
    assert detokenize(toks).strip() == "the cat sat on the mat."


def test_roundtrip_question():
    toks = tokenize("What color is the sky?")
    assert detokenize(toks).strip() == "what color is the sky?"


def test_sentences_split_correctly():
    text = "Alice lives in Paris. Bob lives in Berlin! Where does Carol live?"
    sents = sentences(text)
    assert len(sents) == 3
    assert sents[0].startswith("Alice")
    assert sents[-1].endswith("?")
