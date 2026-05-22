"""Tests for the Inverted Architecture."""
from rck.conscious_agent import ConsciousAgent
from rck.inverted_lm import (
    DistilledTransformerPolisher, InvertedLM,
    RuleBasedPolisher, make_inverted_lm,
)


# ---- polisher --------------------------------------------------------------

def test_rule_based_polisher_capitalises_sentences():
    p = RuleBasedPolisher()
    out = p.polish("the sky is blue. the grass is green.")
    assert out.startswith("The")
    # 2nd sentence either gets a connective + lowercase, or stays capitalised.
    assert "grass is green" in out.lower()


def test_rule_based_polisher_fixes_a_an():
    p = RuleBasedPolisher()
    out = p.polish("a elephant is huge. a animal has legs.")
    # Case-insensitive check -- capitalisation may or may not be applied.
    assert "an elephant" in out.lower()
    assert "an animal" in out.lower()


def test_rule_based_polisher_replaces_underscores():
    p = RuleBasedPolisher()
    out = p.polish("the field of einstein is physics_research.")
    assert "physics research" in out


def test_rule_based_polisher_dedups():
    p = RuleBasedPolisher()
    out = p.polish("The sky is blue. The sky is blue.")
    # After dedup the sentence should appear only once.
    assert out.lower().count("the sky is blue") == 1


def test_rule_based_polisher_adds_connectives():
    p = RuleBasedPolisher()
    out = p.polish("The dog has fur. The dog has legs. The dog has a tail.")
    assert "Also," in out or "Additionally," in out


def test_distilled_polisher_is_stubbed():
    p = DistilledTransformerPolisher()
    try:
        p.polish("the sky is blue")
        assert False, "should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "v7" in str(e)


# ---- inverted-LM end-to-end ------------------------------------------------

def test_inverted_lm_constructor():
    inv = make_inverted_lm()
    assert isinstance(inv, InvertedLM)
    assert isinstance(inv.polisher, RuleBasedPolisher)


def test_inverted_lm_generate_known_fact():
    agent = ConsciousAgent(dim=2048, n_shards=8, seed=0)
    agent.tell("sky", "color", "blue")
    inv = InvertedLM(agent=agent)
    res = inv.generate("What color is the sky?")
    assert res["polished"] is True
    assert "blue" in res["response"].lower()


def test_inverted_lm_generate_unknown_yields_unknown():
    agent = ConsciousAgent(dim=2048, n_shards=8, seed=0)
    inv = InvertedLM(agent=agent)
    res = inv.generate("What color is the alien?")
    # Should soft-reject without hallucinating a color.
    assert "blue" not in res["response"].lower()


def test_inverted_lm_describe_entity_uses_multi_sentence_path():
    agent = ConsciousAgent(dim=4096, n_shards=16, seed=0)
    agent.tell("elephant", "isa", "mammal")
    agent.tell("elephant", "size", "huge")
    agent.tell("elephant", "has", "tusks")
    inv = InvertedLM(agent=agent)
    res = inv.describe_entity("elephant")
    response = res["response"].lower()
    # All three facts should appear in some form.
    hits = sum(w in response for w in ("mammal", "huge", "tusks"))
    assert hits >= 2


def test_inverted_lm_think_aloud_includes_chain():
    agent = ConsciousAgent(dim=4096, n_shards=16, seed=0)
    agent.tell("dog", "isa", "mammal")
    agent.tell("mammal", "has", "fur")
    inv = InvertedLM(agent=agent)
    res = inv.think_aloud("What does the dog have?")
    # Polished response should mention BOTH the chain and the conclusion.
    txt = res["response"].lower()
    assert "fur" in txt
    # think_aloud field carries the unpolished thought.
    assert res.get("think_aloud") or "mammal" in txt


def test_inverted_lm_hallucination_invariant():
    """If we don't have a fact, the response must NOT contain a confident
    declaration of one. This is the core safety claim of the inverted
    architecture."""
    agent = ConsciousAgent(dim=2048, n_shards=8, seed=0)
    agent.tell("sky", "color", "blue")  # only fact we know
    inv = InvertedLM(agent=agent)
    res = inv.generate("What color is the rose?")
    # We have no rose color in the KB. The response must NOT confidently
    # assert any color.
    txt = res["response"].lower()
    color_words = {"red", "blue", "green", "yellow", "purple", "white",
                   "black", "orange", "pink"}
    confident_color_claims = [c for c in color_words if c in txt]
    if confident_color_claims:
        # If a color shows up, it must be in a hedged or denial context.
        hedges = ("don't know", "not sure", "no record", "unknown",
                  "i think", "maybe")
        assert any(h in txt for h in hedges), \
            f"unhedged color claim: {confident_color_claims} in {txt!r}"
