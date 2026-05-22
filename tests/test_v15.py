"""Tests for v1.5: synonyms, multi-sentence answers, think-aloud, sessions."""
import tempfile
from pathlib import Path

from rck.bulk_ingest import bulk_load_triples
from rck.compose_answer import describe
from rck.conscious_agent import ConsciousAgent
from rck.inference import infer
from rck.knowledge_base import ShardedKnowledgeBase
from rck.session import load_session, save_session
from rck.synonyms import canonical_entity, canonical_relation
from rck.think_aloud import narrate


# ---- synonyms --------------------------------------------------------------

def test_relation_synonyms():
    assert canonical_relation("hue") == "color"
    assert canonical_relation("colour") == "color"
    assert canonical_relation("writer") == "author"
    assert canonical_relation("type") == "isa"
    assert canonical_relation("material") == "madeof"


def test_entity_synonyms():
    assert canonical_entity("UK") == "england"
    assert canonical_entity("United States") == "usa"
    assert canonical_entity("Britain") == "england"
    # Unrecognised entity passes through normalized.
    assert canonical_entity("Sweden") == "sweden"


# ---- multi-sentence composition -------------------------------------------

def test_describe_multi_relation_returns_paragraph():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("elephant", "isa", "mammal"),
        ("elephant", "has", "tusks"),
        ("elephant", "size", "huge"),
        ("elephant", "color", "grey"),
    ], symmetrize=False)
    out = describe(kb, "elephant")
    assert "elephant" in out
    # At least 3 relations show up.
    keywords = {"mammal", "tusks", "huge", "grey"}
    hits = sum(1 for k in keywords if k in out.lower())
    assert hits >= 3


def test_describe_handles_unknown_entity():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    out = describe(kb, "nonexistent_entity")
    assert "don't have" in out


# ---- think-aloud ----------------------------------------------------------

def test_narrate_direct_lookup():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [("sky", "color", "blue")], symmetrize=False)
    res = infer(kb, "sky", "color")
    text = narrate("what color is the sky?", res)
    assert "sky" in text.lower() and "blue" in text.lower()
    assert "let me think" in text.lower()


def test_narrate_inherited_chain():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("mammal", "has", "fur"),
    ], symmetrize=False)
    res = infer(kb, "dog", "has")
    text = narrate("what does the dog have?", res)
    # Should mention BOTH the isa step and the has step.
    assert "mammal" in text.lower()
    assert "fur" in text.lower()
    assert "infer" in text.lower() or "so" in text.lower()


def test_narrate_no_answer():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    res = infer(kb, "ufo", "color")
    text = narrate("what color is the ufo?", res)
    assert "don't" in text.lower() or "no" in text.lower()


# ---- session persistence ---------------------------------------------------

def test_session_save_load_roundtrip():
    agent = ConsciousAgent(dim=2048, n_shards=8, seed=0)
    agent.tell("sky", "color", "blue")
    agent.tell("dog", "isa", "mammal")
    agent.ask("What color is the sky?")
    facts_before = agent.knowledge.size()
    turns_before = len(agent.dialogue.history)

    with tempfile.TemporaryDirectory() as td:
        save_session(agent, td)
        agent2 = load_session(td)

    assert agent2.knowledge.size() == facts_before
    assert len(agent2.dialogue.history) == turns_before
    res = agent2.ask("What color is the sky?")
    assert res["answer"] == "blue"


def test_session_preserves_beliefs():
    agent = ConsciousAgent(dim=2048, n_shards=8, seed=0)
    agent.tell_belief("alice", "france", "capital", "paris")
    agent.tell_belief("bob",   "france", "capital", "lyon")

    with tempfile.TemporaryDirectory() as td:
        save_session(agent, td)
        agent2 = load_session(td)

    res = agent2.what_does_x_think("bob", "france", "capital")
    assert res.get("answer") == "lyon"
