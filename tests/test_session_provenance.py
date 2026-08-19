"""Regression: a session must carry the derivation history, not just facts.

Before this was fixed, save_session/load_session persisted knowledge and
belief arrays but NOT provenance, so explain_why() collapsed to a single
"source=unknown (leaf)" node after any reload -- including after
checkpoint(). RCK's core claim is that it can show why it knows something;
a session that drops that is not a complete session.
"""
import json
import rck
from rck.conscious_agent import ConsciousAgent
from rck.session import load_session, save_session


def _derived_agent():
    a = ConsciousAgent(expected_facts=100)
    a.tell("dog", "isa", "mammal")
    a.tell("mammal", "isa", "animal")
    a.induce("dog", "animal")
    return a


def test_derivation_tree_survives_a_session_roundtrip(tmp_path):
    a = _derived_agent()
    before = a.explain_why("dog", "isa", "animal").verbalize()
    assert "source=induced" in before, "setup failed: nothing was derived"

    save_session(a, tmp_path / "s")
    b = load_session(tmp_path / "s")

    assert b.explain_why("dog", "isa", "animal").verbalize() == before, \
        "explain_why lost the derivation across a session roundtrip"


def test_derivation_tree_survives_checkpoint(tmp_path):
    a = ConsciousAgent(expected_facts=100, wal_path=tmp_path / "wal.jsonl")
    a.tell("dog", "isa", "mammal")
    a.tell("mammal", "isa", "animal")
    a.induce("dog", "animal")
    before = a.explain_why("dog", "isa", "animal").verbalize()

    a.checkpoint(tmp_path / "snap")
    b = load_session(tmp_path / "snap")

    assert b.explain_why("dog", "isa", "animal").verbalize() == before, \
        "checkpoint() preserved facts but dropped the derivation history"


def test_session_records_the_real_version(tmp_path):
    """meta.json hardcoded '1.5.0' while the package was 15.3.1."""
    save_session(_derived_agent(), tmp_path / "s")
    meta = json.loads((tmp_path / "s" / "meta.json").read_text())
    assert meta["rck_version"] == rck.__version__


def test_old_sessions_without_provenance_still_load(tmp_path):
    """Backward compat: pre-fix sessions simply lack the aux files."""
    save_session(_derived_agent(), tmp_path / "s")
    for name in ("provenance.jsonl", "skills.jsonl", "query_memory.jsonl"):
        f = tmp_path / "s" / name
        if f.exists():
            f.unlink()
    b = load_session(tmp_path / "s")          # must not raise
    assert b.ask_with_idk({"S": "dog", "R": "isa"}, "O").top_symbol == "mammal"
