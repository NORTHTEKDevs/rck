"""Tests for ProvenanceStore save/load."""
from __future__ import annotations

from rck.provenance import ProvenanceStore


def test_provenance_save_load_roundtrip(tmp_path):
    p = ProvenanceStore()
    p.store("dog", "isa", "mammal", source="user",
            tags={"trusted"}, derivation=[])
    p.store("leaf", "locatedin", "forest", source="induced",
            tags={"induced", "via_2_hops"},
            derivation=[("leaf", "partof", "tree"),
                        ("tree", "locatedin", "forest")])
    f = tmp_path / "prov.jsonl"
    assert p.save(f) == 2
    other = ProvenanceStore()
    assert other.load(f) == 2
    rec = other.get("leaf", "locatedin", "forest")
    assert rec is not None
    assert rec.source == "induced"
    assert "induced" in rec.tags
    assert rec.derivation
    assert rec.derivation[0] == ("leaf", "partof", "tree")


def test_provenance_load_replace_clears(tmp_path):
    p = ProvenanceStore()
    p.store("dog", "isa", "mammal", source="user")
    f = tmp_path / "prov.jsonl"
    p.save(f)
    p.store("cat", "isa", "mammal", source="user")
    p.load(f, replace=True)
    assert p.size() == 1
    assert p.get("cat", "isa", "mammal") is None


def test_conscious_agent_save_load_provenance(tmp_path):
    from rck.conscious_agent import ConsciousAgent
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    a.deny("fish", "isa", "vegetable")
    f = tmp_path / "prov.jsonl"
    n = a.save_provenance(f)
    assert n >= 2

    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    loaded = b.load_provenance(f)
    assert loaded >= 2
    assert b.provenance.get("dog", "isa", "mammal") is not None
    assert b.provenance.get("fish", "not_isa", "vegetable") is not None
