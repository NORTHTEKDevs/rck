"""Tests for agent.save_state / load_state."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_save_state_writes_three_files(tmp_path):
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    p = [("O", "isa")]
    a.skills.record_success(p)
    a.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    out_dir = tmp_path / "state"
    counts = a.save_state(out_dir)
    assert counts["skills"] >= 1
    assert counts["provenance"] >= 1
    assert counts["query_memory"] >= 1
    assert (out_dir / "skills.jsonl").exists()
    assert (out_dir / "provenance.jsonl").exists()
    assert (out_dir / "query_memory.jsonl").exists()


def test_load_state_restores_all_components(tmp_path):
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    p = [("O", "isa")]
    a.skills.record_success(p)
    a.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    out_dir = tmp_path / "state"
    a.save_state(out_dir)

    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    counts = b.load_state(out_dir)
    assert counts["skills"] >= 1
    assert counts["provenance"] >= 1
    assert counts["query_memory"] >= 1
    # Skill present.
    assert b.skills.find_matching(p) is not None
    # Provenance present.
    assert b.provenance.get("dog", "isa", "mammal") is not None
    # Episode present.
    assert b.query_memory.size() >= 1


def test_load_state_missing_dir_returns_zeros(tmp_path):
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    counts = a.load_state(tmp_path / "nope")
    assert all(v == 0 for v in counts.values())
