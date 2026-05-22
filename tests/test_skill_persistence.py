"""Tests for SkillLibrary save/load."""
from __future__ import annotations

from rck.skills import SkillLibrary


def test_skill_save_load_roundtrip(tmp_path):
    lib = SkillLibrary()
    p1 = [("O", "partof"), ("O", "locatedin")]
    p2 = [("O", "isa"), ("O", "isa")]
    for _ in range(3):
        lib.record_success(p1)
    lib.record_success(p2)
    lib.record_failure(p2)
    f = tmp_path / "skills.jsonl"
    n = lib.save(f)
    assert n == 2
    other = SkillLibrary()
    loaded = other.load(f)
    assert loaded == 2
    # Stats preserved.
    s1 = other.find_matching(p1)
    assert s1 is not None
    assert s1.success_count == 3


def test_skill_load_with_replace(tmp_path):
    lib = SkillLibrary()
    p1 = [("O", "partof"), ("O", "locatedin")]
    lib.record_success(p1)
    f = tmp_path / "skills.jsonl"
    lib.save(f)
    # Add another pattern then load with replace=True.
    p2 = [("O", "isa"), ("O", "isa")]
    lib.record_success(p2)
    assert lib.size() if hasattr(lib, "size") else len(lib.skills) == 2
    lib.load(f, replace=True)
    # Only p1 should remain.
    assert len(lib.skills) == 1


def test_conscious_agent_save_load_skills(tmp_path):
    from rck.conscious_agent import ConsciousAgent
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    p = [("O", "partof"), ("O", "locatedin")]
    for _ in range(3):
        a.skills.record_success(p)
    f = tmp_path / "skills.jsonl"
    assert a.save_skills(f) == 1

    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    assert b.load_skills(f) == 1
    assert b.skills.find_matching(p) is not None
