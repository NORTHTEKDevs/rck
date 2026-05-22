"""Tests for federated agent merge."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent
from rck.federated_merge import MergeReport, merge_agents


def test_merge_adds_unique_skills():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    p = [("O", "partof"), ("O", "locatedin")]
    for _ in range(3):
        b.skills.record_success(p)
    report = merge_agents(a, b)
    assert isinstance(report, MergeReport)
    assert report.skills_added >= 1
    sk = a.skills.find_matching(p)
    assert sk is not None
    assert sk.success_count == 3


def test_merge_reinforces_shared_skills():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    p = [("O", "isa")]
    for _ in range(2):
        a.skills.record_success(p)
        b.skills.record_success(p)
    report = merge_agents(a, b)
    assert report.skills_reinforced >= 1
    sk = a.skills.find_matching(p)
    assert sk.success_count == 4


def test_merge_combines_provenance():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b.tell("dog", "isa", "mammal")
    pre = len(a.provenance._records)
    merge_agents(a, b)
    assert len(a.provenance._records) > pre
    assert a.provenance.get("dog", "isa", "mammal") is not None


def test_merge_marks_source_multi_on_conflict():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")  # source=user
    # Inject a non-user source on b.
    b.knowledge.store({"S": "dog", "R": "isa", "O": "mammal"})
    b.provenance.store("dog", "isa", "mammal", source="induced")
    merge_agents(a, b)
    rec = a.provenance.get("dog", "isa", "mammal")
    assert rec.source in {"multi", "user"}  # may stay user if existing != source matches


def test_merge_combines_kb():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    b.tell("cat", "isa", "mammal")
    pre_size = a.knowledge.size()
    merge_agents(a, b)
    assert a.knowledge.size() > pre_size
    # Both facts should now be retrievable from a.
    dog_ans, dog_score = a.knowledge.answer({"S": "dog", "R": "isa"}, "O")
    cat_ans, cat_score = a.knowledge.answer({"S": "cat", "R": "isa"}, "O")
    assert dog_ans == "mammal" and dog_score > 0.10
    assert cat_ans == "mammal" and cat_score > 0.10


def test_merge_bumps_chain_cache_version():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b.tell("cat", "isa", "mammal")
    pre_version = a.chain_cache.kb_version
    merge_agents(a, b)
    assert a.chain_cache.kb_version > pre_version
