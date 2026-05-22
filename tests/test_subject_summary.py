"""Tests for subject summary."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent
from rck.subject_summary import SubjectSummary


def test_summarize_returns_facts():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.tell("dog", "has", "fur")
    s = agent.summarize_subject("dog")
    assert isinstance(s, SubjectSummary)
    assert s.subject == "dog"
    assert s.n_facts >= 2
    rels = {r for r, o in s.facts}
    assert "isa" in rels
    assert "has" in rels
    assert "mammal" in s.parents_via_isa


def test_summarize_lists_negative_facts():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.deny("dog", "isa", "fish")
    s = agent.summarize_subject("dog")
    assert any(o == "fish" for r, o in s.negative_facts)


def test_summarize_finds_siblings():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.tell("cat", "isa", "mammal")
    agent.tell("horse", "isa", "mammal")
    s = agent.summarize_subject("dog")
    assert s.siblings_via_isa
    sib_set = set(s.siblings_via_isa)
    assert "cat" in sib_set or "horse" in sib_set


def test_summarize_counts_query_hits():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    for _ in range(3):
        agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    s = agent.summarize_subject("dog")
    assert s.n_query_hits >= 3


def test_summarize_render_text():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    s = agent.summarize_subject("dog")
    text = s.render()
    assert "dog" in text
    assert "mammal" in text


def test_summarize_unknown_subject_handles_empty():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    s = agent.summarize_subject("nonexistent_thing")
    assert s.n_facts == 0
    text = s.render()
    assert "no facts" in text.lower() or "nonexistent" in text
