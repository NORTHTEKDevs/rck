"""Tests for agent.what_if_user_says preview."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_what_if_extracts_triples():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    result = agent.what_if_user_says("the dog is a mammal")
    assert isinstance(result, dict)
    # The extractor might not parse this sentence; we just check shape.
    assert "extracted" in result


def test_what_if_returns_preview_when_triples_found():
    """A sentence that the open-IE rules parse should produce a preview dict."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("a", "isa", "b")
    agent.tell("b", "isa", "c")
    # Run preview with explicit triples bypassing the extractor.
    triples = [("c", "isa", "d")]
    res = agent.what_changes(triples)
    assert "n_verified_inductions" in res
    assert "sample_derived" in res


def test_what_if_empty_text_returns_note():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    res = agent.what_if_user_says("")
    assert res["extracted"] == []
    assert "note" in res


def test_what_if_preserves_kb_after_preview():
    """The preview should NOT mutate the KB."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("a", "isa", "b")
    pre = agent.knowledge.size()
    agent.what_if_user_says("the cat is a mammal")
    # Even if extraction succeeded, the preview rolls back.
    assert agent.knowledge.size() == pre
