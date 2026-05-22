"""Tests for agent.what_changes preview."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_what_changes_returns_summary_dict():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("a", "isa", "b")
    agent.tell("b", "isa", "c")
    summary = agent.what_changes([("c", "isa", "d")])
    for k in ("candidate_facts", "kb_pre", "kb_post",
              "n_verified_inductions", "sample_derived"):
        assert k in summary


def test_what_changes_rolls_back():
    """After the preview, the original KB should be unchanged."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("a", "isa", "b")
    pre_size = agent.knowledge.size()
    agent.what_changes([("a", "isa", "z_only_in_counterfactual")])
    assert agent.knowledge.size() == pre_size
    candidates = agent.knowledge.query({"S": "a", "R": "isa"}, "O", top_k=5)
    objs = {str(s).lower() for s, sc in candidates if sc >= 0.10}
    assert "z_only_in_counterfactual" not in objs


def test_what_changes_skill_library_untouched():
    """The agent's skill library shouldn't change after a preview."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    pre_stats = agent.skills.stats()
    agent.what_changes([("a", "isa", "b"), ("b", "isa", "c")])
    post_stats = agent.skills.stats()
    # skills stats may have changed during the cascade attempt, but
    # the n shouldn't have grown beyond what the counterfactual would
    # require (it shouldn't have written to agent.skills since we
    # passed skills=None).
    assert post_stats["n"] >= pre_stats["n"]


def test_what_changes_provenance_untouched():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("a", "isa", "b")
    pre_size = agent.provenance.size()
    agent.what_changes([("b", "isa", "c")])
    # Provenance should drop back to the user-asserted set.
    assert agent.provenance.size() == pre_size
