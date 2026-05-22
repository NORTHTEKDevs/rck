"""Tests for provenance-graph explanations."""
from __future__ import annotations

from rck.explain_why import (
    ExplanationNode, explain, explanation_summary,
)
from rck.provenance import ProvenanceStore


def test_user_fact_is_leaf():
    prov = ProvenanceStore()
    prov.store("dog", "isa", "mammal", source="user")
    node = explain(prov, "dog", "isa", "mammal")
    assert node.is_leaf
    assert node.source == "user"


def test_unknown_fact_is_leaf_with_note():
    prov = ProvenanceStore()
    node = explain(prov, "qux", "isa", "thing")
    assert node.is_leaf
    assert node.source == "unknown"
    assert "no provenance" in node.note


def test_induced_fact_expands_to_children():
    prov = ProvenanceStore()
    # Build a 2-hop derivation graph.
    prov.store("a", "partof", "b", source="user")
    prov.store("b", "locatedin", "c", source="user")
    prov.store("a", "locatedin", "c", source="induced",
               derivation=[("a", "partof", "b"), ("b", "locatedin", "c")])
    node = explain(prov, "a", "locatedin", "c")
    assert node.source == "induced"
    assert not node.is_leaf
    assert len(node.children) == 2
    # The children should be the two derivation steps and both LEAVES
    # (their derivations are empty / user-asserted).
    for child in node.children:
        assert child.is_leaf
        assert child.source == "user"


def test_recursive_explanation_two_levels():
    prov = ProvenanceStore()
    # Layer 1: user facts.
    prov.store("a", "partof", "b", source="user")
    prov.store("b", "partof", "c", source="user")
    prov.store("c", "locatedin", "d", source="user")
    # Layer 2: induced shortcut (a, partof, c) from (a partof b, b partof c).
    prov.store("a", "partof", "c", source="induced",
               derivation=[("a", "partof", "b"), ("b", "partof", "c")])
    # Layer 3: induced shortcut (a, locatedin, d) from (a partof c, c locatedin d).
    # Recursively, (a partof c) is itself induced.
    prov.store("a", "locatedin", "d", source="induced",
               derivation=[("a", "partof", "c"), ("c", "locatedin", "d")])

    node = explain(prov, "a", "locatedin", "d", max_depth=5)
    assert node.source == "induced"
    assert node.depth() >= 3  # at least 3 levels deep
    summary = explanation_summary(node)
    assert summary["total_facts_in_tree"] >= 5
    assert summary["sources"].get("user", 0) >= 3


def test_cycle_break():
    prov = ProvenanceStore()
    # A derivation cycle: (a, r, b) depends on (b, r, a) which depends on (a, r, b).
    prov.store("a", "r", "b", source="induced",
               derivation=[("b", "r", "a")])
    prov.store("b", "r", "a", source="induced",
               derivation=[("a", "r", "b")])
    node = explain(prov, "a", "r", "b", max_depth=10)
    # Recursion terminates via cycle break (one child marked cycle).
    found_cycle = any(
        c.source == "cycle" or any(g.source == "cycle" for g in c.children)
        for c in node.children
    )
    assert found_cycle


def test_max_depth_truncates():
    prov = ProvenanceStore()
    prov.store("a", "partof", "b", source="user")
    prov.store("b", "locatedin", "c", source="user")
    prov.store("a", "locatedin", "c", source="induced",
               derivation=[("a", "partof", "b"), ("b", "locatedin", "c")])
    node = explain(prov, "a", "locatedin", "c", max_depth=0)
    # max_depth=0 -> we don't recurse into derivation; treated as leaf.
    assert node.is_leaf


def test_verbalize_includes_subject_relation_obj():
    prov = ProvenanceStore()
    prov.store("dog", "isa", "mammal", source="user")
    node = explain(prov, "dog", "isa", "mammal")
    text = node.verbalize()
    assert "dog" in text and "isa" in text and "mammal" in text
    assert "user" in text


def test_conscious_agent_explain_why():
    """ConsciousAgent.explain_why() wraps the same call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("a", "isa", "b"),
        ("b", "isa", "c"),
    ]:
        agent.tell(s, r, o)
    induced = agent.induce("a", "c")
    if induced is not None and induced.verified:
        node = agent.explain_why("a", induced.relation, "c")
        assert node.source == "induced"
        assert len(node.children) >= 1
