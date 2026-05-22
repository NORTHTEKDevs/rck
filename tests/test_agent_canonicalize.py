"""Tests for the agent's canonicalize helper."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_canonicalize_lowercases_and_strips():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    assert agent.canonicalize("Dog") == "dog"
    assert agent.canonicalize("  dog  ") == "dog"
    assert agent.canonicalize("Saint Bernard") == "saint_bernard"


def test_canonicalize_relation_kind():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    # Relation canonicalisation typically lowercases + handles synonyms.
    result = agent.canonicalize("ISA", kind="relation")
    assert result == "isa"


def test_canonicalize_unknown_passes_through():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    # Unrecognised tokens just get normalised, not rejected.
    assert agent.canonicalize("totally_made_up_entity") == "totally_made_up_entity"
