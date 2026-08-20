"""Tests that the top-level `rck` namespace exposes exactly the frozen
public API, and that everything else still works via direct module
import (with a DeprecationWarning at the package root).
"""
import warnings

import pytest

import rck
from rck.conscious_agent import ConsciousAgent

FROZEN = {
    "ConsciousAgent", "ShardedKnowledgeBase",
    "DecisionRecord", "record_decision", "replay", "state_hash",
    "save_session", "load_session",
    "bulk_load_jsonl", "bulk_load_csv", "bulk_load_triples",
}


def test_all_is_exactly_the_frozen_surface():
    assert set(rck.__all__) == FROZEN


def test_every_name_in_all_actually_resolves():
    for name in rck.__all__:
        assert getattr(rck, name) is not None


def test_demoted_name_still_works_but_warns():
    with pytest.warns(DeprecationWarning, match="bind"):
        _ = rck.bind


def test_direct_module_import_is_unaffected_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # any warning fails the test
        from rck.vsa import bind            # noqa: F401
        from rck.analogy import solve_analogy  # noqa: F401


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = rck.definitely_not_a_real_name


def test_public_api_methods_all_exist_and_are_callable():
    for name in ConsciousAgent.PUBLIC_API:
        assert callable(getattr(ConsciousAgent, name)), name


def test_public_api_is_documented_in_the_class_docstring():
    doc = ConsciousAgent.__doc__ or ""
    for name in ConsciousAgent.PUBLIC_API:
        assert name in doc, f"{name} is in PUBLIC_API but undocumented"
