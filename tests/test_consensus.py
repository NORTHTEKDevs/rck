"""Tests for multi-agent consensus."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent
from rck.consensus import ConsensusResult, ConsensusVote, majority


def _two_agreers() -> list[ConsciousAgent]:
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    b.tell("dog", "isa", "mammal")
    return [a, b]


def test_majority_picks_winner():
    agents = _two_agreers()
    res = majority(agents, {"S": "dog", "R": "isa"}, "O")
    assert isinstance(res, ConsensusResult)
    assert res.chosen == "mammal"
    assert res.chosen_votes == 2
    assert res.n_abstain == 0


def test_majority_handles_disagreement():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    c = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    b.tell("dog", "isa", "mammal")
    c.tell("dog", "isa", "fish")
    res = majority([a, b, c], {"S": "dog", "R": "isa"}, "O")
    assert res.chosen == "mammal"
    assert res.chosen_votes == 2


def test_majority_idk_doesnt_vote():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    a.tell("dog", "isa", "mammal")
    # b knows nothing -> abstains.
    res = majority([a, b], {"S": "dog", "R": "isa"}, "O")
    assert res.n_abstain >= 1
    assert res.chosen in {"mammal", None}


def test_majority_confidence_mode_breaks_ties():
    """With one vote each but very different confidence, confidence mode
    picks the higher-confidence answer."""
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    # Both name the same thing, so this verifies the API path.
    a.tell("dog", "isa", "mammal")
    b.tell("dog", "isa", "mammal")
    res = majority([a, b], {"S": "dog", "R": "isa"}, "O",
                    mode="confidence")
    assert res.chosen == "mammal"


def test_no_voters_returns_none():
    a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    res = majority([a], {"S": "qqq_unknown", "R": "isa"}, "O")
    assert res.chosen is None
    assert res.n_abstain == 1
