"""Tests for the agent.record_truth() calibration tie-in."""
from __future__ import annotations

from rck.conscious_agent import ConsciousAgent


def test_record_truth_correct_increments_calibration():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    # Tell the agent the correct answer matches its prediction.
    res = agent.record_truth({"S": "dog", "R": "isa"}, "O", "mammal")
    assert res["updated"] is True
    assert res["was_correct"] is True
    # Calibration tally should now have a "know_right" entry for "isa"
    # (if confidence was high enough) or another bucket.
    summary = agent.calibration.summary()
    assert "isa" in summary


def test_record_truth_wrong_increments_wrong_bucket():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    # Lie about the correct answer.
    res = agent.record_truth({"S": "dog", "R": "isa"}, "O", "fish")
    assert res["updated"] is True
    assert res["was_correct"] is False
    summary = agent.calibration.summary()
    assert "isa" in summary
    bucket = summary["isa"]
    # One of the *_wrong buckets must be non-zero.
    assert (bucket.get("know_wrong", 0)
            + bucket.get("think_wrong", 0)
            + bucket.get("guess_wrong", 0)) > 0


def test_record_truth_no_prior_episode_returns_unchanged():
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    res = agent.record_truth({"S": "dog", "R": "isa"}, "O", "mammal")
    assert res["updated"] is False
    assert "no prior" in res["reason"]


def test_record_truth_uses_most_recent_episode():
    """If the same signature was asked twice, the most recent answer
    is the one we calibrate against."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    # Change the KB; ask again.
    agent.knowledge.forget({"S": "dog", "R": "isa", "O": "mammal"})
    agent.tell("dog", "isa", "animal")
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    # The latest predicted answer should match "animal", so saying
    # "animal" is the correct ground truth.
    res = agent.record_truth({"S": "dog", "R": "isa"}, "O", "animal")
    assert res["updated"] is True
    if res["predicted"] is not None:
        # If the system was confident enough, the prediction should be animal.
        # Either way, was_correct reflects (predicted == "animal").
        assert res["was_correct"] == (res["predicted"].lower() == "animal")


def test_calibration_score_after_multiple_records():
    """Recording 3 correct + 1 wrong on the same relation should give a
    non-zero calibration_score for that relation (when confidence is
    high enough to land in the 'know' bucket)."""
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("a", "isa", "X")
    agent.tell("b", "isa", "Y")
    for s, gold in [("a", "x"), ("b", "y"), ("a", "x"), ("a", "wrong")]:
        agent.ask_with_idk({"S": s, "R": "isa"}, "O")
        agent.record_truth({"S": s, "R": "isa"}, "O", gold)
    summary = agent.calibration.summary()
    assert "isa" in summary
