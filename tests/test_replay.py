"""Tests for rck.replay: DecisionRecord (Task 2), replay() (Task 3),
and the end-to-end audit scenario (Task 4).

See docs/plans/2026-08-19-replay.md.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rck.conscious_agent import ConsciousAgent
from rck.knowledge_base import ShardedKnowledgeBase
from rck.replay import DecisionRecord, ReplayResult, ReplayStatus, record_decision, replay
from rck.snapshot_hash import state_hash

REPO_ROOT = Path(__file__).parent.parent


def _agent(seed=0, n_shards=8):
    return ConsciousAgent(dim=512, n_shards=n_shards, seed=seed,
                           install_self=False)


# ---- Task 2: DecisionRecord --------------------------------------------

def test_record_decision_captures_answer_and_state_hash():
    a = _agent()
    a.tell("dog", "isa", "mammal")
    rec = record_decision(a, {"S": "dog", "R": "isa"}, "O")

    assert rec.answer["top_symbol"] == "mammal"
    assert rec.state_hash == state_hash(a)
    assert rec.seed == a.seed
    assert rec.query == {"S": "dog", "R": "isa"}
    assert rec.unknown_role == "O"


def test_decision_record_json_roundtrip_is_lossless():
    a = _agent()
    a.tell("dog", "isa", "mammal")
    a.tell("mammal", "isa", "animal")
    a.induce("dog", "animal")
    rec = record_decision(a, {"S": "dog", "R": "isa"}, "O")

    restored = DecisionRecord.from_json(json.loads(json.dumps(rec.to_json())))
    assert restored == rec


def test_top_score_survives_as_exact_float():
    a = _agent()
    a.tell("dog", "isa", "mammal")
    rec = record_decision(a, {"S": "dog", "R": "isa"}, "O")

    restored = DecisionRecord.from_json(json.loads(json.dumps(rec.to_json())))
    # Exact bit-identity is the product claim -- not isclose().
    assert restored.answer["top_score"] == rec.answer["top_score"]


def test_idk_record_has_no_derivation_and_still_roundtrips():
    a = _agent()
    # Nothing has ever been told to this agent -- codebook is empty,
    # guaranteeing an IDK answer with no candidates.
    rec = record_decision(a, {"S": "nonexistent", "R": "isa"}, "O")

    assert rec.answer["top_symbol"] is None
    assert rec.derivation is None
    restored = DecisionRecord.from_json(json.loads(json.dumps(rec.to_json())))
    assert restored == rec


def test_decision_record_save_and_load_roundtrip(tmp_path):
    a = _agent()
    a.tell("dog", "isa", "mammal")
    rec = record_decision(a, {"S": "dog", "R": "isa"}, "O")

    rec.save(tmp_path / "record.json")
    restored = DecisionRecord.load(tmp_path / "record.json")
    assert restored == rec


# ---- Task 3: replay() ---------------------------------------------------

def test_replay_verified_after_record_and_checkpoint(tmp_path):
    a = _agent()
    a.tell("dog", "isa", "mammal")
    rec = record_decision(a, {"S": "dog", "R": "isa"}, "O")
    a.checkpoint(tmp_path / "snap")

    result = replay(rec, tmp_path / "snap")
    assert result.status is ReplayStatus.VERIFIED
    assert result.details["answer"]["top_score"] == rec.answer["top_score"]


def test_replay_state_mismatch_never_runs_the_query(tmp_path, mocker):
    a = _agent()
    a.tell("dog", "isa", "mammal")
    rec = record_decision(a, {"S": "dog", "R": "isa"}, "O")

    a.tell("cat", "isa", "mammal")   # extra fact -> the snapshot has moved
    a.checkpoint(tmp_path / "snap")

    spy = mocker.spy(ShardedKnowledgeBase, "query")
    result = replay(rec, tmp_path / "snap")

    assert result.status is ReplayStatus.STATE_MISMATCH
    assert spy.call_count == 0, "replay ran the query despite a state mismatch"


def test_replay_in_fresh_subprocess_returns_verified(tmp_path):
    """This is the test that actually proves the product claim."""
    a = _agent()
    a.tell("dog", "isa", "mammal")
    a.tell("mammal", "isa", "animal")
    rec = record_decision(a, {"S": "dog", "R": "isa"}, "O")
    a.checkpoint(tmp_path / "snap")
    rec.save(tmp_path / "record.json")

    script = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from rck.replay import DecisionRecord, replay\n"
        f"rec = DecisionRecord.load({str(tmp_path / 'record.json')!r})\n"
        f"result = replay(rec, {str(tmp_path / 'snap')!r})\n"
        "print(result.status.value)\n"
        "print(repr(result.details['answer']['top_score']))\n"
    )
    out = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, check=True)
    lines = out.stdout.strip().splitlines()
    assert lines[0] == "verified"
    assert lines[1] == repr(rec.answer["top_score"]), (
        "fresh-subprocess replay did not reproduce an exact score match"
    )


def test_idk_answer_replays_to_verified(tmp_path):
    a = _agent()
    rec = record_decision(a, {"S": "nonexistent", "R": "isa"}, "O")
    assert rec.answer["state"] == "idk"
    a.checkpoint(tmp_path / "snap")

    result = replay(rec, tmp_path / "snap")
    assert result.status is ReplayStatus.VERIFIED
