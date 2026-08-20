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

    restored = DecisionRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert restored == rec


def test_top_score_survives_as_exact_float():
    a = _agent()
    a.tell("dog", "isa", "mammal")
    rec = record_decision(a, {"S": "dog", "R": "isa"}, "O")

    restored = DecisionRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    # Exact bit-identity is the product claim -- not isclose().
    assert restored.answer["top_score"] == rec.answer["top_score"]


def test_idk_record_has_no_derivation_and_still_roundtrips():
    a = _agent()
    # Nothing has ever been told to this agent -- codebook is empty,
    # guaranteeing an IDK answer with no candidates.
    rec = record_decision(a, {"S": "nonexistent", "R": "isa"}, "O")

    assert rec.answer["top_symbol"] is None
    assert rec.derivation is None
    restored = DecisionRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
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


# ---- Task 4: end-to-end audit scenario -----------------------------------

def test_audit_scenario_multi_hop_replay_then_state_mismatch(tmp_path):
    """The entire feature in one test: an answer is either reproducible
    against pinned state, or you are told plainly that the state moved.

    Only possible because of the 2b3dfac provenance-persistence fix --
    if the derivation comparison fails, that fix regressed. Do not
    weaken this assertion."""
    # First hop is a lifting relation ("isa"), last hop is one of its
    # allowed transfers ("has") -- chain_induction stores the shortcut
    # under "has", NOT "isa", so it never competes with the direct
    # "dog isa mammal" fact for the same (S, R) query. That keeps which
    # fact wins the query deterministic instead of an HRR-crosstalk
    # coin flip between two same-relation candidates.
    a = _agent()
    a.tell("dog", "isa", "mammal")
    a.tell("mammal", "has", "fur")
    induced = a.induce("dog", "fur")
    assert induced is not None and induced.verified, \
        "setup failed: nothing was induced"
    assert induced.relation == "has"

    rec = record_decision(a, {"S": "dog", "R": "has"}, "O")
    assert rec.answer["top_symbol"] == "fur"
    assert rec.derivation is not None
    assert rec.derivation["source"] == "induced", \
        "setup failed: recorded answer wasn't the multi-hop induced fact"
    assert len(rec.derivation["children"]) >= 2, \
        "setup failed: derivation isn't actually multi-hop"

    a.checkpoint(tmp_path / "snap")
    rec.save(tmp_path / "record.json")

    # Replay in a FRESH subprocess -- the derivation tree must be
    # byte-identical across process boundaries, not just in-memory.
    script = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import json\n"
        "from rck.replay import DecisionRecord, replay\n"
        f"rec = DecisionRecord.load({str(tmp_path / 'record.json')!r})\n"
        f"result = replay(rec, {str(tmp_path / 'snap')!r})\n"
        "print(result.status.value)\n"
        "print(json.dumps(result.details.get('derivation')))\n"
    )
    out = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, check=True)
    lines = out.stdout.strip().splitlines()
    assert lines[0] == "verified"
    replayed_derivation = json.loads(lines[1])
    assert replayed_derivation == rec.derivation, (
        "derivation tree diverged across a checkpoint -- the 2b3dfac "
        "provenance-persistence fix regressed; same nodes, same "
        "sources, same confidences are required"
    )

    # Mutate the KB via a real user correction, snapshot again, and
    # replay the SAME record against the NEW snapshot.
    corr = a.correct("Actually, dog is puppy, not mammal.")
    assert corr is not None, "setup failed: correction text did not match"
    a.checkpoint(tmp_path / "snap2")

    result2 = replay(rec, tmp_path / "snap2")
    assert result2.status is ReplayStatus.STATE_MISMATCH, (
        "a record replayed against a moved snapshot must report "
        "STATE_MISMATCH, not silently verify or diverge"
    )


def test_record_survives_a_real_file_roundtrip(tmp_path):
    """The durable path, which is the whole point of a decision record.

    to_dict/from_dict convert to and from DICTS; save/load are the file
    API. They were originally named to_json/from_json, which invited
    `from_json(open(f).read())` -- that looks obviously correct and
    raises. This test pins the API a real auditor would reach for.
    """
    from rck.conscious_agent import ConsciousAgent
    from rck.replay import DecisionRecord, record_decision

    agent = ConsciousAgent(expected_facts=100)
    agent.tell("dog", "isa", "mammal")
    rec = record_decision(agent, {"S": "dog", "R": "isa"}, "O")

    p = tmp_path / "decision.json"
    rec.save(p)
    restored = DecisionRecord.load(p)

    assert restored.state_hash == rec.state_hash
    assert restored.answer["top_score"] == rec.answer["top_score"], \
        "score must survive the file roundtrip EXACTLY -- bit-identity is the claim"
    assert restored.answer["top_symbol"] == rec.answer["top_symbol"]
    assert restored.rck_version == rec.rck_version
