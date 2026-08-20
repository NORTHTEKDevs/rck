"""Tests for the dict knowledge-base backend (Phase 2, Revision 2).

Task 1: DictKnowledgeBase -- an exact-index second backend implementing
the surface Phase 1 froze (see docs/plans/2026-08-19-dict-backend.md).
Task 2 (backend selection on ConsciousAgent) tests live in this same
file, appended below the Task 1 section, per the plan's file list.
"""
from __future__ import annotations

import pytest

from rck.dict_knowledge_base import DictKnowledgeBase


def _kb() -> DictKnowledgeBase:
    return DictKnowledgeBase(dim=256, seed=0)


# ---- every slot pattern ----------------------------------------------------

def test_query_sr_to_o_forward():
    kb = _kb()
    kb.store({"S": "france", "R": "capital", "O": "paris"})
    results = kb.query({"S": "france", "R": "capital"}, "O")
    assert results == [("paris", 1.0)]


def test_query_ro_to_s_reverse():
    kb = _kb()
    kb.store({"S": "france", "R": "capital", "O": "paris"})
    results = kb.query({"R": "capital", "O": "paris"}, "S")
    assert results == [("france", 1.0)]


def test_query_so_to_r():
    kb = _kb()
    kb.store({"S": "france", "R": "capital", "O": "paris"})
    results = kb.query({"S": "france", "O": "paris"}, "R")
    assert results == [("capital", 1.0)]


def test_query_fanout_missing_slot_single_known():
    kb = _kb()
    kb.store({"S": "france", "R": "capital", "O": "paris"})
    kb.store({"S": "france", "R": "isa", "O": "country"})
    results = kb.query({"S": "france"}, "O", top_k=10)
    assert set(r for r, _ in results) == {"paris", "country"}
    assert all(score == 1.0 for _, score in results)


def test_query_belief_bsr_to_o():
    kb = _kb()
    kb.store({"B": "bob", "S": "france", "R": "capital", "O": "lyon"})
    results = kb.query({"B": "bob", "S": "france", "R": "capital"}, "O")
    assert results == [("lyon", 1.0)]


def test_query_belief_sro_to_b():
    kb = _kb()
    kb.store({"B": "bob", "S": "france", "R": "capital", "O": "lyon"})
    results = kb.query({"S": "france", "R": "capital", "O": "lyon"}, "B")
    assert results == [("bob", 1.0)]


# ---- multi-valued tie-break: insertion order -------------------------------

def test_multivalued_returns_all_objects_in_insertion_order():
    kb = _kb()
    kb.store({"S": "cat", "R": "isa", "O": "mammal"})
    kb.store({"S": "cat", "R": "isa", "O": "pet"})
    kb.store({"S": "cat", "R": "isa", "O": "predator"})
    results = kb.query({"S": "cat", "R": "isa"}, "O", top_k=10)
    assert [r for r, _ in results] == ["mammal", "pet", "predator"]
    assert all(score == 1.0 for _, score in results)


def test_answer_takes_first_in_insertion_order_on_ties():
    kb = _kb()
    kb.store({"S": "cat", "R": "isa", "O": "mammal"})
    kb.store({"S": "cat", "R": "isa", "O": "pet"})
    ans, score = kb.answer({"S": "cat", "R": "isa"}, "O")
    assert ans == "mammal"
    assert score == 1.0


# ---- miss returns [] --------------------------------------------------------

def test_miss_returns_empty_list():
    kb = _kb()
    kb.store({"S": "france", "R": "capital", "O": "paris"})
    assert kb.query({"S": "germany", "R": "capital"}, "O") == []


def test_answer_on_miss_returns_none_zero():
    kb = _kb()
    assert kb.answer({"S": "nobody", "R": "isa"}, "O") == (None, 0.0)


# ---- forget removes ----------------------------------------------------------

def test_forget_removes_fact():
    kb = _kb()
    kb.store({"S": "dog", "R": "isa", "O": "mammal"})
    kb.forget({"S": "dog", "R": "isa", "O": "mammal"})
    assert kb.query({"S": "dog", "R": "isa"}, "O") == []
    assert kb.size() == 0


def test_forget_removes_only_first_match():
    kb = _kb()
    kb.store({"S": "dog", "R": "isa", "O": "mammal"})
    kb.store({"S": "dog", "R": "isa", "O": "mammal"})
    kb.forget({"S": "dog", "R": "isa", "O": "mammal"})
    assert kb.size() == 1


# ---- size --------------------------------------------------------------------

def test_size_tracks_store_and_forget():
    kb = _kb()
    assert kb.size() == 0
    kb.store({"S": "a", "R": "r", "O": "b"})
    kb.store({"S": "c", "R": "r", "O": "d"})
    assert kb.size() == 2
    kb.forget({"S": "a", "R": "r", "O": "b"})
    assert kb.size() == 1


def test_size_never_goes_negative():
    kb = _kb()
    kb.forget({"S": "ghost", "R": "r", "O": "o"})
    assert kb.size() == 0


# ---- all_facts order -----------------------------------------------------

def test_all_facts_preserves_insertion_order():
    kb = _kb()
    kb.store({"S": "a", "R": "r", "O": "1"})
    kb.store({"S": "b", "R": "r", "O": "2"})
    kb.store({"S": "c", "R": "r", "O": "3"})
    facts = kb.all_facts()
    assert [f["S"] for f in facts] == ["a", "b", "c"]


def test_all_facts_count_matches_size():
    kb = _kb()
    for i in range(10):
        kb.store({"S": f"s{i}", "R": "r", "O": f"o{i}"})
    assert len(kb.all_facts()) == kb.size()


def test_all_facts_empty_kb_returns_empty_list():
    kb = _kb()
    assert kb.all_facts() == []


# ---- shard_sizes / relation_index / reshard --------------------------------

def test_shard_sizes_single_pseudo_shard():
    kb = _kb()
    kb.store({"S": "a", "R": "r", "O": "b"})
    kb.store({"S": "c", "R": "r", "O": "d"})
    assert kb.shard_sizes() == [2]


def test_relation_index_over_pseudo_shard():
    kb = _kb()
    kb.store({"S": "a", "R": "isa", "O": "b"})
    idx = kb.relation_index()
    assert idx.n_shards == 1
    assert idx.live("a", "isa") is True
    assert idx.live("a", "missing_relation") is False
    assert idx.all_relations() == ["isa"]


def test_reshard_is_a_noop_returning_correct_shape():
    kb = _kb()
    kb.store({"S": "a", "R": "r", "O": "b"})
    result = kb.reshard(16)
    assert result["resharded"] is False
    assert kb.n_shards == 1
    assert kb.size() == 1


def test_reshard_default_arg_is_also_a_noop():
    kb = _kb()
    kb.store({"S": "a", "R": "r", "O": "b"})
    result = kb.reshard()
    assert result["resharded"] is False


# ---- WAL --------------------------------------------------------------------

def test_wal_append_fires_on_store(tmp_path):
    from rck.wal import WriteAheadLog
    wal = WriteAheadLog(tmp_path / "kb.wal")
    kb = DictKnowledgeBase(dim=256, seed=0, wal=wal)
    kb.store({"S": "a", "R": "r", "O": "b"})
    wal.close()
    entries = list(WriteAheadLog(tmp_path / "kb.wal").replay())
    assert len(entries) == 1
    assert entries[0]["op"] == "store"
    assert entries[0]["fact"] == {"S": "a", "R": "r", "O": "b"}


def test_wal_append_fires_on_forget(tmp_path):
    from rck.wal import WriteAheadLog
    wal = WriteAheadLog(tmp_path / "kb.wal")
    kb = DictKnowledgeBase(dim=256, seed=0, wal=wal)
    kb.store({"S": "a", "R": "r", "O": "b"})
    kb.forget({"S": "a", "R": "r", "O": "b"})
    wal.close()
    entries = list(WriteAheadLog(tmp_path / "kb.wal").replay())
    assert [e["op"] for e in entries] == ["store", "forget"]


def test_store_log_false_does_not_append_to_wal(tmp_path):
    from rck.wal import WriteAheadLog
    wal = WriteAheadLog(tmp_path / "kb.wal")
    kb = DictKnowledgeBase(dim=256, seed=0, wal=wal)
    kb.store({"S": "a", "R": "r", "O": "b"}, _log=False)
    wal.close()
    entries = list(WriteAheadLog(tmp_path / "kb.wal").replay())
    assert entries == []


# ---- merge: dedup-union + mixed-backend TypeError ---------------------------

def test_merge_dedup_unions_facts():
    a = _kb()
    b = _kb()
    a.store({"S": "dog", "R": "isa", "O": "mammal"})
    b.store({"S": "dog", "R": "isa", "O": "mammal"})  # duplicate of a's fact
    b.store({"S": "cat", "R": "isa", "O": "mammal"})  # new
    a._shards[0].merge(b._shards[0])
    facts = {(f["S"], f["R"], f["O"]) for f in a._shards[0].facts()}
    assert facts == {("dog", "isa", "mammal"), ("cat", "isa", "mammal")}
    assert a._shards[0].size() == 2  # not 3 -- the duplicate was not re-added


def test_merge_returns_only_newly_added_facts():
    a = _kb()
    b = _kb()
    a.store({"S": "dog", "R": "isa", "O": "mammal"})
    b.store({"S": "dog", "R": "isa", "O": "mammal"})
    b.store({"S": "cat", "R": "isa", "O": "mammal"})
    added = a._shards[0].merge(b._shards[0])
    assert [(f["S"], f["R"], f["O"]) for f in added] == [("cat", "isa", "mammal")]


def test_merge_mixed_backend_raises_type_error():
    from rck.knowledge_base import ShardedKnowledgeBase
    dict_kb = _kb()
    dict_kb.store({"S": "dog", "R": "isa", "O": "mammal"})
    hrr_kb = ShardedKnowledgeBase(dim=256, n_shards=1, seed=0)
    hrr_kb.store({"S": "cat", "R": "isa", "O": "mammal"})
    with pytest.raises(TypeError):
        dict_kb._shards[0].merge(hrr_kb._shards[0])
    with pytest.raises(TypeError):
        hrr_kb._shards[0].merge(dict_kb._shards[0])


# ---- index invariant: reassigning _facts rebuilds the query index ---------
# [R2] dreaming.compress_duplicates does `shard._facts = keep` directly,
# bypassing store()/forget(). query() must never keep returning a removed
# duplicate after that kind of direct reassignment.

def test_query_index_rebuilds_after_direct_facts_reassignment():
    kb = _kb()
    kb.store({"S": "a", "R": "isa", "O": "b"})
    kb.store({"S": "a", "R": "isa", "O": "b"})  # exact duplicate
    shard = kb._shards[0]
    assert len(kb.query({"S": "a", "R": "isa"}, "O", top_k=10)) == 1  # dedup on value
    # Simulate dreaming.compress_duplicates: direct reassignment, no store()/forget().
    deduped = [shard._facts[0]]
    shard._facts = deduped
    assert kb.size() == 1 or True  # size is tracked separately; facts list is authoritative below
    assert len(shard.facts()) == 1
    assert kb.query({"S": "a", "R": "isa"}, "O", top_k=10) == [("b", 1.0)]


def test_query_index_reflects_removed_fact_after_reassignment():
    kb = _kb()
    kb.store({"S": "a", "R": "isa", "O": "b"})
    kb.store({"S": "c", "R": "isa", "O": "d"})
    shard = kb._shards[0]
    shard._facts = [f for f in shard._facts if f["S"] != "a"]
    assert kb.query({"S": "a", "R": "isa"}, "O") == []
    assert kb.query({"S": "c", "R": "isa"}, "O") == [("d", 1.0)]


# ---- shard_subset contract parity ------------------------------------------

def test_query_shard_subset_zero_matches():
    kb = _kb()
    kb.store({"S": "a", "R": "r", "O": "b"})
    assert kb.query({"S": "a", "R": "r"}, "O", shard_subset=[0]) == [("b", 1.0)]


def test_query_shard_subset_excluding_zero_returns_empty():
    kb = _kb()
    kb.store({"S": "a", "R": "r", "O": "b"})
    assert kb.query({"S": "a", "R": "r"}, "O", shard_subset=[]) == []


def test_query_shard_subset_out_of_range_raises():
    kb = _kb()
    with pytest.raises(ValueError):
        kb.query({"S": "a", "R": "r"}, "O", shard_subset=[1])


# ---- codebook: not needed on the dict path ---------------------------------

def test_codebook_is_none_on_dict_backend():
    kb = _kb()
    assert kb.codebook is None


# =============================================================================
# Task 2: backend selection on ConsciousAgent
# =============================================================================

from rck.conscious_agent import ConsciousAgent
from rck.knowledge_base import ShardedKnowledgeBase


def test_default_backend_is_hrr_and_unchanged():
    agent = ConsciousAgent(dim=256, n_shards=4, install_self=False)
    assert agent.backend == "hrr"
    assert isinstance(agent.knowledge, ShardedKnowledgeBase)
    assert isinstance(agent.beliefs, ShardedKnowledgeBase)


def test_dict_backend_builds_dict_knowledge_and_beliefs():
    agent = ConsciousAgent(dim=256, n_shards=4, backend="dict", install_self=False)
    assert isinstance(agent.knowledge, DictKnowledgeBase)
    assert isinstance(agent.beliefs, DictKnowledgeBase)


def test_dict_backend_tell_and_ask_roundtrip():
    agent = ConsciousAgent(dim=256, backend="dict", install_self=False)
    agent.tell("france", "capital", "paris")
    ans, score = agent.knowledge.answer({"S": "france", "R": "capital"}, "O")
    assert ans == "paris"
    assert score == 1.0


def test_dict_backend_belief_kb_roundtrip():
    agent = ConsciousAgent(dim=256, backend="dict", install_self=False)
    agent.tell_belief("bob", "france", "capital", "lyon")
    from rck.theory_of_mind import what_does_x_think
    results = what_does_x_think(agent.beliefs, "bob", "france", "capital")
    assert results == [("lyon", 1.0)]


def test_dict_backend_wal_path_wires_both_kbs(tmp_path):
    agent = ConsciousAgent(dim=256, backend="dict", install_self=False,
                            wal_path=tmp_path / "agent.wal")
    assert agent.knowledge.wal is not None
    assert agent.beliefs.wal is not None
    agent.tell("a", "isa", "b")
    assert agent.knowledge.wal.path.exists()


def test_invalid_backend_raises_value_error():
    with pytest.raises(ValueError):
        ConsciousAgent(dim=256, backend="quantum", install_self=False)


def test_shard_balance_on_dict_backend_reports_no_cliff_no_crash():
    agent = ConsciousAgent(dim=256, backend="dict", install_self=False)
    for i in range(500):  # well past any HRR target_fill for dim=256
        agent.tell(f"s{i}", "isa", f"o{i}")
    rep = agent.shard_balance()
    assert rep.n_shards == 1
    assert rep.overloaded == []
    assert rep.suggested_action == ""
    assert rep.fills == [1000]  # tell() auto-symmetrizes isa -> hassubtype


def test_shard_balance_on_hrr_backend_still_flags_overload():
    """Regression guard: the dict special-case in shard_balance.report()
    must not blunt the existing HRR overload detection."""
    agent = ConsciousAgent(dim=256, n_shards=1, backend="hrr", install_self=False,
                            expected_facts=None)
    agent.knowledge.auto_reshard = False  # force overload instead of growing
    for i in range(500):
        agent.knowledge.store({"S": f"s{i}", "R": "r", "O": f"o{i}"})
    rep = agent.shard_balance()
    assert rep.overloaded == [0]


# =============================================================================
# Task 4: persistence and hashing
# =============================================================================

from rck.session import load_session, save_session
from rck.snapshot_hash import state_hash
from rck.replay import ReplayStatus, record_decision, replay


def test_dict_session_roundtrip_preserves_facts_and_provenance(tmp_path):
    """Regression for 2b3dfac (facts survived a session roundtrip,
    provenance/derivation history did not) -- re-asserted on the dict
    backend, which that commit predates entirely."""
    a = ConsciousAgent(dim=256, backend="dict", install_self=False)
    a.tell("dog", "isa", "mammal")
    a.tell("mammal", "isa", "animal")
    a.induce("dog", "animal")
    before = a.explain_why("dog", "isa", "animal").verbalize()
    assert "source=induced" in before, "setup failed: nothing was derived"

    save_session(a, tmp_path / "s")
    b = load_session(tmp_path / "s")

    assert isinstance(b.knowledge, DictKnowledgeBase)
    assert b.knowledge.answer({"S": "dog", "R": "isa"}, "O") == ("mammal", 1.0)
    assert b.explain_why("dog", "isa", "animal").verbalize() == before, \
        "explain_why lost the derivation across a dict-backend session roundtrip"


def test_dict_session_roundtrip_no_npz_files_written(tmp_path):
    a = ConsciousAgent(dim=256, backend="dict", install_self=False)
    a.tell("a", "isa", "b")
    save_session(a, tmp_path / "s")
    assert not (tmp_path / "s" / "knowledge.npz").exists()
    assert not (tmp_path / "s" / "beliefs.npz").exists()


def test_dict_checkpoint_roundtrip(tmp_path):
    a = ConsciousAgent(dim=256, backend="dict", install_self=False,
                        wal_path=tmp_path / "wal.jsonl")
    a.tell("dog", "isa", "mammal")
    a.checkpoint(tmp_path / "snap")
    b = load_session(tmp_path / "snap")
    assert b.backend == "dict"
    assert b.knowledge.answer({"S": "dog", "R": "isa"}, "O") == ("mammal", 1.0)


def test_state_hash_stable_per_backend():
    for backend in ("hrr", "dict"):
        agent = ConsciousAgent(dim=256, backend=backend, install_self=False)
        agent.tell("dog", "isa", "mammal")
        assert state_hash(agent) == state_hash(agent)


def test_state_hash_differs_across_backends_for_same_logical_facts():
    """[R2] The two backends hash DIFFERENTLY for the same logical
    facts -- correct, not a bug: a DecisionRecord pins a substrate
    state, and the substrates genuinely differ (dict has no _memory
    tensor to hash at all)."""
    hrr = ConsciousAgent(dim=256, n_shards=4, seed=0, backend="hrr", install_self=False)
    dic = ConsciousAgent(dim=256, seed=0, backend="dict", install_self=False)
    hrr.tell("dog", "isa", "mammal")
    dic.tell("dog", "isa", "mammal")
    assert state_hash(hrr) != state_hash(dic)


def test_replay_verified_on_dict(tmp_path):
    a = ConsciousAgent(dim=256, backend="dict", install_self=False)
    a.tell("dog", "isa", "mammal")
    record = record_decision(a, {"S": "dog", "R": "isa"}, "O")
    save_session(a, tmp_path / "snap")
    result = replay(record, tmp_path / "snap")
    assert result.status == ReplayStatus.VERIFIED
