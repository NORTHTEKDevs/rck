"""Tests for rck.wal -- the write-ahead log (Task 3) and the KB-level
WAL hook + recover() (Task 4)."""
from __future__ import annotations

import json

import pytest

from rck.conscious_agent import ConsciousAgent
from rck.wal import WALLockedError, WriteAheadLog


def test_append_then_replay_preserves_order(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.jsonl")
    wal.append("store", {"S": "dog", "R": "isa", "O": "mammal"})
    wal.append("store", {"S": "cat", "R": "isa", "O": "mammal"})
    wal.append("forget", {"S": "dog", "R": "isa", "O": "mammal"})
    entries = list(wal.replay())
    wal.close()
    assert [e["op"] for e in entries] == ["store", "store", "forget"]
    assert entries[0]["fact"] == {"S": "dog", "R": "isa", "O": "mammal"}


def test_missing_log_replays_empty(tmp_path):
    wal = WriteAheadLog(tmp_path / "missing.jsonl")
    assert list(wal.replay()) == []
    wal.close()


def test_torn_final_line_keeps_prior_entries(tmp_path):
    p = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(p)
    wal.append("store", {"S": "dog", "R": "isa", "O": "mammal"})
    wal.append("store", {"S": "cat", "R": "isa", "O": "mammal"})
    wal.close()
    # Simulate a crash mid-write of a third line: partial content, no
    # trailing newline -- a torn write, not corruption.
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"op": "store", "fact": {"S": "fish", "R"')

    wal2 = WriteAheadLog(p)
    entries = list(wal2.replay())
    wal2.close()
    assert len(entries) == 2
    assert entries[1]["fact"]["S"] == "cat"


def test_interior_malformed_line_raises(tmp_path):
    p = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(p)
    wal.append("store", {"S": "dog", "R": "isa", "O": "mammal"})
    wal.close()
    # Corrupt an INTERIOR line by hand (not the last one) -- must raise,
    # not be silently skipped like a torn trailing write.
    with open(p, "a", encoding="utf-8") as f:
        f.write("not valid json at all\n")
        f.write(json.dumps(
            {"op": "store", "fact": {"S": "cat", "R": "isa", "O": "mammal"}},
        ) + "\n")

    wal3 = WriteAheadLog(p)
    with pytest.raises(ValueError):
        list(wal3.replay())
    wal3.close()


def test_truncate_clears(tmp_path):
    p = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(p)
    wal.append("store", {"S": "dog", "R": "isa", "O": "mammal"})
    wal.truncate()
    assert list(wal.replay()) == []
    wal.close()


def test_order_preserved_across_reopen(tmp_path):
    p = tmp_path / "wal.jsonl"
    wal = WriteAheadLog(p)
    wal.append("store", {"S": "dog", "R": "isa", "O": "mammal"})
    wal.close()

    wal2 = WriteAheadLog(p)
    wal2.append("store", {"S": "cat", "R": "isa", "O": "mammal"})
    entries = list(wal2.replay())
    wal2.close()
    assert [e["fact"]["S"] for e in entries] == ["dog", "cat"]


def test_second_concurrent_writer_raises(tmp_path):
    # The lock is acquired lazily, on the first write -- so merely opening
    # a second WriteAheadLog (e.g. for a read-only replay) must NOT
    # contend with a live writer (see the Task 5 checkpoint test, which
    # depends on exactly this). Only a second WRITE attempt must raise.
    p = tmp_path / "wal.jsonl"
    wal1 = WriteAheadLog(p)
    wal1.append("store", {"S": "dog", "R": "isa", "O": "mammal"})
    wal2 = WriteAheadLog(p)  # opening alone must not raise
    try:
        with pytest.raises(WALLockedError):
            wal2.append("store", {"S": "cat", "R": "isa", "O": "mammal"})
    finally:
        wal1.close()
        wal2.close()


def test_context_manager_releases_lock(tmp_path):
    p = tmp_path / "wal.jsonl"
    with WriteAheadLog(p) as wal:
        wal.append("store", {"S": "dog", "R": "isa", "O": "mammal"})
    # Lock released on exit -- a new writer must succeed at actually
    # writing, not just at opening.
    wal2 = WriteAheadLog(p)
    wal2.append("store", {"S": "cat", "R": "isa", "O": "mammal"})
    wal2.close()


# ---- Task 4: KB-level hook + recover() -------------------------------------

def test_wal_is_opt_in_and_absent_by_default(tmp_path):
    a = ConsciousAgent(dim=256, n_shards=4, seed=0, install_self=False)
    a.tell("dog", "isa", "mammal")
    assert a.knowledge.wal is None
    assert a.beliefs.wal is None
    assert list(tmp_path.iterdir()) == []


def test_recover_replays_facts_told_with_no_snapshot(tmp_path):
    wal_path = tmp_path / "knowledge.wal.jsonl"
    a = ConsciousAgent(dim=512, n_shards=4, seed=0, install_self=False,
                        wal_path=wal_path)
    a.tell("dog", "isa", "mammal")
    a.tell("cat", "isa", "mammal")
    a.knowledge.wal.close()
    a.beliefs.wal.close()

    # Simulate a crash: a fresh agent, same wal_path, no snapshot at all.
    b = ConsciousAgent(dim=512, n_shards=4, seed=0, install_self=False,
                        wal_path=wal_path)
    report = b.recover()
    assert report["knowledge"] >= 2
    assert b.knowledge.answer({"S": "dog", "R": "isa"}, "O")[0] == "mammal"
    b.knowledge.wal.close()
    b.beliefs.wal.close()


def test_recover_replays_bulk_ingest_facts(tmp_path):
    from rck.bulk_ingest import bulk_load_triples
    wal_path = tmp_path / "knowledge.wal.jsonl"
    a = ConsciousAgent(dim=512, n_shards=4, seed=0, install_self=False,
                        wal_path=wal_path)
    bulk_load_triples(a.knowledge, [("paris", "capitalof", "france")],
                       symmetrize=False)
    a.knowledge.wal.close()
    a.beliefs.wal.close()

    b = ConsciousAgent(dim=512, n_shards=4, seed=0, install_self=False,
                        wal_path=wal_path)
    b.recover()
    assert b.knowledge.answer({"S": "paris", "R": "capitalof"}, "O")[0] == "france"
    b.knowledge.wal.close()
    b.beliefs.wal.close()


def test_recover_replays_induce_facts(tmp_path):
    wal_path = tmp_path / "knowledge.wal.jsonl"
    a = ConsciousAgent(dim=512, n_shards=4, seed=0, install_self=False,
                        wal_path=wal_path)
    a.tell("leaf", "partof", "tree")
    a.tell("tree", "locatedin", "forest")
    induced = a.induce("leaf", "forest")
    assert induced is not None and induced.verified
    a.knowledge.wal.close()
    a.beliefs.wal.close()

    b = ConsciousAgent(dim=512, n_shards=4, seed=0, install_self=False,
                        wal_path=wal_path)
    b.recover()
    ans = b.knowledge.query({"S": "leaf", "R": "locatedin"}, "O", top_k=1)
    assert ans and str(ans[0][0]) == "forest"
    b.knowledge.wal.close()
    b.beliefs.wal.close()


def test_recover_replays_merge_from_facts(tmp_path):
    wal_path = tmp_path / "knowledge.wal.jsonl"
    target = ConsciousAgent(dim=512, n_shards=4, seed=0, install_self=False,
                             wal_path=wal_path)
    source = ConsciousAgent(dim=512, n_shards=4, seed=1, install_self=False)
    source.tell("whale", "isa", "mammal")
    target.merge_from(source)
    assert target.knowledge.answer({"S": "whale", "R": "isa"}, "O")[0] == "mammal"
    target.knowledge.wal.close()
    target.beliefs.wal.close()

    # Crash target, recover a fresh one from the WAL alone.
    recovered = ConsciousAgent(dim=512, n_shards=4, seed=0, install_self=False,
                                wal_path=wal_path)
    recovered.recover()
    assert recovered.knowledge.answer({"S": "whale", "R": "isa"}, "O")[0] == "mammal"
    recovered.knowledge.wal.close()
    recovered.beliefs.wal.close()
