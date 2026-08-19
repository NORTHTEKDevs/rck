"""Tests for rck.wal -- the write-ahead log (Task 3)."""
from __future__ import annotations

import json

import pytest

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
    p = tmp_path / "wal.jsonl"
    wal1 = WriteAheadLog(p)
    try:
        with pytest.raises(WALLockedError):
            WriteAheadLog(p)
    finally:
        wal1.close()


def test_context_manager_releases_lock(tmp_path):
    p = tmp_path / "wal.jsonl"
    with WriteAheadLog(p) as wal:
        wal.append("store", {"S": "dog", "R": "isa", "O": "mammal"})
    # Lock released on exit -- a new writer must succeed.
    wal2 = WriteAheadLog(p)
    wal2.close()
