"""Tests for episodic query memory."""
from __future__ import annotations

import time

from rck.query_memory import QueryEpisode, QueryMemory


def test_record_and_recall():
    mem = QueryMemory()
    ep = mem.record({"S": "dog", "R": "isa"}, "O",
                     state="known", top_symbol="mammal", top_score=0.7)
    assert isinstance(ep, QueryEpisode)
    assert mem.size() == 1
    assert mem.recent(1)[0].top_symbol == "mammal"


def test_recent_limits_n():
    mem = QueryMemory()
    for i in range(5):
        mem.record({"S": f"q{i}", "R": "r"}, "O", state="known")
    assert len(mem.recent(3)) == 3
    assert mem.recent(0) == []


def test_max_entries_evicts_oldest():
    mem = QueryMemory(max_entries=3)
    for i in range(5):
        mem.record({"S": f"q{i}", "R": "r"}, "O", state="known")
    assert mem.size() == 3
    # Oldest two were evicted.
    syms = [e.known["S"] for e in mem.all()]
    assert "q0" not in syms and "q1" not in syms
    assert "q4" in syms


def test_state_breakdown():
    mem = QueryMemory()
    mem.record({"S": "a", "R": "r"}, "O", state="known")
    mem.record({"S": "b", "R": "r"}, "O", state="known")
    mem.record({"S": "c", "R": "r"}, "O", state="idk")
    bd = mem.state_breakdown()
    assert bd["known"] == 2
    assert bd["idk"] == 1


def test_hot_signatures():
    mem = QueryMemory()
    for _ in range(3):
        mem.record({"S": "dog", "R": "isa"}, "O", state="known")
    mem.record({"S": "cat", "R": "isa"}, "O", state="known")
    hot = mem.hot_signatures(top_k=5)
    # The dog/isa signature should be the most frequent.
    assert hot
    top_sig, count = hot[0]
    assert count == 3


def test_drift_detected_when_state_changes():
    mem = QueryMemory()
    mem.record({"S": "dog", "R": "isa"}, "O",
                state="known", top_symbol="mammal")
    mem.record({"S": "dog", "R": "isa"}, "O",
                state="idk")
    assert mem.drift_detected({"S": "dog", "R": "isa"}, "O") is True


def test_no_drift_when_stable():
    mem = QueryMemory()
    for _ in range(3):
        mem.record({"S": "dog", "R": "isa"}, "O",
                   state="known", top_symbol="mammal", top_score=0.7)
    assert mem.drift_detected({"S": "dog", "R": "isa"}, "O") is False


def test_drift_false_for_single_episode():
    mem = QueryMemory()
    mem.record({"S": "dog", "R": "isa"}, "O", state="known")
    assert mem.drift_detected({"S": "dog", "R": "isa"}, "O") is False


def test_since_filters_by_timestamp():
    mem = QueryMemory()
    t0 = time.time()
    mem.record({"S": "a", "R": "r"}, "O", state="known")
    time.sleep(0.01)
    cutoff = time.time()
    mem.record({"S": "b", "R": "r"}, "O", state="known")
    after = mem.since(cutoff)
    syms = [e.known["S"] for e in after]
    assert "b" in syms
    assert "a" not in syms


def test_transitions_for_signature():
    mem = QueryMemory()
    mem.record({"S": "dog", "R": "isa"}, "O",
                state="known", top_symbol="mammal")
    mem.record({"S": "cat", "R": "isa"}, "O", state="known")
    mem.record({"S": "dog", "R": "isa"}, "O", state="idk")
    history = mem.transitions_for_signature({"S": "dog", "R": "isa"}, "O")
    assert len(history) == 2
    assert history[0][1] == "known"
    assert history[1][1] == "idk"


def test_save_and_load_roundtrip(tmp_path):
    """Save a log to JSONL, load it back, episodes are preserved."""
    mem = QueryMemory()
    mem.record({"S": "dog", "R": "isa"}, "O",
                state="known", top_symbol="mammal", top_score=0.7,
                notes="example")
    mem.record({"S": "qux", "R": "isa"}, "O", state="idk")
    f = tmp_path / "log.jsonl"
    n = mem.save(f)
    assert n == 2
    other = QueryMemory()
    loaded = other.load(f)
    assert loaded == 2
    syms = [e.known["S"] for e in other.all()]
    assert "dog" in syms and "qux" in syms
    # Notes survived.
    dog_ep = next(e for e in other.all() if e.known["S"] == "dog")
    assert dog_ep.notes == "example"


def test_load_with_replace_clears_existing(tmp_path):
    mem = QueryMemory()
    mem.record({"S": "a", "R": "r"}, "O", state="known")
    mem.record({"S": "b", "R": "r"}, "O", state="known")
    f = tmp_path / "log.jsonl"
    mem.save(f)
    # Add a third episode, then load with replace=True.
    mem.record({"S": "c", "R": "r"}, "O", state="known")
    assert mem.size() == 3
    mem.load(f, replace=True)
    assert mem.size() == 2
    syms = [e.known["S"] for e in mem.all()]
    assert "c" not in syms


def test_load_missing_file_returns_zero(tmp_path):
    mem = QueryMemory()
    n = mem.load(tmp_path / "does_not_exist.jsonl")
    assert n == 0


def test_conscious_agent_save_load_query_memory(tmp_path):
    """Agent helpers persist and restore query history across instances."""
    from rck.conscious_agent import ConsciousAgent
    agent_a = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent_a.tell("dog", "isa", "mammal")
    agent_a.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    agent_a.ask_with_idk({"S": "cat", "R": "isa"}, "O")
    f = tmp_path / "qm.jsonl"
    assert agent_a.save_query_memory(f) == 2

    agent_b = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    assert agent_b.query_memory.size() == 0
    assert agent_b.load_query_memory(f) == 2
    assert agent_b.query_memory.size() == 2


def test_drift_report_zero_when_no_drift():
    mem = QueryMemory()
    for _ in range(3):
        mem.record({"S": "dog", "R": "isa"}, "O",
                   state="known", top_symbol="mammal")
    r = mem.drift_report()
    assert r["total_drift_events"] == 0


def test_drift_report_counts_state_changes():
    mem = QueryMemory()
    mem.record({"S": "dog", "R": "isa"}, "O",
                state="known", top_symbol="mammal")
    mem.record({"S": "dog", "R": "isa"}, "O", state="idk")
    mem.record({"S": "dog", "R": "isa"}, "O",
                state="known", top_symbol="animal")
    r = mem.drift_report()
    assert r["total_drift_events"] == 2
    assert r["by_relation"]["isa"] == 2
    assert r["by_signature"]
    top = r["by_signature"][0]
    assert top["n_drift_events"] == 2


def test_drift_report_last_k_limit():
    mem = QueryMemory()
    for _ in range(100):
        mem.record({"S": "dog", "R": "isa"}, "O",
                   state="known", top_symbol="mammal")
    # Now create some drift in the last 5.
    mem.record({"S": "dog", "R": "isa"}, "O", state="idk")
    r = mem.drift_report(last_k=5)
    assert r["total_drift_events"] >= 1


def test_conscious_agent_drift_report():
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    agent.knowledge.forget({"S": "dog", "R": "isa", "O": "mammal"})
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    r = agent.drift_report()
    assert isinstance(r, dict)
    assert "total_drift_events" in r


def test_conscious_agent_query_memory_logs_ask_with_idk():
    """ConsciousAgent records every ask_with_idk into query_memory."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    pre = agent.query_memory.size()
    agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    agent.ask_with_idk({"S": "qqq_unknown", "R": "isa"}, "O")
    assert agent.query_memory.size() == pre + 2
    recent = agent.query_memory.recent(2)
    assert recent[0].state == "known"
    assert recent[1].state == "idk"
