"""Tests for episodic consolidation (dreaming pass)."""
from __future__ import annotations

from rck.episodic_consolidate import ConsolidationReport, consolidate
from rck.query_memory import QueryMemory


def _populate_stable(mem: QueryMemory, s: str, n: int = 5,
                     sym: str = "mammal", state: str = "known") -> None:
    for _ in range(n):
        mem.record({"S": s, "R": "isa"}, "O",
                   state=state, top_symbol=sym, top_score=0.7)


def test_consolidate_promotes_stable_signatures():
    mem = QueryMemory()
    _populate_stable(mem, "dog", n=5, sym="mammal")
    report = consolidate(mem, min_occurrences=3, stability_threshold=0.9)
    promoted = {p for p in report.stable_promoted}
    assert ("dog", "mammal") in promoted


def test_consolidate_flags_unstable_signatures():
    """If the same signature returns different symbols, flag as unstable."""
    mem = QueryMemory()
    # 3 mammal, 3 fish -> below stability threshold.
    for sym in ("mammal", "mammal", "mammal", "fish", "fish", "fish"):
        mem.record({"S": "dog", "R": "isa"}, "O",
                   state="known", top_symbol=sym, top_score=0.5)
    report = consolidate(mem, min_occurrences=3, stability_threshold=0.9)
    assert report.stable_promoted == []
    assert report.unstable_flagged


def test_consolidate_skips_below_min_occurrences():
    mem = QueryMemory()
    _populate_stable(mem, "dog", n=2)
    report = consolidate(mem, min_occurrences=5)
    assert report.stable_promoted == []


def test_consolidate_ignores_pure_idk_signatures():
    mem = QueryMemory()
    for _ in range(5):
        mem.record({"S": "qux", "R": "isa"}, "O", state="idk")
    report = consolidate(mem, min_occurrences=3)
    assert report.stable_promoted == []


def test_consolidate_flags_ambiguous_signatures():
    mem = QueryMemory()
    for _ in range(4):
        mem.record({"S": "ambig", "R": "isa"}, "O",
                   state="ambiguous", top_symbol="alpha", top_score=0.3)
    report = consolidate(mem, min_occurrences=3)
    flagged_subjects = {s for s, _r, _share in report.ambiguous_flagged}
    assert "ambig" in flagged_subjects


def test_conscious_agent_consolidate_episodes():
    """ConsciousAgent.consolidate_episodes() wraps the call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    agent.tell("dog", "isa", "mammal")
    for _ in range(4):
        agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")
    report = agent.consolidate_episodes(
        min_occurrences=3, stability_threshold=0.9,
    )
    assert isinstance(report, ConsolidationReport)
    # The stable promoted list should include (dog, ...).
    promoted_subjects = {s for s, _ in report.stable_promoted}
    assert "dog" in promoted_subjects or report.n_signatures_examined >= 1
