"""Contract tests for rck.snapshot_hash.state_hash.

state_hash() is a SHA-256 over everything that can change an
ask_with_idk answer: hyper (dim, seed, n_shards), each knowledge
shard's `_memory` bytes in shard index order, then the canonical fact
list in stored order. See docs/plans/2026-08-19-replay.md.

[R2] belief_n_shards is deliberately excluded -- ask_with_idk never
reads agent.beliefs (idk_detection.ask_with_idk only calls
kb.query on agent.knowledge), so belief-KB churn must not move the
hash or replay would false-positive STATE_MISMATCH.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rck.conscious_agent import ConsciousAgent
from rck.session import load_session, save_session
from rck.snapshot_hash import state_hash

REPO_ROOT = Path(__file__).parent.parent


def _agent(seed=0, n_shards=8):
    return ConsciousAgent(dim=512, n_shards=n_shards, seed=seed,
                           install_self=False)


def test_same_agent_hashed_twice_is_identical():
    a = _agent()
    a.tell("dog", "isa", "mammal")
    assert state_hash(a) == state_hash(a)


def test_reload_same_process_is_identical(tmp_path):
    a = _agent()
    a.tell("dog", "isa", "mammal")
    a.tell("mammal", "isa", "animal")
    before = state_hash(a)

    save_session(a, tmp_path / "s")
    b = load_session(tmp_path / "s")

    assert state_hash(b) == before


def test_reload_fresh_subprocess_is_identical(tmp_path):
    a = _agent()
    a.tell("dog", "isa", "mammal")
    a.tell("mammal", "isa", "animal")
    before = state_hash(a)
    save_session(a, tmp_path / "s")

    script = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from rck.session import load_session\n"
        "from rck.snapshot_hash import state_hash\n"
        f"print(state_hash(load_session({str(tmp_path / 's')!r})))\n"
    )
    out = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, check=True)
    assert out.stdout.strip() == before


def test_adding_a_fact_changes_the_hash():
    a = _agent()
    a.tell("dog", "isa", "mammal")
    before = state_hash(a)
    a.tell("cat", "isa", "mammal")
    assert state_hash(a) != before


def test_deny_changes_the_hash():
    a = _agent()
    a.tell("dog", "isa", "mammal")
    before = state_hash(a)
    a.deny("dog", "isa", "fish")
    assert state_hash(a) != before


def test_insertion_order_changes_the_hash():
    """Trap 2: bundling is float addition, not associative."""
    a = _agent()
    a.tell("dog", "isa", "mammal")
    a.tell("cat", "isa", "mammal")
    h1 = state_hash(a)

    b = _agent()
    b.tell("cat", "isa", "mammal")
    b.tell("dog", "isa", "mammal")
    h2 = state_hash(b)

    assert h1 != h2


def test_belief_kb_activity_does_not_change_the_hash():
    """[R2] false-STATE_MISMATCH regression guard."""
    a = _agent()
    a.tell("dog", "isa", "mammal")
    before = state_hash(a)
    for i in range(300):
        a.tell_belief("bob", f"subj{i}", "isa", f"cat{i}")
    assert state_hash(a) == before


def test_reshard_does_change_the_hash():
    """reshard() re-bundles _memory in a new shard layout -- verify the
    real behaviour rather than assuming it (plan item 1)."""
    a = _agent(n_shards=8)
    a.tell("dog", "isa", "mammal")
    before = state_hash(a)
    a.knowledge.reshard(64)
    assert state_hash(a) != before
