"""Task 6: a real, hard-killed process loses zero committed facts.

Spawns tests/_wal_crash_child.py as a subprocess that opens a
WAL-enabled agent and tells facts in a loop, then hard-kills it
(TerminateProcess on Windows -- no atexit/finally handlers run) at a
randomized point. Recovers a fresh agent from the WAL alone and checks
every fact the child reported committed is actually present.

Two protocol requirements the child script and this test both honor
(see the durability plan, Task 6):
  - The child prints its count ONLY after wal.append() has returned
    (transitively, after tell() returns) -- so "printed N" means
    "durably on disk" at print time, never before.
  - The parent drains the child's stdout on a background thread while
    waiting out the kill delay -- otherwise the child blocks on a full
    pipe buffer once its stdout buffer fills, and the kill timing
    becomes meaningless (it would just be killing a process stuck on
    a write(), not one actively appending to the WAL).

A third requirement this test adds on top of the plan's two: the kill
delay is measured from a "READY" sentinel the child prints right
before its loop starts, not from process-spawn time. Importing
rck.conscious_agent pulls in ~100 modules, and that import time is
variable enough (observed empirically: some runs killed the child
before it wrote a single fact) to make a spawn-relative delay
meaningless -- the same failure mode the plan's stdout-draining
requirement exists to avoid, just at the import stage instead of the
pipe-buffer stage.
"""
from __future__ import annotations

import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from rck.conscious_agent import ConsciousAgent

CHILD_SCRIPT = Path(__file__).parent / "_wal_crash_child.py"
N_FACTS = 4000


@pytest.mark.slow
def test_hard_killed_process_loses_zero_committed_facts(tmp_path):
    wal_path = tmp_path / "crash.wal.jsonl"

    proc = subprocess.Popen(
        [sys.executable, str(CHILD_SCRIPT), str(wal_path), str(N_FACTS)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    last_reported = [0]
    ready = threading.Event()
    stop_draining = threading.Event()

    def drain() -> None:
        # MUST keep reading -- an undrained pipe fills and blocks the
        # child's print(), which would make the kill delay below land
        # on a stalled child instead of an actively-writing one.
        try:
            for line in proc.stdout:
                line = line.strip()
                if line == "READY":
                    ready.set()
                    continue
                if line.isdigit():
                    last_reported[0] = int(line)
                if stop_draining.is_set():
                    break
        except ValueError:
            pass  # pipe closed under us after kill -- fine

    drain_thread = threading.Thread(target=drain, daemon=True)
    drain_thread.start()

    # Wait for the child to finish importing and reach its loop, THEN
    # start the randomized kill delay -- otherwise variable interpreter
    # startup / import time (rck.conscious_agent pulls in ~100 modules)
    # can eat the whole delay and kill the child before it does any work.
    if not ready.wait(timeout=15):
        proc.kill()
        pytest.fail("child never printed READY within 15s")

    # Randomized kill point, well inside the child's expected runtime
    # (thousands of fsync'd appends at ~1ms each) so it reliably lands
    # mid-loop rather than after the child has already exited.
    time.sleep(random.uniform(0.3, 1.2))
    if proc.poll() is None:
        proc.kill()  # TerminateProcess on Windows
    proc.wait(timeout=10)
    stop_draining.set()
    drain_thread.join(timeout=5)

    reported = last_reported[0]
    assert reported > 0, (
        "child was killed before committing anything -- flaky timing, "
        "widen the kill-delay window or lower N_FACTS"
    )
    assert reported < N_FACTS, (
        "child finished before being killed -- this run didn't actually "
        "exercise a mid-write kill; widen the kill-delay window or "
        "raise N_FACTS"
    )

    recovered = ConsciousAgent(dim=512, expected_facts=N_FACTS, seed=0,
                                install_self=False, wal_path=wal_path)
    report = recovered.recover()
    assert report["knowledge"] >= reported, (
        f"child reported {reported} committed facts but only "
        f"{report['knowledge']} replayed from the WAL -- durability defect"
    )
    for i in range(reported):
        ans = recovered.knowledge.answer(
            {"S": f"entity{i}", "R": "testrel"}, "O")
        assert ans[0] == "thing", (
            f"entity{i} missing after recovery "
            f"(child reported {reported} committed)"
        )
    recovered.knowledge.wal.close()
    recovered.beliefs.wal.close()
