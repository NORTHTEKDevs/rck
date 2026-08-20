"""Helper process for tests/test_crash_recovery.py -- NOT a pytest module
(leading underscore keeps it out of test collection).

Opens a WAL-enabled ConsciousAgent and tells synthetic facts in a loop,
printing the running count after each one. The parent hard-kills this
process at a random point and checks that every count it saw printed
really did make it into the WAL.

Protocol requirement: the count for fact i is printed ONLY after
`a.tell(...)` returns -- and `tell()` -> `knowledge.store()` -> WAL
`append()` already flushes + fsyncs before returning. So "printed N"
means "N is durably on disk" at the moment of printing, never before.
"""
from __future__ import annotations

import sys
from pathlib import Path

from rck.conscious_agent import ConsciousAgent


def main() -> None:
    wal_path = Path(sys.argv[1])
    n = int(sys.argv[2])
    # "testrel" is not in bulk_ingest.INVERSE_PAIRS, so tell() does not
    # auto-symmetrize -- exactly one store() / one WAL entry per tell().
    a = ConsciousAgent(dim=512, expected_facts=n, seed=0,
                        install_self=False, wal_path=wal_path)
    # Signal readiness BEFORE the loop starts, so the parent's randomized
    # kill delay measures time-in-loop, not variable interpreter-startup /
    # import time (importing rck.conscious_agent pulls in ~100 modules).
    print("READY", flush=True)
    for i in range(n):
        a.tell(f"entity{i}", "testrel", "thing")
        print(i + 1, flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
