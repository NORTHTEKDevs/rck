"""Append-only write-ahead log for crash-safe KB mutations.

One JSONL line per (op, fact) mutation. `append()` writes the line,
flushes, and `os.fsync()`s -- deliberately NOT routed through
`rck.atomic`: appending without rewriting the whole file is the entire
point (an atomic rewrite-and-replace on every append would make the WAL
as expensive as the state it protects).

Single-writer enforcement is a correctness requirement, not a nicety.
Measured on this machine: four independent handles appending 300
fsync'd lines each to one path produced 1055 of 1200 lines, with ZERO
unparseable lines -- 12% of committed writes silently vanished, because
Windows `"a"` mode does not give POSIX `O_APPEND`'s atomic seek-and-write
across independent handles. The torn-line check in `replay()` cannot
detect this class of loss. `WriteAheadLog` therefore takes an exclusive
lock on a sibling `.lock` file at open time and raises `WALLockedError`
if another writer already holds it, instead of silently losing writes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


class WALLockedError(RuntimeError):
    """Raised when another writer already holds the WAL's lock file."""


class WriteAheadLog:
    """Single-writer, append-only, crash-safe fact log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = Path(str(self.path) + ".lock")
        self._lock_file = open(self._lock_path, "a+b")
        self._locked = False
        try:
            self._acquire_lock()
        except BaseException:
            self._lock_file.close()
            raise
        self._fh = open(self.path, "a", encoding="utf-8")

    # ---- locking -----------------------------------------------------------

    def _acquire_lock(self) -> None:
        try:
            if sys.platform == "win32":
                import msvcrt
                self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise WALLockedError(
                f"WAL at {self.path} is locked by another writer",
            ) from exc
        self._locked = True

    def _release_lock(self) -> None:
        if not self._locked:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._locked = False

    # ---- core ops ------------------------------------------------------

    def append(self, op: str, fact: dict) -> None:
        """Append one (op, fact) entry, durably."""
        line = json.dumps({"op": op, "fact": fact})
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def replay(self):
        """Yield parsed {"op":..., "fact":...} entries in append order.

        Uses `readlines()` (full lookahead), not a streaming generator,
        so a malformed TRAILING line can be told apart from a malformed
        INTERIOR line: the former is a torn write (skip it), the latter
        is corruption (raise).
        """
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()
        n = len(lines)
        for i, raw in enumerate(lines):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                if i == n - 1:
                    # Torn trailing write -- process died mid-append.
                    return
                raise ValueError(
                    f"corrupt WAL line {i} in {self.path}: {line!r}",
                )
            yield entry

    def truncate(self) -> None:
        """Clear the log after its contents are durably captured elsewhere
        (e.g. a checkpoint snapshot)."""
        from rck.atomic import atomic_write_text
        # Close our own append handle first -- on Windows, os.replace()
        # (inside atomic_write_text) raises PermissionError if this same
        # process still holds an open handle on the destination.
        self._fh.close()
        atomic_write_text(self.path, "")
        self._fh = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
        self._release_lock()
        if not self._lock_file.closed:
            self._lock_file.close()

    def __enter__(self) -> "WriteAheadLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
