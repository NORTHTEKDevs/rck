"""Atomic write primitives.

Guarantees process-crash safety: a process killed mid-write never leaves a
torn or partially-written file at the destination path -- a reader always
sees either the complete old content or the complete new content.

This does NOT establish power-loss durability. The Windows path cannot
fsync a directory handle (`os.open(dir, O_RDONLY)` raises `PermissionError`
there), so the directory-entry fsync is best-effort only; on power loss the
rename itself may not be durable on Windows, only safe against a process
crash. POSIX callers get the directory fsync and therefore stronger
guarantees.

Pattern: write to a temp file in the destination's own directory (so
`os.replace` is same-filesystem and atomic), fsync the temp file, then
`os.replace` over the destination -- with a bounded retry, because on
Windows `os.replace` raises `PermissionError` [WinError 5] if the
destination has an open handle, unlike POSIX which renames over open files
without complaint.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

_REPLACE_ATTEMPTS = 3
_REPLACE_BACKOFF_S = 0.05


def _replace_with_retry(src: str, dst: str) -> None:
    last_exc: OSError | None = None
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt < _REPLACE_ATTEMPTS - 1:
                time.sleep(_REPLACE_BACKOFF_S)
    assert last_exc is not None
    raise last_exc


def _fsync_dir(dir_path: Path) -> None:
    """Best-effort directory-entry fsync. No-op (silently) on Windows."""
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(fd)
    except (OSError, AttributeError):
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write `data` to `path` atomically.

    On any failure the temp file is unlinked and the exception re-raised;
    the destination is left untouched (old content, or absent if it never
    existed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, obj, **kwargs) -> None:
    atomic_write_text(path, json.dumps(obj, **kwargs))
