"""Tests for rck.atomic -- the atomic-write primitive."""
from __future__ import annotations

import json
import os

import pytest

from rck import atomic


def test_atomic_write_bytes_exact_content(tmp_path):
    p = tmp_path / "out.bin"
    atomic.atomic_write_bytes(p, b"hello bytes")
    assert p.read_bytes() == b"hello bytes"


def test_atomic_write_replaces_existing(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text("old")
    atomic.atomic_write_text(p, "new")
    assert p.read_text() == "new"


def test_atomic_write_broken_replace_leaves_original_intact(tmp_path, monkeypatch):
    p = tmp_path / "out.txt"
    p.write_text("original")

    def _boom(*a, **kw):
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        atomic.atomic_write_text(p, "new content")
    assert p.read_text() == "original"


def test_atomic_write_no_temp_files_leaked_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "out.txt"

    def _boom(*a, **kw):
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        atomic.atomic_write_text(p, "content")
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_json_roundtrip(tmp_path):
    p = tmp_path / "out.json"
    atomic.atomic_write_json(p, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2, 3]}
