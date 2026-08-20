# Durability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
>
> **Revision 2, 2026-08-19.** Revision 1 was adversarially reviewed before
> implementation and had a critical defect: its `checkpoint()` would have
> silently destroyed the knowledge base. Six other findings are folded in.
> Review corrections are marked **[R2]** so they are not re-litigated.

**Goal:** Make every RCK write crash-safe, so a process killed mid-write never loses or corrupts committed facts.

**Architecture:** Three layers. (1) An atomic-write primitive - temp file in the destination directory, fsync, `os.replace` - applied to every persistence path. (2) An append-only write-ahead log hooked at `ShardedKnowledgeBase.store()`/`forget()`, so it captures *every* mutation path rather than only `tell()`. (3) A checkpoint that writes a real KB-inclusive snapshot before truncating the log.

**Tech Stack:** Python 3.11+ stdlib only (`os`, `json`, `tempfile`, `msvcrt`/`fcntl`). No new dependencies.

---

## Verified state of the codebase

Each item was confirmed by reading or executing, not assumed.

- **No atomic-write infrastructure exists.** A grep for `fsync|os.replace|atomic|NamedTemporary|mkstemp` across all 126 modules returns only `code_sandbox.py`'s unrelated sandbox temp file and word-matches like `atomic_number`. `persist.py` and `session.py` both write directly to the destination.
- **[R2] Persistence spans FOUR mechanisms, not three:**
  - `rck/persist.py` - `.npz` + `.json` sidecar for `RCKAgent`.
  - `rck/session.py` - `save_session` / `load_session`. **The only one that persists the HRR knowledge base.**
  - `rck/conscious_agent.py:1083` - `save_state` / `load_state`, three JSONL files. Its own docstring: *"Does NOT persist the HRR knowledge base or LM."*
  - **`rck/cli.py:22` - a `pickle.dumps` / `pickle.loads` path.** Undocumented, a different format from `persist.py`, and `pickle.loads` on a caller-supplied path is an arbitrary-code-execution sink. `CRYSTAL.md` claims pickle was replaced in v1.0; it was not, here. Convert to `persist.save`/`persist.load` and note the security fix in the commit message.
  - `rck/identity.py:97` - `UserProfile.save()` via plain `write_text`. Real user state; include it.
- **[R2] `ConsciousAgent.tell()` is NOT the only way facts enter the KB.** `rck/bulk_ingest.py` calls `kb.store(...)` directly at lines 86, 90, 112, 116, 131, 135, 165. `induce`, `merge_from`, `maintain` (cascade induction, rule instantiation, negation propagation, conflict resolution), `correct`, `prune_facts`, and `abstract_facts(commit=True)` all mutate the KB without going through `tell`. A WAL hooked at `tell()` would silently recover a *different, smaller* knowledge base than the one that crashed.
- Facts are bundled in insertion order and nothing sorts them (`relational.py` `store`/`forget`/`merge` never sort; `_memory += fact_hv.astype(np.float32)`). Float addition is not associative, so **replaying facts in a different order does not reproduce the same `_memory` array.** The WAL is a recovery mechanism, not a reproducibility mechanism; snapshots persist the arrays.
- `ShardedKnowledgeBase` auto-reshards during `store()`. `session.py` persists live per-KB shard counts - do not regress that.

## Traps

1. `os.replace` is atomic only within one filesystem. Always `mkstemp(dir=path.parent)`, never the system temp dir.
2. Directory fsync needs `os.open(dir, O_RDONLY)`, which fails on Windows. Guard with `except (OSError, AttributeError)` and skip. **[R2]** Verified: Windows raises `PermissionError`, which *is* an `OSError`, so the guard is correct.
3. **[R2] On Windows, `os.replace` raises `PermissionError [WinError 5]` if the destination has an open handle** - unlike POSIX, which renames happily over open files. Verified by execution. `session.load_session` calls `np.load(...)` and never closes the `NpzFile`, leaving exactly such a window. Close `NpzFile` handles explicitly, and wrap `os.replace` in a bounded retry (3 attempts, 50 ms backoff).
4. **[R2] Reshard must not be double-logged.** `reshard()` re-bundles via `RelationalMemory.store()`, not `ShardedKnowledgeBase.store()`. Hooking the WAL at the *sharded* level is therefore correct and reshard-safe. Do **not** hook `RelationalMemory.store` - every reshard would re-log the entire KB.

---

### Task 1: Atomic write primitive

**Files:** Create `rck/atomic.py`; test `tests/test_atomic.py`.

Implement `atomic_write_bytes` / `atomic_write_text` / `atomic_write_json`: `mkstemp` in the destination's directory, write, flush, `fsync`, `os.replace` (with the **[R2]** bounded retry for `PermissionError`), then best-effort directory fsync. On any failure, unlink the temp file and re-raise.

**[R2]** The module docstring must claim only what is tested: **process-crash safety**, not power-loss durability. The Windows path does not fsync the directory, so power-loss parity with POSIX is not established.

Tests: exact content; replaces existing; a broken `os.replace` leaves the original intact; no temp files leaked on failure; JSON round-trip.

Run `python -m pytest tests/test_atomic.py -v` → expect 5 passed. Commit.

---

### Task 2: Route every writer through the primitive

**Files:** `rck/persist.py`, `rck/session.py`, `rck/provenance.py`, `rck/skills.py`, `rck/query_memory.py`, **[R2]** `rck/cli.py`, `rck/identity.py`.

**Step 1.** Run the grep and triage **every** hit - do not stop at the file list above:

```
Select-String -Path (Get-ChildItem rck -Recurse -Filter *.py) -Pattern 'open\(.*[''"]w|\.write_text\(|\.write_bytes\(|np\.savez|pickle\.dump'
```

Expect ~7 in-scope hits plus a few legitimately out of scope (`conceptnet_loader.py`, `multi_task_corpus.py`, `polisher/*` - corpus/training export, not agent state). **Your report must reconcile the full hit list against what you converted**, with a one-line reason for each exclusion.

**Step 2 - [R2] the test must exercise each writer independently.**

Revision 1's test called `save_state()` once with `os.replace` broken and asserted no output. That test is unsound: `save_state`'s body is a dict literal, Python evaluates dict literals left-to-right, and the first raised exception aborts the rest - so `save_skills` raises and `save_provenance` / `save_query_memory` never run. Four of the five target files could go unconverted with the test still green. (Verified by execution.) It also never touches `persist.py` or `session.py`.

Parametrize over each writer, breaking `os.replace` for each separately:

```python
@pytest.mark.parametrize("writer", ["skills", "provenance", "query_memory",
                                    "persist", "session", "cli", "identity"])
def test_each_writer_is_atomic(writer, tmp_path, monkeypatch):
    setup, call, out = _writer_case(writer, tmp_path)   # one helper per case
    setup()
    monkeypatch.setattr(os, "replace",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        call()
    assert not out.exists(), f"{writer} bypassed rck.atomic and wrote {out}"
```

**Step 3.** Convert each writer: build the full payload in memory, then make exactly one `rck.atomic` call. For `np.savez*`, write into `io.BytesIO` and pass `.getvalue()` to `atomic_write_bytes`. For `cli.py`, replace the pickle path with `rck.persist`. Preserve every existing signature and return value so the 769 existing tests are unaffected.

Run `python -m pytest -q`. Commit.

---

### Task 3: Write-ahead log

**Files:** Create `rck/wal.py`; test `tests/test_wal.py`.

Append-only JSONL. `append(op, fact)` writes one line, `flush()`, `os.fsync()`. `replay()` yields parsed entries. `truncate()` uses `atomic_write_text(path, "")`. Do **not** route `append` through `rck.atomic` - appending without rewriting the file is the whole point.

**[R2] Torn-line rule, and how to implement it.** A malformed *trailing* line is a torn write and must be skipped; a malformed *interior* line is corruption and must raise. `replay()` therefore needs one-line lookahead or `readlines()` - a naive streaming generator cannot tell which line is last and will get this wrong.

**[R2] Single-writer enforcement - a correctness requirement, not a nicety.** Verified by execution on this machine: four handles appending 300 fsync'd lines each to one path produced **1055 of 1200 lines, with zero unparseable lines** - 12% of committed writes silently vanished. Python's `"a"` mode on Windows does not give POSIX `O_APPEND`'s atomic seek-and-write across independent handles. Undetectable by the torn-line logic, and exactly the class of silent corruption this plan exists to prevent.

Take an exclusive lock on open (`msvcrt.locking` on Windows, `fcntl.flock` elsewhere) on a sibling `.lock` file, and raise a clear `WALLockedError` if another writer holds it. Release on `close()`, and make `WriteAheadLog` a context manager.

Tests: append→replay order; missing log replays empty; torn final line keeps prior entries; **interior** malformed line raises; truncate clears; order preserved across reopen; **[R2]** a second concurrent writer raises `WALLockedError` rather than corrupting.

Run `python -m pytest tests/test_wal.py -v`. Commit.

---

### Task 4: Hook the WAL at the KB, not at `tell()`

**Files:** `rck/knowledge_base.py`, `rck/conscious_agent.py`; test `tests/test_wal.py`.

**[R2] Hook `ShardedKnowledgeBase.store()` and `.forget()`**, not `ConsciousAgent.tell()`/`deny()`. Every mutation path - `tell`, `deny`, `induce`, `merge_from`, `maintain`'s cascades, `correct`, `prune_facts`, and `bulk_ingest`'s direct `kb.store` calls - funnels through `store()`/`forget()`. Hooking at `tell()` would make `recover()` silently rebuild a different KB.

Per Trap 4, this level is also reshard-safe: `reshard()` re-bundles through `RelationalMemory.store`, one layer below the hook.

Add an optional `wal: WriteAheadLog | None = None` to `ShardedKnowledgeBase`; append after the in-memory write succeeds. Add `wal_path` to `ConsciousAgent`, wiring it to both `knowledge` and `beliefs` (separate logs - separate KBs). Add `recover()` to replay. Default `None`: no WAL, no new files, no behaviour change.

**[R2] `merge_from` may bypass `store()`** - it is a shard-level bundle sum (`RelationalMemory.merge`). Check this. If it bypasses, either route it through `store()` or log an explicit merge event; if neither is practical, document it as not crash-safe and add a test asserting the gap is *visible* rather than silent.

Tests: facts told with no snapshot survive via `recover()`; **facts added by `bulk_ingest` and by `induce` also survive** (the R2 regression cases); WAL is opt-in and absent by default.

Run `python -m pytest -q`. Commit.

---

### Task 5: Checkpoint - **[R2] the critical fix**

**Files:** `rck/conscious_agent.py`; test `tests/test_wal.py`.

Revision 1 specified `checkpoint(dir)` = `save_state(dir)` + `wal.truncate()`. **That would destroy the knowledge base.** `save_state` writes only `skills.jsonl` / `provenance.jsonl` / `query_memory.jsonl` and explicitly does not persist the HRR KB; truncating the WAL immediately afterwards erases the only other durable record. One `checkpoint()` call and every fact is gone, with a normal-looking return dict and no exception.

`checkpoint(dir)` must call **`rck.session.save_session(self, dir)`** - the only persister that writes the KB - and only truncate the WAL after that returns successfully.

**Required test** (the one whose absence hid the defect - Revision 1's tests deliberately avoided calling `checkpoint` at all):

```python
def test_checkpoint_then_restart_preserves_facts_with_an_empty_wal(tmp_path):
    a = ConsciousAgent(expected_facts=100, wal_path=tmp_path / "wal.jsonl")
    a.tell("dog", "isa", "mammal")
    a.checkpoint(tmp_path / "snap")
    assert list(WriteAheadLog(tmp_path / "wal.jsonl").replay()) == []   # truncated
    b = load_session(tmp_path / "snap")
    assert b.ask_with_idk({"S": "dog", "R": "isa"}, "O").top_symbol == "mammal"
```

Also test: facts told *after* a checkpoint are recovered from the WAL tail on top of the snapshot.

Run `python -m pytest -q`. Commit.

---

### Task 6: Real crash test

**Files:** `tests/test_crash_recovery.py`.

Spawn a subprocess that opens a WAL-enabled agent and writes facts in a loop. Hard-kill it (`proc.kill()`), then `recover()` from a fresh agent and assert every fact the child reported committed is present.

**[R2] Two protocol requirements, or the test is unsound:**
- The child must print its count **only after `wal.append()` returns**. Printing first counts facts that were never durable at kill time, producing false failures.
- The parent must drain the child's stdout on a background thread while waiting out the randomized kill delay. Otherwise the child blocks on a full pipe buffer and the kill timing is meaningless.

`proc.kill()` on Windows is `TerminateProcess` - no `finally`/`atexit` handlers run, which is what we want. Mark `@pytest.mark.slow`, keep under ~30 s.

If it fails, that is a real durability defect: fix `wal.py`, do not weaken the assertion.

---

## Definition of done

- [ ] `python -m pytest -q` green, zero failures
- [ ] Report reconciles the **full** Task-2 grep hit list against what was converted, with reasons for exclusions
- [ ] `cli.py` no longer uses `pickle` (security fix)
- [ ] Each writer independently proven atomic (parametrized, not one aggregate call)
- [ ] A second concurrent WAL writer raises rather than silently losing writes
- [ ] Facts added via `bulk_ingest` and `induce` survive a crash, not just `tell`
- [ ] `checkpoint()` + restart preserves facts with an emptied WAL
- [ ] A hard-killed process loses zero committed facts
- [ ] Default behaviour unchanged: no `wal_path` means no WAL and no new files

## Known gaps to state explicitly in the final report

- Whether `merge_from` is crash-safe (Task 4).
- fsync-per-append costs ~0.8-1.1 ms (measured). `ConsciousAgent.load_jsonl` loops `tell()`, so a WAL-enabled 10k-fact bulk load adds ~10 s. Acceptable, but say so.
- Power-loss (as opposed to process-crash) durability is not established on Windows.

## Not in this plan

The replay format (production-core item 4) builds on this WAL and gets its own plan.
