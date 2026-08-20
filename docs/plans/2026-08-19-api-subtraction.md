# API Subtraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give RCK a small, frozen, obvious public API, so the first question an evaluator asks - "what do I call?" - has an answer.

**Architecture:** Declare a frozen surface. Demote everything else behind a `DeprecationWarning` at the top-level `rck.X` namespace, while leaving direct module imports (`from rck.analogy import ...`) untouched so internals stay reachable and the existing suite keeps passing. This task is almost entirely deletion and re-labelling; no behaviour changes.

**Tech Stack:** Python 3.11+ stdlib. Module-level `__getattr__` (PEP 562).

---

## Measured state

- `ConsciousAgent` exposes **70 public methods**.
- `rck/__init__.py` exports **92 names** but declares only **29** in `__all__` - **63 are accidentally public**.
- The declared `__all__` is almost entirely *substrate primitives* (`bind`, `bundle`, `permute`, `unbind`, `cosine`, `binarize`, `Codebook`, `PCNEncoder`, `LiquidStateMachine`, `TsetlinLayer`, `GlobalWorkspace`, `ActiveInference`, `ColumnEnsemble`, `BigramMemory`). **The declared public API is the research substrate, not the product.** That is the core problem: a new user reading `__all__` learns how the HRR machinery works and nothing about how to use the system.
- 820 tests currently pass. Many import internals by direct module path; that must keep working.

## The frozen surface

Fourteen agent methods, chosen to cover the full product story - teach, ask, justify, correct, reconcile, persist, operate - and nothing else:

| Call | Why it is in |
|---|---|
| `tell` | add a fact |
| `deny` | add a negative fact |
| `ask_with_idk` | the answer path, with calibrated IDK |
| `explain_why` | the derivation tree; the product's core claim |
| `discover` | multi-hop chain discovery |
| `induce` | promote a chain to a direct fact |
| `correct` | fix a fact, with belief revision |
| `detect_conflicts` | surface contradictions |
| `resolve_conflicts` | resolve them by source priority |
| `merge_from` | federated merge |
| `maintain` | the one-call nightly pass |
| `status_report` | operational state |
| `checkpoint` | durable snapshot + WAL truncate |
| `recover` | replay the WAL after a crash |

Plus module-level: `ConsciousAgent`, `ShardedKnowledgeBase`, `DecisionRecord`, `record_decision`, `replay`, `state_hash`, `save_session`, `load_session`, `bulk_load_jsonl`, `bulk_load_csv`, `bulk_load_triples`.

**Everything else stays importable by module path but leaves the top-level namespace.** Nothing is deleted and no method is removed from `ConsciousAgent` - this is about what the package *advertises*, not about removing capability.

---

### Task 1: Freeze the top-level namespace

**Files:** `rck/__init__.py`; test `tests/test_public_api.py` (create).

**Step 1: Write the failing test**

```python
import warnings
import pytest
import rck

FROZEN = {
    "ConsciousAgent", "ShardedKnowledgeBase",
    "DecisionRecord", "record_decision", "replay", "state_hash",
    "save_session", "load_session",
    "bulk_load_jsonl", "bulk_load_csv", "bulk_load_triples",
}


def test_all_is_exactly_the_frozen_surface():
    assert set(rck.__all__) == FROZEN


def test_every_name_in_all_actually_resolves():
    for name in rck.__all__:
        assert getattr(rck, name) is not None


def test_demoted_name_still_works_but_warns():
    with pytest.warns(DeprecationWarning, match="bind"):
        _ = rck.bind


def test_direct_module_import_is_unaffected_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # any warning fails the test
        from rck.vsa import bind            # noqa: F401
        from rck.analogy import solve_analogy  # noqa: F401


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = rck.definitely_not_a_real_name
```

**Step 2: Run it, confirm it fails** (`__all__` currently has 29 substrate names).

**Step 3: Implement**

Set `__all__` to exactly the frozen list. Keep the existing eager imports **only** for frozen names. Move the other 63 behind PEP 562:

```python
_DEMOTED = {
    "bind": "rck.vsa", "bundle": "rck.vsa", ...   # name -> module path
}

def __getattr__(name):
    if name in _DEMOTED:
        warnings.warn(
            f"rck.{name} is not part of the public API and will stop being "
            f"re-exported from the package root. Import it directly: "
            f"from {_DEMOTED[name]} import {name}",
            DeprecationWarning, stacklevel=2,
        )
        import importlib
        return getattr(importlib.import_module(_DEMOTED[name]), name)
    raise AttributeError(f"module 'rck' has no attribute {name!r}")
```

Build `_DEMOTED` from the *actual* current exports - enumerate them, do not hand-write from memory and do not guess module paths. Any name you cannot map to a module is a finding: report it.

**Step 4: Run the full suite.** Baseline **820 passed**. Existing tests that do `from rck import X` for a demoted `X` will now emit a `DeprecationWarning` - that is correct and they should still pass. If any test is configured to turn warnings into errors, update that test to import by module path rather than silencing the warning.

**Step 5: Commit.**

---

### Task 2: Mark the frozen surface on `ConsciousAgent`

**Files:** `rck/conscious_agent.py`; `tests/test_public_api.py`.

Do **not** rename or remove any method. Add a class-level declaration and a docstring section so the surface is discoverable:

```python
    #: The stable, supported API. Everything else on this class is
    #: internal: it may change without a deprecation cycle.
    PUBLIC_API = (
        "tell", "deny", "ask_with_idk", "explain_why", "discover",
        "induce", "correct", "detect_conflicts", "resolve_conflicts",
        "merge_from", "maintain", "status_report", "checkpoint", "recover",
    )
```

**Test:**

```python
def test_public_api_methods_all_exist_and_are_callable():
    for name in ConsciousAgent.PUBLIC_API:
        assert callable(getattr(ConsciousAgent, name)), name


def test_public_api_is_documented_in_the_class_docstring():
    doc = ConsciousAgent.__doc__ or ""
    for name in ConsciousAgent.PUBLIC_API:
        assert name in doc, f"{name} is in PUBLIC_API but undocumented"
```

The second test forces the docstring to stay in sync - a frozen API nobody documented is not frozen.

Commit.

---

### Task 3: Make the quickstart use only frozen calls

**Files:** `README.md`, `docs/guide/01-quickstart.md`; test `tests/test_public_api.py`.

The README's 60-second demo currently calls `ask_with_idk`, `discover`, `induce`, `explain_why` - all frozen, so it should already comply. **Verify rather than assume**, and check `docs/guide/01-quickstart.md` too.

**Test:** extract the python fenced blocks from the README, and assert every `agent.<name>(` call in them is in `ConsciousAgent.PUBLIC_API`. A regex over the fenced blocks is sufficient; keep it simple and skip cleanly if the README moves.

This is the test that keeps the documented surface and the frozen surface from drifting.

If a demo genuinely needs a non-frozen call, that is a signal the frozen list is wrong - report it rather than quietly widening the list.

Commit.

---

## Definition of done

- [ ] `python -m pytest -q` green (baseline **820 passed**)
- [ ] `rck.__all__` is exactly the frozen surface
- [ ] All 63 demoted names still resolve, each with a `DeprecationWarning` naming its real module
- [ ] Direct module imports emit **no** warning
- [ ] `rck.nonexistent` still raises `AttributeError`
- [ ] `ConsciousAgent.PUBLIC_API` exists, all 14 callable, all named in the class docstring
- [ ] Every `agent.*` call in the README quickstart is in `PUBLIC_API`
- [ ] **No method deleted, no behaviour changed** - this task only re-labels

## Report explicitly

- Any exported name that could not be mapped to a source module.
- Any existing test that had to change, and why.
- Any README/guide call that is not in the frozen list (a signal the list is wrong).

## Not in this plan

Deleting or merging methods on `ConsciousAgent`. That is a real API redesign needing its own deprecation cycle; this task establishes the frozen surface first.
