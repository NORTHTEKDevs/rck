# Contributing to RCK

Thanks for your interest. RCK is a young project; contributions of any size
are welcome.

## Quick start

```bash
git clone https://github.com/<your-fork>/rck
cd rck
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest -q
```

The full test suite is ~30-50 seconds on a modern laptop.

## Where to start

Read these in order:

1. `README.md` — the elevator pitch and v15 status.
2. `docs/guide/01-quickstart.md` — your first agent in five minutes.
3. `docs/design/v14-narrative.md` — the architecture decisions that
   produced the current shape.

## Filing issues

* **Bug**: include the failing snippet, what you expected, what happened,
  and `python --version` + `pip show rck`. A failing pytest is ideal.
* **Feature**: open a discussion first; RCK has a small surface and we
  prefer fewer-and-better methods over many one-offs.
* **Question**: stack-overflow-style discussions are fine; the code base
  is small enough to grok in an afternoon.

## Pull requests

* Run `pytest -x -q` before pushing. The suite must stay green.
* Tests live in `tests/` and mirror the `rck/` module layout.
* Style: terse docstrings, no emojis, ASCII art is fine, no `print()`
  in library code.
* For new derivation logic, KEEP the existing filter stack
  (inverse-pair, non-transitive same-relation, lifting-relation,
  intermediate-cycle) intact. These are load-bearing for precision.
* New public methods on `ConsciousAgent` should also wire through to
  `agent.maintain()` if they belong in the nightly pass.

## Design principles

1. **Substrate over surface.** Knowledge lives in the HRR memory; the
   surface layer (LM polisher, NL parser) is replaceable.
2. **Provenance always.** Every stored fact carries a source +
   derivation trail. No silent state.
3. **Filters compose.** Every derivation pipeline (chain induction,
   rule instantiation, abstraction, propagation) applies the same
   filter stack. Don't bypass them.
4. **One vector, many heads.** The agent exposes many methods; they
   should all map to the same underlying retrieval / derivation core.
5. **No GPU, no excuses.** This is a CPU-first system. If your patch
   needs CUDA, it belongs in a different repo.

## Releases

Maintainers cut tagged releases on GitHub. There's no formal release
cadence yet; ping in the issue tracker if you need a release for a
specific change.
