# v14 episodic query memory

Date: 2026-05-21
Status: closed (capability added)

## What

`rck/query_memory.py` -- a bounded FIFO log of every retrieval the
agent performs via `ask_with_idk`. Each entry stores:

  * timestamp
  * the query (known dict + unknown_role)
  * epistemic state (`known` / `ambiguous` / `idk`)
  * top symbol + score
  * optional notes

## Why this matters

We had several pieces individually that needed a query trail:

  * **Calibration**: `metacog.CalibrationTally` tracks
    predicted-vs-actual but doesn't know which queries produced the
    predictions.
  * **Drift detection**: a (S, R) pair that was answerable yesterday
    but IDK today signals KB damage; we never logged enough to detect
    this.
  * **Hot-path profiling**: chain_cache pre-warms specific (start,
    target) pairs but didn't know which ones were actually hot.

QueryMemory closes all three by being the universal sink for
retrieval episodes.

## API

```python
agent.ask_with_idk({"S": "dog", "R": "isa"}, "O")  # auto-logged
agent.query_memory.recent(10)                        # last 10 episodes
agent.query_memory.state_breakdown()                 # {known: N, idk: M, ...}
agent.query_memory.hot_signatures(top_k=5)           # frequent queries
agent.query_memory.drift_detected({...}, "O")        # bool
agent.query_memory.transitions_for_signature({...}, "O")
```

## Open work

* Persistence: serialise the log to disk so a restart doesn't lose
  the audit trail. Out of scope here.
* Tie into calibration: when an answer turns out to be wrong (user
  corrects it), update the metacog tally with the original query's
  predicted state.
* Surface drift in real time: when a fresh episode shows drift
  against history, log a warning.
