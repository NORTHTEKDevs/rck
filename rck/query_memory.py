"""Episodic memory of queries.

Every retrieval the agent performs is logged here: the input
(known + unknown_role), the epistemic state classification, the
chosen top-1 answer, the score, and a timestamp. This is the
audit trail callers need to:

  * Calibrate: did predicted-KNOWN answers actually pan out?
  * Profile: which queries are hot? Worth pre-caching?
  * Detect anomalies: a KNOWN -> IDK transition for the same
    query signals KB drift (something was deleted / replaced).

The store is bounded by `max_entries` (default 4096) with FIFO
eviction. Persistence is out of scope -- this is in-memory only;
callers who need durability serialise the log themselves.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Optional


@dataclass
class QueryEpisode:
    """One row in the episodic query log."""
    timestamp: float
    known: dict[str, Hashable]
    unknown_role: str
    state: str                    # "known" / "ambiguous" / "idk"
    top_symbol: Optional[Hashable]
    top_score: float
    notes: str = ""


@dataclass
class QueryMemory:
    """Bounded FIFO log of retrieval episodes."""
    max_entries: int = 4096
    _entries: deque = field(default_factory=lambda: deque(maxlen=4096))

    def __post_init__(self) -> None:
        # Re-bind the deque to the configured max_entries; default_factory
        # uses the class-level default (4096) which is fine for normal use.
        if self.max_entries != self._entries.maxlen:
            self._entries = deque(self._entries, maxlen=self.max_entries)

    def record(self, known: dict[str, Hashable], unknown_role: str,
               *, state: str, top_symbol: Optional[Hashable] = None,
               top_score: float = 0.0, notes: str = "") -> QueryEpisode:
        ep = QueryEpisode(
            timestamp=time.time(),
            known=dict(known), unknown_role=unknown_role,
            state=state, top_symbol=top_symbol,
            top_score=float(top_score), notes=notes,
        )
        self._entries.append(ep)
        return ep

    def all(self) -> list[QueryEpisode]:
        return list(self._entries)

    def recent(self, n: int = 20) -> list[QueryEpisode]:
        if n <= 0:
            return []
        return list(self._entries)[-n:]

    def since(self, ts: float) -> list[QueryEpisode]:
        return [e for e in self._entries if e.timestamp >= ts]

    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    # ---- analytics -------------------------------------------------------

    def state_breakdown(self) -> dict[str, int]:
        """Counts of episodes by epistemic state."""
        out: dict[str, int] = {}
        for e in self._entries:
            out[e.state] = out.get(e.state, 0) + 1
        return out

    def hot_signatures(self, top_k: int = 10) -> list[tuple[tuple, int]]:
        """Most frequent (known-frozenset, unknown_role) signatures.

        Returns up to `top_k` rows sorted by count descending. Useful
        for picking what to pre-cache.
        """
        from collections import Counter
        counts: Counter = Counter()
        for e in self._entries:
            sig = (tuple(sorted(e.known.items())), e.unknown_role)
            counts[sig] += 1
        return counts.most_common(top_k)

    def transitions_for_signature(self, known: dict[str, Hashable],
                                   unknown_role: str
                                   ) -> list[tuple[float, str, Optional[Hashable]]]:
        """All historical (timestamp, state, top_symbol) for the same query
        signature. Useful for spotting drift (e.g. KNOWN -> IDK)."""
        target_sig = (tuple(sorted(known.items())), unknown_role)
        out = []
        for e in self._entries:
            sig = (tuple(sorted(e.known.items())), e.unknown_role)
            if sig == target_sig:
                out.append((e.timestamp, e.state, e.top_symbol))
        return out

    def drift_report(self, *, last_k: int = 200) -> dict:
        """Aggregate drift across the most recent `last_k` episodes.

        Returns a dict:
          * `by_signature`: list of {signature, transitions, last_state}
            for signatures that drifted in the window.
          * `by_relation`: counts of drift events per relation.
          * `total_drift_events`: integer.

        A "drift event" is an episode whose state/sym differs from the
        most recent prior episode of the SAME signature within the
        window.
        """
        entries = list(self._entries)[-last_k:]
        # Walk in order; keep a running snapshot per signature.
        last_seen: dict = {}
        drift_by_sig: dict = {}
        drift_events_per_relation: dict[str, int] = {}
        total_events = 0
        for e in entries:
            sig = (tuple(sorted(e.known.items())), e.unknown_role)
            prev = last_seen.get(sig)
            curr = (e.state, e.top_symbol)
            if prev is not None and prev != curr:
                drift_by_sig.setdefault(sig, []).append({
                    "prev_state": prev[0],
                    "prev_symbol": prev[1],
                    "now_state": curr[0],
                    "now_symbol": curr[1],
                    "timestamp": e.timestamp,
                })
                rel = str(e.known.get("R", ""))
                drift_events_per_relation[rel] = (
                    drift_events_per_relation.get(rel, 0) + 1
                )
                total_events += 1
            last_seen[sig] = curr
        by_signature_summary = []
        for sig, events in drift_by_sig.items():
            known_pairs, unk = sig
            by_signature_summary.append({
                "known": dict(known_pairs),
                "unknown_role": unk,
                "n_drift_events": len(events),
                "last_event": events[-1],
            })
        by_signature_summary.sort(
            key=lambda r: -r["n_drift_events"],
        )
        return {
            "total_drift_events": total_events,
            "by_signature": by_signature_summary,
            "by_relation": drift_events_per_relation,
        }

    def drift_detected(self, known: dict[str, Hashable],
                       unknown_role: str) -> bool:
        """True iff the query signature has changed epistemic state at
        some point in the log (e.g. known -> idk, or known answer
        changed). Returns False if there's <2 episodes for the
        signature."""
        history = self.transitions_for_signature(known, unknown_role)
        if len(history) < 2:
            return False
        first = (history[0][1], history[0][2])
        for ts, state, sym in history[1:]:
            if (state, sym) != first:
                return True
        return False

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> int:
        """Serialise the log to a JSONL file. One episode per line.
        Returns the number of episodes written."""
        from rck.atomic import atomic_write_text
        path = Path(path)
        lines = []
        for e in self._entries:
            rec = {
                "timestamp": e.timestamp,
                "known": {str(k): str(v) for k, v in e.known.items()},
                "unknown_role": e.unknown_role,
                "state": e.state,
                "top_symbol": (None if e.top_symbol is None
                               else str(e.top_symbol)),
                "top_score": e.top_score,
                "notes": e.notes,
            }
            lines.append(json.dumps(rec))
        atomic_write_text(path, "".join(line + "\n" for line in lines))
        return len(self._entries)

    def load(self, path: str | Path, *, replace: bool = False) -> int:
        """Read episodes from a JSONL file produced by `save`.

        If `replace=True` the current log is cleared first; otherwise
        the loaded episodes are appended to the existing log
        (subject to the FIFO bound).
        Returns the number of episodes loaded.
        """
        path = Path(path)
        if not path.exists():
            return 0
        if replace:
            self._entries.clear()
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ep = QueryEpisode(
                    timestamp=float(rec.get("timestamp", time.time())),
                    known=dict(rec.get("known", {})),
                    unknown_role=str(rec.get("unknown_role", "")),
                    state=str(rec.get("state", "unknown")),
                    top_symbol=rec.get("top_symbol"),
                    top_score=float(rec.get("top_score", 0.0)),
                    notes=str(rec.get("notes", "")),
                )
                self._entries.append(ep)
                n += 1
        return n
