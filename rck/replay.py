"""Decision replay: pin an answer to the exact substrate state that
produced it, then re-execute later and compare exactly.

[R2] Design note -- a record without its snapshot is not replayable.
A `DecisionRecord` is only meaningful paired with the snapshot it was
recorded against (a `checkpoint()`/`save_session` directory). Replay
targets that stored snapshot, NEVER the live agent -- bundling is
float addition and not associative (Trap 2 in the plan), so replay
must load a stored snapshot and must never offer replay-by-re-ingest;
any such API would silently return different results.

A long-lived agent's `state_hash` moves as it grows (auto-reshard,
new facts) -- that does not invalidate old records, because they are
never replayed against the live agent, only against the checkpoint
frozen at record time. Replaying an old record against a *newer*
snapshot correctly returns STATE_MISMATCH: the substrate genuinely
moved, and saying so is the honest answer, not a bug.
"""
from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from rck.atomic import atomic_write_json
from rck.explain_why import ExplanationNode
from rck.snapshot_hash import state_hash


@dataclass
class DecisionRecord:
    """A pinned, replayable answer.

    `created_at` is audit-only metadata -- it is NOT part of
    `state_hash` and never affects replay comparison.
    """

    rck_version: str
    state_hash: str
    query: dict
    unknown_role: str
    answer: dict          # top_symbol, top_score, state, alternatives
    derivation: dict | None   # serialized ExplanationNode tree, or None
    seed: int
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return {
            "rck_version": self.rck_version,
            "state_hash": self.state_hash,
            "query": dict(self.query),
            "unknown_role": self.unknown_role,
            "answer": dict(self.answer),
            "derivation": self.derivation,
            "seed": self.seed,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, data: dict) -> "DecisionRecord":
        return cls(
            rck_version=str(data["rck_version"]),
            state_hash=str(data["state_hash"]),
            query=dict(data["query"]),
            unknown_role=str(data["unknown_role"]),
            answer=dict(data["answer"]),
            derivation=data.get("derivation"),
            seed=int(data["seed"]),
            created_at=float(data["created_at"]),
        )

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_json(), indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "DecisionRecord":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json(data)


def _node_to_dict(node: ExplanationNode) -> dict:
    return {
        "subject": node.subject,
        "relation": node.relation,
        "obj": node.obj,
        "source": node.source,
        "confidence": node.confidence,
        "is_leaf": node.is_leaf,
        "note": node.note,
        "children": [_node_to_dict(c) for c in node.children],
    }


def _node_from_dict(d: dict) -> ExplanationNode:
    return ExplanationNode(
        subject=d["subject"], relation=d["relation"], obj=d["obj"],
        source=d.get("source", "unknown"),
        confidence=float(d.get("confidence", 1.0)),
        is_leaf=bool(d.get("is_leaf", False)),
        note=d.get("note", ""),
        children=[_node_from_dict(c) for c in d.get("children", [])],
    )


def _triple_from(query: dict, unknown_role: str, filled) -> tuple[str, str, str]:
    d = dict(query)
    d[unknown_role] = filled
    return str(d.get("S", "")), str(d.get("R", "")), str(d.get("O", ""))


def record_decision(agent, query: dict, unknown_role: str) -> DecisionRecord:
    """Run `query` against `agent`, capture the answer and (when the
    fact is derivable, i.e. an answer symbol exists) its derivation
    tree, and stamp the current `state_hash`."""
    from rck import __version__ as _rck_version

    ans = agent.ask_with_idk(dict(query), unknown_role)

    derivation = None
    if ans.top_symbol is not None:
        s, r, o = _triple_from(query, unknown_role, ans.top_symbol)
        derivation = _node_to_dict(agent.explain_why(s, r, o))

    return DecisionRecord(
        rck_version=_rck_version,
        state_hash=state_hash(agent),
        query=dict(query),
        unknown_role=unknown_role,
        answer={
            "top_symbol": ans.top_symbol,
            "top_score": ans.top_score,
            "state": ans.state.value,
            "alternatives": [[sym, score] for sym, score in ans.alternatives],
        },
        derivation=derivation,
        seed=agent.seed,
        created_at=time.time(),
    )


class ReplayStatus(enum.Enum):
    VERIFIED = "verified"
    DIVERGED = "diverged"
    STATE_MISMATCH = "state_mismatch"


@dataclass
class ReplayResult:
    """Outcome of replaying a DecisionRecord against a snapshot.

    Never raised -- a replay tool that throws on mismatch is unusable
    for auditing a corpus of records. `details` carries whatever is
    relevant to `status` (hashes on STATE_MISMATCH, both answers/
    derivations on DIVERGED, the reproduced answer on VERIFIED)."""

    status: ReplayStatus
    details: dict = field(default_factory=dict)


def replay(record: DecisionRecord, snapshot_dir: str | Path) -> ReplayResult:
    """Load the snapshot at `snapshot_dir`, and either report that the
    substrate has moved (STATE_MISMATCH, query never run) or re-run
    the query and compare the answer + derivation exactly (VERIFIED /
    DIVERGED).

    Never re-ingests facts -- see the module docstring: a record
    without its snapshot is not replayable, and replay always targets
    a stored snapshot, never a live agent."""
    from rck.session import load_session

    agent = load_session(snapshot_dir)
    current_hash = state_hash(agent)
    if current_hash != record.state_hash:
        return ReplayResult(
            status=ReplayStatus.STATE_MISMATCH,
            details={
                "recorded_state_hash": record.state_hash,
                "snapshot_state_hash": current_hash,
            },
        )

    ans = agent.ask_with_idk(dict(record.query), record.unknown_role)
    derivation = None
    if ans.top_symbol is not None:
        s, r, o = _triple_from(record.query, record.unknown_role, ans.top_symbol)
        derivation = _node_to_dict(agent.explain_why(s, r, o))
    replayed_answer = {
        "top_symbol": ans.top_symbol,
        "top_score": ans.top_score,
        "state": ans.state.value,
        "alternatives": [[sym, score] for sym, score in ans.alternatives],
    }

    if replayed_answer == record.answer and derivation == record.derivation:
        return ReplayResult(status=ReplayStatus.VERIFIED,
                             details={"answer": replayed_answer,
                                      "derivation": derivation})
    return ReplayResult(
        status=ReplayStatus.DIVERGED,
        details={
            "recorded_answer": record.answer,
            "replayed_answer": replayed_answer,
            "recorded_derivation": record.derivation,
            "replayed_derivation": derivation,
        },
    )
