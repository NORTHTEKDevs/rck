"""Conversation persistence -- save and restore a session.

Stores:
  - The ConsciousAgent's sharded KB (via the existing rck.persist
    interface on each shard).
  - The DialogueContext (last entity, last relation, history).
  - The CalibrationTally.

Format: a directory containing one .npz per shard + a meta JSON.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np

from rck.atomic import atomic_write_bytes, atomic_write_text
from rck.conscious_agent import ConsciousAgent


SCHEMA_VERSION = 2


def save_session(agent: ConsciousAgent, path: str | Path) -> dict:
    """Persist the entire ConsciousAgent state to a directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Knowledge shards -- pack codebook + each shard's memory.
    cb = agent.knowledge.codebook
    sym_list = list(cb._atoms.keys())
    cb_matrix = (np.stack([cb._atoms[s] for s in sym_list], axis=0)
                 if sym_list else np.empty((0, cb.dim), dtype=np.int8))
    shard_mems = np.stack([s._memory for s in agent.knowledge._shards], axis=0)
    shard_facts = [s._facts for s in agent.knowledge._shards]

    knowledge_buf = io.BytesIO()
    np.savez_compressed(
        knowledge_buf,
        codebook_matrix=cb_matrix,
        shard_mems=shard_mems,
    )
    atomic_write_bytes(path / "knowledge.npz", knowledge_buf.getvalue())

    # Belief KB (smaller -- same structure).
    bb_syms = list(agent.beliefs.codebook._atoms.keys())
    bb_matrix = (np.stack([agent.beliefs.codebook._atoms[s] for s in bb_syms], axis=0)
                 if bb_syms else np.empty((0, agent.beliefs.dim), dtype=np.int8))
    bb_mems = np.stack([s._memory for s in agent.beliefs._shards], axis=0)
    beliefs_buf = io.BytesIO()
    np.savez_compressed(
        beliefs_buf,
        codebook_matrix=bb_matrix,
        shard_mems=bb_mems,
    )
    atomic_write_bytes(path / "beliefs.npz", beliefs_buf.getvalue())

    meta = {
        "schema": SCHEMA_VERSION,
        "rck_version": "1.5.0",
        "saved_at": time.time(),
        "hyper": {
            "dim": agent.dim,
            # Read the LIVE shard counts, not agent.n_shards / a derived
            # belief count -- ConsciousAgent.n_shards is set once in
            # __post_init__ and never updated when reshard() mutates
            # agent.knowledge.n_shards (or agent.beliefs.n_shards) in
            # place, so it goes stale the moment either KB auto-reshards.
            "n_shards": agent.knowledge.n_shards,
            "belief_n_shards": agent.beliefs.n_shards,
            "seed": agent.seed,
        },
        "knowledge_symbols": [_sym_to_json(s) for s in sym_list],
        "knowledge_facts": shard_facts,
        "belief_symbols": [_sym_to_json(s) for s in bb_syms],
        "belief_facts": [s._facts for s in agent.beliefs._shards],
        "dialogue": {
            "last_entity": agent.dialogue.last_entity,
            "last_relation": agent.dialogue.last_relation,
            "last_answer": agent.dialogue.last_answer,
            "history": list(agent.dialogue.history),
        },
        "calibration": {
            rel: dict(b) for rel, b in agent.calibration.by_relation.items()
        },
    }
    atomic_write_text(path / "meta.json", json.dumps(meta, default=str, indent=2))
    return {"path": str(path), "knowledge_facts": agent.knowledge.size(),
            "belief_facts": agent.beliefs.size(),
            "dialogue_turns": len(agent.dialogue.history)}


def load_session(path: str | Path) -> ConsciousAgent:
    """Reconstruct a ConsciousAgent from a saved session directory."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"unsupported session schema {meta.get('schema')}")
    hyper = meta["hyper"]
    agent = ConsciousAgent(
        dim=int(hyper["dim"]),
        n_shards=int(hyper["n_shards"]),
        seed=int(hyper["seed"]),
        install_self=False,
    )

    # The belief KB can reshard independently of the knowledge KB, so its
    # shard count is NOT reliably `hyper["n_shards"] // 2` (the default
    # ConsciousAgent.__post_init__ derivation) -- it may have grown past
    # that on its own. Resize to the persisted count before loading data.
    # `reshard()` on a still-empty KB just rebuilds the shard array (no
    # facts to lose) and is a no-op when the count already matches.
    belief_n_shards = int(hyper.get("belief_n_shards", agent.beliefs.n_shards))
    if belief_n_shards != agent.beliefs.n_shards:
        agent.beliefs.reshard(belief_n_shards)

    # Knowledge. Close the NpzFile explicitly (not just let it be GC'd) --
    # on Windows an open zip-file handle makes a later os.replace of this
    # same path raise PermissionError [WinError 5].
    with np.load(path / "knowledge.npz") as arr:
        syms = [_sym_from_json(s) for s in meta["knowledge_symbols"]]
        cb_matrix = arr["codebook_matrix"]
        for i, sym in enumerate(syms):
            agent.knowledge.codebook._atoms[sym] = cb_matrix[i].astype(np.int8)
        agent.knowledge.codebook._cache_dirty = True
        shard_mems = arr["shard_mems"]
        for i, shard in enumerate(agent.knowledge._shards):
            shard._memory = shard_mems[i].astype(np.float32)
    for i, facts in enumerate(meta["knowledge_facts"]):
        agent.knowledge._shards[i]._facts = list(facts)
    agent.knowledge._fact_count = sum(len(f) for f in meta["knowledge_facts"])

    # Beliefs.
    with np.load(path / "beliefs.npz") as arr_b:
        bb_syms = [_sym_from_json(s) for s in meta["belief_symbols"]]
        bb_matrix = arr_b["codebook_matrix"]
        for i, sym in enumerate(bb_syms):
            agent.beliefs.codebook._atoms[sym] = bb_matrix[i].astype(np.int8)
        agent.beliefs.codebook._cache_dirty = True
        bb_mems = arr_b["shard_mems"]
        for i, shard in enumerate(agent.beliefs._shards):
            shard._memory = bb_mems[i].astype(np.float32)
    for i, facts in enumerate(meta["belief_facts"]):
        agent.beliefs._shards[i]._facts = list(facts)
    agent.beliefs._fact_count = sum(len(f) for f in meta["belief_facts"])

    # Dialogue state.
    dlg = meta.get("dialogue", {})
    agent.dialogue.last_entity = dlg.get("last_entity")
    agent.dialogue.last_relation = dlg.get("last_relation")
    agent.dialogue.last_answer = dlg.get("last_answer")
    for turn in dlg.get("history", []):
        agent.dialogue.history.append(turn)

    # Calibration.
    for rel, bucket in meta.get("calibration", {}).items():
        agent.calibration.by_relation[rel].update(bucket)

    return agent


def _sym_to_json(s) -> str:
    if isinstance(s, str): return "s:" + s
    if isinstance(s, int): return "i:" + str(s)
    return "r:" + repr(s)


def _sym_from_json(s: str):
    if s.startswith("s:"): return s[2:]
    if s.startswith("i:"): return int(s[2:])
    return eval(s[2:])  # noqa: S307
