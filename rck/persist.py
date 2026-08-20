"""Versioned save/load for RCKAgent.

v1.0 ships an explicit save format instead of relying on pickle. We persist
weights as a single .npz archive plus a JSON sidecar with hyperparameters
and codebook keys. Loading is forward-compatible: future versions can read
v1 archives by ignoring unknown keys.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from rck.agent import RCKAgent
from rck.atomic import atomic_write_bytes, atomic_write_text


SCHEMA_VERSION = 1


def save(agent: RCKAgent, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Header / hyperparameters.
    meta = {
        "schema": SCHEMA_VERSION,
        "version": "1.0.0",
        "hyper": {
            "vocab_size": agent.vocab_size,
            "hv_dim": agent.hv_dim,
            "n_columns": agent.n_columns,
            "reservoir_dim": agent.reservoir_dim,
            "n_clauses": agent.n_clauses,
            "fep_rank": agent.fep_rank,
            "bigram_order": agent.bigram_order,
            "seed": agent.seed,
            "sampling_temperature": agent.sampling_temperature,
            "stochastic_decode": agent.stochastic_decode,
        },
        # Symbols are stored as their string repr; non-string keys are
        # serialised as JSON-encoded representations.
        "symbols": [_sym_to_json(s) for s in agent.codebook.symbols()],
        "symbol_ids": {_sym_to_json(s): i for s, i in agent._symbol_to_id.items()},
        "position": agent._position,
    }
    atomic_write_text(path.with_suffix(".json"), json.dumps(meta, indent=2))

    # Arrays.
    arrays: dict[str, np.ndarray] = {
        "codebook_matrix": np.stack(
            [agent.codebook.encode(s) for s in agent.codebook.symbols()]
        ) if agent.codebook.size() > 0 else np.empty((0, agent.hv_dim), dtype=np.int8),
        "pcn_projector": agent.pcn._projector,
        "lsm_W_in": agent.lsm._W_in,
        "lsm_W_rec": agent.lsm._W_rec,
        "lsm_W_out": agent.lsm._W_out,
        "lsm_P": agent.lsm._P,
        "lsm_state": agent.lsm._state,
        "fep_U": agent.fep._U,
        "fep_V": agent.fep._V,
        "ts_state": agent.tsetlin._ta,
        "ws_state": agent.workspace._ws,
        "bigram_mem": agent.bigram._mem,
        "last_ws": agent._last_ws,
    }
    # PCN weight matrices (variable count).
    for i, W in enumerate(agent.pcn._weights):
        arrays[f"pcn_W_{i}"] = W
    arrays["pcn_n_layers"] = np.array([len(agent.pcn._weights)], dtype=np.int32)

    # Columns: per-column PCN + LSM weights.
    for k, col in enumerate(agent.columns._columns):
        arrays[f"col_{k}_pcn_projector"] = col.pcn._projector
        for i, W in enumerate(col.pcn._weights):
            arrays[f"col_{k}_pcn_W_{i}"] = W
        arrays[f"col_{k}_lsm_W_in"] = col.lsm._W_in
        arrays[f"col_{k}_lsm_W_rec"] = col.lsm._W_rec
        arrays[f"col_{k}_lsm_W_out"] = col.lsm._W_out
        arrays[f"col_{k}_lsm_P"] = col.lsm._P
        arrays[f"col_{k}_lsm_state"] = col.lsm._state

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_write_bytes(path.with_suffix(".npz"), buf.getvalue())


def load(path: str | Path) -> RCKAgent:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema {meta.get('schema')}; expected {SCHEMA_VERSION}")
    hyper = meta["hyper"]
    agent = RCKAgent(**hyper)

    # Close the NpzFile explicitly (not just let it be GC'd) -- on Windows
    # an open zip-file handle makes a later os.replace of this same path
    # raise PermissionError [WinError 5].
    arrs = np.load(path.with_suffix(".npz"))
    # Restore codebook.
    symbols = [_sym_from_json(s) for s in meta["symbols"]]
    if symbols:
        mat = arrs["codebook_matrix"]
        for i, sym in enumerate(symbols):
            agent.codebook._atoms[sym] = mat[i].astype(np.int8)
        agent.codebook._cache_dirty = True
    # Restore symbol ID map.
    for s_json, i in meta["symbol_ids"].items():
        sym = _sym_from_json(s_json)
        agent._symbol_to_id[sym] = i
        agent._id_to_symbol[i] = sym
    agent._position = meta.get("position", 0)

    # PCN.
    n_layers = int(arrs["pcn_n_layers"][0])
    agent.pcn._weights = [arrs[f"pcn_W_{i}"].astype(np.float32) for i in range(n_layers)]
    agent.pcn._projector = arrs["pcn_projector"].astype(np.float32)

    # LSM.
    agent.lsm._W_in = arrs["lsm_W_in"].astype(np.float32)
    agent.lsm._W_rec = arrs["lsm_W_rec"].astype(np.float32)
    agent.lsm._W_out = arrs["lsm_W_out"].astype(np.float32)
    agent.lsm._P = arrs["lsm_P"].astype(np.float32)
    agent.lsm._state = arrs["lsm_state"].astype(np.float32)

    # FEP.
    agent.fep._U = arrs["fep_U"].astype(np.float32)
    agent.fep._V = arrs["fep_V"].astype(np.float32)

    # Tsetlin + Workspace + Bigram.
    agent.tsetlin._ta = arrs["ts_state"].astype(np.int32)
    agent.workspace._ws = arrs["ws_state"].astype(np.float32)
    agent.bigram._mem = arrs["bigram_mem"].astype(np.float32)
    agent._last_ws = arrs["last_ws"].astype(np.int8)

    # Columns.
    for k, col in enumerate(agent.columns._columns):
        col.pcn._projector = arrs[f"col_{k}_pcn_projector"].astype(np.float32)
        # Recover its PCN layer count by counting matching arrays.
        col_layers = []
        i = 0
        while f"col_{k}_pcn_W_{i}" in arrs.files:
            col_layers.append(arrs[f"col_{k}_pcn_W_{i}"].astype(np.float32))
            i += 1
        col.pcn._weights = col_layers
        col.lsm._W_in = arrs[f"col_{k}_lsm_W_in"].astype(np.float32)
        col.lsm._W_rec = arrs[f"col_{k}_lsm_W_rec"].astype(np.float32)
        col.lsm._W_out = arrs[f"col_{k}_lsm_W_out"].astype(np.float32)
        col.lsm._P = arrs[f"col_{k}_lsm_P"].astype(np.float32)
        col.lsm._state = arrs[f"col_{k}_lsm_state"].astype(np.float32)

    arrs.close()
    return agent


# ---- symbol json encoding ---------------------------------------------------
# Codebook supports arbitrary hashable keys. For round-trip we serialise via
# repr; only printable strings + ints + tuples thereof are well supported.

def _sym_to_json(s) -> str:
    if isinstance(s, str):
        return "s:" + s
    if isinstance(s, int):
        return "i:" + str(s)
    return "r:" + repr(s)


def _sym_from_json(s: str):
    if s.startswith("s:"):
        return s[2:]
    if s.startswith("i:"):
        return int(s[2:])
    return eval(s[2:])  # noqa: S307 -- only used on data we wrote
