"""Content hash of substrate state -- everything that can change an
`ask_with_idk` answer, and nothing that cannot.

Hashed, in fixed order:
  1. hyper dict {dim, seed, n_shards} as canonical JSON (sort_keys=True).
  2. Each knowledge shard's `_memory.tobytes()`, in shard index order.
  3. The canonical fact list, in stored order (shard index order, then
     within-shard insertion order).

Excluded, deliberately (see docs/plans/2026-08-19-replay.md):
  - The codebook: atoms are minted deterministically from
    blake2b(seed, symbol), order-independent.
  - The belief KB / belief_n_shards [R2]: idk_detection.ask_with_idk
    only ever calls ShardedKnowledgeBase.query on `agent.knowledge`,
    never touches `agent.beliefs`. Hashing belief activity would
    invalidate records over provably-irrelevant churn (measured: 300
    tell_belief calls grow the belief KB 4 -> 8 shards while
    `knowledge` stays byte-identical) -- a false STATE_MISMATCH.
  - The score calibrator, synonyms, query memory, chain cache: none
    are read by `ask_with_idk`.
  - Wall-clock values (e.g. `saved_at`): would make every hash unique.

Trap 1: do NOT hash snapshot files -- `meta.json` embeds a wall-clock
timestamp and `np.savez_compressed` output is not guaranteed
byte-stable. Hash the CONTENT (this module), not the file bytes.

Trap 2: bundling is float addition, which is not associative. Facts
fed in a different order produce a different `_memory`, so this hash
is insertion-order-sensitive by construction -- that is the intended,
tested behaviour, not a bug.

Trap 3: `_memory` is float32. Assert the dtype and force C-contiguity
before `tobytes()`, or the hash could silently depend on memory
layout rather than content.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def state_hash(agent) -> str:
    """SHA-256 hex digest over everything that can change an
    `ask_with_idk` answer.

    [Phase 2] On backend="dict" there is no `_memory` tensor -- the hash
    is hyper (now including "backend") plus the canonical fact list
    only. The two backends therefore hash DIFFERENTLY for the same
    logical facts. That is correct, not a bug: a DecisionRecord pins a
    substrate state, and the substrates genuinely differ (see
    tests/test_snapshot_hash.py for the explicit cross-backend
    inequality assertion).
    """
    h = hashlib.sha256()

    hyper = {
        "dim": agent.dim,
        "seed": agent.seed,
        "n_shards": agent.knowledge.n_shards,
        "backend": agent.backend,
    }
    h.update(json.dumps(hyper, sort_keys=True).encode("utf-8"))

    if agent.backend != "dict":
        for shard in agent.knowledge._shards:
            mem = shard._memory
            assert mem.dtype == np.float32, (
                f"_memory dtype drifted from float32: {mem.dtype}")
            mem = np.ascontiguousarray(mem)
            h.update(mem.tobytes())

    for shard in agent.knowledge._shards:
        for fact in shard._facts:
            h.update(json.dumps(fact, sort_keys=True).encode("utf-8"))
            h.update(b"\x00")

    return h.hexdigest()
