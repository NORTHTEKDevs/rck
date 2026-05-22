"""Sparse-binary relational memory -- v13 substrate.

The dense `RelationalMemory` stores facts as the multiplicative product
of role-bound filler hypervectors, bundled by summation. This module
provides the SPARSE analog using XOR binding + count-based bundling.

Memory layout:
  * Each fact F_i is a sparse binary HV: bind_xor over (role_j, sym_j)
    for all j.
  * The memory is a length-D int32 counter tensor: each stored fact
    increments the counts at its 1-positions.
  * Query: build the "expected fact" for a candidate answer:
        E(a) = bind_xor over known (role_j, sym_j)
               XOR bind(role_unknown, a)
    The answer is the atom whose expected-fact lights up the memory
    the most: score(a) = sum(memory[E(a).positions]) / |E(a)|.

This works because for the TRUE answer the expected fact equals the
stored fact, so every position is hit (counts >= 1). For wrong answers
the expected fact's positions are random vs. the bundle, giving low
hit rate.

Trade vs dense:
  * Memory per fact: O(k) int32 (e.g. k=160) vs O(D) bipolar (D=4096
    -> 4-16x less RAM per fact).
  * Cleanup is O(|codebook| * k) -- LINEAR in codebook for now;
    optimisations (e.g. positional inverted index) are an obvious win.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Hashable, Iterable

import numpy as np

from rck.sparse_hrr import (
    SparseCodebook, SparseHV, bind_sparse, cosine_sparse,
    name_hashed_sparse,
)


@dataclass
class SparseRelationalMemory:
    """Sparse-binary HRR memory analogous to `RelationalMemory`.

    Args:
        dim:        sparse HV dimensionality (e.g. 8192 -- 2x dense for
                    parity).
        k:          number of 1-positions per sparse HV (e.g. 160 = ~2%).
        seed:       master seed for deterministic role HVs.
        role_names: slot names of the relation (e.g. ('S', 'R', 'O')).
    """

    dim: int = 8192
    k: int = 160
    seed: int = 0
    role_names: tuple[str, ...] = ("S", "R", "O", "B")

    _roles: dict[str, SparseHV] = field(default_factory=dict, init=False)
    _counts: np.ndarray = field(default=None, init=False)
    _facts: list[dict[str, Hashable]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        for name in self.role_names:
            self._roles[name] = self._make_role(name)
        self._counts = np.zeros(self.dim, dtype=np.int32)

    def _make_role(self, name: str) -> SparseHV:
        key = f"sparse-rck-role-v1|{self.seed}|{name}".encode("utf-8")
        salt = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "little")
        return name_hashed_sparse(f"__role__::{name}", dim=self.dim, k=self.k, seed=salt)

    def role(self, name: str) -> SparseHV:
        if name not in self._roles:
            self._roles[name] = self._make_role(name)
        return self._roles[name]

    # ---- core ops ----------------------------------------------------------

    def _fact_hv(self, codebook: SparseCodebook,
                 fact: dict[str, Hashable]) -> SparseHV:
        """XOR-bind every (role, filler) pair."""
        out: SparseHV | None = None
        for role_name, symbol in fact.items():
            sym_hv = codebook.encode(symbol)
            bound = bind_sparse(self.role(role_name), sym_hv)
            out = bound if out is None else bind_sparse(out, bound)
        if out is None:
            return SparseHV(dim=self.dim, positions=np.array([], dtype=np.int32))
        return out

    def store(self, codebook: SparseCodebook,
              fact: dict[str, Hashable]) -> None:
        fact_hv = self._fact_hv(codebook, fact)
        self._counts[fact_hv.positions] += 1
        self._facts.append(dict(fact))

    def store_many(self, codebook: SparseCodebook,
                   facts: Iterable[dict[str, Hashable]]) -> None:
        for f in facts:
            self.store(codebook, f)

    def forget(self, codebook: SparseCodebook,
               fact: dict[str, Hashable]) -> None:
        fact_hv = self._fact_hv(codebook, fact)
        # Decrement, clamped at zero.
        np.subtract.at(self._counts, fact_hv.positions, 1)
        np.maximum(self._counts, 0, out=self._counts)
        for i, f in enumerate(self._facts):
            if f == fact:
                del self._facts[i]
                break

    def query(self, codebook: SparseCodebook,
              known: dict[str, Hashable], unknown_role: str,
              top_k: int = 3) -> list[tuple[Hashable, float]]:
        """Return top-K candidate atoms by expected-fact overlap with memory.

        Score: sum of memory counts at the expected fact's 1-positions,
        normalised by the number of positions probed. Empty memory or
        empty codebook returns [].
        """
        if unknown_role in known:
            raise ValueError("unknown_role cannot also be in known")
        if not codebook._atoms or self._counts.sum() == 0:
            return []

        # Pre-compute the "known half" of the expected-fact key.
        known_half: SparseHV | None = None
        for role_name, sym in known.items():
            sym_hv = codebook.encode(sym)
            bound = bind_sparse(self.role(role_name), sym_hv)
            known_half = bound if known_half is None else bind_sparse(known_half, bound)
        role_unknown = self.role(unknown_role)

        # Baseline lit-set: fraction of positions hit by ANY stored fact.
        lit_mask = self._counts > 0
        baseline = float(lit_mask.sum()) / float(self.dim)

        # Score every codebook atom by boolean-overlap of expected_fact
        # against the lit-set, adjusted by baseline. The right answer
        # whose expected_fact == stored F has overlap 1.0; wrong answers
        # have overlap ~= baseline.
        # Atoms that appear in `known` are excluded -- they create
        # XOR-cancellation degeneracies that produce false ties.
        known_atoms = {str(v) for v in known.values()}
        scores: list[tuple[Hashable, float]] = []
        denom = max(1e-6, 1.0 - baseline)
        for sym, atom_hv in codebook._atoms.items():
            if str(sym) in known_atoms:
                continue
            cand_bound = bind_sparse(role_unknown, atom_hv)
            expected = bind_sparse(known_half, cand_bound) if known_half is not None else cand_bound
            if expected.k == 0:
                continue
            raw = float(lit_mask[expected.positions].sum()) / float(expected.k)
            adjusted = max(0.0, (raw - baseline) / denom)
            scores.append((sym, adjusted))
        scores.sort(key=lambda t: -t[1])
        return scores[:top_k]

    def answer(self, codebook: SparseCodebook,
               known: dict[str, Hashable], unknown_role: str
               ) -> tuple[Hashable | None, float]:
        rs = self.query(codebook, known, unknown_role, top_k=1)
        return rs[0] if rs else (None, 0.0)

    # ---- introspection -----------------------------------------------------

    def size(self) -> int:
        return len(self._facts)

    def facts(self) -> list[dict[str, Hashable]]:
        return list(self._facts)

    def memory_bytes(self) -> int:
        return self._counts.nbytes


# ---------------------------------------------------------------------------
#  Sharded sparse KB
# ---------------------------------------------------------------------------

def _shard_index_sparse(subject: str, relation: str, n_shards: int) -> int:
    key = f"{subject}\x00{relation}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "little") % n_shards


@dataclass
class SparseShardedKnowledgeBase:
    """Sharded KB with the sparse-binary HRR substrate.

    Same API as `ShardedKnowledgeBase` but ~6-10x less RAM per fact
    (empirically; depends on D and k).
    """

    dim: int = 8192
    k: int = 160
    n_shards: int = 64
    seed: int = 0

    codebook: SparseCodebook = field(default=None, init=False)
    _shards: list[SparseRelationalMemory] = field(default_factory=list, init=False)
    _fact_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.codebook = SparseCodebook(dim=self.dim, k=self.k, seed=self.seed)
        self._shards = [
            SparseRelationalMemory(
                dim=self.dim, k=self.k, seed=self.seed,
                role_names=("S", "R", "O", "B"),
            )
            for _ in range(self.n_shards)
        ]

    def store(self, fact: dict[str, Hashable]) -> None:
        s, r = str(fact.get("S", "")), str(fact.get("R", ""))
        idx = _shard_index_sparse(s, r, self.n_shards)
        self._shards[idx].store(self.codebook, fact)
        self._fact_count += 1

    def store_many(self, facts: Iterable[dict[str, Hashable]]) -> int:
        n = 0
        for f in facts:
            self.store(f); n += 1
        return n

    def query(self, known: dict[str, Hashable], unknown_role: str,
              top_k: int = 3) -> list[tuple[Hashable, float]]:
        s, r = str(known.get("S", "")), str(known.get("R", ""))
        if s and r:
            idx = _shard_index_sparse(s, r, self.n_shards)
            return self._shards[idx].query(self.codebook, known, unknown_role, top_k=top_k)
        # Fan-out: merge by score.
        all_results: list[tuple[Hashable, float]] = []
        for shard in self._shards:
            all_results.extend(
                shard.query(self.codebook, known, unknown_role, top_k=top_k)
            )
        all_results.sort(key=lambda x: -x[1])
        seen: set = set()
        out: list[tuple[Hashable, float]] = []
        for sym, sc in all_results:
            if sym in seen:
                continue
            seen.add(sym)
            out.append((sym, sc))
            if len(out) >= top_k:
                break
        return out

    def answer(self, known: dict[str, Hashable], unknown_role: str
               ) -> tuple[Hashable | None, float]:
        rs = self.query(known, unknown_role, top_k=1)
        return rs[0] if rs else (None, 0.0)

    def size(self) -> int:
        return self._fact_count

    def shard_sizes(self) -> list[int]:
        return [s.size() for s in self._shards]

    def memory_bytes(self) -> int:
        codebook_bytes = self.codebook.memory_estimate_bytes()
        shard_bytes = sum(s.memory_bytes() for s in self._shards)
        return codebook_bytes + shard_bytes
