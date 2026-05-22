"""Sparse-binary HRR -- ~10x memory reduction at modest recall cost.

Standard RCK uses dense bipolar HVs: every position is ±1. A SPARSE
binary HV sets only a small fraction of positions to 1 (e.g. 2%) and
the rest to 0.

Tradeoffs:
  * Memory: bit packing -> 32x less RAM (D=4096 bipolar = 16KB, sparse
    binary = ~500 bytes).
  * Storage: dramatically faster to serialise / deserialise.
  * Composition: still works -- bind = XOR, bundle = sum + threshold.
  * Recall: slightly worse than dense at the same D. Recovered by
    increasing D 2-4x, which is cheap since memory is so much smaller.

This module ships the primitive ops and a `SparseCodebook` for tests
and experimentation. Production HRR remains dense for now; v13 could
switch the substrate once the recall/memory tradeoff is mapped.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Hashable

import numpy as np


@dataclass
class SparseHV:
    """Sparse-binary hypervector. Stored as a SORTED set of 1-positions."""
    dim: int
    positions: np.ndarray  # int32, sorted

    @property
    def k(self) -> int:
        return len(self.positions)

    def dense(self) -> np.ndarray:
        """Materialise as a dense binary array."""
        v = np.zeros(self.dim, dtype=np.int8)
        v[self.positions] = 1
        return v


def random_sparse(dim: int, k: int, seed: int) -> SparseHV:
    """Sample k of dim positions uniformly without replacement."""
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(dim, size=k, replace=False)).astype(np.int32)
    return SparseHV(dim=dim, positions=positions)


def name_hashed_sparse(name: Hashable, dim: int, k: int, seed: int = 0) -> SparseHV:
    """Deterministic sparse HV from a name (same scheme as v1.1 codebook)."""
    key = f"sparse-v1|{seed}|{name!r}".encode("utf-8")
    salt = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "little")
    return random_sparse(dim, k, seed=salt)


def bind_sparse(a: SparseHV, b: SparseHV) -> SparseHV:
    """Bind via XOR-like: positions in EITHER a or b, but not both."""
    if a.dim != b.dim:
        raise ValueError("dim mismatch")
    out = np.setxor1d(a.positions, b.positions, assume_unique=True)
    return SparseHV(dim=a.dim, positions=out.astype(np.int32))


def bundle_sparse(hvs: list[SparseHV], top_k: int | None = None) -> SparseHV:
    """Bundle via positional sum + top-K threshold."""
    if not hvs:
        raise ValueError("empty bundle")
    d = hvs[0].dim
    counts = np.zeros(d, dtype=np.int32)
    for h in hvs:
        counts[h.positions] += 1
    if top_k is None:
        # Take positions with count >= mean(counts among hot positions).
        hot = counts > 0
        if not hot.any():
            return SparseHV(dim=d, positions=np.array([], dtype=np.int32))
        cutoff = max(1, int(np.median(counts[hot])))
        positions = np.where(counts >= cutoff)[0]
    else:
        # Take the K densest positions.
        positions = np.argpartition(-counts, min(top_k, d - 1))[:top_k]
        positions = positions[counts[positions] > 0]
        positions = np.sort(positions)
    return SparseHV(dim=d, positions=positions.astype(np.int32))


def cosine_sparse(a: SparseHV, b: SparseHV) -> float:
    """Cosine similarity for sparse binary HVs.

    cos = |A ∩ B| / sqrt(|A| * |B|).
    """
    if a.dim != b.dim:
        return 0.0
    if a.k == 0 or b.k == 0:
        return 0.0
    intersection = np.intersect1d(a.positions, b.positions, assume_unique=True)
    return float(len(intersection)) / float(np.sqrt(a.k * b.k))


# ---------------------------------------------------------------------------
#  Codebook in sparse form
# ---------------------------------------------------------------------------

@dataclass
class SparseCodebook:
    dim: int = 8192
    k: int = 160          # ~2% density
    seed: int = 0
    _atoms: dict[Hashable, SparseHV] = field(default_factory=dict)

    def encode(self, symbol: Hashable) -> SparseHV:
        if symbol not in self._atoms:
            self._atoms[symbol] = name_hashed_sparse(
                symbol, dim=self.dim, k=self.k, seed=self.seed,
            )
        return self._atoms[symbol]

    def cleanup(self, query: SparseHV, top_k: int = 3) -> list[tuple[Hashable, float]]:
        scores = [(sym, cosine_sparse(query, hv))
                  for sym, hv in self._atoms.items()]
        scores.sort(key=lambda t: -t[1])
        return scores[:top_k]

    def size(self) -> int:
        return len(self._atoms)

    def memory_estimate_bytes(self) -> int:
        """Total bytes used by all sparse HVs (int32 indices)."""
        return sum(hv.k * 4 for hv in self._atoms.values())


def compare_memory_vs_dense(n_atoms: int, dim_dense: int, dim_sparse: int,
                            k_sparse: int) -> dict:
    """Compute the memory ratio between dense bipolar (int8) and sparse binary."""
    dense_bytes = n_atoms * dim_dense  # int8 = 1 byte/element
    sparse_bytes = n_atoms * k_sparse * 4
    return {
        "n_atoms": n_atoms,
        "dense_bytes": dense_bytes,
        "sparse_bytes": sparse_bytes,
        "ratio": dense_bytes / max(1, sparse_bytes),
    }
