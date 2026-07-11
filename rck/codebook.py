"""Atomic hypervector codebook with cleanup memory.

Holds the bipolar atoms for every symbol the agent has ever seen.
Atoms are pseudorandom (almost-orthogonal in high D) and assigned on first
sight -- this is the substrate for one-shot vocabulary learning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from rck.vsa import cosine, random_hv


@dataclass
class Codebook:
    """Maps arbitrary symbols (chars, tokens, concept names) to bipolar HVs.

    New symbols are minted on demand. Cleanup memory recovers the nearest
    symbol from a noisy / composed query hypervector.

    The matrix view of all atoms is CACHED so fast_cleanup can run as one
    matmul. Cache is invalidated whenever a new atom is minted.
    """

    dim: int = 10_000
    seed: int = 0
    _atoms: dict[Hashable, np.ndarray] = field(default_factory=dict)
    _matrix_cache: np.ndarray = field(default=None, init=False)
    _matrix_norms: np.ndarray = field(default=None, init=False)
    _symbols_cache: list = field(default=None, init=False)
    _cache_dirty: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        # Per-symbol-name seeded HVs: two codebooks with the same `seed` will
        # mint identical HVs for the same symbol regardless of insertion
        # order. This is the property federated bundling requires.
        pass

    def _make_atom(self, symbol: Hashable) -> np.ndarray:
        # Use hashlib for a PROCESS-STABLE salt. Python's built-in hash()
        # of strings is randomized between processes since 3.3, which would
        # make HVs non-deterministic across runs.
        import hashlib
        key = f"rck-codebook-v1|{self.seed}|{symbol!r}".encode("utf-8")
        salt = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "little")
        rng = np.random.default_rng(salt)
        return random_hv(self.dim, rng)

    def encode(self, symbol: Hashable) -> np.ndarray:
        """Get the HV for a symbol, minting a name-seeded HV if unseen."""
        if symbol not in self._atoms:
            self._atoms[symbol] = self._make_atom(symbol)
            self._cache_dirty = True
        return self._atoms[symbol]

    def has(self, symbol: Hashable) -> bool:
        return symbol in self._atoms

    def cleanup(self, query: np.ndarray, top_k: int = 1) -> list[tuple[Hashable, float]]:
        """Return top_k nearest atoms to query, with cosine similarity."""
        if not self._atoms:
            return []
        scores = [(sym, cosine(query, hv)) for sym, hv in self._atoms.items()]
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]

    def cleanup_one(self, query: np.ndarray) -> tuple[Hashable, float] | None:
        results = self.cleanup(query, top_k=1)
        return results[0] if results else None

    def size(self) -> int:
        return len(self._atoms)

    def symbols(self) -> list[Hashable]:
        return list(self._atoms.keys())

    def _refresh_cache(self) -> None:
        syms = list(self._atoms.keys())
        if not syms:
            self._matrix_cache = np.empty((0, self.dim), dtype=np.float32)
            self._matrix_norms = np.empty(0, dtype=np.float32)
            self._symbols_cache = []
        else:
            mat = np.stack([self._atoms[s] for s in syms], axis=0).astype(np.float32)
            self._matrix_cache = mat
            self._matrix_norms = np.linalg.norm(mat, axis=1) + 1e-12
            self._symbols_cache = syms
        self._cache_dirty = False

    def matrix(self) -> tuple[np.ndarray, list[Hashable]]:
        """Stack of all atoms as a (n_symbols, dim) array + symbol list."""
        if self._cache_dirty:
            self._refresh_cache()
        return self._matrix_cache, self._symbols_cache

    def cleanup_among(self, query: np.ndarray, symbols: list,
                      top_k: int = 1) -> list[tuple[Hashable, float]]:
        """Cleanup restricted to an explicit candidate set.

        Cost is O(len(symbols) * dim) regardless of vocabulary size --
        this is what makes per-query cost independent of total KB
        vocabulary when the caller knows which symbols could possibly
        match (e.g. a shard's own fact-log symbols)."""
        if not symbols:
            return []
        mat = np.stack([self.encode(s) for s in symbols]
                       ).astype(np.float32)
        q = query.astype(np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return []
        norms = np.linalg.norm(mat, axis=1) + 1e-12
        scores = (mat @ q) / (norms * qn)
        k = min(top_k, len(symbols))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(symbols[i], float(scores[i])) for i in idx]

    def fast_cleanup(self, query: np.ndarray, top_k: int = 1) -> list[tuple[Hashable, float]]:
        """Vectorized cleanup using a cached normalised codebook matrix."""
        if self._cache_dirty:
            self._refresh_cache()
        if not self._symbols_cache:
            return []
        q = query.astype(np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return []
        scores = (self._matrix_cache @ q) / (self._matrix_norms * qn)
        idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        syms = self._symbols_cache
        return [(syms[i], float(scores[i])) for i in idx]
