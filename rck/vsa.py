"""Vector Symbolic Architecture ops over bipolar {-1, +1} hypervectors.

All operations preserve the bipolar property (after sign).
Binding is self-inverse: bind(bind(a, b), b) == a.
"""
from __future__ import annotations

import numpy as np


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise multiply. Self-inverse for bipolar HVs."""
    return a * b


def unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Same as bind for bipolar (b * b = 1)."""
    return a * b


def bundle(*hvs: np.ndarray, keep_real: bool = False) -> np.ndarray:
    """Superposition via sum + sign. Returns bipolar HV unless keep_real=True.

    Ties (sum == 0) broken by +1, which is fine in expectation.
    """
    if not hvs:
        raise ValueError("bundle needs at least one hypervector")
    s = np.sum(np.stack(hvs, axis=0), axis=0)
    if keep_real:
        return s
    out = np.sign(s)
    out[out == 0] = 1
    return out.astype(np.int8)


def permute(hv: np.ndarray, n: int = 1) -> np.ndarray:
    """Cyclic shift by n positions. Used to encode sequence order / role."""
    return np.roll(hv, n)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. For bipolar HVs equals (D - 2*hamming)/D."""
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(af, bf) / (na * nb))


def binarize(x: np.ndarray) -> np.ndarray:
    """Project real-valued vector to bipolar {-1, +1}."""
    out = np.sign(x)
    out[out == 0] = 1
    return out.astype(np.int8)


def random_hv(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a single bipolar hypervector uniformly."""
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=dim)
