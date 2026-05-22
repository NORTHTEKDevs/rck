import numpy as np

from rck.codebook import Codebook


def test_encode_is_stable():
    cb = Codebook(dim=1024, seed=42)
    h1 = cb.encode("a")
    h2 = cb.encode("a")
    assert np.array_equal(h1, h2)


def test_new_symbol_mints_new_hv():
    cb = Codebook(dim=2048, seed=42)
    cb.encode("a")
    cb.encode("b")
    assert cb.size() == 2


def test_cleanup_finds_planted_symbol():
    cb = Codebook(dim=4096, seed=0)
    for ch in "abcdefghij":
        cb.encode(ch)
    target = cb.encode("e")
    # Add small noise to target.
    rng = np.random.default_rng(7)
    noise_mask = rng.random(4096) < 0.05
    noisy = target.copy()
    noisy[noise_mask] = -noisy[noisy.shape[0] - 1 if False else noise_mask]
    # That last expression had a typo. Just flip 5% randomly:
    flip = rng.random(4096) < 0.05
    noisy = target.copy()
    noisy[flip] = -noisy[flip]
    sym, score = cb.cleanup_one(noisy)
    assert sym == "e"
    assert score > 0.5


def test_fast_cleanup_matches_naive():
    cb = Codebook(dim=2048, seed=1)
    for ch in "hello world":
        cb.encode(ch)
    q = cb.encode("o")
    naive = cb.cleanup(q, top_k=3)
    fast = cb.fast_cleanup(q, top_k=3)
    # Top-1 must match in both; full top-3 may differ on score ties.
    assert naive[0][0] == fast[0][0]
    assert {s for s, _ in naive} == {s for s, _ in fast}
